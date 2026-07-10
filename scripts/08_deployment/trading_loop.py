"""
Paper-Trading-Loop – Lokaler PC (kein Server).

Minütliche Bewertung aller NASDAQ-100 Aktien mit dem Ensemble-Modell.
Jede Iteration:
  1. Neueste 1-Min-Bars von Alpaca holen
  2. 82 Features berechnen (via FeatureEngine)
  3. 5 Modelle inferieren -> Ensemble-Signal
  4. Exit-Checks (TP/SL/Time-Stop)
  5. Neue Trades eröffnen (max. 3 gleichzeitig)
  6. Alles loggen

Aufruf:
  python trading_loop.py --dry_run          # Keine Orders (Test)
  python trading_loop.py --paper            # Paper Trading
  python trading_loop.py --paper --symbols 20  # Nur Top-20 Symbole
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import signal as unix_signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
import joblib
import lightgbm as lgb
from pytz import timezone as pytz_timezone

# Pfad-Setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_SCRIPT_DIR.parent / "06_model_training"))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "03_pre_split_prep"))

from model import BreakoutModel
from model_sequential import LSTMBreakoutModel, GRUBreakoutModel, CNNBreakoutModel
from utils import get_device, enable_amp, load_model, PROJECT_ROOT as UTIL_ROOT

from feature_engine import MultiSymbolEngine
from data_logger import DataLogger
from account_manager import AccountManager

# ---- Konstanten ----

EASTERN = pytz_timezone("US/Eastern")
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
MARKET_TIMEZONE = "US/Eastern"

SEQ_LEN = 30
FINDER_NAMES = ["lstm", "gru", "cnn", "lightgbm"]
FINDER_THRESHOLDS = {"lstm": 0.32, "gru": 0.334, "cnn": 0.314, "lightgbm": 0.355}

# ---- Signal-Qualitaets-Filter (Meta-Labeling nach Lopez de Prado) ----
# Statt einfachem Threshold: Quality-Score aus MLP-Confidence + Finder-Agreement.
# Research: Entropy-Filtered ML (Serban & Vrinceanu, 2026) + Meta-Labeling
# Nur Top-2 Signale pro Minute werden gehandelt (Qualitaet > Quantitaet)
# Keine kuenstlichen Caps — der Quality-Floor + Regime-Parameter regeln die
# Anzahl auf natuerliche Weise. Wenn 50 Signale mit Quality>0.60 reinkommen,
# sollen die auch gehandelt werden. (Die Realitaet: das passiert nicht.)
FILTER_THRESHOLD = 0.85       # MLP Gate: hoehere Huerde (0.75 -> 0.85)
MAX_SIGNALS_PER_MINUTE = 2     # Top-N Ranking statt Threshold-Filter
FINDER_STD_MAX = 0.08          # Max. StdDev der 4 Finder-Probs (strenger: 0.10 -> 0.08)
MIN_QUALITY_SCORE = 0.35       # Default Quality-Floor (pro Regime ueberschreibbar)
MAX_DAILY_TRADES = 50          # Reines Safety-Net (kein kuenstliches Cap)

# Finder-Gewichte (normalisiert aus F1)
_F1 = {"lstm": 0.642, "gru": 0.647, "cnn": 0.645, "lightgbm": 0.623}
_F1_SUM = sum(_F1.values())
FINDER_WEIGHTS = {k: v / _F1_SUM for k, v in _F1.items()}

# Sektor-Mapping fuer Konzentrations-Limits (NASDAQ-100)
SECTOR_MAP = {
    "AMD": "semis", "INTC": "semis", "NVDA": "semis", "QCOM": "semis",
    "MRVL": "semis", "LRCX": "semis", "AMAT": "semis", "KLAC": "semis",
    "MU": "semis", "MCHP": "semis", "ON": "semis", "GFS": "semis",
    "NXPI": "semis", "ADI": "semis", "TXN": "semis", "AVGO": "semis",
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "META": "tech",
    "CRWD": "security", "ZS": "security", "PANW": "security", "FTNT": "security",
    "DDOG": "software", "TEAM": "software", "WDAY": "software", "ADBE": "software",
    "RIVN": "ev", "LCID": "ev", "TSLA": "ev",
    "MRNA": "biotech", "ALGN": "biotech", "REGN": "biotech", "VRTX": "biotech",
    "ENPH": "energy", "FSLR": "energy",
    "CHTR": "telecom", "CSGP": "realestate", "CSCO": "networking",
}
MAX_POSITIONS_PER_SECTOR = 2   # Max. 2 Positionen pro Sektor gleichzeitig


# ---- CLI ----

def parse_args():
    p = argparse.ArgumentParser(description="USW-TradingModel - Paper Trading Loop")
    p.add_argument("--dry_run", action="store_true",
                   help="Nur Signale generieren, KEINE Orders senden")
    p.add_argument("--paper", action="store_true", default=True,
                   help="Paper-Trading-Modus (default)")
    p.add_argument("--symbols", type=int, default=None,
                   help="Anzahl Symbole (default: alle)")
    p.add_argument("--max_positions", type=int, default=3,
                   help="Max. gleichzeitige Positionen (default: 3)")
    p.add_argument("--risk", type=float, default=0.005,
                   help="Max. Risk pro Trade (default: 0.5%%)")
    p.add_argument("--tp_pct", type=float, default=0.005,
                   help="Take-Profit in %% (default: 0.5%%)")
    p.add_argument("--sl_pct", type=float, default=0.0060,
                   help="Stop-Loss in %% (default: 0.60%% -> R:R ~1:0.83)")
    p.add_argument("--trailing_sl", action="store_true", default=False,
                   help="Trailing Stop Loss AKTIVIEREN (default: AUS)")
    p.add_argument("--no_ratchet", action="store_true",
                   help="Ratchet-Mode deaktivieren (klassischer TP-Exit)")
    p.add_argument("--reentry_cooldown", type=int, default=15,
                   help="Wash-Trade-Sperre in Minuten nach Exit (default: 15)")
    p.add_argument("--sl_time_decay_target", type=float, default=0.0035,
                   help="SL am Ende der Time-Stop-Frist (default: 0.35%%)")
    p.add_argument("--sl_time_decay_grace", type=int, default=10,
                   help="Gnadenfrist in Min bevor Time-Decay einsetzt (default: 10)")
    p.add_argument("--trailing_sl_pct", type=float, default=0.008,
                   help="Trailing-Stop Start-Abstand bei Entry (default: 0.8%%)")
    p.add_argument("--trailing_min_pct", type=float, default=0.004,
                   help="Trailing-Stop Minimal-Abstand bei vollem Profit (default: 0.4%%)")
    p.add_argument("--trailing_ramp_pct", type=float, default=0.008,
                   help="Profit-Level fuer vollen Trail-Lock (default: 0.8%%)")
    p.add_argument("--no_entry_rules", action="store_true",
                   help="Entry-Rules E3-E5 deaktivieren")
    p.add_argument("--position_size_pct", type=float, default=0.05,
                   help="Anteil des Kapitals pro Position (default: 0.05 = 5%%)")
    p.add_argument("--once", action="store_true",
                   help="Nur EINE Iteration ausfuehren und beenden")
    p.add_argument("--start_hour", type=int, default=15,
                   help="Start-Stunde MESZ (default: 15 = 15:30 MESZ)")
    p.add_argument("--max_daily_trades", type=int, default=MAX_DAILY_TRADES,
                   help=f"Safety-Net: Max. Trades pro Tag (default: {MAX_DAILY_TRADES}, rein technische Notbremse)")
    p.add_argument("--mlp_threshold", type=float, default=0.85,
                   help="MLP-Gate-Schwelle (default: 0.85)")
    p.add_argument("--min_quality", type=float, default=MIN_QUALITY_SCORE,
                   help=f"Quality-Floor: Signale darunter werden ignoriert (default: {MIN_QUALITY_SCORE})")
    p.add_argument("--grace_sl_pct", type=float, default=0.015,
                   help="Grace Stufe 1: SL in %% (0-3 Min, default: 1.5%%)")
    p.add_argument("--grace_sl_pct_t2", type=float, default=0.010,
                   help="Grace Stufe 2: SL in %% (3-5 Min, default: 1.0%%)")
    p.add_argument("--entry_grace_minutes", type=int, default=3,
                   help="Grace Stufe 1: Dauer in Minuten (default: 3)")
    p.add_argument("--entry_grace_t2_minutes", type=int, default=5,
                   help="Grace Stufe 2: Ende in Minuten kumulativ (default: 5)")
    return p.parse_args()


# ---- Alpaca Setup ----

def load_alpaca_clients():
    """Initialisiert Alpaca Paper-Trading Clients."""
    keys_path = _PROJECT_ROOT / "conf" / "keys.yaml"
    with open(keys_path) as f:
        keys = yaml.safe_load(f)

    api_key = os.getenv("ALPACA_API_KEY", keys.get("ALPACA", {}).get("API_KEY", ""))
    secret_key = os.getenv("ALPACA_SECRET_KEY", keys.get("ALPACA", {}).get("SECRET_KEY", ""))

    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient

    trading = TradingClient(api_key, secret_key, paper=True)
    data = StockHistoricalDataClient(api_key, secret_key)

    return trading, data


def load_symbols(max_symbols: int = None) -> list[str]:
    """Lädt NASDAQ-100 Symbolliste."""
    sym_path = _PROJECT_ROOT / "data" / "nasdaq100_symbols.csv"
    df = pd.read_csv(sym_path)
    symbols = df["symbol"].tolist() if "symbol" in df.columns else df.iloc[:, 0].tolist()
    if max_symbols:
        symbols = symbols[:max_symbols]
    return symbols


def download_warmup_bars(data_client, symbols: list[str], days: int = 5) -> dict[str, pd.DataFrame]:
    """Lädt die letzten N Handelstage 1-Min-Bars für alle Symbole."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 5)  # Extra-Puffer für Wochenenden

    print(f"\n[Warmup] Lade ~{days} Handelstage 1-Min-Bars fuer {len(symbols)} Symbole...")

    warmup = {}
    for i, sym in enumerate(symbols):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                adjustment="all",
                feed="iex",
            )
            bars = data_client.get_stock_bars(request)
            df = bars.df

            if hasattr(df, 'index') and isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()

            if df.empty:
                continue

            # RTH-Filter (09:30-16:00 ET)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(EASTERN)
                else:
                    df['timestamp'] = df['timestamp'].dt.tz_convert(EASTERN)
                df = df[
                    (df['timestamp'].dt.time >= pd.to_datetime("09:30").time()) &
                    (df['timestamp'].dt.time < pd.to_datetime("16:00").time())
                ]

            warmup[sym] = df

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(symbols)}] {sym:5s} ({len(df)} Bars)")

        except Exception as e:
            print(f"  [WARN] {sym}: Fehler beim Warmup-Download: {e}")

    print(f"  {len(warmup)} Symbole geladen")
    return warmup


def fetch_latest_bars(data_client, symbols: list[str]) -> dict:
    """Holt die neuesten 1-Min-Bars fuer alle Symbole (ein API-Call)."""
    from alpaca.data.requests import StockLatestBarRequest

    try:
        request = StockLatestBarRequest(symbol_or_symbols=symbols)
        response = data_client.get_stock_latest_bar(request)

        bars = {}
        for sym, bar in response.items():
            bars[sym] = {
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "vwap": float(bar.vwap),
            }
        return bars
    except Exception as e:
        print(f"[ERROR] fetch_latest_bars: {e}")
        return {}


def fetch_current_prices(data_client, symbols: list[str]) -> dict[str, float]:
    """Holt aktuelle Preise (fuer Exit-Checks)."""
    bars = fetch_latest_bars(data_client, symbols)
    return {sym: b["close"] for sym, b in bars.items()}


# ---- Model Loading ----

def load_ensemble_models(n_features: int, device: torch.device) -> dict:
    """Lädt alle 5 Modelle."""
    models = {}

    # MLP
    mlp = BreakoutModel(n_features, hidden_sizes=(128,64,32,16), dropout=0.4)
    mlp, _ = load_model(mlp, "mlp_model", device)
    mlp.eval()
    models["mlp"] = mlp

    # LSTM
    lstm = LSTMBreakoutModel(n_features, hidden_size=128, num_layers=2, dropout=0.35, bidirectional=True)
    lstm, _ = load_model(lstm, "lstm_model", device)
    lstm.eval()
    models["lstm"] = lstm

    # GRU
    gru = GRUBreakoutModel(n_features, hidden_size=128, num_layers=2, dropout=0.35)
    gru, _ = load_model(gru, "gru_model", device)
    gru.eval()
    models["gru"] = gru

    # CNN
    cnn = CNNBreakoutModel(n_features, hidden_channels=64, kernel_sizes=(3,5,10), dropout=0.35)
    cnn, _ = load_model(cnn, "cnn_model", device)
    cnn.eval()
    models["cnn"] = cnn

    # LightGBM
    lgb_path = _PROJECT_ROOT / "artifacts" / "models" / "lightgbm_model.txt"
    models["lightgbm"] = lgb.Booster(model_file=str(lgb_path))

    return models


# ---- Ensemble Prediction ----

def ensemble_predict(models: dict, features: np.ndarray, sequence: np.ndarray,
                     device: torch.device) -> tuple[int, float, dict]:
    """Finder Majority + MLP Gate Ensemble.

    Args:
        models: dict of loaded models
        features: (82,) float32 array (aktueller Feature-Vektor, via Scaler transformiert)
        sequence: (30, 82) float32 array (letzte 30 Feature-Vektoren) oder None
        device: torch device

    Returns:
        (signal: 0/1, ensemble_score: float, individual_probs: dict)
    """
    # Flat-Modelle: predict direkt
    x_flat = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, 82)

    with torch.no_grad():
        p_mlp = float(models["mlp"](x_flat).squeeze().cpu().numpy())

    p_lgb = float(models["lightgbm"].predict(features.reshape(1, -1))[0])

    # Sequenz-Modelle: brauchen (1, 30, 82) aus dem Sequence-Buffer
    if sequence is not None:
        x_seq = torch.from_numpy(sequence).unsqueeze(0).to(device)  # (1, 30, 82)

        with torch.no_grad():
            logits_lstm = models["lstm"](x_seq)
            p_lstm = float(torch.sigmoid(logits_lstm).squeeze().cpu().numpy())

            logits_gru = models["gru"](x_seq)
            p_gru = float(torch.sigmoid(logits_gru).squeeze().cpu().numpy())

            logits_cnn = models["cnn"](x_seq)
            p_cnn = float(torch.sigmoid(logits_cnn).squeeze().cpu().numpy())
    else:
        # Fallback wenn Sequence-Buffer noch nicht gefuellt
        p_lstm = p_lgb  # Approximiere mit LightGBM-Wert
        p_gru = p_lgb
        p_cnn = p_lgb

    probs = {"mlp": p_mlp, "lstm": p_lstm, "gru": p_gru, "cnn": p_cnn, "lightgbm": p_lgb}

    # Finder Majority + MLP Gate
    finder_votes = (
        (1 if p_lstm > FINDER_THRESHOLDS["lstm"] else 0) +
        (1 if p_gru > FINDER_THRESHOLDS["gru"] else 0) +
        (1 if p_cnn > FINDER_THRESHOLDS["cnn"] else 0) +
        (1 if p_lgb > FINDER_THRESHOLDS["lightgbm"] else 0)
    )
    finder_score = (
        FINDER_WEIGHTS["lstm"] * p_lstm +
        FINDER_WEIGHTS["gru"] * p_gru +
        FINDER_WEIGHTS["cnn"] * p_cnn +
        FINDER_WEIGHTS["lightgbm"] * p_lgb
    )
    mlp_confirms = p_mlp > FILTER_THRESHOLD
    signal = 1 if (finder_votes >= 2 and mlp_confirms) else 0

    return signal, finder_score, probs


# ---- Signal Quality Scoring (Meta-Labeling + Entropy Filter) ----------

def compute_signal_quality(probs: dict, finder_score: float) -> float:
    """Berechnet Quality-Score fuer ein Signal (0..1, hoeher = besser).

    Research-Basis:
      - Meta-Labeling (Lopez de Prado): Secondary model confidence gewichtet primary signal
      - Entropy-Filtered ML (Serban & Vrinceanu, 2026): Low model disagreement = reliable
      - Hermes-Trader Staged Filtering: Combine multiple quality dimensions

    Formel:
      quality = MLP_confidence * ensemble_score * (1 - 0.5 * finder_std_norm)
        wobei finder_std_norm = min(1.0, finder_std / FINDER_STD_MAX)

    Bei perfekter Uebereinstimmung (std=0): quality = p_mlp * score
    Bei max. Uneinigkeit  (std>=0.10): quality = p_mlp * score * 0.5
    """
    finder_probs = np.array([
        probs["lstm"], probs["gru"], probs["cnn"], probs["lightgbm"]
    ])
    finder_std = float(np.std(finder_probs))
    finder_std_norm = min(1.0, finder_std / FINDER_STD_MAX)  # 0..1
    agreement_factor = 1.0 - 0.5 * finder_std_norm       # 0.5..1.0

    quality = probs["mlp"] * finder_score * agreement_factor
    return float(quality)


def filter_top_signals(
    signals: list[tuple],  # [(sym, price, score, probs, regime), ...]
    max_signals: int = MAX_SIGNALS_PER_MINUTE,
    min_quality: float = MIN_QUALITY_SCORE,
) -> list[tuple]:
    """Rankt Signale nach Quality-Score und nimmt nur die Top-N.

    ZWEI HURDEN:
      1. MIN_QUALITY_SCORE: Alles darunter fliegt raus — auch wenn es das
         einzige Signal der Minute ist. Kein "bester aus einem schlechten
         Haufen"-Prinzip.
      2. MAX_SIGNALS_PER_MINUTE: Aus den verbleibenden nur die Top-N.

    Args:
        signals: Liste von (symbol, price, finder_score, probs, regime)
        max_signals: Max. Anzahl Signale die durchkommen
        min_quality: Harte Quality-Schwelle (default aus Konstanten)

    Returns:
        Gefilterte Liste, sortiert nach Quality (bestes zuerst)
    """
    # Phase 1: Quality berechnen + Floor-Filter
    scored = []
    dropped = 0
    for sym, price, score, probs, regime in signals:
        quality = compute_signal_quality(probs, score)
        if quality < min_quality:
            dropped += 1
            continue
        scored.append((quality, sym, price, score, probs, regime))

    if dropped > 0:
        print(f"  [FILTER] {dropped} Signal(e) unter Quality-Floor ({min_quality}) verworfen")

    if not scored:
        return []

    # Phase 2: Nach Quality ranken, Top-N nehmen
    scored.sort(key=lambda x: x[0], reverse=True)

    if len(scored) > max_signals:
        dropped_topn = len(scored) - max_signals
        print(f"  [FILTER] {dropped_topn} Signal(e) ausserhalb Top-{max_signals} verworfen")
        scored = scored[:max_signals]

    return [(sym, price, score, probs, regime)
            for _, sym, price, score, probs, regime in scored]


# ---- Entry Rules: Regime-basiert (kein binaeres Filtern!) ------------

def _build_feature_index(feature_names: list[str]) -> dict[str, int]:
    """Baut name->index Mapping fuer schnellen Feature-Zugriff."""
    return {name: i for i, name in enumerate(feature_names)}


def get_market_regime(now_et) -> dict:
    """Bestimmt das aktuelle Markt-Regime und gibt Risiko-Multiplikatoren zurueck.

    Research-backed Regime (Intraday Volatility Breakout + ORB):
      - Open Drive (09:30–10:30): Beste Breakout-Qualitaet, volle Positionen
      - Late Morning (10:30–12:00): Gute Bedingungen, normale Parameter
      - Midday (12:00–14:00): KEINE neuen Einstiege! Daten zeigen WR=44-53%
        aber negative avg PnL (-0.15%) = Fake-Breakouts mit grossen Verlusten.
        Research: 50% Fake-Breakout-Rate in Mittagsflaute.
      - Afternoon (14:00–15:00): Erholung, reduzierte Groesse
      - Close (15:00–16:00): KEIN Trading (Liquidation only)

    Returns dict mit Multiplikatoren fuer Schwelle, Positionsgroesse, Time-Stop.
    """
    if now_et is None:
        return {"name": "unknown", "finder_votes_needed": 2, "size_mult": 1.0,
                "time_stop_min": 30, "allow_entries": True, "cooldown_min": 5}

    t = now_et.time()
    morning_start = pd.to_datetime("09:30").time()
    late_morning = pd.to_datetime("10:30").time()
    midday_start = pd.to_datetime("12:00").time()
    afternoon_start = pd.to_datetime("14:00").time()
    close_start = pd.to_datetime("15:00").time()
    market_close = pd.to_datetime("16:00").time()

    if morning_start <= t < late_morning:
        return {"name": "open_drive", "finder_votes_needed": 2, "size_mult": 1.0,
                "time_stop_min": 30, "allow_entries": True, "cooldown_min": 2,
                "min_quality_override": 0.40}  # Bewaehrt: 40% WR, fast break-even
    elif late_morning <= t < midday_start:
        # Late Morning: WR=17% am 09.07. (12 Trades, -$154).
        # Quality-Floor auf 0.50 — nur Signale mit hoher MLP-Confidence
        # UND gutem Finder-Agreement ueberleben. Kein Sektor-Filter,
        # sondern Regime-basierte Qualitaets-Huerde (wie Midday).
        return {"name": "late_morning", "finder_votes_needed": 2, "size_mult": 0.8,
                "time_stop_min": 30, "allow_entries": True, "cooldown_min": 3,
                "min_quality_override": 0.50}
    elif midday_start <= t < afternoon_start:
        # MIDDAY: Einstiege erlaubt, aber DEUTLICH strenger.
        # Daten: 27-37% WR, negative avg PnL (-0.15%) = Fake-Breakout-Zone.
        # Statt blindem Block: Quality-Floor massiv anheben (0.55), 3 Finder-Votes,
        # kleinere Position (40%), kuerzerer Time-Stop (20 Min), laengerer Cooldown.
        # Nur wirklich klare Signale ueberleben diesen Filter.
        return {"name": "midday", "finder_votes_needed": 3, "size_mult": 0.4,
                "time_stop_min": 20, "allow_entries": True, "cooldown_min": 5,
                "min_quality_override": 0.48}
    elif afternoon_start <= t < close_start:
        return {"name": "afternoon", "finder_votes_needed": 2, "size_mult": 0.8,
                "time_stop_min": 25, "allow_entries": True, "cooldown_min": 3}
    else:
        # After 15:00 ET: KEIN Trading (Liquidation only — Daten: negative EV)
        return {"name": "no_trading", "finder_votes_needed": 99, "size_mult": 0.0,
                "time_stop_min": 1, "allow_entries": False, "cooldown_min": 99}


def check_entry_rules(
    features: np.ndarray,
    fidx: dict[str, int],
    finder_votes: int,
    regime: dict,
    enabled: bool = True,
) -> tuple[bool, str]:
    """Regime-basierte Entry-Prüfung (E3–E5).

    E3: Momentum positiv – return_1m > 0
    E4: Kurz-Trend positiv – Slope_close_1 > 0
    E5: Regime-abhaengige Finder-Votes-Schwelle
         (midday=3 votes needed, sonst=2)

    Returns:
        (passed, reason)
    """
    if not enabled:
        return True, "entry_rules_disabled"

    # E3: Momentum
    if fidx.get("return_1m") is not None:
        if features[fidx["return_1m"]] <= 0:
            return False, "E3: return_1m <= 0"

    # E4: Kurz-Trend
    if fidx.get("Slope_close_1") is not None:
        if features[fidx["Slope_close_1"]] <= 0:
            return False, "E4: Slope_close_1 <= 0"

    # E5: Regime-abhaengige Signal-Huerde
    votes_needed = regime.get("finder_votes_needed", 2)
    if finder_votes < votes_needed:
        return False, f"E5: finder_votes={finder_votes} < {votes_needed} (regime={regime['name']})"

    return True, f"passed (regime={regime['name']})"


# ---- Market Hours ----

def is_market_open(trading_client) -> bool:
    """Prueft ob der US-Markt gerade geoeffnet ist."""
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception:
        # Fallback: Manuelle Zeitpruefung
        now = datetime.now(EASTERN)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now < market_close and now.weekday() < 5


def wait_until_market_open(trading_client):
    """Wartet bis zur Marktoeffnung."""
    print("\n[WAIT] Warte auf Marktoeffnung (09:30 ET / 15:30 MESZ)...")
    while not is_market_open(trading_client):
        now = datetime.now(EASTERN)
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now >= next_open:
            break
        wait_seconds = (next_open - now).total_seconds()
        if wait_seconds > 300:
            print(f"  Noch {wait_seconds/60:.0f} Minuten bis Marktoeffnung...")
            time.sleep(min(300, wait_seconds))
        else:
            time.sleep(10)
    print("[READY] Markt ist offen!")


# ---- Main Trading Loop ----

def main():
    args = parse_args()

    # Setup
    device = get_device(verbose=True)
    use_amp = enable_amp(device)

    # Features + Scaler laden
    feat_path = _PROJECT_ROOT / "data" / "processed" / "pre_split" / "features.txt"
    with open(feat_path) as f:
        features_list = [line.strip() for line in f if line.strip()]

    # Feature-Name -> Index (fuer Entry-Rules E3/E4)
    feature_index = _build_feature_index(features_list)

    scaler_path = _PROJECT_ROOT / "data" / "processed" / "global_scaler.pkl"
    scaler = joblib.load(scaler_path)

    # Logger mit vollstaendigen Parametern fuer Reproduzierbarkeit
    run_params = {
        "mode": "dry_run" if args.dry_run else "paper",
        "max_positions": args.max_positions,
        "max_daily_trades": args.max_daily_trades,
        "tp_pct": args.tp_pct,
        "sl_pct": args.sl_pct,
        "rr_ratio": args.tp_pct / args.sl_pct if args.sl_pct > 0 else 0,
        "position_size_pct": args.position_size_pct,
        "trailing_sl": args.trailing_sl,
        "trailing_sl_pct": args.trailing_sl_pct,
        "trailing_min_pct": args.trailing_min_pct,
        "trailing_ramp_pct": args.trailing_ramp_pct,
        "ratchet": not args.no_ratchet,
        "reentry_cooldown_minutes": args.reentry_cooldown,
        "sl_time_decay_target": args.sl_time_decay_target,
        "sl_time_decay_grace": args.sl_time_decay_grace,
        "time_stop_minutes": 30,
        "signal_collapse_threshold": 0.20,
        "max_risk_per_trade": args.risk,
        "entry_rules_enabled": not args.no_entry_rules,
        "mlp_threshold": args.mlp_threshold,
        "finder_thresholds": FINDER_THRESHOLDS,
        "max_signals_per_minute": MAX_SIGNALS_PER_MINUTE,
        "finder_std_max": FINDER_STD_MAX,
        "max_positions_per_sector": MAX_POSITIONS_PER_SECTOR,
        "midday_block": True,
        "grace_sl_pct": args.grace_sl_pct,
        "grace_sl_pct_t2": args.grace_sl_pct_t2,
        "entry_grace_minutes": args.entry_grace_minutes,
        "entry_grace_t2_minutes": args.entry_grace_t2_minutes,
        "symbols": args.symbols or "all",
    }
    logger = DataLogger(params=run_params)

    print(f"\n{'=' * 60}")
    print("USW-TradingModel - Paper Trading Loop")
    print(f"{'=' * 60}")
    print(f"Mode:    {'DRY-RUN' if args.dry_run else 'PAPER'}")
    print(f"Run:     {logger.run_name}")
    print(f"Features: {len(features_list)}")
    print(f"Device:   {device}")

    # Alpaca Clients
    if not args.dry_run:
        trading_client, data_client = load_alpaca_clients()
        print("Alpaca:   verbunden (Paper)")
    else:
        trading_client = None
        _, data_client = load_alpaca_clients()
        print("Alpaca:   Data-Client (Dry-Run)")

    # Symbole laden
    symbols = load_symbols(args.symbols)
    print(f"Symbole:  {len(symbols)}")

    # Modelle laden
    print("\nLade Modelle...")
    models = load_ensemble_models(len(features_list), device)
    print("  5 Modelle geladen.")

    # Account-Manager
    account_mgr = AccountManager(
        trading_client=trading_client,
        max_positions=args.max_positions,
        max_risk_per_trade=args.risk,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        position_size_pct=args.position_size_pct,
        trailing_sl_pct=args.trailing_sl_pct,
        trailing_min_pct=args.trailing_min_pct,
        trailing_ramp_pct=args.trailing_ramp_pct,
        enable_trailing_sl=args.trailing_sl,
        ratchet_mode=not args.no_ratchet,
        reentry_cooldown_minutes=args.reentry_cooldown,
        sl_time_decay_target=args.sl_time_decay_target,
        sl_time_decay_grace=args.sl_time_decay_grace,
        grace_sl_pct=args.grace_sl_pct,
        grace_sl_pct_t2=args.grace_sl_pct_t2,
        entry_grace_minutes=args.entry_grace_minutes,
        entry_grace_t2_minutes=args.entry_grace_t2_minutes,
        logger=logger,
    )

    # Warmup: Historische Bars laden + Feature-Engines initialisieren
    warmup_bars = download_warmup_bars(data_client, symbols, days=5)
    engine = MultiSymbolEngine(symbols, warmup_bars, scaler, features_list)
    ready = engine.ready_symbols()
    print(f"\nFeature-Engines: {len(ready)}/{len(symbols)} Symbole bereit (Z-Norm warm)")

    # Signal-Handler fuer sauberes Shutdown (verkauft alle Positionen)
    running = True
    def shutdown(sig, frame):
        nonlocal running
        print("\n[SHUTDOWN] Strg+C erkannt — verkaüfe alle Positionen...")
        running = False
    unix_signal.signal(unix_signal.SIGINT, shutdown)
    unix_signal.signal(unix_signal.SIGTERM, shutdown)

    # ---- TRADING LOOP ----
    print(f"\n{'=' * 60}")
    print("TRADING LOOP GESTARTET")
    print(f"{'=' * 60}")
    print(f"Market Hours: 09:30-16:00 ET (15:30-22:00 MESZ)")
    print(f"Max Positions: {args.max_positions}")
    print(f"Max Daily Trades: {args.max_daily_trades}")
    print(f"Risk/Trade: {args.risk*100:.1f}%")
    print(f"Strategie: Finder Majority + MLP Gate (Schwelle={args.mlp_threshold:.2f})")
    print(f"Signal-Filter: Top-{MAX_SIGNALS_PER_MINUTE} Quality-Score + Floor={args.min_quality:.2f}")
    print(f"R:R = {args.tp_pct*100:.2f}% / {args.sl_pct*100:.2f}% = 1:{args.tp_pct/args.sl_pct:.1f}")
    print(f"Midday (12-14 ET): ERLAUBT mit Quality-Floor=0.48 + 3 Finder-Votes + 40% Position")
    print(f"Entry-Grace: 0-{args.entry_grace_minutes}Min SL={args.grace_sl_pct*100:.2f}% "
          f"> {args.entry_grace_minutes}-{args.entry_grace_t2_minutes}Min SL={args.grace_sl_pct_t2*100:.2f}% "
          f"> ab {args.entry_grace_t2_minutes}Min SL={args.sl_pct*100:.2f}%")
    if args.trailing_sl:
        print(f"Trailing SL: AN "
              f"({args.trailing_sl_pct*100:.2f}% -> {args.trailing_min_pct*100:.2f}%)")
    else:
        print(f"Trailing SL: AUS (fixer SL + Ratchet)")
    if not args.no_ratchet:
        print(f"Ratchet-Mode: AN – TP={args.tp_pct*100:.2f}% wird zu SL-Boden")
    else:
        print(f"Ratchet-Mode: AUS – klassischer TP-Exit")
    print(f"Time-Decay: {args.sl_time_decay_target*100:.2f}% "
          f"(Grace={args.sl_time_decay_grace}min) | Cooldown={args.reentry_cooldown}min")
    print(f"Druecke Ctrl+C zum Beenden\n")

    iteration = 0
    daily_trades = 0  # Nur fuer Logging/Safety-Net, kein kuenstliches Cap
    last_flush = datetime.now()
    positions_liquidated_today = False

    try:
     while running:
        now = datetime.now(EASTERN)

        # Pre-Close-Liquidation: 15:00 ET (21:00 MESZ) ALLE Positionen verkaufen
        # Daten zeigen: Trades nach 15:00 ET haben negative EV (heute: 26 Trades, -0.11%)
        # Letzte 45 Min vor Close: 21 Trades, -2.26%, Win=48%
        # Median-Haltedauer = 13 Min -> Trades nach 15:30 schaffen keinen vollen Zyklus
        market_close_et = now.replace(hour=16, minute=0, second=0, microsecond=0)
        minutes_until_close = (market_close_et - now).total_seconds() / 60.0
        trading_deadline_et = now.replace(hour=15, minute=0, second=0, microsecond=0)

        if (not positions_liquidated_today
                and account_mgr.open_positions > 0
                and now >= trading_deadline_et):
            print(f"[{now.strftime('%H:%M')}] Trading-Ende (15:00 ET): liquidiere {account_mgr.open_positions} Positionen...")
            account_mgr.close_all_positions()
            positions_liquidated_today = True

        # Marktstatus
        if not is_market_open(trading_client if not args.dry_run else None):
            if iteration == 0 and not args.once:
                # Warte auf Marktoeffnung
                wait_until_market_open(trading_client if not args.dry_run else None)
                continue
            elif args.once:
                print("[INFO] Markt geschlossen, --once Modus beendet.")
                break
            else:
                # Nach Marktschluss: aufraeumen + warten
                if not positions_liquidated_today and account_mgr.open_positions > 0:
                    print(f"[{now.strftime('%H:%M')}] Markt geschlossen — liquidiere {account_mgr.open_positions} Positionen...")
                    account_mgr.close_all_positions()
                    positions_liquidated_today = True

                logger.end_of_day()
                logger.rollover_if_new_day()  # Mitternachts-Rollover
                account_mgr.reset_daily()
                daily_trades = 0
                time.sleep(300)  # Alle 5 Min checken
                continue
        else:
            # Markt ist offen — Liquidierungs-Flag zuruecksetzen
            positions_liquidated_today = False

        iteration += 1

        # Regime-basierte Parameter anpassen (Cooldown, Pre-Close-Schutz)
        now_et = datetime.now(EASTERN)
        current_regime = get_market_regime(now_et)
        account_mgr.set_cooldown(current_regime["cooldown_min"])

        # Pre-Close-Schutz: Keine neuen Einstiege nach 15:00 ET (Daten: negative EV)
        # + Time-Stop-Check: auch vor 15:00 blocken wenn Zyklus nicht mehr passt
        market_close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        minutes_until_close = (market_close_et - now_et).total_seconds() / 60.0
        trading_deadline_et = now_et.replace(hour=15, minute=0, second=0, microsecond=0)
        pre_close_block = (now_et >= trading_deadline_et) or (minutes_until_close < current_regime["time_stop_min"])

        # Warten bis kurz vor der naechsten vollen Minute
        sec = now.second
        if sec < 55:
            time.sleep(max(1, 55 - sec))
            now = datetime.now(EASTERN)

        print(f"\n[{now.strftime('%H:%M:%S')}] Iteration {iteration} "
              f"| Positionen: {account_mgr.open_positions} | "
              f"Ready: {len(engine.ready_symbols())}")

        try:
            # 1. Neueste Bars holen
            latest_bars = fetch_latest_bars(data_client, symbols)
            if not latest_bars:
                print("  [WARN] Keine Bars erhalten, ueberspringe...")
                time.sleep(5)
                continue

            # 2. Features berechnen + Model Inference
            features_map = engine.process_bars(latest_bars)
            print(f"  Bars: {len(latest_bars)} | Features: {len(features_map)}")

            signals = []
            current_prices = {}
            signals_filtered = 0
            for sym, fv in features_map.items():
                bar = latest_bars.get(sym)
                if bar is None:
                    continue

                current_prices[sym] = bar["close"]

                seq = engine.engines[sym].get_sequence() if sym in engine.engines else None
                signal, score, probs = ensemble_predict(models, fv, seq, device)

                # Bar loggen
                logger.log_bar(sym, bar)

                if signal:
                    # Finder-Votes berechnen (fuer Regime-Entscheidung)
                    finder_votes = (
                        (1 if probs["lstm"] > FINDER_THRESHOLDS["lstm"] else 0) +
                        (1 if probs["gru"] > FINDER_THRESHOLDS["gru"] else 0) +
                        (1 if probs["cnn"] > FINDER_THRESHOLDS["cnn"] else 0) +
                        (1 if probs["lightgbm"] > FINDER_THRESHOLDS["lightgbm"] else 0)
                    )

                    # Markt-Regime + Entry-Rules E3–E5
                    now_et = datetime.now(EASTERN)
                    regime = get_market_regime(now_et)
                    rules_ok, rules_reason = check_entry_rules(
                        fv, feature_index, finder_votes, regime,
                        enabled=not args.no_entry_rules,
                    )
                    if not rules_ok:
                        signals_filtered += 1
                        continue  # Signal erfuellt Regime-Huerde nicht

                    signals.append((sym, bar["close"], score, probs, regime))
                    logger.log_signal(
                        timestamp=bar["timestamp"], symbol=sym, target=-1,
                        p_mlp=probs["mlp"], p_lstm=probs["lstm"],
                        p_gru=probs["gru"], p_cnn=probs["cnn"], p_lgb=probs["lightgbm"],
                        ensemble_score=score, ensemble_signal=signal,
                    )

            # 3. Quality-Ranking: Nur Top-N Signale pro Minute + Regime-spezifischer Quality-Floor
            if signals:
                regime_quality_floor = current_regime.get("min_quality_override", args.min_quality)
                signals = filter_top_signals(signals, MAX_SIGNALS_PER_MINUTE, regime_quality_floor)

            # Immer Signal-Status anzeigen (auch bei 0)
            filter_info = f" ({signals_filtered} gefiltert)" if signals_filtered else ""
            if signals:
                q_scores = [f"{sym}={compute_signal_quality(probs, score):.3f}"
                           for sym, _, score, probs, _ in signals]
                print(f"  SIGNALE: {len(signals)}{filter_info} (Top-{len(signals)} Quality)")
                print(f"  Quality: {', '.join(q_scores)}")
            else:
                print(f"  SIGNALE: 0{filter_info}")

            # 4. Finder-Scores aktualisieren (fuer Signal-Kollaps-Check)
            finder_scores = {sym: score for sym, _, score, _, _ in signals}
            account_mgr.update_finder_scores(finder_scores)

            # 5. Exit-Checks
            exits = account_mgr.check_exits(current_prices)
            for exit_action in exits:
                if exit_action.partial_qty > 0:
                    # Teilgewinnmitnahme: Nur einen Teil der Position verkaufen
                    print(f"  EXIT: {exit_action.symbol} (PARTIAL {exit_action.partial_qty}sh) "
                          f"PnL={exit_action.pnl_pct:+.3%}")
                    if not args.dry_run:
                        account_mgr.submit_market_sell(exit_action.symbol, exit_action.partial_qty)
                    account_mgr.register_exit(exit_action)
                else:
                    # Voller Exit
                    print(f"  EXIT: {exit_action.symbol} ({exit_action.reason}) "
                          f"PnL={exit_action.pnl_pct:+.3%}")
                    if not args.dry_run:
                        pos = account_mgr._positions.get(exit_action.symbol)
                        qty = pos.qty if pos else 1
                        account_mgr.submit_market_sell(exit_action.symbol, qty)
                    account_mgr.register_exit(exit_action)

            # 6. Neue Trades eroeffnen
            if not args.dry_run and trading_client:
                account = trading_client.get_account()
                equity = float(account.equity)
            else:
                equity = 100_000.0  # Simuliertes Kapital fuer Dry-Run

            for sym, price, score, probs, regime in signals:
                regime_name = regime.get("name", "unknown")

                # --- Regime-Entry-Check: No-Trading-Zone (nur 15:00-16:00 ET) ---
                if not regime.get("allow_entries", True):
                    continue  # Keine Einstiege in dieser Marktphase (nur no_trading)

                # --- Safety-Net: rein technische Notbremse (default 50, quasi nie erreicht) ---
                if daily_trades >= args.max_daily_trades:
                    continue

                if not account_mgr.can_enter(sym):
                    continue

                # --- Sektor-Konzentrations-Check ---
                sector = SECTOR_MAP.get(sym, "other")
                sector_count = sum(
                    1 for s in account_mgr._positions
                    if SECTOR_MAP.get(s, "other") == sector
                )
                if sector_count >= MAX_POSITIONS_PER_SECTOR:
                    continue  # Zu viele Positionen im gleichen Sektor

                # Pre-Close-Schutz: Kein Einstieg wenn Time-Stop nicht mehr in Marktzeit passt
                regime_time_stop = regime.get("time_stop_min", 30)
                if minutes_until_close < regime_time_stop:
                    continue  # Trade haette keine Chance, den vollen Zyklus zu durchlaufen

                # Regime-abhaengige Positionsgroesse
                size_mult = regime.get("size_mult", 1.0)

                qty = account_mgr.calculate_size(equity, price)
                qty = max(1, int(qty * size_mult))  # Kleinere Position in Risiko-Phasen
                if qty == 0:
                    continue

                tp_price = price * (1.0 + args.tp_pct)
                sl_price = price * (1.0 - args.sl_pct)

                # Quality-Score fuer Logging
                quality = compute_signal_quality(probs, score)
                regime_tag = f" [{regime['name']}]" if regime["name"] not in ("open_drive", "late_morning", "afternoon") else ""

                if args.dry_run:
                    print(f"  [DRY-RUN] {sym}: BUY {qty} @ ${price:.2f} "
                          f"(TP=${tp_price:.2f} SL=${sl_price:.2f} "
                          f"TS={regime_time_stop}min Q={quality:.3f}{regime_tag})")
                    account_mgr.register_entry(sym, price, qty,
                                               order_id=f"dry_{iteration}_{sym}",
                                               finder_score=score,
                                               time_stop_minutes=regime_time_stop)
                else:
                    order_id, fill_price = account_mgr.submit_market_buy(
                        sym, qty, tp_price, sl_price
                    )
                    # ECHTEN Fill-Preis verwenden, nicht bar["close"]
                    entry_px = fill_price if fill_price > 0 else price
                    print(f"  [ORDER] {sym}: BUY {qty} @ ${entry_px:.2f}"
                          f" Q={quality:.3f}{regime_tag}")
                    if order_id:
                        account_mgr.register_entry(sym, entry_px, qty,
                                                   order_id=order_id,
                                                   finder_score=score,
                                                   time_stop_minutes=regime_time_stop)
                daily_trades += 1

            # Safety-Net-Warnung (nur wenn das technische Limit erreicht wird)
            if daily_trades == args.max_daily_trades:
                print(f"  [SAFETY-NET] Technisches Tages-Limit erreicht ({args.max_daily_trades}) "
                      f"- Notbremse greift")

            # 6. Periodisches Flushen (alle 30 Min)
            if (datetime.now() - last_flush).total_seconds() > 1800:
                logger.flush_bars()
                logger.flush_signals()
                logger.flush_orders()
                last_flush = datetime.now()

        except Exception as e:
            print(f"  [ERROR] Iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()

        # --once Modus: nur eine Iteration
        if args.once:
            break

        # Warten bis zur naechsten Minute
        time.sleep(2)

    # ---- SHUTDOWN (verkauft ALLE offenen Positionen) ----
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Strg+C - verkaufe Positionen...")
        exit_reason = "manual_stop"
    except Exception as e:
        print(f"\n[SHUTDOWN] Fehler: {e}")
        import traceback; traceback.print_exc()
        exit_reason = "crash"
    else:
        exit_reason = "normal"
    finally:
        print(f"\n{'=' * 60}")
        print("TRADING LOOP BEENDET")
        print(f"{'=' * 60}")
        print(f"Iterationen: {iteration}")
        account_mgr.close_all_positions()
    logger.end_of_day()
    logger.write_run_exit(exit_reason)
    summary = logger.get_summary()
    print(f"Signale: {summary['n_signals']} | Orders: {summary['n_orders']} | "
          f"Positions: {summary['n_positions']} | Trades heute: {daily_trades}")
    print(f"Gespeichert in: {logger.run_dir}")


if __name__ == "__main__":
    main()
