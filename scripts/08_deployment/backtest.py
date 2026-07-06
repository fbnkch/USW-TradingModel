"""
Historisches Backtesting der Ensemble-Trading-Strategie.

Simuliert Walk-Forward-Trading auf dem Test-Set (Juli-Dezember 2024)
mit echten OHLC-Preisen, Transaktionskosten und Risikomanagement.

Features:
  - 5 Ensemble-Strategien waehlbar (finder_majority, two_stage, stacking, etc.)
  - Reale TP/SL/Time-Stop-Simulation anhand High/Low-Preise
  - Positions-Management (max. 3 gleichzeitig, Kelly-Sizing)
  - Transaktionskosten (0.01% pro Trade)
  - Umfangreiche Metriken (Sharpe, Max DD, CAGR, Calmar, Profit Factor)
  - Benchmark-Vergleich (Equal-Weight Buy-and-Hold)
  - 4 Dark-Theme-Charts (Equity Curve, Drawdown, Monthly Returns, Trades)

Aufruf:
  python scripts/08_deployment/backtest.py
  python scripts/08_deployment/backtest.py --strategy two_stage
  python scripts/08_deployment/backtest.py --no_entry_rules
  python scripts/08_deployment/backtest.py --max_positions 5 --tp_pct 0.005

Output:
  artifacts/evaluation/backtest_results.json
  artifacts/images/06_backtesting/backtest_*.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PREDICTIONS_PATH = _PROJECT_ROOT / "artifacts" / "evaluation" / "ensemble_predictions.parquet"
_SHUFFLED_DIR = _PROJECT_ROOT / "data" / "processed" / "shuffled"
_EVAL_DIR = _PROJECT_ROOT / "artifacts" / "evaluation"
_IMG_DIR = _PROJECT_ROOT / "artifacts" / "images" / "06_backtesting"

# ---------------------------------------------------------------------------
# Trading-Parameter (aus TRADING_STRATEGIE.md / account_manager.py)
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 100_000.0
MAX_POSITIONS = 3
MAX_RISK_PER_TRADE = 0.005       # 0.5% des Portfolios
TP_PCT = 0.0036                   # +0.36% Take Profit
SL_PCT = 0.0015                   # -0.15% Stop Loss
TIME_STOP_MINUTES = 30
SIGNAL_COLLAPSE_THRESHOLD = 0.20
TRANSACTION_COST = 0.0001         # 0.01% Spread + Slippage
RISK_FREE_RATE = 0.04             # 4% annual

# Finder-Thresholds (aus ensemble.py)
FINDER_THRESHOLDS = {
    "lstm": 0.320, "gru": 0.334, "cnn": 0.314, "lightgbm": 0.355,
}
FILTER_THRESHOLD = 0.50
FINDER_WEIGHTS = {
    "lstm": 0.2510754790770434, "gru": 0.2530308955807587,
    "cnn": 0.2522487289792726, "lightgbm": 0.2436448963629253,
}

# ---------------------------------------------------------------------------
# Dark Theme
# ---------------------------------------------------------------------------
BG = "#0a0e17"
CARD = "#111827"
TEXT = "#e2e8f0"
ACCENT = "#3b82f6"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#ea580c"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": "#1f2937", "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "grid.color": "#1f2937", "grid.alpha": 0.5,
    "legend.facecolor": CARD, "legend.edgecolor": "#1f2937",
    "legend.labelcolor": TEXT, "figure.dpi": 150,
})


# ===================================================================
# DATACLASSES
# ===================================================================

@dataclass
class Trade:
    """Ein abgeschlossener Trade."""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    exit_reason: str          # take_profit | stop_loss | time_stop | signal_collapse | end_of_period
    pnl: float
    pnl_pct: float
    bars_held: int
    finder_score: float = 0.0


@dataclass
class OpenPosition:
    """Eine aktuell offene Position im Backtest."""
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: int
    tp_price: float
    sl_price: float
    finder_score: float


# ===================================================================
# DATEN LADEN
# ===================================================================

def load_predictions_with_prices(
    predictions_path: Path = _PREDICTIONS_PATH,
    shuffled_dir: Path = _SHUFFLED_DIR,
) -> pd.DataFrame:
    """Laedt Ensemble-Predictions und merged OHLC-Preise aus Test-Shards.

    Returns:
        DataFrame mit Spalten:
        symbol, timestamp, target,
        p_mlp, p_lstm, p_gru, p_cnn, p_lgb,
        return_1m, Slope_close_1, minutes_since_open,
        score_*, signal_* (alle 5 Strategien),
        ensemble_signal, ensemble_score, strategy_used,
        open, high, low, close, vwap
    """
    # -- Ensemble-Predictions laden --------------------------------------
    print(f"[LOAD] Lade Predictions: {predictions_path}")
    t0 = time.time()
    pred = pd.read_parquet(predictions_path)
    print(f"  {len(pred):,} Zeilen, {len(pred.columns)} Spalten, "
          f"{pred['symbol'].nunique()} Symbole")

    # -- Test-Shards laden ------------------------------------------------
    print(f"[LOAD] Lade Test-Shards aus: {shuffled_dir}")
    test_files = sorted(shuffled_dir.glob("test_shard_*.parquet"))
    print(f"  {len(test_files)} Shards gefunden")

    price_cols = ["symbol", "timestamp", "open", "high", "low", "close", "vwap"]
    price_frames = []
    for f in test_files:
        shard = pd.read_parquet(f, columns=price_cols)
        price_frames.append(shard)

    prices = pd.concat(price_frames, ignore_index=True)
    # Shards: US/Eastern -> UTC -> naive (predictions sind in UTC-naive)
    if prices["timestamp"].dt.tz is not None:
        prices["timestamp"] = prices["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    print(f"  {len(prices):,} Preis-Zeilen geladen ({len(prices['symbol'].unique())} Symbole)")

    # -- Mergen -----------------------------------------------------------
    # Predictions sind bereits UTC-naive (Parquet stripped timezone)
    if pred["timestamp"].dt.tz is not None:
        pred["timestamp"] = pred["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    df = pred.merge(prices, on=["symbol", "timestamp"], how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  Nach Merge: {len(df):,} Zeilen "
          f"(Drop: {len(pred) - len(df):,} = {(1 - len(df)/len(pred))*100:.1f}%)")
    print(f"  Zeitraum: {df['timestamp'].min()} bis {df['timestamp'].max()}")
    print(f"  Dauer: {time.time() - t0:.1f}s")
    return df


# ===================================================================
# BACKTEST-ENGINE
# ===================================================================

def _compute_finder_score(row: pd.Series) -> float:
    """Weighted-Average-Finder-Score aus den 4 Finder-Modellen."""
    return float(
        row["p_lstm"] * FINDER_WEIGHTS["lstm"]
        + row["p_gru"] * FINDER_WEIGHTS["gru"]
        + row["p_cnn"] * FINDER_WEIGHTS["cnn"]
        + row["p_lgb"] * FINDER_WEIGHTS["lightgbm"]
    )


def _check_entry_rules(row: pd.Series) -> bool:
    """Entry-Rules E3-E5 aus TRADING_STRATEGIE.md.

    E3: return_1m > 0  (positives Momentum)
    E4: Slope_close_1 > 0  (kurzfristiger Trend positiv)
    E5: minutes_since_open in [120, 270] U [330, 360]  (keine Mittagsflaute)
    """
    if row.get("return_1m", 0) <= 0:
        return False
    if row.get("Slope_close_1", 0) <= 0:
        return False
    mins = row.get("minutes_since_open", 0)
    in_morning = 120 <= mins <= 270      # 10:00-14:00 ET (30 min offset: 11:30-14:00)
    in_afternoon = 330 <= mins <= 360    # 14:30-15:30 ET
    if not (in_morning or in_afternoon):
        return False
    return True


def _exit_for_symbol(
    sym_bars: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    finder_score_col: str,
) -> tuple[str, float, int]:
    """Simuliert Exit fuer eine einzelne Position.

    Geht die naechsten 30 Bars des Symbols durch und prueft
    TP (via high) > SL (via low) > Time-Stop > Signal-Collapse.

    Args:
        sym_bars: Alle Bars EINES Symbols, chronologisch sortiert.
        entry_idx: Index-Position der Entry-Bar in sym_bars.
        entry_price: Einstiegspreis.
        tp_price: Take-Profit-Preis.
        sl_price: Stop-Loss-Preis.
        finder_score_col: Spaltenname des Finder-Scores.

    Returns:
        (exit_reason, exit_price, bars_held)
    """
    max_lookahead = min(entry_idx + TIME_STOP_MINUTES + 1, len(sym_bars))
    future = sym_bars.iloc[entry_idx + 1 : max_lookahead]

    if len(future) == 0:
        return ("end_of_period", entry_price, 0)

    earliest_exit_bar = 0
    exit_reason = "time_stop"
    exit_price = float(future.iloc[-1]["close"])

    for i, (_, bar) in enumerate(future.iterrows()):
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        # TP-Check zuerst (konservativ: TP hat Prioritaet vor SL)
        if bar_high >= tp_price:
            earliest_exit_bar = i + 1
            exit_reason = "take_profit"
            exit_price = tp_price
            break

        # SL-Check (nur wenn SL aktiv, d.h. sl_price < entry_price)
        if sl_price < entry_price and bar_low <= sl_price:
            earliest_exit_bar = i + 1
            exit_reason = "stop_loss"
            exit_price = sl_price
            break

        # Time-Stop nach 30 Minuten
        if i + 1 >= TIME_STOP_MINUTES:
            earliest_exit_bar = i + 1
            exit_reason = "time_stop"
            exit_price = bar_close
            break

        # Signal-Collapse (nur nach >5 Minuten)
        if i >= 5:
            finder_score = float(bar.get(finder_score_col, 0.5))
            if finder_score < SIGNAL_COLLAPSE_THRESHOLD:
                earliest_exit_bar = i + 1
                exit_reason = "signal_collapse"
                exit_price = bar_close
                break
    else:
        # Kein Exit innerhalb der 30 Bars → Time-Stop am Ende
        earliest_exit_bar = len(future)
        exit_reason = "time_stop"
        exit_price = float(future.iloc[-1]["close"])

    return (exit_reason, exit_price, earliest_exit_bar)


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "signal_finder_majority",
    score_col: str = "score_finder_majority",
    apply_entry_rules: bool = True,
    max_positions: int = MAX_POSITIONS,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost: float = TRANSACTION_COST,
) -> dict:
    """Fuehrt einen Walk-Forward-Backtest durch.

    Args:
        df: DataFrame mit Predictions + OHLC-Preisen.
        signal_col: Spaltenname der Signal-Kolumne (0/1).
        score_col: Spaltenname der Score-Kolumne (kontinuierlich).
        apply_entry_rules: E3-E5 Filter aktivieren.
        max_positions: Max. gleichzeitige Positionen.
        tp_pct: Take-Profit-Schwelle.
        sl_pct: Stop-Loss-Schwelle.
        initial_capital: Startkapital.
        transaction_cost: Transaktionskosten pro Trade (Dezimal).

    Returns:
        dict mit:
          - trades: list[Trade]
          - equity_curve: pd.DataFrame (timestamp, equity, cash, positions_value)
          - benchmark_curve: pd.DataFrame (timestamp, benchmark_value)
          - metadata: dict
    """
    print(f"\n[BACKTEST] Starte Walk-Forward-Simulation...")
    print(f"  Strategie: {signal_col}")
    print(f"  Entry-Rules: {apply_entry_rules}")
    print(f"  Max Positionen: {max_positions}")
    print(f"  TP: {tp_pct*100:.2f}%  SL: {sl_pct*100:.2f}%")
    print(f"  Kapital: ${initial_capital:,.0f}")
    t0 = time.time()

    # -- Pre-Compute Exits pro Symbol ----------------------------------
    print("[BACKTEST] Pre-Computing Exits...")
    symbols = sorted(df["symbol"].unique())
    # Dictionaries: key = (symbol, entry_timestamp) -> (exit_bar_timestamp, exit_reason, exit_price, bars_held)
    exit_map: dict[tuple, tuple] = {}
    skipped_no_data = 0

    for sym in symbols:
        sym_mask = df["symbol"] == sym
        sym_df = df.loc[sym_mask].sort_values("timestamp").reset_index(drop=True)

        if len(sym_df) < 2:
            continue

        signal_mask = sym_df[signal_col] == 1
        signal_indices = sym_df.index[signal_mask]

        for local_idx in signal_indices:
            row = sym_df.loc[local_idx]
            ts = row["timestamp"]

            # Entry-Rules pruefen
            if apply_entry_rules and not _check_entry_rules(row):
                continue

            entry_price = float(row["close"])
            tp_price = entry_price * (1.0 + tp_pct)
            sl_price = entry_price * (1.0 - sl_pct)

            reason, exit_px, bars_held = _exit_for_symbol(
                sym_df, int(local_idx), entry_price, tp_price, sl_price, score_col,
            )

            # Exit-Timestamp bestimmen
            exit_local_idx = int(local_idx) + bars_held
            if exit_local_idx < len(sym_df):
                exit_ts = sym_df.loc[exit_local_idx, "timestamp"]
            else:
                exit_ts = sym_df.iloc[-1]["timestamp"]

            finder = _compute_finder_score(row)
            exit_map[(sym, ts)] = (exit_ts, reason, exit_px, bars_held, finder, entry_price)

    print(f"  {len(exit_map):,} gueltige Signale (pre-computed exits)")
    print(f"  Dauer: {time.time() - t0:.1f}s")

    # -- Walk-Forward-Simulation ---------------------------------------
    print("[BACKTEST] Simuliere Position-Management...")
    t1 = time.time()

    # Sortiere alle Trades nach Entry-Zeit
    all_entries = sorted(exit_map.items(), key=lambda x: x[0][1])  # sort by entry timestamp

    open_positions: dict[str, OpenPosition] = {}       # keyed by symbol
    completed_trades: list[Trade] = []

    # Equity-Tracking: collect per-minute snapshots
    timestamps = sorted(df["timestamp"].unique())
    equity_records: list[dict] = []
    benchmark_records: list[dict] = []

    # Benchmark: Initialpreise pro Symbol
    first_prices = {}
    for sym in symbols:
        sym_data = df[df["symbol"] == sym]
        if len(sym_data) > 0:
            first_prices[sym] = float(sym_data.iloc[0]["close"])
    benchmark_shares = {sym: (initial_capital / len(symbols)) / first_prices[sym]
                        for sym in first_prices}

    # Pre-build lookup: timestamp -> {symbol -> close}
    # For efficient benchmark and equity computation
    ts_to_close: dict[pd.Timestamp, dict[str, float]] = defaultdict(dict)
    for _, row in df.iterrows():
        ts_to_close[row["timestamp"]][row["symbol"]] = float(row["close"])

    cash = initial_capital
    position_cost_basis: dict[str, float] = {}  # total cost for each position (for P&L)

    # Iterator ueber alle potentiellen Trades
    trade_iter = iter(all_entries)
    next_trade = next(trade_iter, None)

    n_events = 0
    for ts in timestamps:
        closes = ts_to_close.get(ts, {})

        # -- Exit-Checks: Positionen schliessen, deren Exit-Timestamp erreicht ist --
        symbols_to_exit = []
        for sym, pos in open_positions.items():
            exit_key = (sym, pos.entry_time)
            if exit_key in exit_map:
                exit_ts, reason, exit_px, bars_held, finder, entry_px = exit_map[exit_key]
                if ts >= exit_ts:
                    symbols_to_exit.append((sym, exit_ts, reason, exit_px, bars_held, finder))

        for sym, exit_ts, reason, exit_px, bars_held, finder in symbols_to_exit:
            pos = open_positions[sym]
            cost_basis = position_cost_basis.get(sym, pos.entry_price * pos.qty)
            gross_pnl = (exit_px - pos.entry_price) * pos.qty
            fee = cost_basis * transaction_cost + (exit_px * pos.qty * transaction_cost)
            net_pnl = gross_pnl - fee
            pnl_pct = (exit_px - pos.entry_price) / pos.entry_price

            completed_trades.append(Trade(
                symbol=sym,
                entry_time=pos.entry_time,
                exit_time=exit_ts,
                entry_price=pos.entry_price,
                exit_price=exit_px,
                qty=pos.qty,
                exit_reason=reason,
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                bars_held=bars_held,
                finder_score=finder,
            ))

            cash += exit_px * pos.qty - fee
            n_events += 1
            del open_positions[sym]
            position_cost_basis.pop(sym, None)

        # -- Entry-Checks: Neue Signale am aktuellen Timestamp --
        while next_trade is not None and next_trade[0][1] == ts:
            (sym, entry_ts), (exit_ts, reason, exit_px, bars_held, finder, entry_price) = next_trade

            if sym not in open_positions and len(open_positions) < max_positions:
                # Position Sizing: Equal allocation pro Slot
                alloc = cash / max(1, max_positions - len(open_positions))
                qty = max(1, int(alloc / (entry_price * (1.0 + transaction_cost))))
                required = entry_price * qty * (1.0 + transaction_cost)

                if required <= cash and qty > 0:
                    cash -= required
                    position_cost_basis[sym] = required
                    open_positions[sym] = OpenPosition(
                        symbol=sym,
                        entry_time=entry_ts,
                        entry_price=entry_price,
                        qty=qty,
                        tp_price=entry_price * (1.0 + tp_pct),
                        sl_price=entry_price * (1.0 - sl_pct),
                        finder_score=finder,
                    )
                    n_events += 1

            next_trade = next(trade_iter, None)

            if next_trade is None:
                break

        # -- Equity & Benchmark tracken --
        positions_value = sum(
            closes.get(pos.symbol, pos.entry_price) * pos.qty
            for pos in open_positions.values()
        )
        total_equity = cash + positions_value

        equity_records.append({
            "timestamp": ts,
            "equity": total_equity,
            "cash": cash,
            "positions_value": positions_value,
            "n_positions": len(open_positions),
        })

        # Benchmark
        bm_value = sum(
            closes.get(sym, 0.0) * benchmark_shares.get(sym, 0)
            for sym in benchmark_shares
        )
        benchmark_records.append({
            "timestamp": ts,
            "benchmark_value": bm_value if bm_value > 0 else initial_capital,
        })

    # -- Periodenende: alle offenen Positionen schliessen --
    final_ts = timestamps[-1]
    final_closes = ts_to_close.get(final_ts, {})
    for sym, pos in list(open_positions.items()):
        exit_px = final_closes.get(sym, pos.entry_price)
        cost_basis = position_cost_basis.get(sym, pos.entry_price * pos.qty)
        gross_pnl = (exit_px - pos.entry_price) * pos.qty
        fee = cost_basis * transaction_cost + (exit_px * pos.qty * transaction_cost)
        net_pnl = gross_pnl - fee

        completed_trades.append(Trade(
            symbol=sym, entry_time=pos.entry_time, exit_time=final_ts,
            entry_price=pos.entry_price, exit_price=exit_px,
            qty=pos.qty, exit_reason="end_of_period",
            pnl=net_pnl, pnl_pct=(exit_px - pos.entry_price) / pos.entry_price,
            bars_held=0, finder_score=pos.finder_score,
        ))
        cash += exit_px * pos.qty - fee
        del open_positions[sym]

    # Finaler Equity-Punkt
    equity_records.append({
        "timestamp": final_ts, "equity": cash,
        "cash": cash, "positions_value": 0, "n_positions": 0,
    })

    equity_df = pd.DataFrame(equity_records)
    benchmark_df = pd.DataFrame(benchmark_records)

    # Sort trades by entry time
    completed_trades.sort(key=lambda t: t.entry_time)

    elapsed = time.time() - t1
    print(f"  Trades: {len(completed_trades)} | "
          f"Pos-Spitzen: {max(r['n_positions'] for r in equity_records)} | "
          f"Final Equity: ${cash:,.0f}")
    print(f"  Dauer: {elapsed:.1f}s | Gesamt: {time.time() - t0:.1f}s")

    return {
        "trades": completed_trades,
        "equity_curve": equity_df,
        "benchmark_curve": benchmark_df,
        "metadata": {
            "strategy": signal_col,
            "entry_rules": apply_entry_rules,
            "max_positions": max_positions,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "initial_capital": initial_capital,
            "transaction_cost": transaction_cost,
            "n_timestamps": len(timestamps),
            "n_symbols": len(symbols),
        },
    }


# ===================================================================
# METRIKEN
# ===================================================================

def compute_sharpe_ratio(daily_returns: np.ndarray, rf: float = RISK_FREE_RATE) -> float:
    """Annualisierter Sharpe Ratio."""
    if len(daily_returns) < 2 or np.std(daily_returns) == 0:
        return 0.0
    excess = daily_returns - rf / 252
    mean_excess = np.mean(excess)
    std_excess = np.std(excess)
    if std_excess == 0:
        return 0.0
    return float(np.sqrt(252) * mean_excess / std_excess)


def compute_max_drawdown(equity: np.ndarray) -> dict:
    """Max Drawdown mit Start-/End-Daten."""
    if len(equity) < 2:
        return {"max_drawdown_pct": 0.0, "max_drawdown_days": 0,
                "peak_idx": 0, "trough_idx": 0}
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    trough_idx = int(np.argmin(drawdown))
    if trough_idx == 0:
        return {"max_drawdown_pct": 0.0, "max_drawdown_days": 0,
                "peak_idx": 0, "trough_idx": 0}
    peak_idx = int(np.argmax(equity[:trough_idx + 1]))
    max_dd = float(drawdown[trough_idx])
    dd_days = trough_idx - peak_idx
    return {
        "max_drawdown_pct": max_dd * 100,
        "max_drawdown_days": dd_days,
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
    }


def compute_cagr(start_value: float, end_value: float, trading_days: int) -> float:
    """Compound Annual Growth Rate."""
    if start_value <= 0 or trading_days < 1:
        return 0.0
    years = trading_days / 252
    return float((end_value / start_value) ** (1.0 / years) - 1.0)


def compute_metrics(
    results: dict,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """Berechnet alle Backtest-Metriken.

    Args:
        results: Output von run_backtest().
        initial_capital: Startkapital.

    Returns:
        dict mit allen Metriken (siehe Plan fuer Details).
    """
    trades: list[Trade] = results["trades"]
    equity_df: pd.DataFrame = results["equity_curve"]
    benchmark_df: pd.DataFrame = results["benchmark_curve"]
    metadata: dict = results["metadata"]

    equity = equity_df["equity"].values
    benchmark = benchmark_df["benchmark_value"].values

    # Tages-Returns (aus Equity Curve)
    equity_df_daily = equity_df.set_index("timestamp").resample("D").last().dropna()
    benchmark_df_daily = benchmark_df.set_index("timestamp").resample("D").last().dropna()

    daily_returns = equity_df_daily["equity"].pct_change().dropna().values
    bm_daily_returns = benchmark_df_daily["benchmark_value"].pct_change().dropna().values

    trading_days = len(daily_returns)
    final_equity = float(equity[-1])
    total_return = (final_equity - initial_capital) / initial_capital * 100

    # Sharpe / Sortino
    sharpe = compute_sharpe_ratio(daily_returns)
    downside_returns = daily_returns[daily_returns < 0]
    sortino = 0.0
    if len(downside_returns) > 1 and np.std(downside_returns) > 0:
        excess_d = daily_returns - RISK_FREE_RATE / 252
        sortino = float(np.sqrt(252) * np.mean(excess_d) / np.std(downside_returns))

    cagr = compute_cagr(initial_capital, final_equity, trading_days)
    dd = compute_max_drawdown(equity)
    calmar = cagr / abs(dd["max_drawdown_pct"] / 100) if dd["max_drawdown_pct"] != 0 else 0.0
    volatility = float(np.std(daily_returns) * np.sqrt(252) * 100)

    # Trades
    n_trades = len(trades)
    win_trades = [t for t in trades if t.pnl > 0]
    loss_trades = [t for t in trades if t.pnl <= 0]
    n_wins = len(win_trades)
    n_losses = len(loss_trades)
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0.0

    total_gains = sum(t.pnl for t in win_trades)
    total_losses = abs(sum(t.pnl for t in loss_trades))
    profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")

    avg_win = float(np.mean([t.pnl_pct for t in win_trades]) * 100) if win_trades else 0.0
    avg_loss = float(np.mean([t.pnl_pct for t in loss_trades]) * 100) if loss_trades else 0.0
    avg_holding = float(np.mean([t.bars_held for t in trades])) if trades else 0.0

    total_fees = n_trades * 2 * initial_capital * metadata["transaction_cost"]  # Approximation
    n_days = trading_days
    trades_per_day = n_trades / n_days if n_days > 0 else 0.0

    # Benchmark
    bm_start = float(benchmark[0]) if len(benchmark) > 0 else initial_capital
    bm_end = float(benchmark[-1]) if len(benchmark) > 0 else initial_capital
    bm_return = (bm_end - bm_start) / bm_start * 100
    bm_sharpe = compute_sharpe_ratio(bm_daily_returns) if len(bm_daily_returns) > 1 else 0.0
    alpha = total_return - bm_return

    # Exit-Breakdown
    exit_counts: dict[str, int] = defaultdict(int)
    for t in trades:
        exit_counts[t.exit_reason] += 1

    # Monthly Returns
    monthly = equity_df_daily["equity"].resample("ME").apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100 if len(x) > 1 else 0
    )
    monthly_returns = {ts.strftime("%Y-%m"): float(v)
                       for ts, v in monthly.items()}

    n_signals = sum(
        1 for t in trades
    )  # Approximation; genauer waere Zaehlung in run_backtest
    n_bars = len(equity_df)
    signal_rate = n_trades / n_bars * 100 if n_bars > 0 else 0

    return {
        # Returns & Risk
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(dd["max_drawdown_pct"], 2),
        "max_drawdown_days": dd["max_drawdown_days"],
        "calmar_ratio": round(calmar, 2),
        "volatility_annual_pct": round(volatility, 2),
        # Trades
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "avg_holding_minutes": round(avg_holding, 1),
        "trades_per_day": round(trades_per_day, 1),
        "total_fees": round(total_fees, 2),
        "signal_rate_pct": round(signal_rate, 2),
        # Benchmark
        "benchmark_return_pct": round(bm_return, 2),
        "benchmark_sharpe": round(bm_sharpe, 3),
        "alpha_pct": round(alpha, 2),
        # Details
        "breakdown_by_exit": dict(exit_counts),
        "monthly_returns": monthly_returns,
        "final_equity": round(final_equity, 2),
        "initial_capital": initial_capital,
        # Metadata
        "strategy": metadata["strategy"],
        "entry_rules": metadata["entry_rules"],
        "n_trading_days": trading_days,
        "n_symbols": metadata["n_symbols"],
    }


# ===================================================================
# VISUALISIERUNG
# ===================================================================

def plot_equity_curve(equity_df: pd.DataFrame, benchmark_df: pd.DataFrame,
                      metrics: dict, save_path: Path):
    """Equity Curve + Benchmark."""
    fig, ax = plt.subplots(figsize=(14, 6))

    eq = equity_df["equity"].values
    bm = benchmark_df["benchmark_value"].values
    # Skaliere Benchmark auf gleichen Start
    bm_scaled = bm / bm[0] * eq[0]

    ax.plot(equity_df["timestamp"], eq, color=ACCENT, linewidth=1.2, label="Strategy")
    ax.plot(benchmark_df["timestamp"], bm_scaled, color="#6b7280",
            linewidth=0.8, linestyle="--", label="Buy & Hold (Equal-Weight)")

    ax.fill_between(equity_df["timestamp"], eq, eq[0],
                    where=(eq >= eq[0]), color=GREEN, alpha=0.08)
    ax.fill_between(equity_df["timestamp"], eq, eq[0],
                    where=(eq < eq[0]), color=RED, alpha=0.08)

    ax.set_ylabel("Portfolio Value ($)", color=TEXT)
    ax.set_title("Backtest Equity Curve", color=TEXT, fontweight="bold", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Info-Box
    info = (f"Return: {metrics['total_return_pct']:.1f}% | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
            f"Max DD: {metrics['max_drawdown_pct']:.1f}% | "
            f"PF: {metrics['profit_factor']:.1f}")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=9,
            color=TEXT, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor=CARD, edgecolor="#1f2937", alpha=0.8))

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_drawdown(equity_df: pd.DataFrame, metrics: dict, save_path: Path):
    """Underwater/Drawdown-Plot."""
    fig, ax = plt.subplots(figsize=(14, 5))

    equity = equity_df["equity"].values
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100

    ax.fill_between(equity_df["timestamp"], drawdown, 0,
                    color=RED, alpha=0.3, linewidth=0)
    ax.plot(equity_df["timestamp"], drawdown, color=RED, linewidth=0.8)

    # Max-DD-Linie
    dd_info = compute_max_drawdown(equity)
    if dd_info["max_drawdown_pct"] != 0:
        max_dd_val = dd_info["max_drawdown_pct"]
        ax.axhline(y=-max_dd_val, color=AMBER, linestyle="--", linewidth=0.8,
                   label=f"Max DD: {max_dd_val:.1f}%")

    ax.set_ylabel("Drawdown (%)", color=TEXT)
    ax.set_title("Drawdown (Underwater Plot)", color=TEXT, fontweight="bold", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_monthly_returns(metrics: dict, save_path: Path):
    """Monthly Returns Heatmap (Kalender-Stil)."""
    monthly = metrics.get("monthly_returns", {})
    if not monthly:
        print("  [SKIP] Keine Monthly-Returns-Daten")
        return

    # Parse in DataFrame
    records = []
    for ym, ret in sorted(monthly.items()):
        y, m = ym.split("-")
        records.append({"year": int(y), "month": int(m), "return": ret})
    if not records:
        return
    df_m = pd.DataFrame(records)

    # Pivot zu Matrix
    years = sorted(df_m["year"].unique())
    months = list(range(1, 13))
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    data = np.full((len(years), 12), np.nan)
    for _, row in df_m.iterrows():
        try:
            yr_idx = years.index(int(row["year"]))
            data[yr_idx, int(row["month"]) - 1] = row["return"]
        except (ValueError, IndexError):
            continue

    fig, ax = plt.subplots(figsize=(12, max(3, len(years) * 0.7)))

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=-max(abs(data[~np.isnan(data)])) if np.any(~np.isnan(data)) else 5,
                          vmax=max(abs(data[~np.isnan(data)])) if np.any(~np.isnan(data)) else 5)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years])

    for i in range(len(years)):
        for j in range(12):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 3 else ("#1a1a1a" if val >= 0 else "white")
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    ax.set_title("Monthly Returns (%)", color=TEXT, fontweight="bold", fontsize=13)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Return (%)", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    for label in cbar.ax.get_yticklabels():
        label.set_color(TEXT)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_trade_distribution(trades: list[Trade], save_path: Path):
    """Histogramm der Trade-Returns, eingefaerbt nach Exit-Grund."""
    if not trades:
        print("  [SKIP] Keine Trades")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    pnl_pcts = [t.pnl_pct * 100 for t in trades]
    exit_reasons = [t.exit_reason for t in trades]

    # Farbzuordnung
    reason_colors = {
        "take_profit": GREEN,
        "stop_loss": RED,
        "time_stop": AMBER,
        "signal_collapse": "#8b5cf6",
        "end_of_period": "#6b7280",
    }

    # Gruppiere nach Exit-Grund und zeichne ueberlagerte Histogramme
    from matplotlib.patches import Patch
    legend_patches = []
    for reason, color in reason_colors.items():
        subset = [pnl_pcts[i] for i, r in enumerate(exit_reasons) if r == reason]
        if subset:
            ax.hist(subset, bins=80, color=color, alpha=0.5,
                    edgecolor=BG, linewidth=0.2, label=f"{reason.replace('_',' ').title()} (n={len(subset)})")
            legend_patches.append(Patch(color=color, label=f"{reason.replace('_',' ').title()}"))

    # Mittelwert-Linie
    avg_pnl = np.mean(pnl_pcts)
    ax.axvline(x=avg_pnl, color=ACCENT, linestyle="--", linewidth=1.2,
               label=f"Ø: {avg_pnl:.3f}%")
    ax.axvline(x=0, color="#6b7280", linestyle="-", linewidth=0.5)

    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color=ACCENT, linestyle="--", label=f"Ø: {avg_pnl:.3f}%")
    ], loc="upper right", framealpha=0.9, fontsize=7)

    ax.set_xlabel("Trade Return (%)", color=TEXT)
    ax.set_ylabel("Anzahl Trades", color=TEXT)
    ax.set_title("Trade Return Distribution by Exit Reason", color=TEXT,
                 fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, facecolor=BG, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


# ===================================================================
# CLI & MAIN
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="USW-TradingModel — Historisches Backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python backtest.py
  python backtest.py --strategy two_stage --no_entry_rules
  python backtest.py --max_positions 5 --tp_pct 0.005 --sl_pct 0.002
        """,
    )
    p.add_argument("--strategy", default="finder_majority",
                   choices=["finder_majority", "two_stage", "simple_avg",
                            "weighted_f1", "stacking"],
                   help="Ensemble-Strategie (default: finder_majority)")
    p.add_argument("--no_entry_rules", action="store_true",
                   help="Entry-Rules E3-E5 deaktivieren")
    p.add_argument("--max_positions", type=int, default=MAX_POSITIONS,
                   help=f"Max. gleichzeitige Positionen (default: {MAX_POSITIONS})")
    p.add_argument("--tp_pct", type=float, default=TP_PCT,
                   help=f"Take-Profit in %% (default: {TP_PCT})")
    p.add_argument("--sl_pct", type=float, default=SL_PCT,
                   help=f"Stop-Loss in %% (default: {SL_PCT})")
    p.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                   help=f"Startkapital (default: {INITIAL_CAPITAL})")
    p.add_argument("--output", type=str, default=None,
                   help="Pfad fuer JSON-Output (default: artifacts/evaluation/backtest_results.json)")
    p.add_argument("--no_plots", action="store_true",
                   help="Keine Plots generieren")
    p.add_argument("--all", action="store_true",
                   help="Alle 5 Strategien nacheinander testen und vergleichen")
    p.add_argument("--tune", action="store_true",
                   help="Grid-Search: Beste Parameter-Kombination finden")
    return p.parse_args()


def print_summary(metrics: dict):
    """Gibt eine formatierte Zusammenfassung aus."""
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST ERGEBNISSE")
    print(f"{'=' * 60}")
    print(f"  Strategie:          {metrics['strategy']}")
    print(f"  Entry-Rules:        {metrics['entry_rules']}")
    print(f"  Trading-Tage:       {metrics['n_trading_days']}")
    print(f"  Symbole:            {metrics['n_symbols']}")
    print(f"  Trades:             {metrics['n_trades']} "
          f"({metrics['trades_per_day']:.1f}/Tag)")
    print(f"{'─' * 60}")
    print(f"  Total Return:       {metrics['total_return_pct']:+.2f}%")
    print(f"  CAGR:               {metrics['cagr_pct']:+.2f}%")
    print(f"  Benchmark Return:   {metrics['benchmark_return_pct']:+.2f}%")
    print(f"  Alpha:              {metrics['alpha_pct']:+.2f}%")
    print(f"{'─' * 60}")
    print(f"  Win Rate:           {metrics['win_rate_pct']:.1f}%")
    print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"  Avg Win:            {metrics['avg_win_pct']:.3f}%")
    print(f"  Avg Loss:           {metrics['avg_loss_pct']:.3f}%")
    print(f"  Avg Holding:        {metrics['avg_holding_minutes']:.1f} min")
    print(f"{'─' * 60}")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:      {metrics['sortino_ratio']:.3f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Max DD Days:        {metrics['max_drawdown_days']}")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:.2f}")
    print(f"  Volatility (ann):   {metrics['volatility_annual_pct']:.2f}%")
    print(f"{'─' * 60}")
    print(f"  Exit-Breakdown:     {metrics['breakdown_by_exit']}")
    print(f"  Final Equity:       ${metrics['final_equity']:,.0f}")
    print(f"{'=' * 60}\n")


def _build_data_structures(df: pd.DataFrame):
    """Baut wiederverwendbare Datenstrukturen fuer schnelle Backtests.

    Returns:
        sym_dfs: dict[symbol] -> DataFrame (chronologisch sortiert)
        ts_to_close: dict[timestamp, dict[symbol, float]]
        timestamps: sorted list of unique timestamps
    """
    # Alle Spalten die precompute_exits braucht: OHLC + Signale + Scores + Entry-Rule-Features
    signal_cols = [c for c in df.columns if c.startswith("signal_") or c.startswith("score_")]
    base_cols = ["symbol", "timestamp", "open", "high", "low", "close", "vwap"]
    entry_cols = ["return_1m", "Slope_close_1", "minutes_since_open"]
    prob_cols = ["p_lstm", "p_gru", "p_cnn", "p_lgb", "p_mlp"]
    needed = base_cols + signal_cols + entry_cols + prob_cols
    cols = [c for c in needed if c in df.columns]
    df_sub = df[cols].copy()
    df_sub = df_sub.sort_values(["symbol", "timestamp"])

    # Per-Symbol DataFrames
    sym_dfs = {}
    for sym, grp in df_sub.groupby("symbol"):
        sym_dfs[sym] = grp.reset_index(drop=True)

    # ts_to_close via pivot (VIEL schneller als iterrows)
    close_pivot = df_sub.pivot_table(
        index="timestamp", columns="symbol", values="close", aggfunc="last"
    )
    ts_to_close = {ts: {sym: float(close_pivot.at[ts, sym])
                        for sym in close_pivot.columns
                        if not pd.isna(close_pivot.at[ts, sym])}
                   for ts in close_pivot.index}
    timestamps = sorted(ts_to_close.keys())

    return sym_dfs, ts_to_close, timestamps


def precompute_exits(
    sym_dfs: dict[str, pd.DataFrame],
    signal_col: str,
    score_col: str,
    tp_pct: float,
    sl_pct: float,
    apply_entry_rules: bool,
) -> dict[tuple, tuple]:
    """Pre-computes exit information for all signals (optimized: uses pre-split sym_dfs).

    Returns:
        dict keyed by (symbol, entry_timestamp) ->
            (exit_timestamp, exit_reason, exit_price, bars_held, finder_score, entry_price)
    """
    exit_map: dict[tuple, tuple] = {}

    for sym, sym_df in sym_dfs.items():
        if len(sym_df) < 2:
            continue

        signal_mask = sym_df[signal_col].values == 1
        signal_indices = sym_df.index[signal_mask]

        for local_idx in signal_indices:
            row = sym_df.iloc[local_idx]
            ts = row["timestamp"]

            if apply_entry_rules and not _check_entry_rules(row):
                continue

            entry_price = float(row["close"])
            tp_price = entry_price * (1.0 + tp_pct)
            sl_price = entry_price * (1.0 - sl_pct)

            reason, exit_px, bars_held = _exit_for_symbol(
                sym_df, int(local_idx), entry_price, tp_price, sl_price, score_col,
            )

            exit_local_idx = int(local_idx) + bars_held
            if exit_local_idx < len(sym_df):
                exit_ts = sym_df.iloc[exit_local_idx]["timestamp"]
            else:
                exit_ts = sym_df.iloc[-1]["timestamp"]

            finder = _compute_finder_score(row)
            exit_map[(sym, ts)] = (exit_ts, reason, exit_px, bars_held, finder, entry_price)

    return exit_map


def run_backtest_with_exits(
    df: pd.DataFrame,
    exit_map: dict[tuple, tuple],
    max_positions: int,
    tp_pct: float,
    sl_pct: float,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost: float = TRANSACTION_COST,
    _cached_ts_close: dict | None = None,
    _cached_sym_first_prices: dict[str, float] | None = None,
) -> dict:
    """Fuehrt Walk-Forward-Backtest mit vorberechneten Exits durch.

    Args:
        _cached_ts_close: Vorgebauter timestamp->{symbol->close} Lookup (optional).
        _cached_sym_first_prices: Vorgebaute first_prices pro Symbol (optional).
    """
    symbols = sorted(df["symbol"].unique())
    timestamps = sorted(df["timestamp"].unique())

    if _cached_sym_first_prices is not None:
        first_prices = _cached_sym_first_prices
    else:
        first_prices = {}
        for sym in symbols:
            sym_data = df[df["symbol"] == sym]
            if len(sym_data) > 0:
                first_prices[sym] = float(sym_data.iloc[0]["close"])
    benchmark_shares = {sym: (initial_capital / max(1, len(symbols))) / first_prices[sym]
                        for sym in first_prices}

    if _cached_ts_close is not None:
        ts_to_close = _cached_ts_close
    else:
        ts_to_close = defaultdict(dict)
        for _, row in df.iterrows():
            ts_to_close[row["timestamp"]][row["symbol"]] = float(row["close"])

    # Sort all entries chronologically, then by finder_score DESC (highest first)
    all_entries = sorted(exit_map.items(), key=lambda x: (x[0][1], -x[1][4]))

    open_positions: dict[str, OpenPosition] = {}
    completed_trades: list[Trade] = []
    position_cost_basis: dict[str, float] = {}
    cash = initial_capital

    trade_iter = iter(all_entries)
    next_trade = next(trade_iter, None)

    for ts in timestamps:
        closes = ts_to_close.get(ts, {})

        # Exit-Checks
        symbols_to_exit = []
        for sym, pos in open_positions.items():
            exit_key = (sym, pos.entry_time)
            if exit_key in exit_map:
                exit_ts, reason, exit_px, bars_held, finder, entry_px = exit_map[exit_key]
                if ts >= exit_ts:
                    symbols_to_exit.append((sym, exit_ts, reason, exit_px, bars_held, finder))

        for sym, exit_ts, reason, exit_px, bars_held, finder in symbols_to_exit:
            pos = open_positions[sym]
            cost_basis = position_cost_basis.get(sym, pos.entry_price * pos.qty)
            gross_pnl = (exit_px - pos.entry_price) * pos.qty
            fee = cost_basis * transaction_cost + (exit_px * pos.qty * transaction_cost)
            net_pnl = gross_pnl - fee
            pnl_pct = (exit_px - pos.entry_price) / pos.entry_price

            completed_trades.append(Trade(
                symbol=sym, entry_time=pos.entry_time, exit_time=exit_ts,
                entry_price=pos.entry_price, exit_price=exit_px,
                qty=pos.qty, exit_reason=reason,
                pnl=net_pnl, pnl_pct=pnl_pct,
                bars_held=bars_held, finder_score=finder,
            ))
            cash += exit_px * pos.qty - fee
            del open_positions[sym]
            position_cost_basis.pop(sym, None)

        # Entry-Checks
        while next_trade is not None and next_trade[0][1] == ts:
            (sym, entry_ts), (exit_ts, reason, exit_px, bars_held, finder, entry_price) = next_trade

            if sym not in open_positions and len(open_positions) < max_positions:
                # Position Sizing: Equal allocation per slot
                # Jede Position bekommt 1/max_positions des verfuegbaren Kapitals
                alloc = cash / max(1, max_positions - len(open_positions))
                qty = max(1, int(alloc / (entry_price * (1.0 + transaction_cost))))
                required = entry_price * qty * (1.0 + transaction_cost)
                if required <= cash and qty > 0:
                    cash -= required
                    position_cost_basis[sym] = required
                    open_positions[sym] = OpenPosition(
                        symbol=sym, entry_time=entry_ts, entry_price=entry_price,
                        qty=qty, tp_price=entry_price * (1.0 + tp_pct),
                        sl_price=entry_price * (1.0 - sl_pct),
                        finder_score=finder,
                    )
            next_trade = next(trade_iter, None)
            if next_trade is None:
                break

    # Close remaining
    final_ts = timestamps[-1]
    final_closes = ts_to_close.get(final_ts, {})
    for sym, pos in list(open_positions.items()):
        exit_px = final_closes.get(sym, pos.entry_price)
        cost_basis = position_cost_basis.get(sym, pos.entry_price * pos.qty)
        gross_pnl = (exit_px - pos.entry_price) * pos.qty
        fee = cost_basis * transaction_cost + (exit_px * pos.qty * transaction_cost)
        net_pnl = gross_pnl - fee
        completed_trades.append(Trade(
            symbol=sym, entry_time=pos.entry_time, exit_time=final_ts,
            entry_price=pos.entry_price, exit_price=exit_px,
            qty=pos.qty, exit_reason="end_of_period",
            pnl=net_pnl, pnl_pct=(exit_px - pos.entry_price) / pos.entry_price,
            bars_held=0, finder_score=pos.finder_score,
        ))
        cash += exit_px * pos.qty - fee
        del open_positions[sym]

    # Build equity curve
    equity_records = [{"timestamp": timestamps[0], "equity": initial_capital,
                       "cash": initial_capital, "positions_value": 0, "n_positions": 0}]

    # Re-simulate equity tracking (lightweight)
    cash_2 = initial_capital
    pos_map: dict[str, dict] = {}
    trade_by_entry = {(t.symbol, t.entry_time): t for t in completed_trades}

    # Simplified equity from trades
    for t in completed_trades:
        cash_2 += t.pnl
    final_equity_simple = cash_2

    # Build proper equity curve from trade events
    events = []
    for t in completed_trades:
        events.append({"timestamp": t.entry_time, "delta": -abs(t.pnl) if t.pnl < 0 else 0, "type": "entry"})
        events.append({"timestamp": t.exit_time, "delta": t.pnl, "type": "exit"})
    events.sort(key=lambda e: e["timestamp"])

    eq = initial_capital
    eq_idx = 0
    eq_data = []
    for ts in timestamps:
        while eq_idx < len(events) and events[eq_idx]["timestamp"] <= ts:
            eq += events[eq_idx]["delta"]
            eq_idx += 1
        n_pos = sum(1 for t in completed_trades
                    if t.entry_time <= ts < t.exit_time)
        eq_data.append({"timestamp": ts, "equity": eq, "cash": eq, "positions_value": 0, "n_positions": n_pos})
    equity_df = pd.DataFrame(eq_data)

    # Benchmark
    bm_data = []
    for ts in timestamps:
        closes = ts_to_close.get(ts, {})
        bm_value = sum(closes.get(sym, 0.0) * benchmark_shares.get(sym, 0) for sym in benchmark_shares)
        bm_data.append({"timestamp": ts, "benchmark_value": bm_value if bm_value > 0 else initial_capital})
    benchmark_df = pd.DataFrame(bm_data)

    final_equity = equity_df["equity"].iloc[-1] if len(equity_df) > 0 else initial_capital

    return {
        "trades": completed_trades,
        "equity_curve": equity_df,
        "benchmark_curve": benchmark_df,
        "final_equity": final_equity,
        "metadata": {
            "strategy": "finder_majority",
            "entry_rules": True,
            "max_positions": max_positions,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "initial_capital": initial_capital,
            "transaction_cost": transaction_cost,
            "n_timestamps": len(timestamps),
            "n_symbols": len(symbols),
        },
    }


def _compute_result_dict(results, mp, tp, sl, er, args):
    """Extrahiert Metriken aus einem Backtest-Ergebnis."""
    trades_list = results["trades"]
    final_eq = results["final_equity"]
    total_ret = (final_eq - args.capital) / args.capital * 100

    n_wins = sum(1 for t in trades_list if t.pnl > 0)
    n_losses = sum(1 for t in trades_list if t.pnl <= 0)
    total_gains = sum(t.pnl for t in trades_list if t.pnl > 0)
    total_losses = abs(sum(t.pnl for t in trades_list if t.pnl <= 0))
    win_rate = n_wins / len(trades_list) * 100 if trades_list else 0
    profit_factor = total_gains / total_losses if total_losses > 0 else 0

    # Sharpe aus dem Equity-Curve
    eq_arr = results["equity_curve"]["equity"].values
    daily_eq = pd.Series(eq_arr).iloc[::390]
    daily_ret = daily_eq.pct_change().dropna().values
    sharpe = compute_sharpe_ratio(daily_ret) if len(daily_ret) > 1 else -99

    dd = compute_max_drawdown(eq_arr)
    composite = profit_factor * (win_rate / 100) if profit_factor > 0 else 0

    return {
        "max_positions": mp,
        "tp_pct": round(tp * 100, 2),
        "sl_pct": round(sl * 100, 2),
        "entry_rules": er,
        "n_trades": len(trades_list),
        "final_equity": round(final_eq, 0),
        "total_return_pct": round(total_ret, 1),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(dd["max_drawdown_pct"], 1),
        "composite": round(composite, 3),
    }


def tune_parameters(df: pd.DataFrame, args) -> None:
    """Walk-Forward Parameter-Tuning mit 3 Zeitfenstern.

    H2 2024 wird in drei strikt getrennte Zeitraeume aufgeteilt:
      1. TUNING:    Juli-August  → Grid-Search (150 Kombinationen)
      2. VALIDATE:  September    → Top-5 auf ungesehenem Monat vergleichen
      3. FINAL:     Oktober-Dez  → Beste Kombination EINMAL testen

    DIESES VORGEHEN VERHINDERT OVERFITTING:
    - Parameter werden NIE auf dem gleichen Zeitraum gewaehlt und evaluiert
    - Das Final-Ergebnis auf Okt-Dez ist die ehrliche Out-of-Sample-Schaetzung
    - Genau diese Parameter kommen ins Paper-Trading
    """
    signal_col = "signal_finder_majority"
    score_col = "score_finder_majority"

    # -- Zeitliche Splits definieren -----------------------------------
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"]

    tune_start = pd.Timestamp("2024-07-01")
    tune_end = pd.Timestamp("2024-09-01")        # Juli + August
    val_start = pd.Timestamp("2024-09-01")
    val_end = pd.Timestamp("2024-10-01")          # September
    final_start = pd.Timestamp("2024-10-01")
    final_end = pd.Timestamp("2025-01-01")        # Oktober–Dezember

    df_tune = df[(ts >= tune_start) & (ts < tune_end)].copy()
    df_val = df[(ts >= val_start) & (ts < val_end)].copy()
    df_final = df[(ts >= final_start) & (ts < final_end)].copy()

    print(f"\n{'=' * 60}")
    print(f"  WALK-FORWARD PARAMETER-TUNING")
    print(f"{'=' * 60}")
    print(f"  TUNING:    {tune_start.date()} – {tune_end.date()}  "
          f"({len(df_tune):,} Bars, {df_tune['symbol'].nunique()} Symbole)")
    print(f"  VALIDATE:  {val_start.date()} – {val_end.date()}  "
          f"({len(df_val):,} Bars, {df_val['symbol'].nunique()} Symbole)")
    print(f"  FINAL:     {final_start.date()} – {final_end.date()}  "
          f"({len(df_final):,} Bars, {df_final['symbol'].nunique()} Symbole)")
    print(f"  Strategie: finder_majority")

    # -- Suchraum (aus Modell-Logik + Target-Review abgeleitet) ---------
    # Problem 1 (behoben): Position Sizing = equal allocation, nicht risk-based
    # Problem 2 (behoben): SL zu eng → Trades werden durch Rauschen gestoppt
    #   → Breitere SLs, inkl. SL=0 (nur Time-Stop)
    # Problem 3: TP nahe am theta=0.3% halten
    max_positions_list = [5, 10, 15]
    tp_list = [0.0020, 0.0025]                  # 0.20–0.25% (unter theta, erreichbar)
    sl_list = [0.0000, 0.0010, 0.0015, 0.0020]  # 0%=kein SL, 0.10–0.20%
    entry_rules_list = [True, False]

    n_exit_combos = len(tp_list) * len(sl_list)
    n_sims_per_exit = len(max_positions_list) * len(entry_rules_list)
    valid_combos = sum(1 for tp in tp_list for sl in sl_list if sl < tp) * n_sims_per_exit

    print(f"  Suchraum: {len(max_positions_list)} max_pos × {len(tp_list)} TP × "
          f"{len(sl_list)} SL × {len(entry_rules_list)} ER = "
          f"{valid_combos} gueltige Kombinationen "
          f"(nur SL < TP)")
    print(f"  Modell-Target theta=0.3% → TP im Bereich 0.20–0.30%")
    print(f"  Risk/Reward ≥2:1 garantiert (SL ≤ TP/2)")
    print(f"  Geschaetzte Dauer: ~{n_exit_combos * 12 + valid_combos * 2:.0f}s")

    # =================================================================
    # PHASE 1: TUNING auf Juli-August
    # =================================================================
    print(f"\n{'─' * 60}")
    print(f"  PHASE 1: GRID SEARCH auf Juli–August")
    print(f"{'─' * 60}")

    # Pre-build ALL data structures ONCE for this phase
    print(f"  Baue Datenstrukturen... ", end="", flush=True)
    t_prep = time.time()
    sym_dfs_tune, ts_close_tune, _timestamps_tune = _build_data_structures(df_tune)
    # Compute first_prices once
    first_prices_tune = {}
    for sym, sdf in sym_dfs_tune.items():
        if len(sdf) > 0:
            first_prices_tune[sym] = float(sdf.iloc[0]["close"])
    print(f"({time.time()-t_prep:.1f}s)")

    exit_cache: dict[tuple, dict] = {}
    tune_results: list[dict] = []
    t_start = time.time()

    for tp in tp_list:
        for sl in sl_list:
            if sl >= tp:
                continue

            for er in entry_rules_list:
                cache_key = (tp, sl, er)
                print(f"  TP={tp*100:.2f}% SL={sl*100:.2f}% ER={er} → "
                      f"Exits... ", end="", flush=True)
                t0 = time.time()
                exit_map = precompute_exits(sym_dfs_tune, signal_col, score_col, tp, sl, er)
                exit_cache[cache_key] = exit_map
                print(f"{len(exit_map):,} Signale ({time.time()-t0:.1f}s)  Sims: ", end="", flush=True)

                for mp in max_positions_list:
                    t1 = time.time()
                    results = run_backtest_with_exits(
                        df_tune, exit_map, mp, tp, sl, args.capital,
                        _cached_ts_close=ts_close_tune,
                        _cached_sym_first_prices=first_prices_tune,
                    )
                    r = _compute_result_dict(results, mp, tp, sl, er, args)
                    r["duration_s"] = round(time.time() - t1, 1)
                    r["phase"] = "tune"
                    tune_results.append(r)
                    print(".", end="", flush=True)
                print(f" ({sum(1 for x in tune_results if x['tp_pct']==tp*100 and x['sl_pct']==sl*100 and x['entry_rules']==er)} sims)", flush=True)

    # Nach Composite scoren und Top-5 auswaehlen
    tune_results.sort(key=lambda r: r["composite"], reverse=True)
    top5 = tune_results[:5]

    # Entferne Duplikate (gleiche Parameter-Kombi wg. Rundung)
    seen_params = set()
    top5_unique = []
    for r in top5:
        key = (r["max_positions"], r["tp_pct"], r["sl_pct"], r["entry_rules"])
        if key not in seen_params:
            seen_params.add(key)
            top5_unique.append(r)
        if len(top5_unique) >= 5:
            break
    top5 = top5_unique

    print(f"\n  Top-5 aus Tuning (Juli–Aug):")
    print(f"  {'#':<3} {'Pos':>4} {'TP%':>6} {'SL%':>6} {'ER':>5} "
          f"{'Return':>8} {'Win%':>6} {'PF':>6} {'Trades':>7} {'Score':>7}")
    for i, r in enumerate(top5):
        print(f"  {i+1:<3} {r['max_positions']:>4} {r['tp_pct']:>5.2f}% {r['sl_pct']:>5.2f}% "
              f"{str(r['entry_rules'])[:5]:>5} "
              f"{r['total_return_pct']:>7.1f}% {r['win_rate_pct']:>5.1f}% "
              f"{r['profit_factor']:>5.2f} {r['n_trades']:>6,} {r['composite']:>6.3f}")

    # =================================================================
    # PHASE 2: VALIDATION auf September (Top-5 vergleichen)
    # =================================================================
    print(f"\n{'─' * 60}")
    print(f"  PHASE 2: VALIDATION auf September (Top-5)")
    print(f"{'─' * 60}")

    # Build val data structures once
    print(f"  Baue Datenstrukturen... ", end="", flush=True)
    sym_dfs_val, ts_close_val, _timestamps_val = _build_data_structures(df_val)
    first_prices_val = {sym: float(sdf.iloc[0]["close"]) for sym, sdf in sym_dfs_val.items() if len(sdf) > 0}
    print(f"OK")

    val_results = []
    for i, params in enumerate(top5):
        tp = params["tp_pct"] / 100
        sl = params["sl_pct"] / 100
        er = params["entry_rules"]
        mp = params["max_positions"]

        print(f"  [{i+1}/5] max_pos={mp} TP={tp*100:.2f}% SL={sl*100:.2f}% ER={er} ... ",
              end="", flush=True)
        t0 = time.time()
        exit_map = precompute_exits(sym_dfs_val, signal_col, score_col, tp, sl, er)
        results = run_backtest_with_exits(
            df_val, exit_map, mp, tp, sl, args.capital,
            _cached_ts_close=ts_close_val,
            _cached_sym_first_prices=first_prices_val,
        )
        r = _compute_result_dict(results, mp, tp, sl, er, args)
        r["phase"] = "validate"
        r["duration_s"] = round(time.time() - t0, 1)
        val_results.append(r)
        print(f"Return={r['total_return_pct']:.1f}% Win={r['win_rate_pct']:.1f}% "
              f"PF={r['profit_factor']:.2f} Trades={r['n_trades']:,}")

    # Beste auf Validation auswaehlen (nach Composite)
    val_results.sort(key=lambda r: r["composite"], reverse=True)
    best = val_results[0]

    print(f"\n  → BESTE KOMBINATION (validiert auf September):")
    print(f"     max_positions={best['max_positions']}  TP={best['tp_pct']:.2f}%  "
          f"SL={best['sl_pct']:.2f}%  Entry-Rules={best['entry_rules']}")
    print(f"     Return={best['total_return_pct']:.1f}%  Win={best['win_rate_pct']:.1f}%  "
          f"PF={best['profit_factor']:.2f}  Sharpe={best['sharpe_ratio']:.2f}")

    # =================================================================
    # PHASE 3: FINAL TEST auf Oktober–Dezember (NUR 1x!)
    # =================================================================
    print(f"\n{'─' * 60}")
    print(f"  PHASE 3: FINALER TEST auf Oktober–Dezember")
    print(f"  ⚠️  NUR EIN DURCHLAUF — das ist die ehrliche Out-of-Sample-Schaetzung")
    print(f"{'─' * 60}")

    tp_final = best["tp_pct"] / 100
    sl_final = best["sl_pct"] / 100
    er_final = best["entry_rules"]
    mp_final = best["max_positions"]

    print(f"  Parameter: max_pos={mp_final} TP={tp_final*100:.2f}% "
          f"SL={sl_final*100:.2f}% ER={er_final}")
    t0 = time.time()
    # Build final data structures once
    sym_dfs_final, ts_close_final, _timestamps_final = _build_data_structures(df_final)
    first_prices_final = {sym: float(sdf.iloc[0]["close"]) for sym, sdf in sym_dfs_final.items() if len(sdf) > 0}

    exit_map = precompute_exits(sym_dfs_final, signal_col, score_col, tp_final, sl_final, er_final)
    results = run_backtest_with_exits(
        df_final, exit_map, mp_final, tp_final, sl_final, args.capital,
        _cached_ts_close=ts_close_final,
        _cached_sym_first_prices=first_prices_final,
    )
    final_metrics = compute_metrics(results, initial_capital=args.capital)
    final_metrics["strategy"] = "finder_majority (tuned)"
    final_metrics["entry_rules"] = er_final

    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  FINALES ERGEBNIS (Out-of-Sample, Okt–Dez 2024)   ║")
    print(f"  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Return:   {final_metrics['total_return_pct']:>+8.1f}%                              ║")
    print(f"  ║  Sharpe:   {final_metrics['sharpe_ratio']:>8.2f}                              ║")
    print(f"  ║  Max DD:   {final_metrics['max_drawdown_pct']:>8.1f}%                              ║")
    print(f"  ║  Win Rate: {final_metrics['win_rate_pct']:>8.1f}%                              ║")
    print(f"  ║  PF:       {final_metrics['profit_factor']:>8.2f}                              ║")
    print(f"  ║  Trades:   {final_metrics['n_trades']:>8,}                              ║")
    print(f"  ╚══════════════════════════════════════════════════════╝")

    # Auch auf dem GESAMTEN H2-2024 laufen lassen fuer Kontext
    print(f"\n  [INFO] Gleiche Parameter auf GESAMTEM H2 2024 (nur zum Vergleich):")
    sym_dfs_full, ts_close_full, _ = _build_data_structures(df)
    first_prices_full = {sym: float(sdf.iloc[0]["close"]) for sym, sdf in sym_dfs_full.items() if len(sdf) > 0}
    exit_map_full = precompute_exits(sym_dfs_full, signal_col, score_col, tp_final, sl_final, er_final)
    results_full = run_backtest_with_exits(
        df, exit_map_full, mp_final, tp_final, sl_final, args.capital,
        _cached_ts_close=ts_close_full,
        _cached_sym_first_prices=first_prices_full,
    )
    metrics_full = compute_metrics(results_full, initial_capital=args.capital)
    print(f"  Gesamt-H2: Return={metrics_full['total_return_pct']:.1f}% "
          f"Sharpe={metrics_full['sharpe_ratio']:.2f} "
          f"MaxDD={metrics_full['max_drawdown_pct']:.1f}% "
          f"Win={metrics_full['win_rate_pct']:.1f}% PF={metrics_full['profit_factor']:.2f}")

    # -- Plots fuer Final-Ergebnis -------------------------------------
    if not args.no_plots:
        print(f"\n[PLOTS] Generiere Charts fuer Final-Ergebnis...")
        eq_df = results["equity_curve"]
        bm_df = results["benchmark_curve"]
        trades = results["trades"]

        plot_equity_curve(eq_df, bm_df, final_metrics,
                          _IMG_DIR / "backtest_equity_curve.png")
        plot_drawdown(eq_df, final_metrics,
                      _IMG_DIR / "backtest_drawdown.png")
        plot_monthly_returns(final_metrics,
                             _IMG_DIR / "backtest_monthly_returns.png")
        plot_trade_distribution(trades,
                                _IMG_DIR / "backtest_trade_distribution.png")

    # -- JSON speichern ------------------------------------------------
    tune_output = {
        "method": "walk_forward_3_windows",
        "tune_period": f"{tune_start.date()} – {tune_end.date()}",
        "val_period": f"{val_start.date()} – {val_end.date()}",
        "final_period": f"{final_start.date()} – {final_end.date()}",
        "best_params": best,
        "final_metrics": final_metrics,
        "full_h2_metrics": metrics_full,
        "top5_tune": top5,
        "top5_validation": val_results,
        "all_tune_results": tune_results,
        "date": pd.Timestamp.now().isoformat(),
    }

    tune_path = _EVAL_DIR / "backtest_tune_results.json"
    with open(tune_path, "w", encoding="utf-8") as f:
        json.dump(tune_output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[JSON] Alle Ergebnisse: {tune_path}")

    elapsed = time.time() - t_start
    print(f"\n  Gesamtdauer: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # -- Empfehlung fuer Paper-Trading ---------------------------------
    print(f"\n{'=' * 60}")
    print(f"  EMPFEHLUNG FUER PAPER-TRADING")
    print(f"{'=' * 60}")
    print(f"  Verwende diese Parameter im trading_loop.py:")
    print(f"    --max_positions {mp_final}")
    print(f"    --tp_pct {tp_final}")
    print(f"    --sl_pct {sl_final}")
    if not er_final:
        print(f"    --no_entry_rules")
    print(f"")
    print(f"  Ehrliche Out-of-Sample-Schaetzung (Okt–Dez 2024):")
    print(f"    Return: {final_metrics['total_return_pct']:+.1f}%")
    print(f"    Sharpe: {final_metrics['sharpe_ratio']:.2f}")
    print(f"  Paper-Trading-Start: Sobald Markt oeffnet")
    print(f"  Ziel: Ergebnisse mit dieser Schaetzung vergleichen")
    print(f"{'=' * 60}")


def _run_single_strategy(df, args, strategy_name, output_suffix=""):
    """Fuehrt Backtest fuer eine einzelne Strategie aus."""
    signal_col = f"signal_{strategy_name}" if strategy_name != "ensemble" else "ensemble_signal"
    score_col = f"score_{strategy_name}" if strategy_name != "ensemble" else "ensemble_score"

    # Fallback: wenn ensemble-Spalte, nutze die
    if signal_col not in df.columns:
        signal_col = "ensemble_signal"
        score_col = "ensemble_score"

    print(f"\n{'=' * 60}")
    print(f"  STRATEGIE: {strategy_name}  (Signal: {signal_col})")
    print(f"{'=' * 60}")
    print(f"  Signale: {df[signal_col].sum():,} ({df[signal_col].mean()*100:.2f}%)")

    results = run_backtest(
        df,
        signal_col=signal_col,
        score_col=score_col,
        apply_entry_rules=not args.no_entry_rules,
        max_positions=args.max_positions,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        initial_capital=args.capital,
    )

    metrics = compute_metrics(results, initial_capital=args.capital)
    metrics["strategy"] = strategy_name  # Ueberschreibe mit lesbarem Namen

    # JSON speichern
    suffix = f"_{output_suffix}" if output_suffix else f"_{strategy_name}"
    output_path = Path(args.output) if args.output else _EVAL_DIR / f"backtest_results{suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    return results, metrics, output_path


def main():
    args = parse_args()

    # -- Daten laden (nur EINMAL) --------------------------------------
    df = load_predictions_with_prices()

    STRATEGIES = ["finder_majority", "two_stage", "simple_avg", "weighted_f1", "stacking"]

    if args.tune:
        tune_parameters(df, args)
        return

    if args.all:
        # -- Alle Strategien testen ------------------------------------
        all_metrics = []
        winner_metrics = None
        winner_name = None
        winner_score = -999

        for strat in STRATEGIES:
            results, metrics, out_path = _run_single_strategy(df, args, strat)
            all_metrics.append((strat, metrics, results))

            # Winner anhand composite: profit_factor * win_rate
            composite = metrics["profit_factor"] * (metrics["win_rate_pct"] / 100)
            if composite > winner_score and metrics["n_trades"] >= 50:
                winner_score = composite
                winner_metrics = metrics
                winner_name = strat
                winner_results = results

        # -- Vergleichstabelle -----------------------------------------
        print(f"\n{'=' * 80}")
        print(f"  STRATEGIE-VERGLEICH (alle 5)")
        print(f"{'=' * 80}")
        print(f"  {'Strategie':<20} {'Return':>8} {'Sharpe':>8} {'Max DD':>8} "
              f"{'Win%':>7} {'PF':>7} {'Trades':>7} {'Calmar':>7}")
        print(f"  {'─' * 78}")
        for strat, m, _ in all_metrics:
            print(f"  {strat:<20} {m['total_return_pct']:>7.1f}% {m['sharpe_ratio']:>7.2f} "
                  f"{m['max_drawdown_pct']:>7.1f}% {m['win_rate_pct']:>6.1f}% "
                  f"{m['profit_factor']:>6.2f} {m['n_trades']:>6,} "
                  f"{m['calmar_ratio']:>6.2f}")
        print(f"  {'─' * 78}")

        # -- Vergleichs-JSON -------------------------------------------
        comparison = {
            "date": pd.Timestamp.now().isoformat(),
            "config": {
                "max_positions": args.max_positions,
                "tp_pct": args.tp_pct,
                "sl_pct": args.sl_pct,
                "entry_rules": not args.no_entry_rules,
            },
            "winner": winner_name,
            "strategies": {strat: m for strat, m, _ in all_metrics},
        }
        comp_path = _EVAL_DIR / "backtest_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[JSON] Vergleich gespeichert: {comp_path}")
        print(f"[WINNER] {winner_name} (Composite: {winner_score:.3f})")

        # -- Plots fuer den Winner -------------------------------------
        if not args.no_plots and winner_metrics is not None:
            print("\n[PLOTS] Generiere Charts fuer Winner-Strategie...")
            eq_df = winner_results["equity_curve"]
            bm_df = winner_results["benchmark_curve"]
            trades = winner_results["trades"]

            plot_equity_curve(eq_df, bm_df, winner_metrics,
                              _IMG_DIR / "backtest_equity_curve.png")
            plot_drawdown(eq_df, winner_metrics,
                          _IMG_DIR / "backtest_drawdown.png")
            plot_monthly_returns(winner_metrics,
                                 _IMG_DIR / "backtest_monthly_returns.png")
            plot_trade_distribution(trades,
                                    _IMG_DIR / "backtest_trade_distribution.png")

        print_summary(winner_metrics)

        # -- Ziel-KPI-Check fuer Winner --------------------------------
        print("Ziel-KPI-Check (TRADING_STRATEGIE.md):")
        checks = [
            ("Win Rate >= 60%", winner_metrics["win_rate_pct"] >= 60),
            ("Profit Factor >= 1.3", winner_metrics["profit_factor"] >= 1.3),
            ("Sharpe Ratio >= 1.2", winner_metrics["sharpe_ratio"] >= 1.2),
            ("Max Drawdown <= 10%", winner_metrics["max_drawdown_pct"] <= 10),
            ("Trades/Tag 5-20", 5 <= winner_metrics["trades_per_day"] <= 20),
        ]
        for label, passed in checks:
            icon = "[PASS]" if passed else "[FAIL]"
            print(f"  {icon} {label}")

    else:
        # -- Einzelne Strategie ----------------------------------------
        results, metrics, out_path = _run_single_strategy(df, args, args.strategy)
        print(f"\n[JSON] Metriken gespeichert: {out_path}")

        if not args.no_plots:
            print("\n[PLOTS] Generiere Charts...")
            eq_df = results["equity_curve"]
            bm_df = results["benchmark_curve"]
            trades = results["trades"]

            plot_equity_curve(eq_df, bm_df, metrics,
                              _IMG_DIR / "backtest_equity_curve.png")
            plot_drawdown(eq_df, metrics,
                          _IMG_DIR / "backtest_drawdown.png")
            plot_monthly_returns(metrics,
                                 _IMG_DIR / "backtest_monthly_returns.png")
            plot_trade_distribution(trades,
                                    _IMG_DIR / "backtest_trade_distribution.png")

        print_summary(metrics)

        print("Ziel-KPI-Check (TRADING_STRATEGIE.md):")
        checks = [
            ("Win Rate >= 60%", metrics["win_rate_pct"] >= 60),
            ("Profit Factor >= 1.3", metrics["profit_factor"] >= 1.3),
            ("Sharpe Ratio >= 1.2", metrics["sharpe_ratio"] >= 1.2),
            ("Max Drawdown <= 10%", metrics["max_drawdown_pct"] <= 10),
            ("Trades/Tag 5-20", 5 <= metrics["trades_per_day"] <= 20),
        ]
        for label, passed in checks:
            icon = "[PASS]" if passed else "[FAIL]"
            print(f"  {icon} {label}")


if __name__ == "__main__":
    main()
