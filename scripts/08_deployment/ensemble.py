"""
Ensemble Predictor -Kombiniert 5 Modelle zu einem Trading-Signal.

Zwei-Stufen-Architektur (Finder -> Filter):
  1. Finder (GRU, LSTM, CNN, LightGBM) screenen Breakout-Kandidaten (~87% Recall)
  2. Filter (MLP V2) bestaetigt mit hoher Precision (60%)
  3. Nur wenn BEIDE feuern -> Trade-Signal

Implementiert und vergleicht 5 Ensemble-Strategien:
  A) Two-Stage Finder->Filter (primar, wie in TRADING_STRATEGIE.md)
  B) Simple Average aller 5 Modelle
  C) Weighted Average (nach Validation-F1)
  D) Finder Majority + MLP Gate
  E) Stacking (Logistic Regression Meta-Learner)

Aufruf:
  python scripts/08_deployment/ensemble.py [--max_files N] [--threshold_optimize]

Output:
  artifacts/evaluation/ensemble_evaluation.json  -Strategie-Vergleich + Winner
  artifacts/evaluation/ensemble_predictions.parquet -Alle Predictions + Signale
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
)
from sklearn.linear_model import LogisticRegression

# --- Pfad zu den Training-Scripts -----------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_SCRIPT_DIR.parent / "06_model_training"))

from model import BreakoutModel
from model_sequential import LSTMBreakoutModel, GRUBreakoutModel, CNNBreakoutModel
from utils import (
    PROJECT_ROOT,
    get_device,
    enable_amp,
    load_features,
    load_scaler,
    load_class_balance,
    load_model,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Konstanten -----------------------------------------------------
SEQ_LEN = 30
INFER_BATCH_FLAT = 8192       # MLP / LightGBM (flache Vektoren)
INFER_BATCH_SEQ = 1024        # LSTM / GRU / CNN (Sequenzen)
DEVICE = None                  # Wird in main() gesetzt
USE_AMP = False

# Optimale Thresholds pro Modell (aus model_comparison.json)
FINDER_THRESHOLDS = {
    "lstm": 0.320,
    "gru": 0.334,
    "cnn": 0.314,
    "lightgbm": 0.355,
}
FILTER_THRESHOLD = 0.75       # MLP V2 (angehoben von 0.50 — war zu niedrig kalibriert)

# Validation F1-Scores (aus model_comparison.json)
F1_VAL = {
    "mlp": 0.563,
    "lstm": 0.642,
    "gru": 0.647,
    "cnn": 0.645,
    "lightgbm": 0.623,
}

# Finder-Gewichte fur Two-Stage (normalisiert aus F1 der Finder)
_F1_FINDERS = {k: v for k, v in F1_VAL.items() if k != "mlp"}
_F1_SUM = sum(_F1_FINDERS.values())
FINDER_WEIGHTS = {k: v / _F1_SUM for k, v in _F1_FINDERS.items()}

# Gewichte fur Weighted Average (alle 5 Modelle)
_F1_ALL_SUM = sum(F1_VAL.values())
WEIGHTS_ALL = {k: v / _F1_ALL_SUM for k, v in F1_VAL.items()}

# Output-Pfade
EVAL_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
PRE_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "pre_split"


# =======================================================================
# MODELL-LADEN
# =======================================================================
def load_all_models(n_features: int, device: torch.device) -> dict:
    """Ladt alle 5 trainierten Modelle.

    Returns:
        {"mlp": BreakoutModel, "lstm": LSTMBreakoutModel,
         "gru": GRUBreakoutModel, "cnn": CNNBreakoutModel,
         "lightgbm": lgb.Booster}
    """
    print("\n" + "=" * 60)
    print("LADE MODELLE")
    print("=" * 60)

    models = {}

    # -- MLP V2 (Filter) ------------------------------------------
    print("\n[1/5] MLP V2 (Filter)...")
    mlp = BreakoutModel(
        input_size=n_features,
        hidden_sizes=(128, 64, 32, 16),
        dropout=0.4,
        use_batch_norm=True,
    )
    mlp, meta_mlp = load_model(mlp, "mlp_model", device)
    mlp.eval()
    models["mlp"] = mlp
    if meta_mlp:
        print(f"  Best Val Loss: {meta_mlp.get('best_val_loss', 'N/A'):.4f}"
              if isinstance(meta_mlp.get('best_val_loss'), float)
              else f"  Best Val Loss: {meta_mlp.get('best_val_loss', 'N/A')}")
        print(f"  Epochen: {meta_mlp.get('total_epochs', 'N/A')}")

    # -- LSTM (Finder) --------------------------------------------
    print("\n[2/5] LSTM (Finder)...")
    lstm = LSTMBreakoutModel(
        input_size=n_features,
        hidden_size=128,
        num_layers=2,
        dropout=0.35,
        bidirectional=True,
    )
    lstm, _ = load_model(lstm, "lstm_model", device)
    lstm.eval()
    models["lstm"] = lstm

    # -- GRU (Finder) ---------------------------------------------
    print("\n[3/5] GRU (Finder)...")
    gru = GRUBreakoutModel(
        input_size=n_features,
        hidden_size=128,
        num_layers=2,
        dropout=0.35,
    )
    gru, _ = load_model(gru, "gru_model", device)
    gru.eval()
    models["gru"] = gru

    # -- CNN (Finder) ---------------------------------------------
    print("\n[4/5] CNN (Finder)...")
    cnn = CNNBreakoutModel(
        input_size=n_features,
        hidden_channels=64,
        kernel_sizes=(3, 5, 10),
        dropout=0.35,
    )
    cnn, _ = load_model(cnn, "cnn_model", device)
    cnn.eval()
    models["cnn"] = cnn

    # -- LightGBM (Finder) ----------------------------------------
    print("\n[5/5] LightGBM (Finder)...")
    lgb_path = PROJECT_ROOT / "artifacts" / "models" / "lightgbm_model.txt"
    if not lgb_path.exists():
        raise FileNotFoundError(f"LightGBM-Modell nicht gefunden: {lgb_path}")
    lgb_model = lgb.Booster(model_file=str(lgb_path))
    models["lightgbm"] = lgb_model
    print(f"  Baume: {lgb_model.num_trees()}")

    n_params = sum(
        sum(p.numel() for p in m.parameters())
        for name, m in models.items()
        if name != "lightgbm"
    )
    print(f"\n  PyTorch-Parameter gesamt: {n_params:,}")
    print("=" * 60)
    return models


# =======================================================================
# PER-SYMBOL INFERENCE
# =======================================================================
def predict_symbol(
    df: pd.DataFrame,
    scaler,
    features: list[str],
    models: dict,
    device: torch.device,
) -> pd.DataFrame | None:
    """Fuhrt Inference aller 5 Modelle auf einem Symbol durch.

    Data Flow:
      1. Sortiere chronologisch, entferne NaN-Targets
      2. Scaler.transform() -> X_scaled (N, 82)
      3. Flat-Modelle (MLP, LightGBM): predict auf X_scaled[30:] -> (N-30,)
      4. Sequenz-Modelle (LSTM, GRU, CNN): Sliding-Window [i-30:i], predict -> (N-30,)
      5. Alignment: Alle Outputs haben Lange N-30, target = y[30:]
      6. Return DataFrame mit allen Probabilities

    Returns:
        DataFrame mit [symbol, timestamp, target, p_mlp, p_lstm, p_gru, p_cnn, p_lgb,
                        return_1m, slope_close_1, minutes_since_open]
        oder None wenn zu wenige Zeilen.
    """
    symbol = str(df["symbol"].iloc[0])

    # Chronologisch sortieren
    df = df.sort_values("timestamp").reset_index(drop=True)

    # NaN-Targets entfernen (letzte 30 Zeilen haben unvollstandiges Forward-Window)
    df = df[df["breakout_30m"].notna()].reset_index(drop=True)
    n_rows = len(df)
    if n_rows <= SEQ_LEN:
        return None

    # -- Entry-Rule-Features VOR dem Scaling extrahieren -----------
    # return_1m und Slope_close_1: Rohwerte aus der Parquet-Datei
    #   (return_1m ist NICHT rolling-Z-normalisiert -> Rohwert)
    #   (Slope_close_1 > 0 im Z-Score-Raum = Trend ueber rollendem Mittel)
    # minutes_since_open: Aus Timestamp berechnen (in der Parquet-Datei
    #   bereits Z-normalisiert -> unbrauchbar fuer harte Schwellwerte)
    raw_return_1m = df["return_1m"].to_numpy(dtype=np.float32)
    raw_slope_close_1 = df["Slope_close_1"].to_numpy(dtype=np.float32)

    # Roh-Minuten seit 09:30 ET aus Timestamp rekonstruieren
    ts = pd.to_datetime(df["timestamp"])
    raw_minutes = ((ts.dt.hour - 9) * 60 + ts.dt.minute - 30).to_numpy(dtype=np.float32)

    # -- Features skalieren ---------------------------------------
    feature_array = scaler.transform(df[features].to_numpy(dtype="float64"))
    feature_array = feature_array.astype(np.float32)
    y_true = df["breakout_30m"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].values

    n_windows = n_rows - SEQ_LEN  # Anzahl gultiger Sequenz-Fenster

    # -- Flat-Modelle (MLP + LightGBM): Predict auf X[SEQ_LEN:] --
    X_flat = feature_array[SEQ_LEN:]  # (n_windows, 82)
    y_aligned = y_true[SEQ_LEN:]       # (n_windows,)
    ts_aligned = timestamps[SEQ_LEN:]  # (n_windows,)

    # Entry-Rule-Features alignen (gleicher Offset wie Predictions)
    return_1m_vals = raw_return_1m[SEQ_LEN:]
    slope_close_1_vals = raw_slope_close_1[SEQ_LEN:]
    minutes_since_open_vals = raw_minutes[SEQ_LEN:]

    # MLP: Batch-Inference auf GPU/CPU
    p_mlp = _predict_flat_torch(models["mlp"], X_flat, device, INFER_BATCH_FLAT)

    # LightGBM: CPU-Inference
    p_lgb = models["lightgbm"].predict(X_flat)  # numpy array

    # -- Sequenz-Modelle (LSTM, GRU, CNN): Sliding Windows --------
    # sliding_window_view erzeugt View in O(1), dann eine kontinuierliche Kopie
    from numpy.lib.stride_tricks import sliding_window_view

    X_view = sliding_window_view(feature_array, (SEQ_LEN,), axis=0)  # (N-29, 82, 30)
    X_seq = np.ascontiguousarray(
        X_view[:n_windows].transpose(0, 2, 1),  # (n_windows, SEQ_LEN, 82)
        dtype=np.float32,
    )

    p_lstm = _predict_seq_torch(models["lstm"], X_seq, device, INFER_BATCH_SEQ)
    p_gru = _predict_seq_torch(models["gru"], X_seq, device, INFER_BATCH_SEQ)
    p_cnn = _predict_seq_torch(models["cnn"], X_seq, device, INFER_BATCH_SEQ)

    # -- DataFrame bauen ------------------------------------------
    result = pd.DataFrame({
        "symbol": symbol,
        "timestamp": ts_aligned,
        "target": y_aligned,
        "p_mlp": p_mlp,
        "p_lstm": p_lstm,
        "p_gru": p_gru,
        "p_cnn": p_cnn,
        "p_lgb": p_lgb,
        "return_1m": return_1m_vals,
        "Slope_close_1": slope_close_1_vals,
        "minutes_since_open": minutes_since_open_vals,
    })

    return result


def _predict_flat_torch(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Batch-Inference fur flache PyTorch-Modelle (MLP)."""
    n = len(X)
    probs = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(X[start:end]).to(device, non_blocking=True)

            if USE_AMP and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = model(xb).squeeze(-1)
            else:
                out = model(xb).squeeze(-1)

            probs[start:end] = out.cpu().numpy()

    return probs


def _predict_seq_torch(
    model: torch.nn.Module,
    X: np.ndarray,   # (n, 30, 82)
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Batch-Inference fur sequenzielle PyTorch-Modelle (LSTM/GRU/CNN).

    WICHTIG: Diese Modelle geben rohe Logits aus (kein Sigmoid!).
    Wir mussen torch.sigmoid() anwenden.
    """
    n = len(X)
    probs = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(X[start:end]).to(device, non_blocking=True)

            if USE_AMP and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(xb).squeeze(-1)
            else:
                logits = model(xb).squeeze(-1)

            probs[start:end] = torch.sigmoid(logits).cpu().numpy()

    return probs


# =======================================================================
# ALLE TEST-DATEN VERARBEITEN
# =======================================================================
def predict_all_test_data(
    models: dict,
    scaler,
    features: list[str],
    device: torch.device,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Iteriert uber alle *_test.parquet Dateien und sammelt Predictions.

    Verarbeitet ein Symbol nach dem anderen fur Speichereffizienz.
    """
    test_files = sorted(PRE_SPLIT_PATH.glob("*_test.parquet"))
    if not test_files:
        raise FileNotFoundError(
            f"Keine Test-Dateien in {PRE_SPLIT_PATH} gefunden.\n"
            "Bitte zuerst: python scripts/03_pre_split_prep/main.py"
        )

    if max_files:
        test_files = test_files[:max_files]

    print(f"\n{'=' * 60}")
    print(f"INFERENCE auf {len(test_files)} Test-Symbolen")
    print(f"{'=' * 60}")

    all_dfs = []
    n_skipped = 0
    t_start = time.time()

    for i, fpath in enumerate(test_files):
        symbol = fpath.stem.replace("_test", "")
        df = pd.read_parquet(fpath)

        if len(df) == 0:
            n_skipped += 1
            continue

        result = predict_symbol(df, scaler, features, models, device)

        if result is None:
            n_skipped += 1
            continue

        all_dfs.append(result)

        if (i + 1) % 20 == 0 or i == len(test_files) - 1:
            elapsed = time.time() - t_start
            n_done = i + 1 - n_skipped
            print(f"  [{i+1}/{len(test_files)}] {symbol:5s}  "
                  f"({len(result):,} Zeilen)  "
                  f"{elapsed:.0f}s")

    t_total = time.time() - t_start

    if not all_dfs:
        raise RuntimeError("Keine gultigen Predictions -alle Symbole zu klein?")

    full_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n  Fertig in {t_total:.0f}s ({t_total/60:.1f} Min)")
    print(f"  Symbole: {len(all_dfs)} (ubersprungen: {n_skipped})")
    print(f"  Predictions: {len(full_df):,}")
    print(f"  Klassen-Balance: {full_df['target'].mean():.2%} Breakouts")
    print("=" * 60)

    return full_df


# =======================================================================
# ENSEMBLE-STRATEGIEN
# =======================================================================

def strategy_two_stage(df: pd.DataFrame) -> np.ndarray:
    """A) Two-Stage Finder->Filter (PRIMAER -wie in TRADING_STRATEGIE.md)

    Stage 1 -Finder (gewichteter Durchschnitt):
      finder_score = w_gru*p_gru + w_lstm*p_lstm + w_cnn*p_cnn + w_lgb*p_lgb
      candidate = finder_score > 0.33

    Stage 2 -Filter (MLP V2):
      confirmed = p_mlp > 0.50

    Signal = candidate AND confirmed
    """
    finder_score = (
        FINDER_WEIGHTS["gru"] * df["p_gru"].values
        + FINDER_WEIGHTS["lstm"] * df["p_lstm"].values
        + FINDER_WEIGHTS["cnn"] * df["p_cnn"].values
        + FINDER_WEIGHTS["lightgbm"] * df["p_lgb"].values
    )
    candidate = finder_score > 0.33
    confirmed = df["p_mlp"].values > FILTER_THRESHOLD
    signal = (candidate & confirmed).astype(int)
    return signal, finder_score


def strategy_simple_average(df: pd.DataFrame) -> np.ndarray:
    """B) Simple Average aller 5 Modelle.

    avg_score = mean(p_mlp, p_lstm, p_gru, p_cnn, p_lgb)
    Signal = avg_score > threshold (optimiert via F1)
    """
    avg_score = (
        df["p_mlp"].values
        + df["p_lstm"].values
        + df["p_gru"].values
        + df["p_cnn"].values
        + df["p_lgb"].values
    ) / 5.0
    return avg_score  # Threshold wird spater via F1-Optimierung bestimmt


def strategy_weighted_f1(df: pd.DataFrame) -> np.ndarray:
    """C) Weighted Average nach Validation-F1.

    weighted = w_mlp*p_mlp + w_lstm*p_lstm + w_gru*p_gru + w_cnn*p_cnn + w_lgb*p_lgb
    """
    weighted = (
        WEIGHTS_ALL["mlp"] * df["p_mlp"].values
        + WEIGHTS_ALL["lstm"] * df["p_lstm"].values
        + WEIGHTS_ALL["gru"] * df["p_gru"].values
        + WEIGHTS_ALL["cnn"] * df["p_cnn"].values
        + WEIGHTS_ALL["lightgbm"] * df["p_lgb"].values
    )
    return weighted


def strategy_finder_majority(df: pd.DataFrame) -> np.ndarray:
    """D) Finder Majority Vote + MLP Gate.

    Jeder Finder voted mit seinem optimierten Threshold.
    Signal = (votes >= 2) AND (p_mlp > 0.50)
    """
    votes = (
        (df["p_lstm"].values > FINDER_THRESHOLDS["lstm"]).astype(int)
        + (df["p_gru"].values > FINDER_THRESHOLDS["gru"]).astype(int)
        + (df["p_cnn"].values > FINDER_THRESHOLDS["cnn"]).astype(int)
        + (df["p_lgb"].values > FINDER_THRESHOLDS["lightgbm"]).astype(int)
    )
    mlp_confirms = df["p_mlp"].values > FILTER_THRESHOLD
    signal = (votes >= 2) & mlp_confirms
    return signal.astype(int), votes / 4.0  # Normalisierter Score [0,1]


def strategy_stacking(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> np.ndarray:
    """E) Stacking mit Logistic Regression Meta-Learner.

    Trainiert auf Validation-Daten, predicted auf Test-Daten.
    Features: 5 Modell-Probabilities
    """
    feature_cols = ["p_mlp", "p_lstm", "p_gru", "p_cnn", "p_lgb"]

    X_stack_train = df_train[feature_cols].values
    y_stack_train = df_train["target"].values

    X_stack_test = df_test[feature_cols].values

    meta = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    meta.fit(X_stack_train, y_stack_train)

    y_prob = meta.predict_proba(X_stack_test)[:, 1]
    return y_prob


# =======================================================================
# ENTRY RULES (E3-E5 aus TRADING_STRATEGIE.md)
# =======================================================================
def apply_entry_rules(df: pd.DataFrame, signal: np.ndarray) -> np.ndarray:
    """Wendet Entry-Regeln E3-E5 auf Ensemble-Signale an.

    E3: return_1m > 0 (Momentum positiv)
    E4: Slope_close_1 > 0 (Kurz-Trend positiv)
    E5: minutes_since_open in [120, 270] oder [330, 360]
        (10:00-12:00 ET oder 14:00-15:30 ET -keine Mittagsflaute)
    """
    e3 = df["return_1m"].values > 0
    e4 = df["Slope_close_1"].values > 0
    mins = df["minutes_since_open"].values
    e5 = ((mins >= 120) & (mins <= 270)) | ((mins >= 330) & (mins <= 360))

    filtered_signal = signal.copy()
    filtered_signal[~(e3 & e4 & e5)] = 0
    return filtered_signal


# =======================================================================
# EVALUATION
# =======================================================================

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Findet den F1-optimalen Schwellwert fur die Breakout-Klasse."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        p, r = precision[i], recall[i]
        if p + r > 0:
            f1_scores[i] = 2 * p * r / (p + r)

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    f1_default = float(f1_score(y_true, (y_prob > 0.5).astype(int)))

    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "f1_at_0.5": f1_default,
        "f1_improvement": best_f1 - f1_default,
    }


def estimate_profit_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tp_pct: float = 0.0036,
    fp_pct: float = -0.0015,
) -> dict:
    """Schatzt Trading-Profit-Metriken basierend auf Confusion Matrix.

    Vereinfachte Annahmen:
      - Jeder TP bringt +0.36% (Take Profit)
      - Jeder FP kostet -0.15% (Stop Loss)
      - FN und TN haben keine direkten Kosten (kein Trade)
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    n_trades = int(tp + fp)
    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_per_trade_pct": 0.0,
            "total_profit_pct": 0.0,
            "profit_factor": 0.0,
            "expected_return_bps": 0.0,
        }

    win_rate = tp / n_trades if n_trades > 0 else 0.0
    total_profit = tp * tp_pct + fp * fp_pct
    profit_per_trade = total_profit / n_trades

    total_gain = tp * tp_pct
    total_loss = abs(fp * fp_pct) if fp > 0 else 0.0
    profit_factor = total_gain / total_loss if total_loss > 0 else float("inf")

    return {
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "profit_per_trade_pct": float(profit_per_trade),
        "total_profit_pct": float(total_profit),
        "profit_factor": float(profit_factor),
        "expected_return_bps": float(profit_per_trade * 10000),  # Basispunkte
        "tp_count": int(tp),
        "fp_count": int(fp),
    }


def evaluate_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    strategy_name: str,
    baseline_acc: float,
    threshold_info: dict | None = None,
) -> dict:
    """Berechnet alle Metriken fur eine Ensemble-Strategie."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    n = len(y_true)
    accuracy = (tp + tn) / n if n > 0 else 0.0
    signal_rate = (tp + fp) / n if n > 0 else 0.0

    report = classification_report(
        y_true, y_pred,
        target_names=["Kein Breakout", "Breakout"],
        output_dict=True,
        zero_division=0,
    )

    profit = estimate_profit_metrics(y_true, y_pred)

    result = {
        "strategy_name": strategy_name,
        "n_samples": int(n),
        "accuracy": float(accuracy),
        "baseline_accuracy": float(baseline_acc),
        "improvement_pp": float((accuracy - baseline_acc) * 100),
        "signal_rate": float(signal_rate),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "classification_report": report,
        "trading_metrics": profit,
    }

    if threshold_info:
        result["threshold_info"] = threshold_info

    return result


# =======================================================================
# MAIN
# =======================================================================
def main():
    global DEVICE, USE_AMP

    parser = argparse.ArgumentParser(description="Ensemble Predictor -USW-TradingModel")
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Maximale Anzahl Test-Symbole (fur schnelle Tests)"
    )
    parser.add_argument(
        "--threshold_optimize", action="store_true", default=True,
        help="F1-optimale Thresholds fur kontinuierliche Strategien finden"
    )
    parser.add_argument(
        "--no_threshold_optimize", action="store_false", dest="threshold_optimize",
        help="Threshold-Optimierung uberspringen (nutze 0.5)"
    )
    parser.add_argument(
        "--entry_rules", action="store_true",
        help="Entry-Rules E3-E5 zusaetzlich zum Ensemble-Signal anwenden"
    )
    args = parser.parse_args()

    # -- Setup ----------------------------------------------------
    DEVICE = get_device(verbose=True)
    USE_AMP = enable_amp(DEVICE)

    features = load_features()
    scaler = load_scaler()
    balance = load_class_balance()
    baseline_acc = max(balance["positive_ratio"], 1 - balance["positive_ratio"])

    print(f"Features: {len(features)}")
    print(f"Baseline Accuracy: {baseline_acc:.2%}")
    print(f"Test-Symbole: {'alle' if args.max_files is None else args.max_files}")

    # -- Modelle laden --------------------------------------------
    models = load_all_models(len(features), DEVICE)

    # -- Inference auf Test-Daten ---------------------------------
    df_test = predict_all_test_data(
        models, scaler, features, DEVICE,
        max_files=args.max_files,
    )

    # -- Inference auf Validation-Daten (fur Stacking) ------------
    print(f"\n{'=' * 60}")
    print("INFERENCE auf Validation-Daten (fur Stacking)")
    print(f"{'=' * 60}")
    val_files = sorted(PRE_SPLIT_PATH.glob("*_validation.parquet"))
    if args.max_files:
        val_files = val_files[:args.max_files]

    val_dfs = []
    for i, fpath in enumerate(val_files):
        df = pd.read_parquet(fpath)
        if len(df) == 0:
            continue
        result = predict_symbol(df, scaler, features, models, DEVICE)
        if result is not None:
            val_dfs.append(result)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(val_files)}]")

    df_val = pd.concat(val_dfs, ignore_index=True) if val_dfs else df_test.copy()
    print(f"  Validation-Predictions: {len(df_val):,}")
    print("=" * 60)

    # -- Ensemble-Strategien evaluieren ---------------------------
    print(f"\n{'=' * 60}")
    print("ENSEMBLE-STRATEGIEN VERGLEICH")
    print(f"{'=' * 60}")

    y_test = df_test["target"].values.astype(int)
    strategies = {}

    # A) Two-Stage Finder -> Filter
    print("\n[A] Two-Stage Finder->Filter...")
    signal_a, score_a = strategy_two_stage(df_test)
    if args.entry_rules:
        signal_a = apply_entry_rules(df_test, signal_a)
    threshold_a = find_optimal_threshold(y_test, score_a)
    strategies["two_stage"] = evaluate_strategy(
        y_test, signal_a, score_a,
        "Two-Stage Finder->Filter", baseline_acc, threshold_a,
    )
    _print_strategy_summary(strategies["two_stage"])

    # B) Simple Average
    print("\n[B] Simple Average...")
    score_b = strategy_simple_average(df_test)
    threshold_b = find_optimal_threshold(y_test, score_b)
    tau_b = threshold_b["best_threshold"] if args.threshold_optimize else 0.5
    signal_b = (score_b > tau_b).astype(int)
    if args.entry_rules:
        signal_b = apply_entry_rules(df_test, signal_b)
    strategies["simple_average"] = evaluate_strategy(
        y_test, signal_b, score_b,
        "Simple Average (5 Models)", baseline_acc, threshold_b,
    )
    _print_strategy_summary(strategies["simple_average"])

    # C) Weighted by F1
    print("\n[C] Weighted Average (F1)...")
    score_c = strategy_weighted_f1(df_test)
    threshold_c = find_optimal_threshold(y_test, score_c)
    tau_c = threshold_c["best_threshold"] if args.threshold_optimize else 0.5
    signal_c = (score_c > tau_c).astype(int)
    if args.entry_rules:
        signal_c = apply_entry_rules(df_test, signal_c)
    strategies["weighted_f1"] = evaluate_strategy(
        y_test, signal_c, score_c,
        "Weighted Average (F1)", baseline_acc, threshold_c,
    )
    _print_strategy_summary(strategies["weighted_f1"])

    # D) Finder Majority + MLP
    print("\n[D] Finder Majority + MLP Gate...")
    signal_d, score_d = strategy_finder_majority(df_test)
    if args.entry_rules:
        signal_d = apply_entry_rules(df_test, signal_d)
    threshold_d = find_optimal_threshold(y_test, score_d)
    strategies["finder_majority"] = evaluate_strategy(
        y_test, signal_d, score_d,
        "Finder Majority + MLP Gate", baseline_acc, threshold_d,
    )
    _print_strategy_summary(strategies["finder_majority"])

    # E) Stacking (Logistic Regression)
    print("\n[E] Stacking (Logistic Regression Meta-Learner)...")
    score_e = strategy_stacking(df_val, df_test)
    threshold_e = find_optimal_threshold(y_test, score_e)
    tau_e = threshold_e["best_threshold"] if args.threshold_optimize else 0.5
    signal_e = (score_e > tau_e).astype(int)
    if args.entry_rules:
        signal_e = apply_entry_rules(df_test, signal_e)
    strategies["stacking"] = evaluate_strategy(
        y_test, signal_e, score_e,
        "Stacking (LogReg Meta-Learner)", baseline_acc, threshold_e,
    )
    _print_strategy_summary(strategies["stacking"])

    # -- Winner selektieren ---------------------------------------
    # Trading-Score: Profit-Faktor priorisiert (maximale Profitabilitat)
    # Fallback: F1 x Precision als Composite Score
    winner = None
    winner_trading_score = -1.0
    for name, s in strategies.items():
        report = s.get("classification_report", {})
        breakout = report.get("Breakout", {})
        prec = breakout.get("precision", 0)
        f1 = breakout.get("f1-score", 0)
        s["composite_score"] = float(f1 * prec)

        # Trading-Score priorisiert Profit-Faktor (positiv getraded!)
        trad = s.get("trading_metrics", {})
        pf = trad.get("profit_factor", 0)
        wr = trad.get("win_rate", 0)
        n_trades = trad.get("n_trades", 0)
        # Profit-Faktor dominiert, Win-Rate als Tiebreaker, Mindest-Trades als Gate
        if n_trades < 100:
            trading_score = 0.0  # Nicht genug Trades fur statistische Signifikanz
        else:
            trading_score = pf * (1.0 + wr)  # PF gewichtet mit Win-Rate-Boost
        s["trading_score"] = float(trading_score)

        if trading_score > winner_trading_score:
            winner_trading_score = trading_score
            winner = name

    print(f"\n{'=' * 60}")
    print(f"WINNER: {winner}")
    ws_temp = strategies[winner]
    print(f"  Trading Score (PF x (1+Win%)): {winner_trading_score:.2f}")
    print(f"  Profit-Faktor: {ws_temp['trading_metrics']['profit_factor']:.2f}")
    print(f"  Win-Rate: {ws_temp['trading_metrics']['win_rate']:.2%}")
    print(f"{'=' * 60}")

    # -- Detaillierte Winner-Analyse ------------------------------
    ws = strategies[winner]
    print(f"\n  Accuracy:      {ws['accuracy']:.2%} (+{ws['improvement_pp']:.1f} PP)")
    print(f"  Precision:     {ws['classification_report']['Breakout']['precision']:.3f}")
    print(f"  Recall:        {ws['classification_report']['Breakout']['recall']:.3f}")
    print(f"  F1:            {ws['classification_report']['Breakout']['f1-score']:.3f}")
    print(f"  Signal-Rate:   {ws['signal_rate']:.2%}")
    tm = ws["trading_metrics"]
    print(f"  Trades:        {tm['n_trades']:,}")
    print(f"  Win-Rate:      {tm['win_rate']:.2%}")
    print(f"  Profit/Trade:  {tm['profit_per_trade_pct']:.4%}")
    print(f"  Profit-Faktor: {tm['profit_factor']:.2f}")
    print(f"  Exp. Return:   {tm['expected_return_bps']:.1f} bps/Trade")

    # -- Strategy-Vergleichstabelle -------------------------------
    print(f"\n{'=' * 90}")
    print("STRATEGIE-VERGLEICH")
    print(f"{'=' * 90}")
    print(f"{'Strategie':<32} {'Prec':>6} {'Recall':>6} {'F1':>6} "
          f"{'Acc':>6} {'Trades':>8} {'Win%':>6} {'PF':>6} {'Comp.':>6}")
    print("-" * 90)
    for name in ["two_stage", "simple_average", "weighted_f1", "finder_majority", "stacking"]:
        s = strategies[name]
        br = s["classification_report"]["Breakout"]
        tm = s["trading_metrics"]
        marker = " *" if name == winner else ""
        print(f"{name+marker:<32} "
              f"{br['precision']:>6.3f} {br['recall']:>6.3f} {br['f1-score']:>6.3f} "
              f"{s['accuracy']:>6.2%} {tm['n_trades']:>8,} "
              f"{tm['win_rate']:>6.2%} {tm['profit_factor']:>6.2f} "
              f"{s['composite_score']:>6.3f}")
    print("=" * 90)

    # -- Ergebnisse speichern -------------------------------------
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluation JSON
    ensemble_eval = {
        "metadata": {
            "n_test_samples": int(len(df_test)),
            "n_val_samples": int(len(df_val)),
            "n_symbols": int(df_test["symbol"].nunique()),
            "class_balance": float(df_test["target"].mean()),
            "baseline_accuracy": float(baseline_acc),
            "finder_weights": FINDER_WEIGHTS,
            "weights_all": WEIGHTS_ALL,
            "finder_thresholds": FINDER_THRESHOLDS,
            "filter_threshold": FILTER_THRESHOLD,
            "entry_rules_applied": args.entry_rules,
            "date": pd.Timestamp.now().isoformat(),
        },
        "strategies": strategies,
        "winner": winner,
        "winner_composite_score": float(strategies[winner].get("composite_score", 0)),
        "winner_trading_score": float(strategies[winner].get("trading_score", 0)),
    }

    eval_path = EVAL_DIR / "ensemble_evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(ensemble_eval, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nEvaluation gespeichert: {eval_path}")

    # -- Predictions als Parquet speichern ------------------------
    # Fuge alle Strategie-Signale zum DataFrame hinzu
    df_out = df_test.copy()

    # Two-Stage
    _, score_ts = strategy_two_stage(df_test)
    df_out["score_two_stage"] = score_ts
    df_out["signal_two_stage"] = signal_a

    # Simple Average
    df_out["score_simple_avg"] = score_b
    df_out["signal_simple_avg"] = signal_b

    # Weighted F1
    df_out["score_weighted_f1"] = score_c
    df_out["signal_weighted_f1"] = signal_c

    # Finder Majority
    _, score_fm = strategy_finder_majority(df_test)
    df_out["score_finder_majority"] = score_fm
    df_out["signal_finder_majority"] = signal_d

    # Stacking
    df_out["score_stacking"] = score_e
    df_out["signal_stacking"] = signal_e

    # Winner-Signal als primares Ensemble-Signal
    df_out["ensemble_signal"] = df_out[f"signal_{winner}"]
    df_out["ensemble_score"] = df_out[f"score_{winner}"]
    df_out["strategy_used"] = winner

    pred_path = EVAL_DIR / "ensemble_predictions.parquet"
    df_out.to_parquet(pred_path, index=False)
    print(f"Predictions gespeichert: {pred_path}")
    print(f"  {len(df_out):,} Zeilen x {len(df_out.columns)} Spalten")
    print(f"  Grosse: {pred_path.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\n{'=' * 60}")
    print("ENSEMBLE PREDICTOR FERTIG")
    print(f"{'=' * 60}")
    print(f"  Beste Strategie: {winner}")
    print(f"  Signale: {df_out['ensemble_signal'].sum():,} Trades")
    print(f"  Signal-Rate: {df_out['ensemble_signal'].mean():.2%}")
    print(f"  Nachster Schritt: Backtesting mit ensemble_predictions.parquet")


def _print_strategy_summary(result: dict):
    """Gibt eine einzeilige Zusammenfassung der Strategie-Metriken."""
    br = result["classification_report"]["Breakout"]
    tm = result["trading_metrics"]
    print(f"  Prec={br['precision']:.3f}  Recall={br['recall']:.3f}  "
          f"F1={br['f1-score']:.3f}  Trades={tm['n_trades']:,}  "
          f"Win%={tm['win_rate']:.2%}  PF={tm['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
