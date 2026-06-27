"""
LightGBM Training für Breakout-Vorhersage.

LightGBM ist oft stärker als einfache MLPs auf tabellarischen Daten:
  - Gradient-Boosted Trees können nicht-lineare Interaktionen automatisch erfassen
  - Robust gegen Skalierungsunterschiede
  - Integriertes Feature-Importance-Ranking
  - Schnelles Training (CPU, kein GPU-Bedarf)

Aufruf:
  python scripts/06_model_training/train_lightgbm.py
"""

from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from pathlib import Path

from utils import (
    PROJECT_ROOT, params,
    load_features, load_scaler, load_class_balance,
    load_shards, N_SHARDS,
)


EVAL_DIR = PROJECT_ROOT / "artifacts" / "evaluation"


def find_optimal_threshold_lgb(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Findet den F1-optimalen Schwellwert."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        p, r = precision[i], recall[i]
        if p + r > 0:
            f1_scores[i] = 2 * p * r / (p + r)

    best_idx = np.argmax(f1_scores)
    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1_scores[best_idx]),
        "f1_at_0.5": float(f1_score(y_true, (y_prob > 0.5).astype(int))),
    }


def train_lightgbm(
    early_stopping_rounds: int = 50,
    num_boost_round: int = 5000,
    learning_rate: float = 0.05,
    max_bin: int = 255,
    num_leaves: int = 127,
    max_depth: int = 8,
    min_child_samples: int = 100,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 0.1,
    n_val_shards: int = 10,
) -> dict:
    """Trainiert ein LightGBM-Modell auf allen Trainings-Shards.

    Verwendet die geshuffelten Shards (keine zeitliche Ordnung nötig).
    """
    t_start = time.time()

    # ── Features & Scaler ──────────────────────────────────────
    features = load_features()
    scaler = load_scaler()
    balance = load_class_balance()
    baseline_acc = max(balance["positive_ratio"], 1 - balance["positive_ratio"])
    scale_pos_weight = balance["scale_pos_weight"]

    print(f"Features: {len(features)}")
    print(f"Klassen-Balance: {balance['positive_ratio']:.2%} Breakout "
          f"(scale_pos_weight={scale_pos_weight:.4f})")
    print(f"Baseline Accuracy: {baseline_acc:.2%}\n")

    # ── Trainingsdaten laden (Shards) ──────────────────────────
    print("=" * 60)
    print("Lade Trainingsdaten...")
    train_df = load_shards("train", features)
    X_train_np = scaler.transform(train_df[features].to_numpy(dtype="float64"))
    y_train_np = train_df["breakout_30m"].to_numpy(dtype="float32")
    print(f"  Train: {X_train_np.shape[0]:,} Samples, "
          f"{X_train_np.shape[1]} Features")

    # ── Validierungsdaten laden (nur n_val_shards für Speed) ──
    print("\nLade Validierungsdaten...")
    val_dfs = []
    for k in range(min(n_val_shards, N_SHARDS)):
        fpath = Path(params["SPLIT_DATA"]["SHUFFLED_PATH"]) / f"validation_shard_{k}.parquet"
        val_dfs.append(pd.read_parquet(fpath))
    val_df_full = pd.concat(val_dfs, ignore_index=True)

    X_val_np = scaler.transform(val_df_full[features].to_numpy(dtype="float64"))
    y_val_np = val_df_full["breakout_30m"].to_numpy(dtype="float32")
    print(f"  Val: {X_val_np.shape[0]:,} Samples ({n_val_shards}/{N_SHARDS} Shards)")

    # ── LightGBM Datasets ──────────────────────────────────────
    print("\nErstelle LightGBM Datasets...")
    dtrain = lgb.Dataset(
        X_train_np, label=y_train_np,
        feature_name=features,
        params={"max_bin": max_bin},
    )
    dval = lgb.Dataset(
        X_val_np, label=y_val_np,
        reference=dtrain,
        params={"max_bin": max_bin},
    )

    # ── Parameter ──────────────────────────────────────────────
    params_lgb = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "scale_pos_weight": scale_pos_weight,
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "verbosity": 0,
        "num_threads": -1,  # Alle CPU-Kerne
    }

    print(f"\nLightGBM Parameter:")
    for k, v in params_lgb.items():
        print(f"  {k}: {v}")
    print(f"  num_boost_round: {num_boost_round}")
    print(f"  early_stopping_rounds: {early_stopping_rounds}")

    # ── Training ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Starte LightGBM Training...")

    callbacks = [
        lgb.early_stopping(early_stopping_rounds),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params=params_lgb,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    train_time = time.time() - t_start
    print(f"\nTraining abgeschlossen in {train_time/60:.1f} Min")
    print(f"  Beste Iteration: {model.best_iteration}")
    print(f"  Bester Score:    {model.best_score}")

    # ── Evaluation auf Validation ──────────────────────────────
    print(f"\n{'=' * 60}")
    print("Evaluierung auf Validation-Set...")

    y_prob = model.predict(X_val_np)
    threshold_info = find_optimal_threshold_lgb(y_val_np, y_prob)
    threshold = threshold_info["best_threshold"]
    y_pred = (y_prob > threshold).astype(int)

    cm = confusion_matrix(y_val_np, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    accuracy = (tp + tn) / len(y_val_np)

    print(f"\nOptimaler Threshold: {threshold:.4f}")
    print(f"  F1 bei Threshold:  {threshold_info['best_f1']:.4f}")
    print(f"  F1 bei 0.5:        {threshold_info['f1_at_0.5']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")

    print(f"\nAccuracy: {accuracy:.2%} (Baseline: {baseline_acc:.2%}, "
          f"+{(accuracy - baseline_acc)*100:.1f} PP)")

    print(f"\nClassification Report:")
    print(classification_report(
        y_val_np, y_pred,
        target_names=["Kein Breakout", "Breakout"],
        digits=4,
    ))

    # ── Speichern ──────────────────────────────────────────────
    model_dir = PROJECT_ROOT / "artifacts" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lightgbm_model.txt"
    model.save_model(str(model_path))
    print(f"\nModell gespeichert: {model_path}")

    # Feature Importance speichern
    importance = model.feature_importance(importance_type="gain")
    feature_imp = sorted(
        zip(features, importance),
        key=lambda x: x[1], reverse=True
    )
    imp_path = EVAL_DIR / "lightgbm_feature_importance.json"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(imp_path, "w", encoding="utf-8") as f:
        json.dump([
            {"feature": name, "importance_gain": float(imp)}
            for name, imp in feature_imp
        ], f, indent=2, ensure_ascii=False)
    print(f"Feature Importance gespeichert: {imp_path}")

    # Top-20 Features ausgeben
    print(f"\nTop-20 Features (Gain):")
    for i, (name, imp) in enumerate(feature_imp[:20], 1):
        print(f"  {i:2d}. {name:40s} {imp:>12.2f}")

    # ── Ergebnisse ─────────────────────────────────────────────
    results = {
        "model_type": "lightgbm",
        "params": params_lgb,
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,
        "threshold": threshold,
        "threshold_info": threshold_info,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_acc,
        "improvement_pp": (accuracy - baseline_acc) * 100,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "train_time_s": train_time,
        "n_features": len(features),
        "n_train_samples": int(X_train_np.shape[0]),
        "n_val_samples": int(X_val_np.shape[0]),
    }

    results_path = EVAL_DIR / "lightgbm_evaluation.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nErgebnisse gespeichert: {results_path}")
    return results


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LightGBM Training für Breakout-Vorhersage")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning Rate")
    parser.add_argument("--leaves", type=int, default=127, help="num_leaves")
    parser.add_argument("--depth", type=int, default=8, help="max_depth")
    parser.add_argument("--rounds", type=int, default=5000, help="num_boost_round")
    parser.add_argument("--early_stop", type=int, default=50, help="Early stopping rounds")
    parser.add_argument("--val_shards", type=int, default=10, help="Anzahl Validation-Shards")
    args = parser.parse_args()

    train_lightgbm(
        learning_rate=args.lr,
        num_leaves=args.leaves,
        max_depth=args.depth,
        num_boost_round=args.rounds,
        early_stopping_rounds=args.early_stop,
        n_val_shards=args.val_shards,
    )
