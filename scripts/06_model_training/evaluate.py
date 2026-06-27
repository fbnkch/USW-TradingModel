"""
Evaluierung für das MLP-Modell auf dem Test-Set.

Aufruf:
  python scripts/06_model_training/evaluate.py

Berechnet:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1)
  - Accuracy vs. Baseline
  - Threshold-Optimierung (F1-maximierend)
  - Speichert Ergebnisse als JSON für Modellvergleich
"""

from __future__ import annotations

import json
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
)

from model import BreakoutModel
from utils import (
    PROJECT_ROOT, params,
    get_device, enable_amp,
    load_features, load_scaler, load_class_balance,
    load_shards, prepare_tensors, load_model,
)


EVAL_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
THRESHOLD = 0.5


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Findet den F1-optimalen Schwellwert für die Breakout-Klasse."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        p, r = precision[i], recall[i]
        if p + r > 0:
            f1_scores[i] = 2 * p * r / (p + r)

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    f1_default = float(f1_score(y_true, (y_prob > THRESHOLD).astype(int)))

    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "f1_at_0.5": f1_default,
        "f1_improvement": best_f1 - f1_default,
    }


def evaluate_mlp() -> dict:
    """Evaluiert das gespeicherte MLP-Modell auf dem Test-Set."""
    device = get_device(verbose=True)
    use_amp = enable_amp(device)

    # ── Features & Scaler ──────────────────────────────────────
    features = load_features()
    scaler = load_scaler()
    balance = load_class_balance()
    baseline_acc = max(balance["positive_ratio"], 1 - balance["positive_ratio"])

    print(f"Features: {len(features)}")
    print(f"Baseline Accuracy: {baseline_acc:.2%}\n")

    # ── Modell laden ───────────────────────────────────────────
    model = BreakoutModel(
        input_size=len(features),
        hidden_sizes=(128, 64, 32, 16),
        dropout=0.4,
        use_batch_norm=True,
    )
    model, metadata = load_model(model, "mlp_model", device)

    if metadata:
        print("Trainings-Metadaten:")
        print(f"  Best Val Loss: {metadata.get('best_val_loss', 'N/A'):.6f}"
              if isinstance(metadata.get('best_val_loss'), float) else
              f"  Best Val Loss: {metadata.get('best_val_loss', 'N/A')}")
        print(f"  Epochen:       {metadata.get('total_epochs', 'N/A')}")
        print(f"  Trainingszeit: {metadata.get('train_time_s', 0)/60:.1f} Min\n")

    model.eval()

    # ── Test-Daten laden ───────────────────────────────────────
    print("Lade Test-Daten...")
    test_df = load_shards("test", features)
    print(f"  Test: {len(test_df):,} Samples")

    X_test, y_test_np = prepare_tensors(test_df, features, scaler, device)
    y_test = y_test_np.cpu().numpy()
    del test_df

    # ── Vorhersagen ────────────────────────────────────────────
    print("Mache Vorhersagen...")
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(X_test).squeeze()
                y_prob = torch.sigmoid(logits).cpu().numpy()
        else:
            logits = model(X_test).squeeze()
            y_prob = torch.sigmoid(logits).cpu().numpy()

    # ── Threshold-Optimierung ──────────────────────────────────
    print("\nOptimiere Entscheidungsschwelle (F1-maximierend)...")
    threshold_info = find_optimal_threshold(y_test, y_prob)
    optimal_threshold = threshold_info["best_threshold"]
    print(f"  Optimaler Threshold: {optimal_threshold:.4f}")
    print(f"  F1 bei Threshold:   {threshold_info['best_f1']:.4f}")
    print(f"  F1 bei {THRESHOLD}: {threshold_info['f1_at_0.5']:.4f}")
    print(f"  Verbesserung:       {threshold_info['f1_improvement']:.4f}")

    # ── Confusion Matrix (mit optimalem Threshold) ─────────────
    y_pred_opt = (y_prob > optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # ── Metriken ───────────────────────────────────────────────
    accuracy = (tp + tn) / len(y_test)

    print(f"\n{'=' * 60}")
    print(f"EVALUIERUNG: MLP V2 auf Test-Set")
    print(f"{'=' * 60}")
    print(f"  Samples:      {len(y_test):,}")
    print(f"  Threshold:    {optimal_threshold:.4f}")
    print(f"  Accuracy:     {accuracy:.2%}")
    print(f"  Baseline:     {baseline_acc:.2%}")
    print(f"  Verbesserung: +{(accuracy - baseline_acc) * 100:.1f} PP")
    print()
    print("Confusion Matrix:")
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")
    print()
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred_opt,
        target_names=["Kein Breakout", "Breakout"],
        digits=4,
    ))

    # ── Ergebnisse speichern ───────────────────────────────────
    results = {
        "model_type": "mlp",
        "split": "test",
        "n_samples": int(len(y_test)),
        "threshold": optimal_threshold,
        "threshold_info": threshold_info,
        "accuracy": float(accuracy),
        "baseline_accuracy": float(baseline_acc),
        "improvement_pp": float((accuracy - baseline_acc) * 100),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "classification_report": classification_report(
            y_test, y_pred_opt,
            target_names=["Kein Breakout", "Breakout"],
            output_dict=True,
        ),
        "model_metadata": metadata,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EVAL_DIR / "mlp_evaluation.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nErgebnisse gespeichert: {results_path}")
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    evaluate_mlp()
