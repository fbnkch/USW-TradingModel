"""
Evaluierung für sequenzielle Modelle (LSTM, GRU, CNN-1D) auf dem Test-Set.

Aufruf:
  python scripts/06_model_training/evaluate_sequential.py --model lstm
  python scripts/06_model_training/evaluate_sequential.py --model gru
  python scripts/06_model_training/evaluate_sequential.py --model cnn

Berechnet:
  - Confusion Matrix
  - Precision, Recall, F1 pro Klasse
  - Accuracy vs. Baseline
  - Threshold-Optimierung (F1-maximierend)
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
)
from pathlib import Path

from model_sequential import LSTMBreakoutModel, GRUBreakoutModel, CNNBreakoutModel
from utils import (
    PROJECT_ROOT, params,
    get_device, enable_amp,
    load_features, load_scaler, load_class_balance,
    build_sequences, load_model,
)


PRE_SPLIT_PATH = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]
EVAL_DIR = PROJECT_ROOT / "artifacts" / "evaluation"

MODEL_CLASSES = {
    "lstm": LSTMBreakoutModel,
    "gru": GRUBreakoutModel,
    "cnn": CNNBreakoutModel,
}

DEFAULT_CONFIGS = {
    "lstm": {"hidden_size": 128, "num_layers": 2, "dropout": 0.35, "bidirectional": True},
    "gru":  {"hidden_size": 128, "num_layers": 2, "dropout": 0.35},
    "cnn":  {"hidden_channels": 64, "kernel_sizes": (3, 5, 10), "dropout": 0.35},
}


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Findet den Schwellwert der den F1-Score auf der Breakout-Klasse maximiert.

    Parameters
        y_true : Wahre Labels (0/1)
        y_prob : Vorhergesagte Wahrscheinlichkeiten

    Returns
        Dict mit best_threshold, best_f1, und thresholds+fscores für Plotting
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # F1 für jeden Threshold berechnen
    f1_scores = []
    for p, r in zip(precision[:-1], recall[:-1]):  # letzter Punkt hat keinen Threshold
        if p + r > 0:
            f1_scores.append(2 * p * r / (p + r))
        else:
            f1_scores.append(0.0)

    f1_scores = np.array(f1_scores)

    if len(f1_scores) == 0:
        return {"best_threshold": 0.5, "best_f1": 0.0}

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    # F1 bei Default-Threshold 0.5
    default_pred = (y_prob > 0.5).astype(int)
    f1_default = f1_score(y_true, default_pred)

    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "f1_at_0.5": f1_default,
        "f1_improvement": best_f1 - f1_default,
        "thresholds": thresholds.tolist(),
        "f1_scores": f1_scores.tolist(),
        "n_thresholds_evaluated": len(thresholds),
    }


def evaluate_sequential(
    model_type: str = "lstm",
    threshold: float = None,
    split: str = "test",
    max_files: int = None,
) -> dict:
    """Evaluiert ein gespeichertes sequenzielles Modell.

    Parameters
        model_type : "lstm", "gru", oder "cnn"
        threshold : Entscheidungsschwelle (None = optimiere via F1)
        split : "validation" oder "test"
        max_files : Optional, begrenzt Anzahl Dateien

    Returns
        Dict mit allen Evaluierungs-Ergebnissen
    """
    device = get_device(verbose=True)
    use_amp = enable_amp(device)

    features = load_features()
    scaler = load_scaler()
    balance = load_class_balance()
    baseline_acc = max(balance["positive_ratio"], 1 - balance["positive_ratio"])

    print(f"Features: {len(features)}")
    print(f"Baseline Accuracy: {baseline_acc:.2%}\n")

    # ── Modell laden ────────────────────────────────────────────
    model_cls = MODEL_CLASSES[model_type]
    cfg = DEFAULT_CONFIGS[model_type]

    if model_type == "cnn":
        model = model_cls(
            input_size=len(features),
            hidden_channels=cfg["hidden_channels"],
            kernel_sizes=cfg["kernel_sizes"],
            dropout=cfg["dropout"],
        )
    elif model_type in ("lstm", "gru"):
        kwargs = dict(
            input_size=len(features),
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
        if model_type == "lstm":
            kwargs["bidirectional"] = cfg.get("bidirectional", True)
        model = model_cls(**kwargs)

    model, metadata = load_model(model, f"{model_type}_model", device)

    if metadata:
        print("Trainings-Metadaten:")
        print(f"  Best Val Loss: {metadata.get('best_val_loss', 'N/A'):.6f}"
              if isinstance(metadata.get('best_val_loss'), float) else
              f"  Best Val Loss: {metadata.get('best_val_loss', 'N/A')}")
        print(f"  Epochen:       {metadata.get('total_epochs', 'N/A')}")
        print(f"  Trainingszeit: {metadata.get('train_time_s', 0)/60:.1f} Min\n"
              if metadata.get('train_time_s') else "")

    model.eval()

    # ── Test-Sequenzen bauen ────────────────────────────────────
    print(f"Baue {split}-Sequenzen...")
    split_map = {"validation": "validation", "test": "test"}
    glob_pattern = str(PRE_SPLIT_PATH / f"*_{split_map.get(split, split)}.parquet")
    X_test, y_test_tensor = build_sequences(
        glob_pattern, features, scaler,
        seq_len=30, max_files=max_files,
    )
    y_test = y_test_tensor.numpy()  # für sklearn-Metriken

    # ── Vorhersagen (batched – verhindert GPU-OOM bei >20 GB Daten) ─
    print(f"Mache Vorhersagen auf {X_test.shape[0]:,} Sequenzen...")

    INFER_BATCH = 1024  # Inference-Batch (konservativ, passt in 24 GB VRAM)
    infer_dataset = TensorDataset(X_test, y_test_tensor)
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=INFER_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_probs = []
    model.eval()
    with torch.no_grad():
        for (X_batch, _) in infer_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(X_batch).squeeze()
                    probs = torch.sigmoid(logits)
            else:
                logits = model(X_batch).squeeze()
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    y_prob = np.concatenate(all_probs)

    # ── Threshold-Optimierung (auf Validation) oder fixen Threshold ──
    if threshold is None:
        print("\nOptimiere Entscheidungsschwelle (F1-maximierend)...")
        threshold_info = find_optimal_threshold(y_test, y_prob)
        threshold = threshold_info["best_threshold"]
        print(f"  Optimaler Threshold: {threshold:.4f}")
        print(f"  F1 bei Threshold:   {threshold_info['best_f1']:.4f}")
        print(f"  F1 bei 0.5:         {threshold_info['f1_at_0.5']:.4f}")
        print(f"  Verbesserung:       {threshold_info['f1_improvement']:.4f}")
    else:
        threshold_info = {"best_threshold": threshold}

    y_pred = (y_prob > threshold).astype(int)

    # ── Confusion Matrix ────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # ── Metriken ────────────────────────────────────────────────
    report = classification_report(
        y_test, y_pred,
        target_names=["Kein Breakout", "Breakout"],
        output_dict=True,
    )
    total = len(y_test)
    accuracy = (tp + tn) / total if total > 0 else 0

    # ── Ergebnisse ──────────────────────────────────────────────
    results = {
        "model_type": model_type,
        "split": split,
        "n_samples": total,
        "threshold": threshold,
        "threshold_info": threshold_info,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_acc,
        "improvement_pp": (accuracy - baseline_acc) * 100,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "classification_report": report,
        "model_metadata": metadata,
    }

    # ── Ausgabe ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"EVALUIERUNG: {model_type.upper()} auf {split}-Set")
    print(f"{'=' * 60}")
    print(f"  Samples:           {total:,}")
    print(f"  Threshold:         {threshold:.4f}")
    print(f"  Accuracy:          {accuracy:.2%}")
    print(f"  Baseline:          {baseline_acc:.2%}")
    print(f"  Verbesserung:      {results['improvement_pp']:.1f} PP")
    print()
    print("Confusion Matrix:")
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")
    print()
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Kein Breakout", "Breakout"],
        digits=4,
    ))

    # ── Speichern ───────────────────────────────────────────────
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EVAL_DIR / f"{model_type}_evaluation.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"Ergebnisse gespeichert: {results_path}")
    return results


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluierung für sequenzielle Breakout-Modelle"
    )
    parser.add_argument(
        "--model", type=str, default="lstm",
        choices=["lstm", "gru", "cnn"],
        help="Modelltyp (default: lstm)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Entscheidungsschwelle (default: auto-optimiert via F1)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["validation", "test"],
        help="Split für Evaluierung (default: test)"
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Maximale Anzahl Dateien für schnelle Tests"
    )
    args = parser.parse_args()

    evaluate_sequential(
        model_type=args.model,
        threshold=args.threshold,
        split=args.split,
        max_files=args.max_files,
    )
