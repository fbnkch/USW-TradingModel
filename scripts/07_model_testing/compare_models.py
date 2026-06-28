"""
Modellvergleich: Vergleicht alle trainierten Modelle auf dem Test-Set.

Lädt Evaluierungs-JSONs und erstellt eine Vergleichstabelle + Visualisierung.

Aufruf:
  python scripts/06_model_training/compare_models.py
"""

from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
IMG_DIR = PROJECT_ROOT / "artifacts" / "images" / "03_modeling"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#dee2e6",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

FARBEN = {
    "mlp": "#2563eb",
    "lstm": "#dc2626",
    "gru": "#ea580c",
    "cnn": "#7c3aed",
    "lightgbm": "#16a34a",
    "baseline": "#6b7280",
}


def load_eval(model_type: str) -> dict:
    """Lädt die Evaluierungs-JSON eines Modells."""
    path = EVAL_DIR / f"{model_type}_evaluation.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def create_comparison_chart(model_results: dict) -> Path:
    """Erstellt eine Vergleichs-Visualisierung aller Modelle."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    model_names = list(model_results.keys())
    n_models = len(model_names)

    # Farben zuweisen
    colors = [FARBEN.get(m, "#94a3b8") for m in model_names]

    # ── 1. Accuracy vs Baseline ────────────────────────────────
    ax = axes[0, 0]
    accuracies = [model_results[m]["accuracy"] * 100 for m in model_names]
    baseline = model_results[model_names[0]]["baseline_accuracy"] * 100

    x = np.arange(n_models)
    bars = ax.bar(x, accuracies, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    ax.axhline(y=baseline, color=FARBEN["baseline"], linestyle="--", linewidth=2,
               label=f"Baseline ({baseline:.1f}%)")

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", fontweight="bold", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in model_names], fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy im Vergleich zur Baseline", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(baseline - 1, max(accuracies) + 2)

    # ── 2. F1-Score (Breakout-Klasse) ──────────────────────────
    ax = axes[0, 1]
    f1_scores = []
    for m in model_names:
        report = model_results[m].get("classification_report", {})
        f1 = report.get("Breakout", {}).get("f1-score", 0)
        f1_scores.append(f1)

    bars = ax.bar(x, f1_scores, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    ax.axhline(y=0.5, color=FARBEN["baseline"], linestyle="--", linewidth=2, label="Zufall (0.50)")

    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f"{f1:.3f}", ha="center", fontweight="bold", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in model_names], fontweight="bold")
    ax.set_ylabel("F1-Score (Breakout)")
    ax.set_title("F1-Score der Breakout-Klasse", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0.48, max(f1_scores) + 0.03)

    # ── 3. Precision vs Recall (Breakout-Klasse) ───────────────
    ax = axes[1, 0]
    precisions = []
    recalls = []
    for m in model_names:
        report = model_results[m].get("classification_report", {})
        precisions.append(report.get("Breakout", {}).get("precision", 0))
        recalls.append(report.get("Breakout", {}).get("recall", 0))

    x_scatter = np.arange(n_models)
    width = 0.35
    ax.bar(x_scatter - width / 2, precisions, width, label="Precision",
           color="#3b82f6", edgecolor="white", linewidth=0.5)
    ax.bar(x_scatter + width / 2, recalls, width, label="Recall",
           color="#f59e0b", edgecolor="white", linewidth=0.5)

    for i, (p, r) in enumerate(zip(precisions, recalls)):
        ax.text(i - width / 2, p + 0.005, f"{p:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, r + 0.005, f"{r:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x_scatter)
    ax.set_xticklabels([m.upper() for m in model_names], fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall (Breakout-Klasse)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, max(max(precisions), max(recalls)) + 0.08)

    # ── 4. Verbesserung über Baseline (Prozentpunkte) ──────────
    ax = axes[1, 1]
    improvements = [model_results[m]["improvement_pp"] for m in model_names]

    bars = ax.barh([m.upper() for m in model_names], improvements,
                    color=colors, edgecolor="white", linewidth=1.5, height=0.5)

    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2.,
                f"+{imp:.1f} PP", va="center", fontweight="bold", fontsize=11)

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Verbesserung über Baseline (Prozentpunkte)")
    ax.set_title("Performance-Gewinn pro Modell", fontweight="bold")

    # ── Gesamttitel ────────────────────────────────────────────
    fig.suptitle("Modellvergleich: Breakout-Vorhersage (30 Minuten)",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    output_path = IMG_DIR / "00_model_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Vergleichs-Chart gespeichert: {output_path}")
    return output_path


def main():
    """Hauptfunktion: Lädt Evaluierungen, vergleicht, erstellt Charts."""
    print("=" * 60)
    print("MODELLVERGLEICH: Breakout-Vorhersage")
    print("=" * 60)

    model_types = ["mlp", "lstm", "gru", "cnn", "lightgbm"]
    model_results = {}

    for mt in model_types:
        result = load_eval(mt)
        if result:
            model_results[mt] = result
            report = result.get("classification_report", {})
            breakout = report.get("Breakout", {})
            print(f"\n{mt.upper():>12s}: "
                  f"Acc={result['accuracy']:.2%} | "
                  f"F1={breakout.get('f1-score', 0):.3f} | "
                  f"Prec={breakout.get('precision', 0):.3f} | "
                  f"Recall={breakout.get('recall', 0):.3f} | "
                  f"+{result['improvement_pp']:.1f} PP")
        else:
            print(f"\n{mt.upper():>12s}: [NICHT GEFUNDEN] – Bitte zuerst trainieren")

    if len(model_results) < 2:
        print("\n[INFO] Mindestens 2 Modelle für Vergleich nötig. "
              "Bitte trainiere zuerst die Modelle.")
        return

    # Vergleichstabelle als JSON
    comparison = {
        "baseline_accuracy": model_results[list(model_results.keys())[0]]["baseline_accuracy"],
        "models": {
            name: {
                "accuracy": res["accuracy"],
                "improvement_pp": res["improvement_pp"],
                "f1_breakout": res.get("classification_report", {}).get("Breakout", {}).get("f1-score", 0),
                "precision_breakout": res.get("classification_report", {}).get("Breakout", {}).get("precision", 0),
                "recall_breakout": res.get("classification_report", {}).get("Breakout", {}).get("recall", 0),
                "threshold": res.get("threshold", 0.5),
                "confusion_matrix": res.get("confusion_matrix", {}),
            }
            for name, res in model_results.items()
        }
    }

    comp_path = EVAL_DIR / "model_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Vergleich gespeichert: {comp_path}")

    # Visualisierung
    create_comparison_chart(model_results)

    # ── Empfehlung ─────────────────────────────────────────────
    best_model = max(model_results.items(),
                     key=lambda x: x[1]["improvement_pp"])
    print(f"\n{'=' * 60}")
    print(f"EMPFOHLENES MODELL: {best_model[0].upper()}")
    print(f"  +{best_model[1]['improvement_pp']:.1f} PP über Baseline")
    print(f"  Accuracy: {best_model[1]['accuracy']:.2%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
