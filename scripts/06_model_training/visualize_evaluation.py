"""
Evaluations-Visualisierung
--------------------------
Erstellt aussagekraeftige Plots zur Modellbewertung:
1. Confusion Matrix (Heatmap)
2. Trainingsverlauf (Loss & Accuracy ueber Epochen)
3. Metriken-Vergleich pro Klasse (Precision, Recall, F1)
4. Modell vs. Baseline Vergleich
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Matplotlib fuer nicht-interaktive Umgebung
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DATEN (aus deinen Trainings-/Evaluations-Outputs)
# ---------------------------------------------------------------------------

# Trainingsverlauf
epochs = list(range(1, 11))
val_loss = [0.6731, 0.6728, 0.6742, 0.6721, 0.6736, 0.6710, 0.6686, 0.6718, 0.6713, 0.6701]
val_acc  = [0.587, 0.588, 0.585, 0.587, 0.585, 0.593, 0.598, 0.595, 0.592, 0.595]

# Confusion Matrix: [[TN, FP], [FN, TP]]
cm = np.array([[837443, 468692],
               [439523, 499033]])

# Abgeleitete Metriken
TN, FP = cm[0, 0], cm[0, 1]
FN, TP = cm[1, 0], cm[1, 1]
total = cm.sum()

# Metriken pro Klasse (aus classification_report)
metrics = {
    "Kein Breakout (0)": {"Precision": 0.66, "Recall": 0.64, "F1-Score": 0.65},
    "Breakout (1)":     {"Precision": 0.52, "Recall": 0.53, "F1-Score": 0.52},
}

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
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
    "blue":   "#2563eb",
    "green":  "#16a34a",
    "red":    "#dc2626",
    "orange": "#ea580c",
    "purple": "#7c3aed",
    "gray":   "#6b7280",
}

# ---------------------------------------------------------------------------
# FIGURE 1: CONFUSION MATRIX + METRIKEN-BARS (2×1)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# --- 1a: Confusion Matrix Heatmap ---
ax = axes[0]
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

# Beschriftung
klassen = ["Kein Breakout\n(Vorhersage 0)", "Breakout\n(Vorhersage 1)"]
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(klassen)
ax.set_yticklabels(["Tatsaechlich\nKein Breakout", "Tatsaechlich\nBreakout"])

# Zahlen in Zellen
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        pct = val / total * 100
        farbe = "white" if cm[i, j] > cm.max() * 0.5 else "black"
        ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center",
                fontsize=12, fontweight="bold", color=farbe)

# Zusaetzliche Annotationen (TP, TN, FP, FN)
label_map = {(0,0): "TN", (0,1): "FP", (1,0): "FN", (1,1): "TP"}
for i in range(2):
    for j in range(2):
        ax.text(j, i - 0.36, label_map[(i,j)], ha="center", va="top",
                fontsize=9, color=FARBEN["gray"], fontstyle="italic")

ax.set_title("Confusion Matrix (Test-Set: 2.244.691 Samples)", fontweight="bold")

# --- 1b: Metriken pro Klasse (Grouped Bar) ---
ax = axes[1]
metric_names = ["Precision", "Recall", "F1-Score"]
x = np.arange(len(metric_names))
width = 0.35

bars_0 = [metrics["Kein Breakout (0)"][m] for m in metric_names]
bars_1 = [metrics["Breakout (1)"][m] for m in metric_names]

b0 = ax.bar(x - width/2, bars_0, width, label="Kein Breakout (0)",
            color=FARBEN["blue"], edgecolor="white", linewidth=0.5)
b1 = ax.bar(x + width/2, bars_1, width, label="Breakout (1)",
            color=FARBEN["orange"], edgecolor="white", linewidth=0.5)

# Werte auf Balken
for bar in b0 + b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 0.85)
ax.set_ylabel("Score")
ax.set_title("Metriken pro Klasse", fontweight="bold")
ax.legend(loc="lower right")
ax.axhline(y=0.50, color=FARBEN["gray"], linestyle="--", linewidth=1, alpha=0.7)
ax.text(2.35, 0.505, "Zufall (0.50)", fontsize=8, color=FARBEN["gray"], ha="right")

fig.suptitle("Modell-Evaluation: Breakout-Vorhersage (30 Minuten)",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "01_confusion_matrix_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 01_confusion_matrix_metrics.png")

# ---------------------------------------------------------------------------
# FIGURE 2: TRAININGSVERLAUF (Loss + Accuracy, dual axis)
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 5))

color_loss = FARBEN["red"]
color_acc  = FARBEN["blue"]

ax1.plot(epochs, val_loss, "o-", color=color_loss, linewidth=2, markersize=8, label="Val Loss (BCE)")
ax1.set_xlabel("Epoche")
ax1.set_ylabel("Binary Cross Entropy Loss", color=color_loss)
ax1.tick_params(axis="y", labelcolor=color_loss)
ax1.set_xticks(epochs)

# Loss Range anpassen
loss_min = min(val_loss) - 0.002
loss_max = max(val_loss) + 0.002
ax1.set_ylim(loss_min, loss_max)

# Beste Loss markieren
best_loss_epoch = epochs[val_loss.index(min(val_loss))]
best_loss_val   = min(val_loss)
ax1.annotate(f"Beste: {best_loss_val:.4f}",
             xy=(best_loss_epoch, best_loss_val),
             xytext=(best_loss_epoch + 0.5, best_loss_val + 0.0015),
             arrowprops=dict(arrowstyle="->", color=color_loss),
             fontsize=9, color=color_loss, fontweight="bold")

ax2 = ax1.twinx()
ax2.plot(epochs, [a*100 for a in val_acc], "s--", color=color_acc, linewidth=2, markersize=8, label="Val Accuracy")
ax2.set_ylabel("Accuracy (%)", color=color_acc)
ax2.tick_params(axis="y", labelcolor=color_acc)

# Beste Accuracy markieren
best_acc_epoch = epochs[val_acc.index(max(val_acc))]
best_acc_val   = max(val_acc) * 100
ax2.annotate(f"Beste: {best_acc_val:.1f}%",
             xy=(best_acc_epoch, best_acc_val),
             xytext=(best_acc_epoch + 0.5, best_acc_val + 0.3),
             arrowprops=dict(arrowstyle="->", color=color_acc),
             fontsize=9, color=color_acc, fontweight="bold")

# Baseline einzeichnen
ax2.axhline(y=50.22, color=FARBEN["gray"], linestyle=":", linewidth=1.5, alpha=0.8)
ax2.text(10.2, 50.22, "Baseline 50.22%", fontsize=8, color=FARBEN["gray"], va="center")

# Kombinierte Legende
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

ax1.set_title("Trainingsverlauf ueber 10 Epochen", fontweight="bold")
fig.suptitle("Validierungs-Performance waehrend des Trainings", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "02_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 02_training_curves.png")

# ---------------------------------------------------------------------------
# FIGURE 3: MODELL vs. BASELINE
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

kategorien = ["Baseline\n(Majority Class)", "Modell\n(Validation Best)", "Modell\n(Test Set)"]
acc_values = [50.22, 59.8, 59.56]  # test accuracy = (TN+TP)/total = (837443+499033)/2244691
farben = [FARBEN["gray"], FARBEN["blue"], FARBEN["green"]]

bars = ax.bar(kategorien, acc_values, color=farben, width=0.5, edgecolor="white", linewidth=1)

for bar, val in zip(bars, acc_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=14, fontweight="bold")

# Differenz-Pfeile
ax.annotate("",
            xy=(1, 59.8), xytext=(0, 50.22),
            arrowprops=dict(arrowstyle="<->", color=FARBEN["purple"], lw=2))
ax.text(0.5, 55.3, "+9.6 PP", ha="center", fontsize=11,
        color=FARBEN["purple"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_ylim(0, 68)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Modell-Performance vs. Baseline", fontweight="bold")
ax.axhline(y=50.0, color="black", linestyle="--", linewidth=0.8, alpha=0.3)

# Zusaetzliche Info-Box
info_text = (
    f"Trainings-Samples: 10.394.874\n"
    f"Test-Samples:       2.244.691\n"
    f"Features:           Preis-MA, EMA, Slopes, Z-Scores\n"
    f"Modell:             MLP (Input->64->32->1)"
)
ax.text(0.98, 0.95, info_text, transform=ax.transAxes, fontsize=9,
        va="top", ha="right", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", edgecolor="#cbd5e1"))

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "03_model_vs_baseline.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 03_model_vs_baseline.png")

# ---------------------------------------------------------------------------
# FIGURE 4: HANDELSKONTEXT -- Was bedeuten die Metriken?
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# --- 4a: Fehlerarten im Handelskontext ---
ax = axes[0]
fehler_labels = [
    "True Positive\n(Breakout erkannt)",
    "True Negative\n(Kein Breakout erkannt)",
    "False Positive\n(Fehlalarm)",
    "False Negative\n(Breakout verpasst)",
]
fehler_counts = [TP, TN, FP, FN]
fehler_colors = [FARBEN["green"], FARBEN["blue"], FARBEN["orange"], FARBEN["red"]]
explode = (0.02, 0.02, 0.05, 0.05)

wedges, texts, autotexts = ax.pie(
    fehler_counts, labels=fehler_labels, colors=fehler_colors,
    autopct="%1.1f%%", explode=explode, startangle=90,
    textprops={"fontsize": 9}
)
for at in autotexts:
    at.set_fontweight("bold")
    at.set_fontsize(10)

ax.set_title("Aufteilung der Vorhersagen\n(Test-Set: 2.244.691 Samples)", fontweight="bold")

# --- 4b: Trading-Impact -- Kosten von Fehlern ---
ax = axes[1]

# Annahmen fuer Trading-Kosten (fiktiv, zur Veranschaulichung)
# False Positive: Du gehst long, aber es gibt keinen Breakout -> Verlust durch Spread/Commission + ggf. Stop-Loss
# False Negative: Du verpasst einen Breakout -> Opportunitaetskosten
fp_cost = 0.5   # relative Kosten pro FP
fn_cost = 1.0   # relative Kosten pro FN (Opportunitaetskosten)

impact_labels = ["Richtige\nVorhersagen", "Fehlalarme\n(FP)", "Verpasste\nBreakouts (FN)"]
impact_values = [TP + TN, FP, FN]
impact_colors = [FARBEN["green"], FARBEN["orange"], FARBEN["red"]]

bars = ax.barh(impact_labels, impact_values, color=impact_colors, edgecolor="white", height=0.5)

for bar, val in zip(bars, impact_values):
    pct = val / total * 100
    ax.text(bar.get_width() + 20000, bar.get_y() + bar.get_height()/2.,
            f"{val:,} ({pct:.1f}%)", va="center", fontsize=11, fontweight="bold")

ax.set_xlabel("Anzahl Samples")
ax.set_title("Fehlerkosten im Trading-Kontext", fontweight="bold")
ax.set_xlim(0, total * 0.75)

# Annotationsbox
cost_text = (
    "Trading-Implikation:\n"
    f"- {FP:,} Fehlalarme -> Trading-Kosten\n"
    f"- {FN:,} verpasste Breakouts -> Entgangene Gewinne\n"
    f"- Precision (Breakout) = 52% -> Jeder 2. Trade ein Fehlalarm\n"
    f"- Recall (Breakout) = 53% -> Fast jeder 2. Breakout verpasst"
)
ax.text(0.98, 0.02, cost_text, transform=ax.transAxes, fontsize=9,
        va="bottom", ha="right", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef2f2", edgecolor="#fca5a5"))

fig.suptitle("Trading-Relevanz der Modell-Performance", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "04_trading_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 04_trading_impact.png")

# ---------------------------------------------------------------------------
# FIGURE 5: ZUSAMMENFASSUNG -- Dashboard
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8))

# Titel
fig.suptitle("Breakout-Modell -- Gesamtbewertung",
             fontsize=16, fontweight="bold", y=0.98)

# KPI-Kacheln oben
kpi_data = [
    ("Accuracy\n(Test Set)", "59.6%", FARBEN["blue"], "+9.3 PP\nvs. Baseline"),
    ("Precision\nBreakout", "52%", FARBEN["orange"], "Jeder 2.\nTrade falsch"),
    ("Recall\nBreakout", "53%", FARBEN["red"], "47% der Breakouts\nverpasst"),
    ("F1-Score\nBreakout", "0.52", FARBEN["purple"], "Schwaches\nSignal"),
]

for i, (label, value, color, note) in enumerate(kpi_data):
    ax = fig.add_axes([0.08 + i * 0.22, 0.55, 0.18, 0.30])
    ax.set_facecolor("#f8f9fa")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#dee2e6")
        spine.set_linewidth(2)

    ax.text(0.5, 0.72, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, fontweight="bold", color=FARBEN["gray"])
    ax.text(0.5, 0.35, value, transform=ax.transAxes, ha="center", va="center",
            fontsize=28, fontweight="bold", color=color)
    ax.text(0.5, 0.08, note, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color=FARBEN["gray"], fontstyle="italic")

# Unterer Bereich: Fazit-Text
ax_text = fig.add_axes([0.08, 0.05, 0.84, 0.45])
ax_text.axis("off")

fazit = (
    "┌─────────────────────────────────────────────────────────────────────────────┐\n"
    "│  [===] GESAMTBEWERTUNG                                                          │\n"
    "├─────────────────────────────────────────────────────────────────────────────┤\n"
    "│                                                                              │\n"
    "│  [+] POSITIV                                                                  │\n"
    "│  - Das Modell schlaegt die Baseline (50.22%) mit +9.3 Prozentpunkten          │\n"
    "│  - Das Modell lernt -- auch wenn der Fortschritt gering ist                   │\n"
    "│  - Die Architektur (MLP) und Features sind sinnvoll gewaehlt                  │\n"
    "│  - Keine Anzeichen von Overfitting (Val Loss sinkt leicht)                   │\n"
    "│                                                                              │\n"
    "│  [!]  KRITISCH                                                                │\n"
    "│  - Breakout-Precision = 52% -> Kaum besser als Muenzwurf fuer Trades            │\n"
    "│  - Breakout-Recall = 53% -> Fast die Haelfte aller Breakouts wird verpasst     │\n"
    "│  - Trainings-Verlust stagniert (0.673 -> 0.670) -> Modell am Limit             │\n"
    "│  - Keine Verbesserung ab Epoche 7 -> Mehr Epochen bringen nichts              │\n"
    "│                                                                              │\n"
    "│  [>>] NAeCHSTE SCHRITTE (Empfehlung)                                            │\n"
    "│  - 1. Feature Engineering: Volumen, Order Book, Markt-Mikrostruktur          │\n"
    "│  - 2. Komplexere Architektur: Mehr Layer, BatchNorm, Residual Connections    │\n"
    "│  - 3. Klassen-Gewichtung: Hoehere Gewichte fuer die Breakout-Klasse            │\n"
    "│  - 4. Threshold-Tuning: Statt 0.5 den Schwellwert optimieren (z.B. per F1)  │\n"
    "│  - 5. Alternative Modelle: XGBoost, LightGBM als staerkere Baseline           │\n"
    "│  - 6. Anderer Horizont: 30min koennte zu kurz/lang sein -- 5min/15min testen  │\n"
    "│                                                                              │\n"
    "└─────────────────────────────────────────────────────────────────────────────┘"
)

ax_text.text(0.5, 0.5, fazit, transform=ax_text.transAxes, va="center", ha="center",
             fontsize=9.5, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#dee2e6"))

fig.savefig(OUTPUT_DIR / "05_dashboard_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 05_dashboard_summary.png")

print(f"\n[OK] Alle Visualisierungen gespeichert in: {OUTPUT_DIR}")
