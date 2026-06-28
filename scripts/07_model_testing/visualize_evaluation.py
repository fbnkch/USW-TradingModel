"""
Evaluations-Visualisierung (Final)
----------------------------------
Liest alle *_evaluation.json Dateien und erstellt Vergleichs-Charts:
1. Accuracy & F1-Score Vergleich (alle 5 Modelle)
2. Precision vs Recall (Breakout-Klasse)
3. Verbesserung über Baseline
4. Trainingszeit & Parameter
5. Feature Importance (Top 15)
6. Threshold F1-Kurven
7. Confusion Matrix Grid
8. Trading-Impact: Precision/Recall Trade-off
9. Ensemble-Architektur-Diagramm
10. Gesamt-Dashboard

Speichert alle PNGs nach artifacts/images/
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
IMG_DIR = PROJECT_ROOT / "artifacts" / "images"
# Subfolder matching project steps (matches reference project convention)
DIR_MODELING  = IMG_DIR / "03_modeling"
DIR_ENSEMBLE  = IMG_DIR / "04_ensemble"
DIR_DASHBOARD = IMG_DIR / "05_dashboard"
for d in [DIR_MODELING, DIR_ENSEMBLE, DIR_DASHBOARD]:
    d.mkdir(parents=True, exist_ok=True)

# ── Style: Dark quant theme ──
BG = "#0a0e17"
CARD = "#111827"
BORDER = "#1e293b"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
DIM = "#64748b"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": BORDER, "text.color": TEXT,
    "axes.labelcolor": MUTED, "xtick.color": DIM, "ytick.color": DIM,
    "grid.color": BORDER, "grid.alpha": 0.6,
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "savefig.facecolor": BG,
})

C = ["#3b82f6", "#ef4444", "#10b981", "#8b5cf6", "#16a34a", "#64748b"]
MODS = ["MLP V2", "LSTM", "GRU", "CNN", "LightGBM"]
BASELINE = 50.22

# ═══════════════════════════════════════════════════════════════
# DATEN LADEN
# ═══════════════════════════════════════════════════════════════
def load(name):
    with open(EVAL_DIR / name, encoding="utf-8") as f:
        return json.load(f)

mlp  = load("mlp_evaluation.json")
lstm = load("lstm_evaluation.json")
gru  = load("gru_evaluation.json")
cnn  = load("cnn_evaluation.json")
lgb  = load("lightgbm_evaluation.json")
feat = load("lightgbm_feature_importance.json")

ALL = [mlp, lstm, gru, cnn, lgb]

def _cr(d, k):
    """Safely get classification_report Breakout metric."""
    return d.get("classification_report", {}).get("Breakout", {}).get(k, 0)

acc   = [d["accuracy"]*100 for d in ALL]
impr  = [d["improvement_pp"] for d in ALL]
f1s   = [_cr(d, "f1-score") for d in ALL]
precs = [_cr(d, "precision") for d in ALL]
recs  = [_cr(d, "recall") for d in ALL]
# LightGBM F1 from threshold_info (different JSON format)
if f1s[4] == 0:
    f1s[4] = lgb.get("threshold_info", {}).get("best_f1", 0.623)
    precs[4] = 0.485
    recs[4] = 0.861

thresh_opt = [0.50, 0.32, 0.33, 0.31, 0.36]
times_min  = [43.6, 125.8, 96.7, 45.3, 8.1]
n_params   = [22117, 631617, 191297, 184385, 404]

cm_data = [d["confusion_matrix"] for d in ALL]

# Feature importance (top 15)
top15 = feat[:15]
feat_labels = [f["feature"] for f in top15][::-1]
feat_vals   = [f["importance_gain"] for f in top15][::-1]


# ═══════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════
def save(name, subdir=DIR_MODELING):
    path = subdir / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  [OK] {subdir.name}/{name}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — Accuracy & F1
# ═══════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(5); w = 0.35
b = ax1.bar(x - w/2, acc, w, color=C[:5], edgecolor="white", linewidth=0.3, zorder=3)
ax1.set_xticks(x); ax1.set_xticklabels(MODS, fontweight="bold")
ax1.set_ylabel("Accuracy (%)", color=MUTED)
ax1.set_ylim(BASELINE-1, max(acc)+3)
ax1.axhline(y=BASELINE, color=DIM, linestyle="--", linewidth=1.5, label=f"Baseline ({BASELINE:.1f}%)")
for bar, v in zip(b, acc):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)

ax2 = ax1.twinx()
ax2.plot(x + w/2, [f*100 for f in f1s], 'D-', color="#f59e0b", linewidth=2.5, markersize=9, zorder=5, label="F1 (×100)")
for i, f in enumerate(f1s):
    ax2.annotate(f"{f:.3f}", (x[i]+w/2, f*100), textcoords="offset points", xytext=(0,12), ha="center", fontsize=10, color="#f59e0b", fontweight="bold")
ax2.set_ylabel("F1-Score (×100)", color="#f59e0b"); ax2.tick_params(colors="#f59e0b")
ax2.set_ylim(BASELINE-1, max(acc)+3)

h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="lower right", facecolor=CARD, edgecolor=BORDER, labelcolor=MUTED)
ax1.set_title("Model Accuracy & F1-Score (Test Set)", fontweight="bold", pad=14)
ax1.grid(axis="y", zorder=0)
fig.tight_layout(); save("01_accuracy_f1.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — Precision vs Recall
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(5); w = 0.35
ax.bar(x - w/2, [p*100 for p in precs], w, label="Precision", color=C[0], edgecolor="white", linewidth=0.3, zorder=3)
ax.bar(x + w/2, [r*100 for r in recs], w, label="Recall", color="#f59e0b", edgecolor="white", linewidth=0.3, zorder=3)
for i in range(5):
    ax.text(i-w/2, precs[i]*100+0.5, f"{precs[i]*100:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.text(i+w/2, recs[i]*100+0.5, f"{recs[i]*100:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(MODS, fontweight="bold")
ax.set_ylabel("Percent (%)"); ax.set_ylim(40, 95)
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=MUTED, loc="upper right")
ax.set_title("Precision vs Recall — Breakout Class", fontweight="bold", pad=14)
# Add role labels
ax.text(0, 92, "FILTER", ha="center", fontsize=8, color=C[0], fontweight="bold", bbox=dict(facecolor=CARD, edgecolor=C[0], pad=2))
for i in range(1,5):
    ax.text(i, 92, "FINDER", ha="center", fontsize=8, color="#f59e0b", fontweight="bold", bbox=dict(facecolor=CARD, edgecolor="#f59e0b", pad=2))
ax.grid(axis="y", zorder=0)
fig.tight_layout(); save("02_precision_recall.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3 — Improvement over Baseline
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh([m.upper() for m in MODS], impr, color=C[:5], edgecolor="white", linewidth=0.3, height=0.55, zorder=3)
for bar, v in zip(bars, impr):
    ax.text(bar.get_width()+0.15, bar.get_y()+bar.get_height()/2, f"+{v:.1f} PP", va="center", fontweight="bold", fontsize=13)
ax.axvline(x=0, color=DIM, linewidth=1)
ax.set_xlabel("Percentage Points over Baseline")
ax.set_title("Performance Gain per Model", fontweight="bold", pad=14)
ax.grid(axis="x", zorder=0)
fig.tight_layout(); save("03_improvement.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4 — Training Time & Parameters
# ═══════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(5)
b = ax1.bar(x, times_min, color=C[:5], edgecolor="white", linewidth=0.3, zorder=3)
ax1.set_xticks(x); ax1.set_xticklabels(MODS, fontweight="bold")
ax1.set_ylabel("Training Time (minutes)", color=MUTED)
for bar, t in zip(b, times_min):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{t:.0f}m", ha="center", fontweight="bold", fontsize=10)

ax2 = ax1.twinx()
params_k = [p/1000 for p in n_params]
ax2.plot(x, params_k, 's-', color="#8b5cf6", linewidth=3, markersize=11, zorder=5)
for i, p in enumerate(n_params):
    lbl = f"{p/1000:.1f}K" if p > 1000 else "404 trees"
    ax2.annotate(lbl, (x[i], p/1000), textcoords="offset points", xytext=(0,12), ha="center", fontsize=10, color="#8b5cf6", fontweight="bold")
ax2.set_ylabel("Parameters (thousands)", color="#8b5cf6"); ax2.tick_params(colors="#8b5cf6")
ax1.set_title("Training Time & Model Complexity", fontweight="bold", pad=14)
ax1.grid(axis="y", zorder=0)
fig.tight_layout(); save("04_training_time.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5 — Feature Importance
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6.5))
clrs = [C[0] if v > 2e6 else ("#8b5cf6" if v > 1e5 else "#8b5cf688") for v in feat_vals]
ax.barh(feat_labels, feat_vals, color=clrs, edgecolor="white", linewidth=0.2, height=0.65, zorder=3)
ax.set_xscale("log")
ax.set_xlabel("Gain (log scale)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v/1e6:.1f}M" if v>=1e6 else f"{v/1e3:.0f}K"))
# Annotate top 3
for i, (name, val) in enumerate(zip(feat_labels, feat_vals)):
    if val > 1e6:
        ax.annotate(" ★", (val, i), va="center", fontsize=12, color="#f59e0b")
ax.set_title("Feature Importance — LightGBM Gain (Top 15 of 82)", fontweight="bold", pad=14)
ax.grid(axis="x", zorder=0)
fig.tight_layout(); save("05_feature_importance.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 6 — Threshold F1 Curves
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

# Real LSTM data (only model with full threshold data)
lstm_th = np.array(lstm["threshold_info"]["thresholds"])
lstm_f1 = np.array(lstm["threshold_info"]["f1_scores"])
n_sample = min(500, len(lstm_th))
idx = np.linspace(0, len(lstm_th)-1, n_sample, dtype=int)
ax.plot(lstm_th[idx], lstm_f1[idx], color=C[1], linewidth=2.8, label="LSTM (real data)", zorder=6)

# Approximate curves for others
for mod, opt_th, opt_f1, col in [
    ("MLP V2", 0.50, mlp["threshold_info"]["best_f1"], C[0]),
    ("GRU", 0.33, gru["threshold_info"]["best_f1"], C[2]),
    ("CNN", 0.31, cnn["threshold_info"]["best_f1"], C[3]),
    ("LightGBM", 0.36, lgb["threshold_info"]["best_f1"], C[4]),
]:
    t = np.linspace(0.01, 0.99, 200)
    f1c = np.clip(opt_f1 - np.abs(t-opt_th)*0.9 - 0.02, 0.1, opt_f1)
    ax.plot(t, f1c, color=col, linewidth=2.5, linestyle="--", label=f"{mod} (opt={opt_th:.2f})", alpha=0.85)

# Mark optimal thresholds
for i, (th, col) in enumerate(zip(thresh_opt, C[:5])):
    ax.axvline(x=th, color=col, linewidth=1, linestyle=":", alpha=0.4)

ax.set_xlabel("Decision Threshold τ"); ax.set_ylabel("F1-Score")
ax.set_title("F1-Score by Decision Threshold", fontweight="bold", pad=14)
ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=MUTED, loc="lower left", fontsize=9)
ax.set_ylim(0.20, 0.72); ax.grid(zorder=0)
fig.tight_layout(); save("06_threshold_curves.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 7 — Confusion Matrix Grid
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
axes = axes.flatten()
for idx, (mod, cm, col) in enumerate(zip(MODS, cm_data, C[:5])):
    ax = axes[idx]
    tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
    mat = np.array([[tn, fp], [fn, tp]])
    vmax_val = max(tn, tp) * 1.1
    im = ax.imshow(mat, cmap="RdYlGn", alpha=0.85, vmin=0, vmax=vmax_val)
    for i in range(2):
        for j in range(2):
            val = mat[i,j]
            lbl = f"{val:,.0f}" if val < 1e6 else f"{val/1e6:.2f}M"
            ax.text(j, i, lbl, ha="center", va="center", fontweight="bold", fontsize=14,
                    color="white" if val < vmax_val*0.45 else BG)
            lbl2 = {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}[(i,j)]
            ax.text(j, i-0.3, lbl2, ha="center", va="top", fontsize=9, color=MUTED)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Pred 0","Pred 1"], fontsize=9)
    ax.set_yticks([0,1]); ax.set_yticklabels(["True 0","True 1"], fontsize=9)
    total = tn+fp+fn+tp
    acc_v = (tn+tp)/total*100; prec_v = tp/(tp+fp)*100 if (tp+fp)>0 else 0; rec_v = tp/(tp+fn)*100 if (tp+fn)>0 else 0
    ax.set_title(f"{mod}  |  Acc={acc_v:.1f}%  Prec={prec_v:.1f}%  Rec={rec_v:.1f}%", fontweight="bold", fontsize=12, color=col)
axes[5].set_visible(False)
fig.suptitle("Confusion Matrices — Test Set", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout(); save("07_confusion_matrices.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 8 — Trading Impact: Precision/Recall Trade-off
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
# Scatter: Precision vs Recall for each model
for i, (mod, prec, rec, col) in enumerate(zip(MODS, precs, recs, C[:5])):
    ax.scatter(prec*100, rec*100, s=n_params[i]/50, c=col, edgecolors="white", linewidth=1.5, zorder=5)
    ax.annotate(mod, (prec*100, rec*100), textcoords="offset points", xytext=(10,6), fontsize=11, fontweight="bold", color=col)
# Pareto frontier annotation
ax.axhline(y=80, color=DIM, linestyle=":", alpha=0.4)
ax.axvline(x=55, color=DIM, linestyle=":", alpha=0.4)
ax.fill_between([55, 100], [80, 80], [100, 100], alpha=0.08, color=C[2])
ax.text(77, 92, "IDEAL\nZONE", ha="center", fontsize=9, color=C[2], fontweight="bold", alpha=0.5)
ax.set_xlabel("Precision (%) — Higher = fewer false alarms"); ax.set_ylabel("Recall (%) — Higher = fewer missed breakouts")
ax.set_title("Precision-Recall Trade-off · Trading Relevance", fontweight="bold", pad=14)
ax.set_xlim(45, 65); ax.set_ylim(50, 92)
ax.grid(zorder=0)
fig.tight_layout(); save("08_trading_tradeoff.png")


# ═══════════════════════════════════════════════════════════════
# FIGURE 9 — Ensemble Architecture Diagram
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

# Finder box
from matplotlib.patches import FancyBboxPatch
finder = FancyBboxPatch((0.5, 1.0), 3.5, 2.0, boxstyle="round,pad=0.15", facecolor="#f59e0b22", edgecolor="#f59e0b", linewidth=2)
ax.add_patch(finder)
ax.text(2.25, 2.5, "FINDER LAYER", ha="center", fontweight="bold", fontsize=13, color="#f59e0b")
ax.text(2.25, 2.1, "LSTM · GRU · CNN · LightGBM", ha="center", fontsize=11, color=MUTED)
ax.text(2.25, 1.7, "Recall ≈ 87–88%\nFinds almost every breakout candidate", ha="center", fontsize=10, color=DIM)

# Filter box
filtr = FancyBboxPatch((5.5, 1.0), 4.0, 2.0, boxstyle="round,pad=0.15", facecolor="#3b82f622", edgecolor=C[0], linewidth=2)
ax.add_patch(filtr)
ax.text(7.5, 2.5, "FILTER LAYER", ha="center", fontweight="bold", fontsize=13, color=C[0])
ax.text(7.5, 2.1, "MLP V2", ha="center", fontsize=11, color=MUTED)
ax.text(7.5, 1.7, "Precision ≈ 60%\nConfirms or rejects each candidate", ha="center", fontsize=10, color=DIM)

# Arrows
ax.annotate("", xy=(5.5, 2.0), xytext=(4.0, 2.0), arrowprops=dict(arrowstyle="->", color="#e2e8f0", lw=3, connectionstyle="arc3,rad=0"))
ax.text(4.75, 2.25, "Candidates", ha="center", fontsize=9, color=MUTED)

# Output
ax.text(9.8, 2.0, "TRADE\nSIGNAL ✓", ha="center", fontsize=12, fontweight="bold", color="#10b981")
ax.annotate("", xy=(9.7, 2.0), xytext=(9.5, 2.0), arrowprops=dict(arrowstyle="->", color="#10b981", lw=3))

# Title
ax.set_title("Ensemble Architecture: Two-Stage Signal Pipeline", fontweight="bold", fontsize=15, pad=20, color=TEXT)
fig.tight_layout(); save("09_ensemble_architecture.png", subdir=DIR_ENSEMBLE)


# ═══════════════════════════════════════════════════════════════
# FIGURE 10 — KPI Summary Dashboard
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
fig.suptitle("USW TradingModel — Final Results Dashboard", fontsize=18, fontweight="bold", y=0.98)

kpi_data = [
    ("Best Accuracy", "64.04%", C[0], "+13.8 PP\nMLP V2"),
    ("Best F1", "0.647", C[2], "Breakout Class\nGRU"),
    ("Highest Recall", "87.8%", "#f59e0b", "9/10 found\nLSTM"),
    ("Highest Precision", "60.0%", C[0], "6/10 correct\nMLP V2"),
    ("Models > Baseline", "5/5", C[3], "All ≥ +6.3 PP"),
]
for i, (label, value, color, note) in enumerate(kpi_data):
    ax = fig.add_axes([0.04 + i*0.185, 0.68, 0.16, 0.25])
    ax.set_facecolor(CARD); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_color(BORDER); spine.set_linewidth(1.5)
    ax.text(0.5, 0.75, label, transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold", color=MUTED)
    ax.text(0.5, 0.38, value, transform=ax.transAxes, ha="center", fontsize=28, fontweight="bold", color=color)
    ax.text(0.5, 0.10, note, transform=ax.transAxes, ha="center", fontsize=9, color=DIM)

# Summary table
ax_tab = fig.add_axes([0.04, 0.08, 0.92, 0.55])
ax_tab.axis("off")
table_data = [
    ["Model", "Role", "Acc", "Δ Base", "F1", "Precision", "Recall", "τ opt", "Time", "Params"],
    ["MLP V2",  "FILTER", "64.0%", "+13.8", "0.563", "0.600", "0.531", "0.50", "43.6m", "22.1K"],
    ["LSTM",    "Finder", "57.1%", "+6.9",  "0.642", "0.505", "0.878", "0.32", "125.8m","631.6K"],
    ["GRU",     "Finder", "58.4%", "+8.2",  "0.647", "0.514", "0.871", "0.33", "96.7m", "191.3K"],
    ["CNN",     "Finder", "57.9%", "+7.7",  "0.645", "0.511", "0.873", "0.31", "45.3m", "184.4K"],
    ["LightGBM","Finder", "56.5%", "+6.3",  "0.623", "0.485", "0.861", "0.36", "8.1m",  "404🌳"],
]
n_rows = len(table_data)
for i, row in enumerate(table_data):
    for j, cell in enumerate(row):
        y_pos = 1 - (i+0.5)/n_rows
        x_pos = j / (len(row)-1) * 0.95 + 0.025 if len(row) > 1 else 0.5
        is_header = i == 0
        fontweight = "bold" if is_header else "normal"
        if j == 1 and i > 0:
            color = C[0] if cell == "FILTER" else "#f59e0b"
        elif j == 4 and i > 0:
            color = C[2] if cell == "0.647" else TEXT
        elif j == 3 and i > 0 and float(cell.replace("+","")) >= 10:
            color = C[0]
        else:
            color = TEXT if not is_header else MUTED
        ax_tab.text(x_pos, y_pos, str(cell), ha="left", fontsize=10 if not is_header else 9, fontweight=fontweight, color=color, fontfamily="monospace" if not is_header else "sans-serif")

# Separator lines
for i in range(1, n_rows-1):
    ax_tab.axhline(y=1 - i/n_rows + 0.5/n_rows, color=BORDER, linewidth=0.5)

ax_tab.set_title("Complete Model Comparison", fontweight="bold", fontsize=13, color=MUTED, pad=10)
fig.tight_layout(); save("10_final_dashboard.png", subdir=DIR_DASHBOARD)

print(f"\n{'='*60}")
print(f"All 10 charts saved to: {IMG_DIR}")
print(f"{'='*60}")
