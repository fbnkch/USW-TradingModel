"""
Ergaenzende Feature-Rechtfertigungs-Plots fuer fehlende Features:
  - RSI_14 + BB_position: Extremwerte als Breakout-Indikatoren
  - EMA_cross: Trendwechsel-Signale
  - volume_price_divergence: Fake-Breakout-Warnung
  - Pipeline-Flowchart: Gesamtarchitektur
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "data" / "processed" / "pre_split"
OUT_DIR  = PROJECT_ROOT / "artifacts" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "AAPL"
COLORS = ["#2c3e50", "#e74c3c", "#2ecc71", "#3498db", "#e67e22", "#9b59b6"]
TARGET = "breakout_30m"


def load():
    df = pd.read_parquet(PROC_DIR / f"{SYMBOL}_train.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df


def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 9: RSI_14 + BB_position — Extremwerte als Breakout-Indikatoren
# ═══════════════════════════════════════════════════════════════════════

def plot_rsi_bb(df):
    """RSI und BB-Position zeigen Extremzustaende. Breakouts passieren
    haeufig wenn beide Indikatoren extreme Werte annehmen."""
    dates = sorted(df["date"].unique())
    d = df[df["date"] == dates[35]].copy().reset_index(drop=True)

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), dpi=120, sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.5, 1.5, 2.5]})

    # Panel 1: Close + Bollinger Bands
    ax = axes[0]
    close = d["close"]
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    ax.plot(d.index, close, color=COLORS[0], linewidth=1.5, label="Close")
    ax.plot(d.index, bb_mid, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.fill_between(d.index, bb_mid + 2*bb_std, bb_mid - 2*bb_std,
                    alpha=0.1, color="blue", label="Bollinger Bands (+2σ)")
    ax.fill_between(d.index, close, bb_mid + 2*bb_std,
                    where=(close > bb_mid + 2*bb_std),
                    color="darkgreen", alpha=0.3, label="Preis > oberes Band")

    bo = d[TARGET] == 1
    in_bo = False; s = None
    for i, v in enumerate(bo):
        if v and not in_bo: s = i; in_bo = True
        elif not v and in_bo:
            ax.axvspan(s, i-1, alpha=0.12, color="gold")
            in_bo = False
    if in_bo: ax.axvspan(s, len(d)-1, alpha=0.12, color="gold", label="Breakout")
    ax.set_ylabel("Preis ($)")
    ax.set_title("Close + Bollinger Bänder", fontsize=11)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel 2: BB_position
    ax = axes[1]
    bb_pos = d["BB_position"]
    colors_bb = [COLORS[2] if v > 1 else (COLORS[1] if v < 0 else "gray") for v in bb_pos]
    ax.bar(d.index, bb_pos.clip(-0.5, 1.5), color=colors_bb, width=1.0, alpha=0.7)
    ax.axhline(y=1.0, color=COLORS[2], linestyle="--", linewidth=1.0, label="Oberes Band (Breakout)")
    ax.axhline(y=0.0, color=COLORS[1], linestyle="--", linewidth=1.0, label="Unteres Band")
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, label="Mittelwert")
    ax.set_ylabel("Position")
    ax.set_title("BB_position: > 1 = Preis bricht oberes Band (gruen)", fontsize=11)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel 3: RSI_14
    ax = axes[2]
    rsi = d["RSI_14"]
    colors_rsi = [COLORS[1] if v > 1.5 else (COLORS[2] if v < -1.5 else "gray") for v in rsi]
    ax.bar(d.index, rsi.clip(-3, 3), color=colors_rsi, width=1.0, alpha=0.7)
    ax.axhline(y=1.5, color=COLORS[1], linestyle="--", linewidth=1.0, label="Extrem ueberkauft (Z > 1.5)")
    ax.axhline(y=-1.5, color=COLORS[2], linestyle="--", linewidth=1.0, label="Extrem ueberverkauft (Z < -1.5)")
    ax.set_ylabel("Z-Score")
    ax.set_title("RSI_14: Extremwerte zeigen Momentum", fontsize=11)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel 4: Close + Breakouts
    ax = axes[3]
    ax.plot(d.index, close, color=COLORS[0], linewidth=1.5)
    in_bo = False; s = None
    for i, v in enumerate(d[TARGET] == 1):
        if v and not in_bo: s = i; in_bo = True
        elif not v and in_bo:
            ax.axvspan(s, i-1, alpha=0.12, color="gold")
            in_bo = False
    if in_bo: ax.axvspan(s, len(d)-1, alpha=0.12, color="gold", label="Breakout")
    ax.set_xlabel("Minute des Handelstages")
    ax.set_ylabel("Preis ($)")
    ax.set_title("Breakout-Bereiche (gold) zum Vergleich", fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    fig.suptitle("Feature-Rechtfertigung 9: RSI_14 + BB_position (Extremwerte)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "justify_09_rsi_bb.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 10: EMA_CROSS — Trendwechsel-Signale
# ═══════════════════════════════════════════════════════════════════════

def plot_ema_cross(df):
    """EMA_cross misst den Abstand zwischen kurzem und langem EMA.
    Kreuzung durch 0 = Trendwechsel. Breakouts folgen oft nach Kreuzung."""
    dates = sorted(df["date"].unique())
    d = df[df["date"] == dates[42]].copy().reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), dpi=120, sharex=True)

    # Panel 1: Close + EMA_5 + EMA_20
    ax = axes[0]
    ax.plot(d.index, d["close"], color=COLORS[0], linewidth=1.2, label="Close")
    ax.plot(d.index, d["EMA_5"], color=COLORS[3], linewidth=1.2, alpha=0.8, label="EMA 5 (schnell)")
    ax.plot(d.index, d["EMA_20"], color=COLORS[4], linewidth=1.2, alpha=0.8, label="EMA 20 (langsam)")
    # Kreuzungen markieren
    diff = d["EMA_5"] - d["EMA_20"]
    for i in range(1, len(d)):
        if diff.iloc[i-1] <= 0 and diff.iloc[i] > 0:
            ax.axvline(x=i, color=COLORS[2], linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Preis ($)")
    ax.set_title("Close + EMA 5 (schnell) + EMA 20 (langsam) | Gruene Linien = Crossover", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 2: EMA_cross_5_20
    ax = axes[1]
    cross = d["EMA_cross_5_20"]
    colors_cross = [COLORS[2] if v >= 0 else COLORS[1] for v in cross]
    ax.bar(d.index, cross, color=colors_cross, width=1.0, alpha=0.6)
    ax.axhline(y=0, color="black", linewidth=1.0)
    ax.set_ylabel("Normalisierter Abstand")
    ax.set_title("EMA_cross_5_20: gruen = EMA_5 > EMA_20 (bullish), rot = bearish", fontsize=11)
    ax.grid(alpha=0.25)

    # Panel 3: Breakout
    ax = axes[2]
    ax.plot(d.index, d["close"], color=COLORS[0], linewidth=1.5)
    bo = d[TARGET] == 1
    in_bo = False; s = None
    for i, v in enumerate(bo):
        if v and not in_bo: s = i; in_bo = True
        elif not v and in_bo:
            ax.axvspan(s, i-1, alpha=0.12, color="gold")
            in_bo = False
    if in_bo: ax.axvspan(s, len(d)-1, alpha=0.12, color="gold", label="Breakout")
    ax.set_xlabel("Minute des Handelstages")
    ax.set_ylabel("Preis ($)")
    ax.set_title("Breakout-Bereiche (gold) — folgen oft nach EMA-Crossover", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Feature-Rechtfertigung 10: EMA_cross (Trendwechsel-Signal)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "justify_10_ema_cross.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 11: VOLUME-PRICE DIVERGENCE — Fake-Breakout-Warnung
# ═══════════════════════════════════════════════════════════════════════

def plot_volume_price_divergence(df):
    """Wenn der Preis steigt aber das Volumen faellt, fehlt dem Breakout
    der 'Treibstoff'. volume_price_divergence detektiert genau das."""
    dates = sorted(df["date"].unique())

    # Suche einen Tag mit klarer Divergenz
    best_day = None
    best_neg = 0
    for day in dates[10:80]:
        d = df[df["date"] == day].copy().reset_index(drop=True)
        div = d["volume_price_divergence_5"]
        neg_ratio = (div < -0.5).sum() / max(len(d), 1)
        if neg_ratio > 0.05 and len(d) > 300:
            best_day = day
            break
    if best_day is None:
        best_day = dates[40]

    d = df[df["date"] == best_day].copy().reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), dpi=120, sharex=True)

    # Panel 1: Close
    ax = axes[0]
    ax.plot(d.index, d["close"], color=COLORS[0], linewidth=1.5)
    bo = d[TARGET] == 1
    in_bo = False; s = None
    for i, v in enumerate(bo):
        if v and not in_bo: s = i; in_bo = True
        elif not v and in_bo:
            ax.axvspan(s, i-1, alpha=0.12, color="gold")
            in_bo = False
    if in_bo: ax.axvspan(s, len(d)-1, alpha=0.12, color="gold", label="Breakout")
    ax.set_ylabel("Preis ($)")
    ax.set_title(f"Close + Breakout – {best_day}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 2: Slope_close_5 + Slope_volume_5
    ax = axes[1]
    ax.plot(d.index, d["Slope_close_5"], color=COLORS[0], linewidth=1.2, label="Slope Preis (5 Min)")
    ax.plot(d.index, d["Slope_volume_5"], color=COLORS[4], linewidth=1.2, label="Slope Volumen (5 Min)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Slope")
    ax.set_title("Preis-Slope vs. Volumen-Slope", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 3: volume_price_divergence
    ax = axes[2]
    div = d["volume_price_divergence_5"]
    colors_div = [COLORS[1] if v < -0.3 else (COLORS[2] if v > 0.3 else "gray") for v in div]
    ax.bar(d.index, div.clip(-3, 3), color=colors_div, width=1.0, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhspan(-0.3, 0.3, alpha=0.05, color="gray", label="Neutrale Zone")

    # Rot markieren: Preis steigt, Volumen faellt
    neg_mask = div < 0
    if neg_mask.any():
        neg_sections = []
        in_neg = False; ns = None
        for i, v in enumerate(neg_mask):
            if v and not in_neg: ns = i; in_neg = True
            elif not v and in_neg:
                neg_sections.append((ns, i-1))
                in_neg = False
        if in_neg: neg_sections.append((ns, len(d)-1))
        for ns, ne in neg_sections[:5]:
            ax.axvspan(ns, ne, alpha=0.15, color=COLORS[1])

    ax.set_xlabel("Minute des Handelstages")
    ax.set_ylabel("Divergenz")
    ax.set_title("volume_price_divergence_5: ROT = Preis up + Volumen down = Fake-Breakout-Warnung",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Feature-Rechtfertigung 11: Volume-Price Divergence (Fake-Breakout-Detektor)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "justify_11_divergence.png")

if __name__ == "__main__":
    print("=" * 60)
    print("ERGÄNZENDE PLOTS")
    print("=" * 60)
    df = load()
    print(f"  Processed: {len(df):,} bars\n")
    plot_rsi_bb(df)
    plot_ema_cross(df)
    plot_volume_price_divergence(df)
    print(f"\nAlle Plots gespeichert in: {OUT_DIR}")
