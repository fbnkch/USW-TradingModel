"""
plots_feature_justification.py
-------------------------------
Erzeugt Plots, die zeigen WARUM jedes Feature in unsere Pipeline aufgenommen wurde.

Jeder Plot illustriert eine spezifische Frage:
  1. Warum normalisieren wir?         → close_norm, volume_norm
  2. Warum VWAP?                      → vwap_distance
  3. Warum Cumulative Delta?          → cumulative_delta
  4. Warum Opening Range Breakout?    → opening_range_position
  5. Warum Volume Spike Ratio?        → volume_spike_ratio
  6. Warum MACD?                      → MACD_histogram
  7. Warum Slopes 2. Ordnung?         → Slope_Slope (Beschleunigung)
  8. Warum Lagged Features?           → close_norm_lag5/10/15

Verwendung:
    python scripts/02_data_understanding/plots_feature_justification.py
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR  = PROJECT_ROOT / "data" / "raw" / "Bars_1m_adj"
PROC_DIR = PROJECT_ROOT / "data" / "processed" / "pre_split"
OUT_DIR  = PROJECT_ROOT / "artifacts" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "AAPL"
COLORS = ["#2c3e50", "#e74c3c", "#2ecc71", "#3498db", "#e67e22", "#9b59b6"]

_BREAKOUT_COL = "breakout_30m"


def load_data():
    raw = pd.read_parquet(RAW_DIR / f"{SYMBOL}.parquet")
    raw["timestamp"] = pd.to_datetime(raw.index)
    proc = pd.read_parquet(PROC_DIR / f"{SYMBOL}_train.parquet")
    proc["timestamp"] = pd.to_datetime(proc["timestamp"])
    return raw, proc


def save_and_close(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1: NORMIERUNG — Warum Z-Score?
# ═══════════════════════════════════════════════════════════════════════════

def plot_why_normalize(raw):
    """Zeigt Roh-Preise vs. Z-Score für 3 Aktien mit unterschiedlichen
    Preisniveaus. Ohne Normierung kann ein ML-Modell nicht
    symboluebergreifend lernen."""
    symbols = ["AAPL", "NVDA", "MSFT"]
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), dpi=120)

    for sym, c in zip(symbols, COLORS):
        df = pd.read_parquet(RAW_DIR / f"{sym}.parquet")
        df.index = pd.to_datetime(df.index)

        # Roh-Preis
        subset = df.iloc[:390 * 5]  # 5 Handelstage
        axes[0].plot(subset.index, subset["close"], color=c, linewidth=1.0,
                     alpha=0.8, label=f"{sym} (${subset['close'].iloc[-1]:.0f})")

        # Z-Score
        z = (subset["close"] - subset["close"].rolling(390).mean()) / (
            subset["close"].rolling(390).std() + 1e-12)
        axes[1].plot(subset.index, z, color=c, linewidth=1.0,
                     alpha=0.8, label=sym)

    for i, (title, ylabel) in enumerate([
        ("Ohne Normierung: Roh-Preise (unvergleichbar)", "Preis ($)"),
        ("Mit Rolling Z-Score: Vergleichbare Skala", "Z-Score (σ)"),
    ]):
        ax = axes[i]
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8) if i == 1 else None

    fig.suptitle("Feature-Rechtfertigung 1: Normierung (close_norm, volume_norm)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_and_close(fig, "justify_01_normalization.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2: VWAP als Intraday-Magnet
# ═══════════════════════════════════════════════════════════════════════════

def plot_why_vwap(raw):
    """VWAP wirkt wie ein Magnet: Preis pendelt um VWAP, Breakouts
    passieren wenn sich der Preis loest. vwap_distance misst genau das."""
    df = raw.copy()
    if "timestamp" not in df.columns:
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Zwei repräsentative Tage
    dates = sorted(pd.to_datetime(df["timestamp"]).dt.date.unique())
    day1, day2 = dates[10], dates[20]
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), dpi=120)

    for i, day in enumerate([day1, day2]):
        ax = axes[i]
        mask = df["date"] == day
        d = df[mask].copy()
        if len(d) == 0:
            continue

        # VWAP berechnen
        d["cum_pv"] = (d["close"] * d["volume"]).cumsum()
        d["cum_vol"] = d["volume"].cumsum()
        d["vwap_calc"] = d["cum_pv"] / d["cum_vol"]

        # Close vs VWAP
        ax.plot(d.index - d.index.min(), d["close"], color=COLORS[0],
                linewidth=1.5, label="Close-Preis")
        ax.plot(d.index - d.index.min(), d["vwap_calc"], color=COLORS[3],
                linewidth=2.0, linestyle="--", label="VWAP")

        # Abstand färben
        diff = (d["close"] - d["vwap_calc"]) / d["vwap_calc"] * 100
        for j in range(len(d) - 1):
            color = COLORS[2] if diff.iloc[j] > 0 else COLORS[1]
            ax.fill_between([j, j + 1],
                            d["close"].iloc[j], d["vwap_calc"].iloc[j],
                            color=color, alpha=0.15)

        # Breakout-Bars markieren (aus processed data)
        try:
            p = pd.read_parquet(PROC_DIR / f"{SYMBOL}_train.parquet")
            p["ts"] = pd.to_datetime(p["timestamp"])
            p["d"] = p["ts"].dt.date
            p_day = p[p["d"] == day]
            if len(p_day) > 0:
                bo_bars = p_day.index[p_day[_BREAKOUT_COL] == 1]
                if len(bo_bars) > 0:
                    bo_first = bo_bars[0] - p_day.index[0]
                    bo_last = bo_bars[-1] - p_day.index[0]
                    ax.axvspan(bo_first, bo_last, alpha=0.15, color="gold",
                               label="Breakout (breakout_30m=1)")
        except:
            pass

        ax.set_title(f"{day}", fontsize=12)
        ax.set_ylabel("Preis ($)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)

    axes[0].set_xlabel("Minute des Handelstages")
    axes[1].set_xlabel("Minute des Handelstages")
    fig.suptitle("Feature-Rechtfertigung 2: VWAP (vwap_distance)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_and_close(fig, "justify_02_vwap.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3: CUMULATIVE DELTA — Smart Money Flow
# ═══════════════════════════════════════════════════════════════════════════

def plot_why_cumulative_delta(proc):
    """Cumulative Delta addiert Kauf-/Verkaufsdruck auf. Positives Delta
    zeigt institutionelle Kaeuferaktivitaet, oft VOR einem Breakout."""
    df = proc.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    dates = sorted(df["date"].unique())
    pick_date = dates[15]

    mask = df["date"] == pick_date
    d = df[mask].copy().reset_index(drop=True)

    bo_mask = d[_BREAKOUT_COL] == 1

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=120, sharex=True)

    # Panel 1: Close + Breakout-Bereiche
    ax = axes[0]
    ax.plot(d.index, d["close"], color=COLORS[0], linewidth=1.2, label="Close")
    # Breakout-Bereiche markieren
    in_breakout = False
    start_bo = None
    for i, is_bo in enumerate(bo_mask):
        if is_bo and not in_breakout:
            start_bo = i
            in_breakout = True
        elif not is_bo and in_breakout:
            ax.axvspan(start_bo, i - 1, alpha=0.15, color="gold")
            in_breakout = False
    if in_breakout:
        ax.axvspan(start_bo, len(d) - 1, alpha=0.15, color="gold",
                   label="Breakout-Clusters")
    ax.set_ylabel("Close ($)")
    ax.set_title(f"Tag {pick_date} — Breakout-Clusters (gold)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 2: Cumulative Delta
    ax = axes[1]
    ax.fill_between(d.index, 0, d["cumulative_delta"],
                    where=(d["cumulative_delta"] >= 0), color=COLORS[2], alpha=0.5,
                    label="Kaufdruck (+)")
    ax.fill_between(d.index, 0, d["cumulative_delta"],
                    where=(d["cumulative_delta"] < 0), color=COLORS[1], alpha=0.5,
                    label="Verkaufsdruck (-)")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Kumulierte Delta")
    ax.set_title("Cumulative Delta (institutionelle Aktivitaet)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 3: Volume + Volume Norm
    ax = axes[2]
    ax.bar(d.index, d["volume_norm"].clip(-2, 10), color=COLORS[3], alpha=0.5,
           width=1.0, label="volume_norm")
    ax.axhline(y=2, color=COLORS[1], linestyle="--", linewidth=1.0,
               label="Schwelle: sehr hohes Volumen")
    ax.set_xlabel("Minute des Handelstages")
    ax.set_ylabel("Z-Score")
    ax.set_title("Volume Norm (Z-Score)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Feature-Rechtfertigung 3: Cumulative Delta + Volume Norm",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_and_close(fig, "justify_03_cumulative_delta.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4: OPENING RANGE BREAKOUT
# ═══════════════════════════════════════════════════════════════════════════

def plot_why_opening_range(proc):
    """Die ersten 30 Minuten setzen die Range fuer den Tag.
    Preis ueber dem Opening-Range-Hoch = klassisches Breakout-Signal."""
    df = proc.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["bar"] = df.groupby("date").cumcount()

    dates = sorted(df["date"].unique())
    pick_date = dates[18]
    d = df[df["date"] == pick_date].copy().reset_index(drop=True)

    # Opening Range (erste 30 Bars)
    or_high = d["high"].iloc[:30].max()
    or_low = d["low"].iloc[:30].min()

    fig, ax = plt.subplots(figsize=(16, 5), dpi=120)

    ax.plot(d.index, d["close"], color=COLORS[0], linewidth=1.5, label="Close")

    # Opening Range
    ax.axvspan(0, 30, alpha=0.08, color="gray")
    ax.text(15, ax.get_ylim()[1] if ax.get_ylim()[1] is not None else d["high"].max(),
            "Opening\nRange", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(y=or_high, color=COLORS[1], linestyle="--", linewidth=1.2,
               label=f"Opening Range High = ${or_high:.2f}")
    ax.axhline(y=or_low, color=COLORS[2], linestyle="--", linewidth=1.2,
               label=f"Opening Range Low  = ${or_low:.2f}")

    # Breakouts markieren
    bo_bars = d.index[d[_BREAKOUT_COL] == 1]
    if len(bo_bars) > 0:
        bo_min = bo_bars[0]
        bo_max = bo_bars[-1]
        ax.axvspan(bo_min, bo_max, alpha=0.12, color="gold",
                   label=f"Breakout-Bereich")

    # Pfeil: Preis bricht OR High
    cross_idx = np.where((d["close"] > or_high) & (d.index >= 30))[0]
    if len(cross_idx) > 0:
        first_cross = cross_idx[0]
        ax.annotate("Preis > OR-High →\nBreakout-Signal",
                    xy=(first_cross, d["close"].iloc[first_cross]),
                    xytext=(first_cross + 40, d["close"].iloc[first_cross] + 1.5),
                    arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.5),
                    fontsize=10, color=COLORS[1], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Minute des Handelstages")
    ax.set_ylabel("Preis ($)")
    ax.set_title(f"Feature-Rechtfertigung 4: Opening Range Breakout – {pick_date}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    save_and_close(fig, "justify_04_opening_range.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5: VOLUME SPIKE RATIO — Normales vs. Anomales Volumen
# ═══════════════════════════════════════════════════════════════════════════

def plot_why_volume_spike(proc):
    """Volume Spike Ratio vergleicht aktuelles Volumen mit Tagesdurchschnitt.
    Spikes (>2x Durchschnitt) fallen oft mit Breakouts zusammen."""
    df = proc.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    dates = sorted(df["date"].unique())
    pick_date = dates[25]
    d = df[df["date"] == pick_date].copy().reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), dpi=120, sharex=True)

    # Panel 1: Volume + Spike Ratio
    ax = axes[0]
    ax.bar(d.index, d["volume"], color="lightgray", alpha=0.7, width=1.0, label="Volumen (Roh)")
    ax.set_ylabel("Volumen (Roh)", color="gray")
    ax.tick_params(axis="y", labelcolor="gray")

    ax2 = ax.twinx()
    ax2.plot(d.index, d["volume_spike_ratio"], color=COLORS[0], linewidth=1.5,
             label="Volume Spike Ratio")
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Normal = 1.0")
    ax2.axhline(y=2.0, color=COLORS[1], linestyle="--", linewidth=1.0,
                label="Spike-Schwelle = 2.0")
    ax2.set_ylabel("Volume Spike Ratio", color=COLORS[0])
    ax2.tick_params(axis="y", labelcolor=COLORS[0])

    # Breakouts markieren
    bo_mask = d[_BREAKOUT_COL] == 1
    in_bo = False
    start = None
    for i, is_bo in enumerate(bo_mask):
        if is_bo and not in_bo:
            start = i; in_bo = True
        elif not is_bo and in_bo:
            ax.axvspan(start, i - 1, alpha=0.1, color="gold")
            in_bo = False
    if in_bo:
        ax.axvspan(start, len(d) - 1, alpha=0.1, color="gold")

    ax.set_title(f"Volume Spike Ratio – {pick_date}", fontsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel 2: Close-Preis
    ax = axes[1]
    ax.plot(d.index, d["close"], color=COLORS[0], linewidth=1.5)
    # Volumen-Spikes als rote Punkte
    spike_mask = d["volume_spike_ratio"] > 2.5
    spike_idx = d.index[spike_mask]
    ax.scatter(spike_idx, d.loc[spike_mask, "close"], color=COLORS[1], s=20,
               alpha=0.6, label="Volumen-Spike (Ratio > 2.5)")
    ax.set_xlabel("Minute des Handelstages")
    ax.set_ylabel("Preis ($)")
    ax.set_title("Close + Volumen-Spikes (rot markiert)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Feature-Rechtfertigung 5: Volume Spike Ratio",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_and_close(fig, "justify_05_volume_spike.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 6: MACD HISTOGRAM — Trendwechsel vor Breakout
# ═══════════════════════════════════════════════════════════════════════════

def plot_why_macd(proc):
    """MACD-Histogramm kreuzt die Nulllinie BEVOR ein Breakout passiert.
    Das ist ein klassisches Vorlaufsignal fuer Trendwechsel."""
    df = proc.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    dates = sorted(df["date"].unique())
    pick_date = dates[30]
    d = df[df["date"] == pick_date].copy().reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=120, sharex=True)

    # Panel 1: Close + EMAs
    ax = axes[0]
    close = d["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ax.plot(d.index, d["close"], color=COLORS[0], linewidth=1.2, label="Close")
    ax.plot(d.index, ema12, color=COLORS[3], linewidth=1.0, alpha=0.7, label="EMA 12")
    ax.plot(d.index, ema26, color=COLORS[4], linewidth=1.0, alpha=0.7, label="EMA 26")
    ax.set_ylabel("Preis ($)")
    ax.set_title(f"Close + EMA 12/26 – {pick_date}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 2: MACD Histogram
    ax = axes[1]
    colors_macd = [COLORS[2] if v >= 0 else COLORS[1] for v in d["MACD_histogram"]]
    ax.bar(d.index, d["MACD_histogram"], color=colors_macd, width=1.0, alpha=0.6)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("MACD Histogram")
    ax.set_title("MACD Histogram: gruen = positiv (Aufwaertsdruck), rot = negativ", fontsize=11)
    ax.grid(alpha=0.25)

    # Panel 3: Close + Breakout
    ax = axes[2]
    ax.plot(d.index, d["close"], color=COLORS[0], linewidth=1.5)
    bo_mask = d[_BREAKOUT_COL] == 1
    in_bo = False; start = None
    for i, is_bo in enumerate(bo_mask):
        if is_bo and not in_bo: start = i; in_bo = True
        elif not is_bo and in_bo:
            ax.axvspan(start, i - 1, alpha=0.12, color="gold")
            in_bo = False
    if in_bo:
        ax.axvspan(start, len(d) - 1, alpha=0.12, color="gold")
    ax.set_xlabel("Minute des Handelstages")
    ax.set_ylabel("Preis ($)")
    ax.set_title("Close + Breakout-Bereiche (gold)", fontsize=11)
    ax.grid(alpha=0.25)

    fig.suptitle("Feature-Rechtfertigung 6: MACD Histogram (Trendwechsel-Signal)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_and_close(fig, "justify_06_macd.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 7: SLOPES 2. ORDNUNG — Beschleunigung vor dem Preis
# ═══════════════════════════════════════════════════════════════════════════

def plot_why_slope_acceleration(proc):
    """2. Ableitung (Slope of Slope) erkennt Momentum-Aenderungen
    FRUEHER als die 1. Ableitung oder der Preis selbst."""
    df = proc.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    dates = sorted(df["date"].unique())
    pick_date = dates[35]
    d = df[df["date"] == pick_date].copy().reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=120, sharex=True)

    # Panel 1: Close-Preis
    ax = axes[0]
    ax.plot(d.index, d["close"], color=COLORS[0], linewidth=1.5)
    # Bo markieren
    bo_mask = d[_BREAKOUT_COL] == 1
    in_bo = False; start = None
    for i, is_bo in enumerate(bo_mask):
        if is_bo and not in_bo: start = i; in_bo = True
        elif not is_bo and in_bo:
            ax.axvspan(start, i - 1, alpha=0.12, color="gold")
            in_bo = False
    if in_bo:
        ax.axvspan(start, len(d) - 1, alpha=0.12, color="gold")
    ax.set_ylabel("Preis ($)")
    ax.set_title("Close + Breakout (gold)", fontsize=11)
    ax.grid(alpha=0.25)

    # Panel 2: Slope 1. Ordnung (Geschwindigkeit)
    ax = axes[1]
    ax.plot(d.index, d["Slope_close_1"], color=COLORS[3], linewidth=1.2,
            label="Slope_close_1 (1. Ableitung: Geschwindigkeit)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Slope")
    ax.set_title("1. Ableitung: Geschwindigkeit des Preises", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 3: Slope 2. Ordnung (Beschleunigung)
    ax = axes[2]
    ax.plot(d.index, d["Slope_Slope_close_1_1"], color=COLORS[1], linewidth=1.5,
            label="Slope_Slope_close_1_1 (2. Ableitung: Beschleunigung)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Minute des Handelstages")
    ax.set_ylabel("Slope of Slope")
    ax.set_title("2. Ableitung: Beschleunigung — aendert sich FRUEHER als der Preis",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Feature-Rechtfertigung 7: Slopes 2. Ordnung (Beschleunigung)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_and_close(fig, "justify_07_slope_acceleration.png")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 8: LAGGED FEATURES — Geschichte der letzten 15 Minuten
# ═══════════════════════════════════════════════════════════════════════════

def plot_why_lagged_features(proc):
    """Vergleicht close_norm HEUTE mit close_norm vor 5/10/15 Minuten.
    Breakouts kuendigen sich ueber mehrere Minuten an — nicht instantan."""
    df = proc.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    dates = sorted(df["date"].unique())
    pick_date = dates[40]
    d = df[df["date"] == pick_date].copy().reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), dpi=120, sharex=True)

    # Panel 1: Close-Norm heute + vor 5/10/15 Min
    ax = axes[0]
    ax.plot(d.index, d["close_norm"], color=COLORS[0], linewidth=2.0,
            label="close_norm (heute)")
    ax.plot(d.index, d["close_norm_lag5"], color=COLORS[3], linewidth=1.0,
            alpha=0.7, linestyle="--", label="close_norm (vor 5 Min)")
    ax.plot(d.index, d["close_norm_lag10"], color=COLORS[4], linewidth=1.0,
            alpha=0.7, linestyle="--", label="close_norm (vor 10 Min)")
    ax.plot(d.index, d["close_norm_lag15"], color=COLORS[5], linewidth=1.0,
            alpha=0.7, linestyle="--", label="close_norm (vor 15 Min)")

    # Vertikale Linie: "Jetzt"
    mid = len(d) // 2
    ax.axvline(x=mid, color=COLORS[1], linestyle="-", linewidth=1.5)

    # Annotation: was "jetzt" sieht vs. was vorher war
    ax.annotate("Heute\n(Jetzt)",
                xy=(mid, d["close_norm"].iloc[mid]),
                xytext=(mid + 15, d["close_norm"].iloc[mid] + 0.5),
                arrowprops=dict(arrowstyle="->"), fontsize=9, fontweight="bold")
    ax.annotate("Vor 15 Min:\nPreis war tiefer\n→ Aufwärtstrend\nbaut sich auf",
                xy=(mid, d["close_norm_lag15"].iloc[mid]),
                xytext=(mid - 60, d["close_norm_lag15"].iloc[mid] - 1.0),
                arrowprops=dict(arrowstyle="->"), fontsize=8,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_ylabel("Z-Score")
    ax.set_title("close_norm: Heute vs. vor 5/10/15 Minuten", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    # Panel 2: Differenz (heute - vor 5 Min) = Momentum-Indikator
    ax = axes[1]
    mom = d["close_norm"] - d["close_norm_lag5"]
    colors_mom = [COLORS[2] if v >= 0 else COLORS[1] for v in mom]
    ax.bar(d.index, mom, color=colors_mom, alpha=0.5, width=1.0)
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.set_xlabel("Minute des Handelstages")
    ax.set_ylabel("Differenz (heute - vor 5 Min)")
    ax.set_title("Momentum: close_norm(heute) - close_norm(vor 5 Min)", fontsize=12)
    ax.grid(alpha=0.25)

    fig.suptitle("Feature-Rechtfertigung 8: Lagged Features (Gedaechtnis)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_and_close(fig, "justify_08_lagged_features.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE JUSTIFICATION PLOTS")
    print("=" * 60)

    raw, proc = load_data()
    print(f"  Raw: {len(raw):,} bars | Processed: {len(proc):,} bars\n")

    plot_why_normalize(raw)
    plot_why_vwap(raw)
    plot_why_cumulative_delta(proc)
    plot_why_opening_range(proc)
    plot_why_volume_spike(proc)
    plot_why_macd(proc)
    plot_why_slope_acceleration(proc)
    plot_why_lagged_features(proc)

    print(f"\nAlle Plots gespeichert in: {OUT_DIR}")
