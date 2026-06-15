import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from base_utils import ensure_artifacts_dir, ARTIFACTS_DIR, PROJECT_ROOT

# ─── Plot 1: Breakout-Wahrscheinlichkeit nach Tageszeit ────────────────────

def plot_breakout_by_time_of_day(symbol="AAPL"):
    """Breakout-Rate pro Minute des Handelstages.

    Zeigt die Marktmikrostruktur: Breakouts haeufen sich zur Eroeffnung
    und vor allem zum Handelsschluss. Mittags passieren kaum Breakouts.
    """
    path = PROJECT_ROOT / "data" / "processed" / "pre_split" / f"{symbol}_train.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nicht gefunden. Bitte zuerst scripts/03_pre_split_prep/main.py ausfuehren."
        )

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["minute_of_day"] = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - 570)

    # Breakout-Rate pro Minute (nur Minuten mit >= 50 Bars auswerten)
    by_minute = df.groupby("minute_of_day")["breakout_30m"].agg(["mean", "count"])
    by_minute = by_minute[by_minute["count"] >= 50]
    by_minute["mean_pct"] = by_minute["mean"] * 100

    overall = df["breakout_30m"].mean() * 100

    fig, ax = plt.subplots(figsize=(14, 5), dpi=120)

    ax.fill_between(by_minute.index, by_minute["mean_pct"], alpha=0.3, color="steelblue")
    ax.plot(by_minute.index, by_minute["mean_pct"], color="steelblue", linewidth=1.5)

    # Market-Open und Market-Close hervorheben
    ax.axvspan(0, 30, alpha=0.08, color="orange", label="Open (09:30–10:00)")
    ax.axvspan(360, 390, alpha=0.08, color="darkred", label="Close (15:30–16:00)")

    # Durchschnittslinie
    ax.axhline(y=overall, color="gray", linestyle="--", linewidth=1.2,
               label=f"Durchschnittliche Breakout-Rate: {overall:.1f}%")

    # X-Achse: lesbare Uhrzeiten
    tick_minutes = [0, 60, 120, 180, 240, 300, 360, 390]
    tick_labels = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]
    ax.set_xticks(tick_minutes)
    ax.set_xticklabels(tick_labels)

    ax.set_ylabel("Breakout-Wahrscheinlichkeit (%)")
    ax.set_xlabel("Uhrzeit (ET)")
    ax.set_title(f"Breakout-Rate nach Tageszeit – {symbol} (Trainingsdaten)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # Min- und Max-Werte annotieren
    max_row = by_minute.loc[by_minute["mean_pct"].idxmax()]
    min_row = by_minute.loc[by_minute["mean_pct"].idxmin()]
    max_time = f"{9 + int(max_row.name // 60):02d}:{int(max_row.name % 60):02d}"
    min_time = f"{9 + int(min_row.name // 60):02d}:{int(min_row.name % 60):02d}"
    ax.annotate(f"Max: {max_row['mean_pct']:.1f}% ({max_time})",
                xy=(max_row.name, max_row["mean_pct"]),
                xytext=(max_row.name + 20, max_row["mean_pct"] + 2),
                arrowprops=dict(arrowstyle="->", color="darkred"), fontsize=9, color="darkred")
    ax.annotate(f"Min: {min_row['mean_pct']:.1f}% ({min_time})",
                xy=(min_row.name, min_row["mean_pct"]),
                xytext=(min_row.name + 20, min_row["mean_pct"] - 4),
                arrowprops=dict(arrowstyle="->", color="darkgreen"), fontsize=9, color="darkgreen")

    ensure_artifacts_dir()
    out = ARTIFACTS_DIR / f"{symbol}_breakout_by_time.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"[OK] Plot gespeichert: {out}")


# ─── Plot 2: Breakout-Dauer-Verteilung ─────────────────────────────────────

def plot_breakout_duration(symbol="AAPL"):
    """Wie lange dauern Breakouts? Histogramm der Dauer.

    Breakouts sind ueberlappende forward-looking Labels: Wenn Minute t
    ein Breakout meldet und Minute t+1 ebenfalls, gehoeren beide zum
    selben Breakout-Cluster. Ein Cluster ist eine durchgehende Folge
    von breakout_30m = 1.
    """
    path = PROJECT_ROOT / "data" / "processed" / "pre_split" / f"{symbol}_train.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nicht gefunden. Bitte zuerst scripts/03_pre_split_prep/main.py ausfuehren."
        )

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Cluster identifizieren: Jede aufeinanderfolgende Kette von 1en ist ein Cluster
    df["is_new_cluster"] = (df["breakout_30m"].diff() == 1) | (
        (df["breakout_30m"] == 1) & (df.index == df.index[0])
    )
    df["cluster_id"] = df["is_new_cluster"].cumsum() * (df["breakout_30m"] == 1)
    df["cluster_id"] = df["cluster_id"].replace(0, np.nan)

    # Dauer jedes Clusters
    durations = df.dropna(subset=["cluster_id"]).groupby("cluster_id").size()

    print(f"  Breakout-Cluster gefunden: {len(durations)}")
    print(f"  Dauer – Min: {durations.min()}, Median: {durations.median():.0f}, "
          f"Max: {durations.max()}, Mean: {durations.mean():.1f}")

    # Nur Cluster bis 30 Minuten plotten (der Rest ist sehr selten)
    max_plot = 30
    durations_plot = durations[durations <= max_plot]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)

    counts = durations_plot.value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color="steelblue", alpha=0.8, edgecolor="white")

    # Median markieren
    med = durations.median()
    ax.axvline(x=med, color="darkred", linestyle="--", linewidth=1.5,
               label=f"Median: {med:.0f} Minuten")

    # Werte auf die Balken schreiben (nur bei den haeufigsten)
    top_n = counts.nlargest(3)
    for dur, cnt in top_n.items():
        ax.text(dur, cnt + max(counts.values) * 0.02, str(int(cnt)), ha="center", fontsize=9)

    ax.set_xlabel("Dauer des Breakout-Clusters (Minuten)")
    ax.set_ylabel("Anzahl Cluster")
    ax.set_title(f"Breakout-Dauer-Verteilung – {symbol} (Trainingsdaten)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.set_xlim(0.5, max_plot + 0.5)

    # Inset: Verteilung ueber 30 Minuten (Long-Tail)
    if len(durations[durations > max_plot]) > 0:
        ax_inset = ax.inset_axes([0.55, 0.45, 0.40, 0.50])
        long_dur = durations[durations > max_plot]
        ax_inset.hist(long_dur, bins=20, color="darkorange", alpha=0.7, edgecolor="white")
        ax_inset.set_title(f"> {max_plot} Min ({len(long_dur)} Cluster)", fontsize=8)
        ax_inset.set_xlabel("Minuten", fontsize=7)
        ax_inset.set_ylabel("Anzahl", fontsize=7)
        ax_inset.tick_params(labelsize=7)
        ax_inset.grid(alpha=0.3, axis="y")

    ensure_artifacts_dir()
    out = ARTIFACTS_DIR / f"{symbol}_breakout_duration.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"[OK] Plot gespeichert: {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("BREAKOUT DOMAIN PLOTS")
    print("=" * 60)
    plot_breakout_by_time_of_day("AAPL")
    plot_breakout_duration("AAPL")
    print("\nFertig.")
