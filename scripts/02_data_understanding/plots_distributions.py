import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from base_utils import load_symbol_df, ensure_artifacts_dir, ARTIFACTS_DIR


def plot_return_and_volume_distributions(symbol):
    df = load_symbol_df(symbol)

    # Returns berechnen
    df["return_1m"] = df["close"].pct_change()

    ensure_artifacts_dir()

    # Histogramm Returns (Full Range)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    sns.histplot(df["return_1m"].dropna(), bins=100, kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Verteilung der 1-Minuten-Returns für {symbol}")
    ax.set_xlabel("Return (pct_change)")
    ax.set_ylabel("Häufigkeit")
    out_path = ARTIFACTS_DIR / f"{symbol}_returns_hist.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Plot gespeichert: {out_path}")

    # Histogramm Returns (Zoom -1% bis +1%)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    sns.histplot(df["return_1m"].dropna(), bins=100, kde=True, ax=ax, color="steelblue")
    ax.set_xlim(-0.01, 0.01)
    ax.set_title(f"Verteilung der 1-Minuten-Returns (Zoom, -1% bis +1%) – {symbol}")
    ax.set_xlabel("Return (pct_change)")
    ax.set_ylabel("Häufigkeit")
    out_path = ARTIFACTS_DIR / f"{symbol}_returns_hist_zoom.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Plot gespeichert: {out_path}")

    # Histogramm Volumen (linear)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    sns.histplot(df["volume"].dropna(), bins=100, kde=False, ax=ax, color="darkorange")
    ax.set_title(f"Verteilung des Volumens für {symbol}")
    ax.set_xlabel("Volumen")
    ax.set_ylabel("Häufigkeit")
    out_path = ARTIFACTS_DIR / f"{symbol}_volume_hist.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Plot gespeichert: {out_path}")

    # Histogramm Volumen (log10)
    volume_pos = df["volume"].dropna()
    volume_pos = volume_pos[volume_pos > 0]
    if not volume_pos.empty:
        vol_log = np.log10(volume_pos)

        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        sns.histplot(vol_log, bins=100, kde=False, ax=ax, color="darkorange")
        ax.set_title(f"Verteilung des Volumens (log10) für {symbol}")
        ax.set_xlabel("log10(Volumen)")
        ax.set_ylabel("Häufigkeit")
        out_path = ARTIFACTS_DIR / f"{symbol}_volume_hist_log.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] Plot gespeichert: {out_path}")
    else:
        print(f"[WARN] Kein positives Volumen für {symbol}, Log-Histogramm übersprungen.")


if __name__ == "__main__":
    plot_return_and_volume_distributions("AAPL")