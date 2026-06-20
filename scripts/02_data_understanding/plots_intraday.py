import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import timedelta

from base_utils import load_symbol_df, ensure_artifacts_dir, ARTIFACTS_DIR


def plot_intraday_close(symbol, start_date=None, days=3):
    df = load_symbol_df(symbol)

    # Falls Startdatum nicht gesetzt, nimm das erste Datum im Datensatz
    if start_date is None:
        start_date = df.index.min().date()

    # Zeitraum definieren
    tz = df.index.tz  # z.B. US/Eastern
    start_ts = pd.Timestamp(start_date).tz_localize(tz)
    end_ts = start_ts + timedelta(days=days)

    df_plot = df.loc[(df.index >= start_ts) & (df.index < end_ts)].copy()

    if df_plot.empty:
        print(f"[WARN] Kein Plot möglich für {symbol}: Zeitraum leer.")
        return

    fig, ax = plt.subplots(figsize=(14, 4), dpi=120)

    ax.plot(df_plot.index, df_plot["close"], color="steelblue", linewidth=0.8, label="Close")

    # Tagesgrenzen markieren
    dates = df_plot.index.normalize().unique()
    for date in dates:
        ax.axvline(x=date, color="green", linestyle="--", linewidth=0.8, alpha=0.7)

    # Referenzbalken in der Mitte
    mid_idx = len(df_plot) // 2
    ax.axvline(x=df_plot.index[mid_idx], color="red", linestyle="--", linewidth=1.2, label="Referenz")

    ax.set_title(f"{symbol} – 1-Minuten-Close (Intraday)")
    ax.set_xlabel("Zeit (Datum + Uhrzeit)")
    ax.set_ylabel("Preis in USD")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=45)
    ax.legend()

    ensure_artifacts_dir()
    out_path = ARTIFACTS_DIR / f"{symbol}_intraday_close_sample.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[OK] Plot gespeichert: {out_path}")


if __name__ == "__main__":
    plot_intraday_close("AAPL", days=3)
    plot_intraday_close("MSFT", days=3)