import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

from base_utils import load_symbol_df, ensure_artifacts_dir, ARTIFACTS_DIR


def plot_intraday_patterns(symbol):
    df = load_symbol_df(symbol)

    # Zeitzone sicherstellen
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    # Uhrzeit extrahieren
    df["time"] = df.index.time

    # Durchschnittliches Volumen pro Uhrzeit
    avg_volume = df.groupby("time")["volume"].mean()

    # Intraday-Volatility: std der 1-Minuten-Returns pro Uhrzeit
    df["return_1m"] = df["close"].pct_change()
    avg_volatility = df.groupby("time")["return_1m"].std()

    ensure_artifacts_dir()

    # Plot Volumen
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    ax.plot(avg_volume.index.astype(str), avg_volume.values, color="steelblue")
    ax.set_title(f"Durchschnittliches Volumen pro Uhrzeit – {symbol}")
    ax.set_xlabel("Uhrzeit (HH:MM)")
    ax.set_ylabel("Durchschnittliches Volumen")

    # X Ticks ausdünnen
    times = avg_volume.index.astype(str).tolist()
    step = max(1, len(times) // 20)
    ax.set_xticks(range(0, len(times), step))
    ax.set_xticklabels(times[::step], rotation=45, ha="right")

    out_path = ARTIFACTS_DIR / f"{symbol}_avg_volume_by_time.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Plot gespeichert: {out_path}")

    # Plot Volatility
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    ax.plot(avg_volatility.index.astype(str), avg_volatility.values, color="darkred")
    ax.set_title(f"Durchschnittliche Intraday-Volatility pro Uhrzeit – {symbol}")
    ax.set_xlabel("Uhrzeit (HH:MM)")
    ax.set_ylabel("Std. 1-Minuten-Returns")

    # X Ticks ausdünnen
    times = avg_volatility.index.astype(str).tolist()
    step = max(1, len(times) // 20)
    ax.set_xticks(range(0, len(times), step))
    ax.set_xticklabels(times[::step], rotation=45, ha="right")

    out_path = ARTIFACTS_DIR / f"{symbol}_avg_volatility_by_time.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Plot gespeichert: {out_path}")


if __name__ == "__main__":
    plot_intraday_patterns("AAPL")