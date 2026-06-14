"""

Dieses Script ist KEIN Teil der Pipeline. Wir müssen das hier manuell ausführen, um nach main.py zu prüfen ob die Features plausibel aussehen:
  - Sind EMAs glätter als der Close-Preis?
  - Ist das Slope-Signal sinnvoll (positiv bei Anstieg, negativ bei Fall)?
  - Ist close_norm ungefähr um 0 zentriert?
  - Stimmt das breakout-Label an der markierten Stelle mit dem Chart überein?

"""

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from pathlib import Path


# Features die geplottet werden (müssen in features.txt vorhanden sein)
FEATURES_TO_PLOT = [
    "close_norm",    # Z-normierter Close: sollte um 0 pendeln
    "volume_norm",   # Z-normiertes Volumen: Spikes sichtbar machen
    "EMA_20",        # Mittelfristiger Trend
    "EMA_60",        # Laengerfristiger Trend
    "Slope_close_1", # Momentane Richtung des Close
]


def plot_features(
    df,
    index,
    window_before=300,
    window_after=300,
    symbol=None,
    target_col="breakout_30m",
):
    """Plottet engineerte Features um einen gewählten Row-Index.

    Parameters
    -
    df : pd.DataFrame
        DataFrame mit 'timestamp' und Feature-Spalten.
    index : int
        Positionale Zeile, um die das Fenster zentriert wird.
    window_before : int
        Bars vor index die angezeigt werden.
    window_after : int
        Bars nach index die angezeigt werden.
    symbol : str, optional
        Symbol-String für Titel und Labels.
    target_col : str
        Name der Target-Spalte; Wert wird im Legendentitel angezeigt.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    start  = max(0, index - window_before)
    end    = min(len(df) - 1, index + window_after)
    subset = df.iloc[start : end + 1].copy().reset_index(drop=True)

    present = [c for c in FEATURES_TO_PLOT if c in subset.columns]
    if not present:
        raise KeyError(
            f"Keine der erwarteten Features gefunden.\n"
            f"Erwartet: {FEATURES_TO_PLOT}\n"
            f"Vorhanden: {subset.columns.tolist()}"
        )

    target_val = df.loc[index, target_col] if target_col in df.columns else "N/A"
    if target_val != "N/A":
        target_val = int(target_val)

    fig, ax = plt.subplots(figsize=(18, 6), dpi=120)

    for feat in present:
        label = f"{feat} ({symbol})" if symbol else feat
        ax.plot(subset.index, subset[feat], label=label, linewidth=1.2)

    # Ziel-Row rot markieren
    target_idx_in_subset = index - start
    ax.axvline(
        x=target_idx_in_subset,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Row {index} | {target_col} = {target_val}",
    )

    # Tageswechsel grün markieren
    day_changes = (
        subset["timestamp"].dt.date
        .ne(subset["timestamp"].dt.date.shift())
        .fillna(False)
    )
    first_day_line = True
    for i, is_new_day in enumerate(day_changes):
        if is_new_day and i != 0:
            ax.axvline(
                x=i,
                color="green",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label="Tageswechsel" if first_day_line else None,
            )
            first_day_line = False

    # X-Achse mit lesbaren Timestamps
    tick_positions = subset.index[:: max(1, len(subset) // 10)]
    tick_labels    = subset.loc[tick_positions, "timestamp"].dt.strftime("%m-%d %H:%M")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_xlabel("Bar-Index")
    ax.set_ylabel("Feature-Wert (z-normiert)")

    ts_str = df.loc[index, "timestamp"].strftime("%Y-%m-%d %H:%M")
    sym_prefix = f"{symbol} - " if symbol else ""
    ax.set_title(
        f"{sym_prefix}Engineerte Features | "
        f"+-{window_before} Bars um Row {index} ({ts_str})"
    )

    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


# 
# Direktausführung
# 
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    params         = yaml.safe_load(open(PROJECT_ROOT / "conf" / "params.yaml"))
    processed_path = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]
    symbols_csv    = params["SYMBOLS_PATH"]

    SYMBOL = "AAPL"  # Hier Symbol eintragen

    features_file = processed_path / f"{SYMBOL}_train.parquet"
    if not features_file.exists():
        raise FileNotFoundError(
            f"Parquet nicht gefunden: {features_file}\n"
            "Bitte zuerst main.py ausführen."
        )

    df = pd.read_parquet(str(features_file))

    # Row 2500 in der Trainingspartition als Beispiel plotten
    plot_features(df, index=2500, window_before=300, window_after=300, symbol=SYMBOL)