"""
plot_lagged_correlation.py
--------------------------
Berechnet und plottet die Lagged Korrelation ausgewählter Features mit dem
Breakout-Target (breakout_30m) für Lags von 0 bis 30 Minuten.

Frage die beantwortet wird:
  "Wie gut sagt Feature X vor t Minuten voraus, ob in den nächsten 30 Minuten
   ein Breakout passiert?"

Interpretation:
  - Lag 0  = Feature und Target zur selben Minute (wie stark korreliert das
             Feature überhaupt mit dem Target?)
  - Lag 5  = Feature von vor 5 Minuten vs. heutiges Breakout-Label
  - Lag 30 = Feature von vor 30 Minuten vs. heutiges Breakout-Label

  Hohe positive Korrelation bei Lag t -> Feature steigt typischerweise t
  Minuten bevor ein Breakout passiert -> wertvolles Frühwarnsignal.

Ausgabe:
  artifacts/images/lagged_correlation.png

Verwendung:
  python plot_lagged_correlation.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Pfad zur Trainingsdatei
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pre_split" / "AAPL_train.parquet"

# Ausgabepfad für den Plot
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "images" / "lagged_correlation.png"

# Target-Spalte
TARGET_COL = "breakout_30m"

# Features deren Lagged Korrelation wir untersuchen
FEATURES = [
    "close_norm",
    "volume_norm",
    "Slope_close_1",
    "RSI_14",
    "BB_position",
    "vwap_distance",
    "cumulative_delta",
]

# Lags in Minuten (0 = kein Lag, 30 = Feature von vor 30 Minuten)
MAX_LAG = 30
LAGS = list(range(0, MAX_LAG + 1))

# Farben pro Feature (seaborn-Palette, automatisch zugewiesen)
PALETTE = sns.color_palette("tab10", n_colors=len(FEATURES))

# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------

print(f"Lade Trainingsdaten: {DATA_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Parquet nicht gefunden: {DATA_PATH}\n"
        "Bitte zuerst main.py ausführen um die pre-split Daten zu erzeugen."
    )

df = pd.read_parquet(DATA_PATH)
print(f"  -> {len(df):,} Bars geladen | Spalten: {df.shape[1]}")

# Prüfen ob alle benötigten Spalten vorhanden sind
missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
if missing:
    raise KeyError(f"Fehlende Spalten im DataFrame: {missing}")

# ---------------------------------------------------------------------------
# Lagged Korrelation berechnen
# ---------------------------------------------------------------------------
# Für jeden Lag t: verschiebe das Feature um t Zeilen nach unten (shift(t))
# -> Zeile i bekommt den Feature-Wert von Zeile i-t
# Dann berechne die Pearson-Korrelation mit dem Target (Lag 0)
#
# Wichtig: shift() arbeitet über Tagesgrenzen hinweg. In echten Daten mit
# RTH-Filterung ist das unkritisch, da die ersten paar Rows jedes Tages
# sowieso durch z_norm-Warmup und dropna() bereinigt wurden.

print("Berechne Lagged Korrelationen ...")

# Ergebnistabelle: Zeilen = Lags, Spalten = Features
corr_matrix = pd.DataFrame(index=LAGS, columns=FEATURES, dtype=float)

target = df[TARGET_COL]

for feat in FEATURES:
    for lag in LAGS:
        # Feature um lag Minuten nach hinten verschieben
        shifted = df[feat].shift(lag)
        # Pearson-Korrelation (NaNs durch shift werden automatisch ausgelassen)
        corr_matrix.loc[lag, feat] = shifted.corr(target)

print("  -> Fertig.")

# ---------------------------------------------------------------------------
# Plot erstellen
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.1)

fig, ax = plt.subplots(figsize=(13, 7), dpi=120)

# Eine Linie pro Feature
for feat, color in zip(FEATURES, PALETTE):
    corr_values = corr_matrix[feat].astype(float)

    # Linie zeichnen
    ax.plot(
        LAGS,
        corr_values,
        label=feat,
        color=color,
        linewidth=2.0,
        alpha=0.85,
    )

    # Punkt beim Lag mit der stärksten (absoluten) Korrelation markieren
    best_lag = corr_values.abs().idxmax()
    best_val = corr_values[best_lag]
    ax.scatter(
        best_lag,
        best_val,
        color=color,
        s=80,           # Punktgröße
        zorder=5,       # über den Linien zeichnen
        edgecolors="white",
        linewidths=1.2,
    )

# Nulllinie als Referenz
ax.axhline(y=0, color="black", linestyle="--", linewidth=1.2, alpha=0.5, label="Korrelation = 0")

# Achsen beschriften
ax.set_xlabel("Lag (Minuten)", fontsize=13)
ax.set_ylabel(f"Pearson-Korrelation mit `{TARGET_COL}`", fontsize=13)
ax.set_title(
    "Lagged Korrelation: Features vs. Breakout-Target (AAPL, Training)\n"
    "Punkt = stärkster Lag je Feature",
    fontsize=14,
    pad=15,
)

# X-Achse: ganzzahlige Lags, alle 5 Minuten ein Tick
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

# Grid nur auf Hauptticks
ax.grid(True, which="major", alpha=0.35)
ax.grid(True, which="minor", alpha=0.12)

# Legende außerhalb des Plots (rechts), damit Linien nicht verdeckt werden
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    borderaxespad=0,
    fontsize=10,
    framealpha=0.9,
    title="Features",
    title_fontsize=11,
)

plt.tight_layout()

# ---------------------------------------------------------------------------
# Speichern
# ---------------------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUTPUT_PATH), bbox_inches="tight")
print(f"Plot gespeichert: {OUTPUT_PATH}")

plt.show()