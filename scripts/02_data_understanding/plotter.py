"""
Step 2: Data Understanding
----------------------------------
- Laedt 1-Minuten-Bars fuer ein Symbol aus Parquet
- Plottet die Intraday-Preisreihe (Close-Preis)
- Markiert Tagesgrenzen mit gruenen gestrichelten Linien
- Speichert Plot als PNG in artifacts/images/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─── Konfiguration ─────────────────────────────────────────────────────────────

SYMBOL     = "AAPL"           # Symbol das geplottet werden soll
DATA_PATH  = "data/raw/Bars_1m_adj"
OUTPUT_DIR = "artifacts/images"
N_BARS     = 500              # Anzahl Bars die angezeigt werden

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Daten laden ───────────────────────────────────────────────────────────────

file_path = os.path.join(DATA_PATH, f"{SYMBOL}.parquet")
df = pd.read_parquet(file_path)

print(f"Geladen: {len(df)} Bars fuer {SYMBOL}")
print(df.head())

# Subset der ersten N Bars nehmen
df_plot = df.iloc[:N_BARS].copy()

# ─── Plot erstellen ────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 4), dpi=100)

# Close-Preis plotten
ax.plot(df_plot.index, df_plot["close"], color="steelblue", linewidth=0.8, label="Close")

# Tagesgrenzen als gruene gestrichelte Linien einzeichnen
dates = df_plot.index.normalize().unique()
for date in dates:
    ax.axvline(x=date, color="green", linestyle="--", linewidth=0.8, alpha=0.7)

# Referenzbalken in der Mitte rot markieren
mid_idx = len(df_plot) // 2
ax.axvline(x=df_plot.index[mid_idx], color="red", linestyle="--", linewidth=1.2, label="Referenzbalken")

ax.set_title(f"{SYMBOL} – 1-Minuten-Bars (Close-Preis)", fontsize=13)
ax.set_xlabel("Zeit (ET)")
ax.set_ylabel("Preis (USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()

# Speichern
output_file = os.path.join(OUTPUT_DIR, f"{SYMBOL}_intraday.png")
plt.savefig(output_file)
plt.close()

print(f"\nPlot gespeichert: {output_file}")
