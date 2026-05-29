"""
Step 1: Data Acquisition
----------------------------------
- Liest API-Keys aus conf/keys.yaml
- Liest Konfiguration aus conf/params.yaml
- Laedt NASDAQ-100 Symbole aus data/nasdaq100_symbols.csv
- Fuer jedes Symbol: holt 1-Minuten-Bars von Alpaca API
- Filtert auf regulaere Handelszeiten (9:30 - 16:00 ET)
- Speichert Daten als Parquet-Datei pro Symbol
"""

import os
import yaml
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ─── Konfiguration laden ───────────────────────────────────────────────────────

with open("conf/keys.yaml", "r") as f:
    keys = yaml.safe_load(f)

with open("conf/params.yaml", "r") as f:
    params = yaml.safe_load(f)

API_KEY    = keys["ALPACA"]["API_KEY"]
SECRET_KEY = keys["ALPACA"]["SECRET_KEY"]

DATA_PATH    = params["DATA_ACQUISITION"]["DATA_PATH"]
START_DATE   = params["DATA_ACQUISITION"]["START_DATE"]
END_DATE     = params["DATA_ACQUISITION"]["END_DATE"]
SYMBOLS_PATH = params["DATA_ACQUISITION"]["SYMBOLS_PATH"]

# Zielordner erstellen falls nicht vorhanden
OUTPUT_PATH = os.path.join(DATA_PATH, "Bars_1m_adj")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─── Symbole laden ─────────────────────────────────────────────────────────────

symbols_df = pd.read_csv(SYMBOLS_PATH)
symbols = symbols_df["Symbol"].tolist()

# Fuer erste Tests nur 5 Symbole verwenden
# Wenn Pipeline funktioniert, diese Zeile auskommentieren
symbols = symbols[:5]
print(f"Symbole geladen: {symbols}")

# ─── Alpaca Client erstellen ───────────────────────────────────────────────────

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Handelszeiten (Eastern Time)
ET = ZoneInfo("America/New_York")
MARKET_OPEN  = 9 * 60 + 30   # 9:30 in Minuten
MARKET_CLOSE = 16 * 60        # 16:00 in Minuten

# ─── Daten fuer jedes Symbol herunterladen ─────────────────────────────────────

for symbol in symbols:
    print(f"\nLade Daten fuer {symbol} ...")

    try:
        # API-Anfrage definieren
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=datetime.strptime(START_DATE, "%Y-%m-%d"),
            end=datetime.strptime(END_DATE, "%Y-%m-%d"),
            adjustment="all",   # Bereinigt um Splits und Dividenden
            feed="iex"          # Kostenloser Feed (IEX Exchange)
        )

        # Daten abrufen
        bars = client.get_stock_bars(request)
        df = bars.df

        # Multi-Index aufloesen falls vorhanden (symbol, timestamp) -> timestamp
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)

        # Timestamp in Eastern Time konvertieren
        df.index = df.index.tz_convert(ET)

        # Nur regulaere Handelszeiten behalten (9:30 - 16:00 ET)
        minutes = df.index.hour * 60 + df.index.minute
        df = df[(minutes >= MARKET_OPEN) & (minutes < MARKET_CLOSE)]

        # Als Parquet speichern
        output_file = os.path.join(OUTPUT_PATH, f"{symbol}.parquet")
        df.to_parquet(output_file)

        print(f"  -> {len(df)} Bars gespeichert: {output_file}")

    except Exception as e:
        print(f"  -> Fehler bei {symbol}: {e}")

print("\nFertig!")
