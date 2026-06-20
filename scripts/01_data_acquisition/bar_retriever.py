"""
Step 1: Data Acquisition
----------------------------------
- Liest API-Keys aus conf/keys.yaml
- Liest Konfiguration aus conf/params.yaml
- Lädt NASDAQ-100 Symbole aus data/nasdaq100_symbols.csv
- Für jedes Symbol: holt 1-Minuten-Bars von Alpaca API
- Filtert auf reguläre Handelszeiten mit Alpaca Handelskalender
- Speichert Daten als Parquet-Datei pro Symbol
"""

import os
import yaml
import pandas as pd
from datetime import datetime
import pytz
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ─── Konfiguration laden ───────────────────────────────────────────────────────

with open("conf/keys.yaml", "r") as f:
    keys = yaml.safe_load(f)

with open("conf/params.yaml", "r") as f:
    params = yaml.safe_load(f)

API_KEY = keys["ALPACA"]["API_KEY"]
SECRET_KEY = keys["ALPACA"]["SECRET_KEY"]

DATA_PATH = params["DATA_ACQUISITION"]["DATA_PATH"]
START_DATE = params["DATA_ACQUISITION"]["START_DATE"]
END_DATE = params["DATA_ACQUISITION"]["END_DATE"]
SYMBOLS_PATH = params["DATA_ACQUISITION"]["SYMBOLS_PATH"]

# Zielordner erstellen falls nicht vorhanden
OUTPUT_PATH = os.path.join(DATA_PATH, "Bars_1m_adj")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─── Symbole laden ─────────────────────────────────────────────────────────────

symbols_df = pd.read_csv(SYMBOLS_PATH)
symbols = symbols_df["Symbol"].tolist()

# Für erste Tests nur 5 Symbole verwenden
# Wenn Pipeline funktioniert, diese Zeile auskommentieren
# symbols = symbols[:5]
print(f"Symbole geladen: {symbols}")

# ─── Alpaca Client erstellen + Kalender laden ──────────────────────────────────

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY)

print("Lade Handelskalender...")
cal_request = GetCalendarRequest(
    start=datetime.strptime(START_DATE, "%Y-%m-%d").date(),
    end=datetime.strptime(END_DATE, "%Y-%m-%d").date()
)
calendar = trading_client.get_calendar(cal_request)

# Lookup Tabelle für Öffnungszeiten bauen
eastern = pytz.timezone("US/Eastern")
cal_map = {}
for c in calendar:
    # Setze explizit die Zeitzone auf US/Eastern
    open_dt = eastern.localize(c.open) if not c.open.tzinfo else c.open.astimezone(eastern)
    close_dt = eastern.localize(c.close) if not c.close.tzinfo else c.close.astimezone(eastern)
    cal_map[c.date] = (open_dt, close_dt)


def is_market_open(ts):
    # Wandelt den Zeitstempel in Eastern Time um
    ts_eastern = ts.tz_convert(eastern) if ts.tzinfo else ts.tz_localize("UTC").astimezone(eastern)
    d = ts_eastern.date()
    # Ist der Tag im Kalender vermerkt?
    if d not in cal_map:
        return False
    # Ist die konkrete Minute innerhalb der echten Öffnungszeit des jeweiligen Tages?
    return cal_map[d][0] <= ts_eastern < cal_map[d][1]


# ─── Daten für jedes Symbol herunterladen ─────────────────────────────────────

for symbol in symbols:
    print(f"\nLade Daten für {symbol} ...")

    try:
        # API-Anfrage definieren
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=datetime.strptime(START_DATE, "%Y-%m-%d"),
            end=datetime.strptime(END_DATE, "%Y-%m-%d"),
            adjustment="all",  # Bereinigt um Splits und Dividenden
            feed="iex"  # Kostenloser Feed (IEX Exchange)
        )

        # Daten abrufen
        bars = client.get_stock_bars(request)
        df = bars.df

        # Multi-Index auflösen falls vorhanden (symbol, timestamp) -> timestamp
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)

        # Index explizit als Eastern Time definieren (zur Sicherheit für die Parquet Speicherung)
        df.index = df.index.tz_convert(eastern) if df.index.tzinfo else df.index.tz_localize("UTC").tz_convert(eastern)

        # Nur reguläre Handelszeiten behalten mithilfe der Kalenderfunktion (ersetzt alte Minutengrenzen)
        df["is_open"] = df.index.map(is_market_open)
        df = df[df["is_open"]].drop(columns=["is_open"])

        # Als Parquet speichern
        output_file = os.path.join(OUTPUT_PATH, f"{symbol}.parquet")
        df.to_parquet(output_file)

        print(f"  -> {len(df)} Bars gespeichert: {output_file}")

    except Exception as e:
        print(f"  -> Fehler bei {symbol}: {e}")

print("\nFertig!")