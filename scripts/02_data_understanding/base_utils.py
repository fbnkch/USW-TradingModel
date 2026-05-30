import os
from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Bars_1m_adj"
SYMBOLS_CSV = PROJECT_ROOT / "data" / "nasdaq100_symbols.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "images"


def ensure_artifacts_dir():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_symbols(n=5, explicit_list=None):
    if explicit_list:
        return explicit_list
    symbols_df = pd.read_csv(SYMBOLS_CSV)
    symbols = symbols_df["Symbol"].tolist()
    return symbols[:n]


def load_symbol_df(symbol):
    file_path = DATA_DIR / f"{symbol}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet-Datei nicht gefunden: {file_path}")
    df = pd.read_parquet(file_path)
    return df


def basic_checks(symbols):
    print("=" * 70)
    print("BASIS-CHECKS (Zeilenanzahl, Zeitspanne, NaNs, Zeitzone)")
    print("=" * 70)

    for symbol in symbols:
        df = load_symbol_df(symbol)

        row_count = len(df)
        ts_min = df.index.min()
        ts_max = df.index.max()

        close_na = df["close"].isna().sum() if "close" in df.columns else "N/A"
        volume_na = df["volume"].isna().sum() if "volume" in df.columns else "N/A"

        tzinfo = df.index.tz

        print(f"\nSymbol: {symbol}")
        print(f"  Zeilenanzahl: {row_count}")
        print(f"  Zeitraum: {ts_min} bis {ts_max}")
        print(f"  Missing close: {close_na}")
        print(f"  Missing volume: {volume_na}")
        print(f"  Index-Zeitzone: {tzinfo}")


def data_quality_checks(symbols, expected_minutes_per_day=390):
    print("=" * 70)
    print("DATA-QUALITY-CHECKS (Duplikate, Luecken, RTH-Minuten)")
    print("=" * 70)

    for symbol in symbols:
        df = load_symbol_df(symbol)

        # 1) Duplikate im Index
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            print(f"[WARN] {symbol}: {dup_count} doppelte Zeitstempel gefunden.")
        else:
            print(f"[OK]   {symbol}: Keine Duplikate im Index.")

        # 2) Lücken innerhalb von Handelstagen
        #    Gruppieren nach Datum (lokale Zeit im Index)
        if df.index.tz is None:
            dates = df.index.date
        else:
            dates = df.index.tz_convert("America/New_York").date

        counts_per_day = pd.Series(dates).value_counts().sort_index()
        missing_days = counts_per_day[counts_per_day < expected_minutes_per_day]

        if len(missing_days) > 0:
            print(f"[WARN] {symbol}: Tage mit weniger als {expected_minutes_per_day} Bars gefunden:")
            print(missing_days.head(5))
        else:
            print(f"[OK]   {symbol}: Keine offensichtlichen Lücken pro Tag.")