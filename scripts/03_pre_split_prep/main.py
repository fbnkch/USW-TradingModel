"""

  1. Lade 1-Minuten-Bars (adjusted) aus Parquet
  2. Berechne forward-looking Breakout-Target (breakout_30m)
  3. Berechne normierte technische Features
  4. Droppe NaN-Zeilen (entstehen durch Rolling/Slope/Target-Tail)
  5. Splitte zeitbasiert in Train / Validation / Test
  6. Speichere je Split als Parquet

Reihenfolge ist zwingend

  Targets und Features sind forward-looking: Sie brauchen Zukunftsdaten.
  Wenn wir erst splitten, fliessen Val/Test-Informationen in die
  Train-Daten ein (Data Leakage). Deshalb:
    Laden -> Targets -> Features -> dropna -> Split -> Speichern

Artefakte
  data/processed/pre_split/features.txt
  data/processed/pre_split/{SYMBOL}_train.parquet
  data/processed/pre_split/{SYMBOL}_validation.parquet
  data/processed/pre_split/{SYMBOL}_test.parquet

Verwendung
  Aus dem Projektroot ausführen:
    python scripts/03_pre_split_prep/main.py

  Für Chunked-Runs (z.B. bei Unterbrechung) den Offset-Counter anpassen:
    counter = 42  # bei Symbol 42 weitermachen
"""

import os

import pandas as pd
import yaml

import targets
import features
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 
# Konfiguration laden
# 
params     = yaml.safe_load(open(PROJECT_ROOT / "conf" / "params.yaml"))
symbols_df = pd.read_csv(PROJECT_ROOT / params["SYMBOLS_PATH"])

#  Data-Prep-Parameter 
horizon       = params["DATA_PREP"]["HORIZON"]
theta         = params["DATA_PREP"]["THETA"]
ema_periods   = params["DATA_PREP"]["EMA_PERIODS"]
slope_periods = params["DATA_PREP"]["SLOPE_PERIODS"]
z_norm_window = params["DATA_PREP"]["Z_NORM_WINDOW"]

#  Pfade 
data_path      = PROJECT_ROOT / params["DATA_ACQUISITION"]["DATA_PATH"]
processed_path = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]
os.makedirs(processed_path, exist_ok=True)

#  Datumsgrenzen für den Split
train_date      = params["DATA_PREP"]["TRAIN_DATE"]
validation_date = params["DATA_PREP"]["VALIDATION_DATE"]
test_date       = params["DATA_PREP"]["TEST_DATE"]


# 
# Offset für Chunked-Runs
# Bei Unterbrechung hier auf den letzten verarbeiteten Index setzen.
# 
counter = 0


# 
# Hauptschleife: ein Symbol nach dem anderen
# 
for symbol in symbols_df["Symbol"][counter:]:
    print(f"{counter}. Verarbeite {symbol} ...")
    counter += 1

    # 
    # 1) Rohdaten laden
    # 
    bars_file = os.path.join(data_path, f"{symbol}.parquet")
    if not os.path.exists(bars_file):
        print(f"   [WARN] Parquet nicht gefunden: {bars_file} - uebersprungen.")
        continue

    df = pd.read_parquet(bars_file)
    df["symbol"] = symbol

    # Timestamp-Spalte sicherstellen.
    # Manche Parquet-Dateien haben Timestamp als Index, andere als Spalte.
    if "timestamp" not in df.columns:
        df = df.reset_index()
        for candidate in ["index", "datetime", "time"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "timestamp"})
                break
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 
    # 2) Forward-looking Breakout-Target berechnen
    # 
    df = targets.add_breakout_target(df, horizon=horizon, theta=theta)

    # 
    # 3) Feature Engineering
    # 
    df, features_added = features.generate_features(
        df,
        ema_periods=ema_periods,
        slope_periods=slope_periods,
        z_norm_window=z_norm_window,
    )

    # 
    # 4) NaN-Zeilen droppen
    # Entstehen durch Rolling-Fenster, Slope-Shift und Target-Tail.
    # Erst HIER droppen (nicht in targets.py / features.py), damit
    # beide Module denselben Index sehen
    # 
    df = df.dropna().reset_index(drop=True)

    # 
    # 5) Feature-Liste einmalig als Artefakt speichern
    # Wird später vom Training-Loader genutzt, um genau die richtigen
    # Spalten als Features zu laden.
    # 
    features_file = os.path.join(processed_path, "features.txt")
    if not os.path.exists(features_file):
        with open(features_file, "w") as f:
            for feat in features_added:
                f.write(f"{feat}\n")
        print(f"   [OK]   features.txt geschrieben ({len(features_added)} Features).")

    # 
    # 6) Zeitbasierter Split (nur über Timestamp-Grenzen).
    # 
    tz = df["timestamp"].dt.tz
    train_date_tz = pd.Timestamp(train_date, tz=tz)
    validation_date_tz = pd.Timestamp(validation_date, tz=tz)
    test_date_tz = pd.Timestamp(test_date, tz=tz)

    train = df[df["timestamp"] <= train_date_tz]
    validation = df[
        (df["timestamp"] > train_date_tz) & (df["timestamp"] <= validation_date_tz)
        ]
    test = df[
        (df["timestamp"] > validation_date_tz) & (df["timestamp"] <= test_date_tz)
        ]
    # 
    # 7) Splits als Parquet speichern
    # 
    train.to_parquet(
        os.path.join(processed_path, f"{symbol}_train.parquet"), index=False
    )
    validation.to_parquet(
        os.path.join(processed_path, f"{symbol}_validation.parquet"), index=False
    )
    test.to_parquet(
        os.path.join(processed_path, f"{symbol}_test.parquet"), index=False
    )

    print(
        f"   [OK]   Train: {len(train):>7,} | "
        f"Val: {len(validation):>6,} | "
        f"Test: {len(test):>6,} Bars"
    )

print("\nFertig! Alle Symbole verarbeitet.")
