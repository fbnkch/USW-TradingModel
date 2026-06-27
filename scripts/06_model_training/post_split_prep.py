# DEPRECATED: use 05_post_split_prep/global_scaler.py instead.
# This script fits a scaler on only 5 symbols (hardcoded).
# 05_post_split_prep/global_scaler.py uses partial_fit on ALL 100 training shards
# and is the canonical scaler for the pipeline.
#
# Kept for reference / backward compatibility during development.

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
params = yaml.safe_load(open(PROJECT_ROOT / "conf" / "params.yaml"))
processed_path = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]

SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

# Features laden (die Liste die main.py gespeichert hat)
features = open(processed_path / "features.txt").read().splitlines()

# Mittelwert und Standardabweichung NUR aus Trainingsdaten berechnen
all_train = pd.concat([
    pd.read_parquet(processed_path / f"{symbol}_train.parquet")
    for symbol in SYMBOLS
])

mean = all_train[features].mean()
std  = all_train[features].std().replace(0, 1)  # Division durch 0 verhindern

# Scaler speichern (wird später beim Deployment gebraucht)
scaler = {"mean": mean, "std": std}
pickle.dump(scaler, open(PROJECT_ROOT / "artifacts" / "scaler.pkl", "wb"))

print(f"Scaler gespeichert | {len(features)} Features skaliert")