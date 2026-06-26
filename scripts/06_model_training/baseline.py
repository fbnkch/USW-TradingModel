import pandas as pd
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
params = yaml.safe_load(open(PROJECT_ROOT / "conf" / "params.yaml"))
processed_path = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]

SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

for symbol in SYMBOLS:
    df = pd.read_parquet(processed_path / f"{symbol}_train.parquet")

    # Wie oft ist breakout_30m = 1?
    rate = df["breakout_30m"].mean()

    print(f"{symbol}: Breakout-Rate = {rate:.1%} | Baseline Accuracy = {max(rate, 1 - rate):.1%}")