import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import yaml
from model import BreakoutModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
params = yaml.safe_load(open(PROJECT_ROOT / "conf" / "params.yaml"))
processed_path = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]

SYMBOLS   = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
THRESHOLD = 0.5

# Features laden
features = open(processed_path / "features.txt").read().splitlines()

# Validierungsdaten laden
val_df = pd.concat([pd.read_parquet(processed_path / f"{s}_validation.parquet") for s in SYMBOLS])

X_val = torch.tensor(val_df[features].values, dtype=torch.float32)
y_val = val_df["breakout_30m"].values

# Modell laden
model = BreakoutModel(input_size=len(features))
model.load_state_dict(torch.load(PROJECT_ROOT / "artifacts" / "models" / "model.pt"))
model.eval()

# Vorhersagen
with torch.no_grad():
    y_pred_prob = model(X_val).squeeze().numpy()
    y_pred = (y_pred_prob > THRESHOLD).astype(int)

# Ergebnisse
print("=== Confusion Matrix ===")
print(confusion_matrix(y_val, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_val, y_pred, target_names=["Kein Breakout", "Breakout"]))