import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import yaml
import pickle
from model import BreakoutModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
params = yaml.safe_load(open(PROJECT_ROOT / "conf" / "params.yaml"))
processed_path = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]

SYMBOLS   = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
EPOCHS    = 10
THRESHOLD = 0.5  # ab wann gilt Vorhersage als Breakout

# Features laden
features = open(processed_path / "features.txt").read().splitlines()

# Daten laden und zusammenführen
train_df = pd.concat([pd.read_parquet(processed_path / f"{s}_train.parquet") for s in SYMBOLS])
val_df   = pd.concat([pd.read_parquet(processed_path / f"{s}_validation.parquet") for s in SYMBOLS])

X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
y_train = torch.tensor(train_df["breakout_30m"].values, dtype=torch.float32)
X_val   = torch.tensor(val_df[features].values, dtype=torch.float32)
y_val   = torch.tensor(val_df["breakout_30m"].values, dtype=torch.float32)

# DataLoader (teilt Daten in kleine Pakete auf)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)

# Modell, Loss, Optimizer
model     = BreakoutModel(input_size=len(features))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(EPOCHS):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        pred = model(X_batch).squeeze()
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

    # Nach jeder Epoche auf Validation checken
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).squeeze()
        val_loss = criterion(val_pred, y_val).item()
        val_acc  = ((val_pred > THRESHOLD) == y_val.bool()).float().mean().item()

    print(f"Epoche {epoch+1}/{EPOCHS} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.1%}")

# Modell speichern
torch.save(model.state_dict(), PROJECT_ROOT / "artifacts" / "models" / "model.pt")
print("\nModell gespeichert.")