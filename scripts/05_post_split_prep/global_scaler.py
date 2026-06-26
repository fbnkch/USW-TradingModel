"""
Fittet einen globalen StandardScaler auf allen Trainings-Shards.

Der Scaler wird als zweite Normalisierungsstufe NACH der Rolling-Z-Norm
angewendet. Waehrend die Rolling-Z-Norm pro Symbol arbeitet, gleicht der
GlobalScaler die Verteilungen symboluebergreifend an.

Technik:
  sklearn.preprocessing.StandardScaler mit partial_fit().
  Verarbeitet Shard fuer Shard – konstanter Speicherverbrauch.
  KEIN Rewrite der Shards – Transformation erfolgt spaeter im DataLoader.

Output:
  data/processed/global_scaler.pkl (joblib)
"""

from __future__ import annotations

import pandas as pd
import yaml
from pathlib import Path

from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "conf" / "params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

SHUFFLED_PATH  = PROJECT_ROOT / params["SPLIT_DATA"]["SHUFFLED_PATH"]
SCALER_PATH    = PROJECT_ROOT / params["POST_SPLIT"]["SCALER_PATH"]
FEATURES_PATH  = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"] / "features.txt"


def _shard_number(fpath: Path) -> int:
    """Extrahiert die Shard-Nummer aus dem Dateinamen."""
    return int(fpath.stem.split("_")[-1])


def _load_feature_names() -> list[str]:
    """Liest die Feature-Namen aus features.txt."""
    with open(FEATURES_PATH, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def fit_global_scaler() -> None:
    """Fittet einen StandardScaler inkrementell auf allen Trainings-Shards
    und speichert ihn als .pkl fuer spaetere Transformation im DataLoader.
    """
    feature_names = _load_feature_names()
    shard_files = sorted(SHUFFLED_PATH.glob("train_shard_*.parquet"),
                          key=_shard_number)

    if not shard_files:
        raise FileNotFoundError(
            f"Keine train_shard_*.parquet in {SHUFFLED_PATH} gefunden. "
            f"Bitte zuerst shuffle_shard.py ausfuehren."
        )

    scaler = StandardScaler()
    total_rows = 0

    print(f"Fitte GlobalScaler ueber {len(shard_files)} Trainings-Shards "
          f"({len(feature_names)} Features) ...")

    for i, fpath in enumerate(shard_files):
        cols_to_read = [c for c in feature_names if c not in ("symbol",)]
        df = pd.read_parquet(fpath, columns=cols_to_read)

        # Sicherstellen, dass alle Feature-Spalten vorhanden sind
        missing = [c for c in cols_to_read if c not in df.columns]
        if missing:
            print(f"  [WARN] {fpath.name}: fehlende Spalten: {missing}")
            df = df.reindex(columns=cols_to_read, fill_value=0.0)

        X = df[cols_to_read].to_numpy(dtype="float64")
        scaler.partial_fit(X)
        total_rows += len(df)

        if (i + 1) % 20 == 0 or i == len(shard_files) - 1:
            print(f"  [{i + 1:>3}/{len(shard_files)}] {fpath.name}  "
                  f"({len(df):>8,} Zeilen)")

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nGlobalScaler gespeichert: {SCALER_PATH}")
    print(f"  Trainings-Samples gesamt: {total_rows:,}")
    print(f"  Feature-Dimension:        {len(feature_names)}")


if __name__ == "__main__":
    fit_global_scaler()
