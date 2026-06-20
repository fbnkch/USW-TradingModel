"""
Berechnet die Klassenverteilung des break_30m Labels im Trainings-Set.

Ergebnis (class_balance.json):
  total_samples      : Gesamtzahl Trainings-Bars
  positive_samples   : Anzahl Breakout-Events (Label=1)
  negative_samples   : Anzahl Nicht-Breakouts (Label=0)
  positive_ratio     : Anteil Breakouts
  scale_pos_weight   : neg/pos  – fuer LightGBM scale_pos_weight
  class_weight_0     : Gewicht  – fuer sklearn class_weight='balanced'
  class_weight_1     : Gewicht  – fuer sklearn class_weight='balanced'

Wird ausschliesslich auf Trainings-Shards berechnet (kein Val/Test-Leakage).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "conf" / "params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

SHUFFLED_PATH = PROJECT_ROOT / params["SPLIT_DATA"]["SHUFFLED_PATH"]
BALANCE_PATH  = PROJECT_ROOT / params["POST_SPLIT"]["BALANCE_PATH"]
TARGET_COL    = f"breakout_{params['DATA_PREP']['HORIZON']}m"


def _shard_number(fpath: Path) -> int:
    """Extrahiert die Shard-Nummer aus dem Dateinamen (z.B. 42 aus train_shard_42.parquet)."""
    return int(fpath.stem.split("_")[-1])


def compute_class_balance() -> None:
    """Iteriert ueber alle train_shard Parquets und ermittelt
    die Klassenverteilung des Breakout-Targets.
    """
    shard_files = sorted(SHUFFLED_PATH.glob("train_shard_*.parquet"),
                          key=_shard_number)

    if not shard_files:
        raise FileNotFoundError(
            f"Keine train_shard_*.parquet in {SHUFFLED_PATH} gefunden. "
            f"Bitte zuerst shuffle_shard.py ausfuehren."
        )

    pos_total = 0
    neg_total = 0

    for fpath in shard_files:
        df = pd.read_parquet(fpath, columns=[TARGET_COL])
        counts = df[TARGET_COL].value_counts()
        pos_total += int(counts.get(1, 0))
        neg_total += int(counts.get(0, 0))

    total = pos_total + neg_total

    if pos_total == 0:
        print("[WARN] Keine positiven Breakout-Labels im Trainings-Set gefunden.")
        scale_pos_weight = 1.0
        cw0, cw1 = 1.0, 1.0
    else:
        scale_pos_weight = neg_total / pos_total
        cw0 = total / (2.0 * neg_total) if neg_total > 0 else 1.0
        cw1 = total / (2.0 * pos_total)

    pos_ratio = pos_total / total if total > 0 else 0.0

    stats = {
        "total_samples": int(total),
        "positive_samples": int(pos_total),
        "negative_samples": int(neg_total),
        "positive_ratio": round(float(pos_ratio), 6),
        "scale_pos_weight": round(float(scale_pos_weight), 4),
        "class_weight_0": round(float(cw0), 4),
        "class_weight_1": round(float(cw1), 4),
    }

    BALANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BALANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Klassenverteilung Trainings-Set:")
    print(f"  Gesamt                    {total:>10,}")
    print(f"  Breakout (1)              {pos_total:>10,}  ({pos_ratio:.2%})")
    print(f"  Kein Breakout (0)         {neg_total:>10,}  ({1 - pos_ratio:.2%})")
    print(f"  scale_pos_weight          {scale_pos_weight:>10.4f}")
    print(f"  class_weight (0 / 1)      {cw0:>10.4f} / {cw1:.4f}")


if __name__ == "__main__":
    compute_class_balance()
