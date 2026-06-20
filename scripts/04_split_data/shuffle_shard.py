"""
Shuffelt und shardet die pro-Symbol Train/Val/Test Parquet-Dateien.

Ablauf pro Split (train/validation/test):
  1) Lade alle {symbol}_{split}.parquet aus dem Pre-Split-Verzeichnis
  2) Shuffle jede Datei einzeln (pandas.sample)
  3) Verteile Zeilen per Round-Robin auf n_shards Shards
  4) Finaler Shuffle jedes Shards (mischt Symbole)
  5) Schreibe {split}_shard_{k}.parquet

Jeder Shard enthaelt einen repräsentativen Querschnitt aller Symbole und
Zeitpunkte. Zeitliche Ordnung wurde bereits in den Features explizit codiert
(Rolling-Fenster, Slopes, Lags) – Shuffeln zerstoert keine Information.

Fuer schnelle Testlaeufe kann SUBSET_SYMBOLS in params.yaml gesetzt werden.
Bei null werden alle Symbole verarbeitet.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "conf" / "params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

PROCESSED_PATH = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]
N_SHARDS       = params["SPLIT_DATA"]["N_SHARDS"]
SHUFFLED_PATH  = PROJECT_ROOT / params["SPLIT_DATA"]["SHUFFLED_PATH"]
SUBSET         = params.get("SUBSET_SYMBOLS", None)

SEED = 42


def _find_split_files(split: str) -> list[Path]:
    """Findet alle pro-Symbol Parquet-Dateien fuer einen Split."""
    files = sorted(PROCESSED_PATH.glob(f"*_{split}.parquet"))
    if SUBSET is not None:
        files = files[:SUBSET]
    return files


def shuffle_and_shard(split: str) -> None:
    """Shuffelt alle per-Symbol Dateien eines Splits und verteilt sie
    gleichmaessig auf N_SHARDS globale Shard-Dateien.

    Parameters
    ----------
    split : str
        Einer von {'train', 'validation', 'test'}.
    """
    files = _find_split_files(split)
    if not files:
        print(f"[WARN] Keine Dateien fuer Split '{split}' gefunden.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Split: {split}  ({len(files)} Symbole, {N_SHARDS} Shards)")
    print(f"{'=' * 60}")

    # Akkumuliere Zeilen pro Shard (Liste von DataFrames)
    shard_parts: dict[int, list[pd.DataFrame]] = {k: [] for k in range(N_SHARDS)}
    total_rows = 0

    for i, fpath in enumerate(files):
        df = pd.read_parquet(fpath)
        symbol = os.path.basename(fpath).split("_")[0]
        n_rows = len(df)
        total_rows += n_rows

        # In-File-Shuffle: vollstaendige Permutation
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

        # Round-Robin: gleichmaessig auf Shards verteilen
        shard_ids = np.arange(n_rows) % N_SHARDS
        for k in range(N_SHARDS):
            mask = shard_ids == k
            if mask.any():
                chunk = df.loc[mask].copy()
                shard_parts[k].append(chunk)

        if (i + 1) % 20 == 0 or i == len(files) - 1:
            avg = total_rows / (i + 1)
            print(f"  [{i + 1:>3}/{len(files)}] {symbol:<5}  "
                  f"({n_rows:>8,} Zeilen)  |  "
                  f"~{total_rows / 1e6:.1f}M total  ({avg / 1e3:.0f}k/Symbol)")

    # Shards zusammenbauen und final shufflen
    os.makedirs(SHUFFLED_PATH, exist_ok=True)

    for k in range(N_SHARDS):
        if not shard_parts[k]:
            continue
        shard_df = pd.concat(shard_parts[k], ignore_index=True)
        # Finaler Shuffle ueber alle Symbole hinweg (pro Shard eigener Seed)
        shard_df = shard_df.sample(frac=1, random_state=SEED + k).reset_index(drop=True)

        out_file = SHUFFLED_PATH / f"{split}_shard_{k}.parquet"
        shard_df.to_parquet(out_file, index=False)

        if k < 5 or k == N_SHARDS - 1:
            print(f"  [OK] {out_file.name}  ({len(shard_df):>8,} Zeilen)")

    mean_per_shard = total_rows / N_SHARDS
    print(f"  => Gesamt: {total_rows:,} Zeilen (~{mean_per_shard:,.0f}/Shard)\n")


if __name__ == "__main__":
    for split_name in ["train", "validation", "test"]:
        shuffle_and_shard(split_name)

    print("Fertig. Shards gespeichert in:", SHUFFLED_PATH)
