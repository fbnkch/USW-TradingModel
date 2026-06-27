"""
Gemeinsame Hilfsfunktionen für alle Trainings- und Evaluierungsskripte.

Bietet:
  - GPU-Setup (CUDA/CPU Auto-Detect, Mixed Precision)
  - Scaler-Laden (global_scaler.pkl)
  - Feature-Listen laden
  - Daten-Loading (Shards oder Per-Symbol)
  - Sequenzbau für LSTM/GRU/CNN
  - Early Stopping & Checkpointing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from glob import glob
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
import joblib


# ─── Pfade ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "conf" / "params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

PROCESSED_PATH = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]
SHUFFLED_PATH  = PROJECT_ROOT / params["SPLIT_DATA"]["SHUFFLED_PATH"]
SCALER_PATH    = PROJECT_ROOT / params["POST_SPLIT"]["SCALER_PATH"]
BALANCE_PATH   = PROJECT_ROOT / params["POST_SPLIT"]["BALANCE_PATH"]
N_SHARDS       = params["SPLIT_DATA"]["N_SHARDS"]


# ─── GPU-Setup ────────────────────────────────────────────────────
def get_device(verbose: bool = True) -> torch.device:
    """Erkennt GPU (CUDA) oder fällt auf CPU zurück.

    Returns
        torch.device: "cuda" wenn GPU verfügbar, sonst "cpu"
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            props = torch.cuda.get_device_properties(0)
            print(f"[GPU] {torch.cuda.get_device_name(0)} "
                  f"({props.total_memory / 1e9:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[CPU] Keine GPU gefunden – Training läuft auf CPU")
    return device


def enable_amp(device: torch.device) -> bool:
    """Aktiviert Automatic Mixed Precision wenn GPU verfügbar."""
    return device.type == "cuda"


# ─── Features & Scaler ────────────────────────────────────────────
def load_features() -> list[str]:
    """Liest die Feature-Namen aus features.txt."""
    feat_path = PROCESSED_PATH / "features.txt"
    if not feat_path.exists():
        raise FileNotFoundError(f"features.txt nicht gefunden: {feat_path}")
    with open(feat_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_scaler():
    """Lädt den globalen StandardScaler (gefi tted auf allen Train-Shards)."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"GlobalScaler nicht gefunden: {SCALER_PATH}\n"
            f"Bitte zuerst: python scripts/05_post_split_prep/global_scaler.py"
        )
    return joblib.load(SCALER_PATH)


def load_class_balance() -> dict:
    """Lädt die Klassenverteilung aus class_balance.json."""
    with open(BALANCE_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─── Daten-Loading (Shards) ───────────────────────────────────────
def load_shards(split: str, features: list[str]) -> pd.DataFrame:
    """Lädt alle Shards eines Splits und gibt einen einzigen DataFrame zurück.

    Parameters
        split : "train", "validation", oder "test"
        features : Liste der Feature-Namen

    Returns
        pd.DataFrame mit allen Features + breakout_30m-Spalte
    """
    pattern = f"{split}_shard_*.parquet"
    files = sorted(SHUFFLED_PATH.glob(pattern),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(
            f"Keine {split}-Shards in {SHUFFLED_PATH} gefunden.\n"
            f"Bitte zuerst: python scripts/04_split_data/shuffle_shard.py"
        )

    print(f"Lade {len(files)} {split}-Shards...")
    dfs = []
    for fpath in files:
        cols = features + ["breakout_30m"]
        available = [c for c in cols if c != "breakout_30m"]  # features only
        # Prüfe ob target existiert
        df = pd.read_parquet(fpath)
        missing_features = [c for c in features if c not in df.columns]
        if missing_features:
            print(f"  [WARN] {fpath.name}: fehlende Features: {missing_features[:5]}...")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def prepare_tensors(
    df: pd.DataFrame,
    features: list[str],
    scaler=None,
    device: torch.device = None,
    target_col: str = "breakout_30m",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Konvertiert DataFrame zu PyTorch-Tensoren, optional mit Scaler-Transformation.

    Parameters
        df : DataFrame mit Features und Target
        features : Feature-Namen
        scaler : Optional, sklearn StandardScaler
        device : Optional, Ziel-Gerät für die Tensoren
        target_col : Name der Zielspalte

    Returns
        (X_tensor, y_tensor)
    """
    X = df[features].to_numpy(dtype="float64")
    if scaler is not None:
        X = scaler.transform(X)
    y = df[target_col].to_numpy(dtype="float32")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    if device is not None:
        X_tensor = X_tensor.to(device)
        y_tensor = y_tensor.to(device)

    return X_tensor, y_tensor


# ─── Sequenzbau für LSTM/GRU/CNN ─────────────────────────────────
def build_sequences(
    file_glob: str,
    features: list[str],
    scaler,
    seq_len: int = 30,
    target_col: str = "breakout_30m",
    max_files: Optional[int] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Baut (seq_len, features)-Sequenzen aus per-symbol Parquet-Dateien.

    WICHTIG: Verwendet die per-symbol Dateien (pre_split/),
    NICHT die geshuffelten Shards! Die Shards haben die zeitliche
    Reihenfolge zerstört – LSTM/GRU brauchen aber Sequenzen.

    Verwendet Sliding Window: Für Bar i wird das Fenster [i-seq_len, i-1]
    als Input verwendet und das Label an Position i als Target.
    So gibt es KEINEN Data Leakage (die Zukunft wird nicht ins Input-Fenster
    gegeben).

    Parameters
        file_glob : Glob-Pattern für Parquet-Dateien (z.B. "*_train.parquet")
        features : Feature-Namen
        scaler : Gefitteter StandardScaler
        seq_len : Länge der Sequenz (Default: 30 Minuten)
        target_col : Name der Zielspalte
        max_files : Optional, begrenzt Anzahl Dateien (für schnelle Tests)

    Returns
        (X_tensor, y_array) wobei X shape = (N, seq_len, n_features)
    """
    files = sorted(glob(file_glob))
    if not files:
        raise FileNotFoundError(f"Keine Dateien gefunden mit: {file_glob}")

    if max_files:
        files = files[:max_files]
        print(f"  [TEST-MODUS] Nur {len(files)} Dateien")

    print(f"  Baue Sequenzen aus {len(files)} Dateien (seq_len={seq_len})...")

    X_list, y_list = [], []
    skipped = 0

    for f_idx, fpath in enumerate(files):
        df = pd.read_parquet(fpath)

        if len(df) <= seq_len:
            skipped += 1
            continue

        # Zeitliche Sortierung sicherstellen
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        # Scaler anwenden
        feature_array = scaler.transform(df[features].to_numpy(dtype="float64"))
        labels = df[target_col].to_numpy(dtype="float32")

        # Sliding Window: [i-seq_len : i] → Label i
        # So sieht das Modell nur Vergangenheitswerte
        for i in range(seq_len, len(feature_array)):
            X_list.append(feature_array[i - seq_len : i])
            y_list.append(labels[i])

        if (f_idx + 1) % 20 == 0:
            print(f"    [{f_idx + 1}/{len(files)}] {Path(fpath).stem} "
                  f"→ {len(X_list):,} Sequenzen")

    if skipped:
        print(f"  {skipped} Dateien übersprungen (zu kurz)")

    X = np.stack(X_list)  # (N, seq_len, n_features)
    y = np.array(y_list)  # (N,)

    print(f"  Fertig: {X.shape[0]:,} Sequenzen, Shape={X.shape}")
    return torch.tensor(X, dtype=torch.float32), y


def build_sequences_fast(
    file_glob: str,
    features: list[str],
    scaler,
    seq_len: int = 30,
    target_col: str = "breakout_30m",
    max_files: Optional[int] = None,
    sample_every: int = 1,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Wie build_sequences, aber mit Subsampling für schnelleres Training.

    Parameters
        sample_every : Nur jeden N-ten sequenz verwenden (1 = alle)
    """
    X, y = build_sequences(file_glob, features, scaler, seq_len,
                           target_col, max_files)
    if sample_every > 1:
        X = X[::sample_every]
        y = y[::sample_every]
        print(f"  Subsampled (every {sample_every}): {X.shape[0]:,} Sequenzen")
    return X, y


# ─── Training Utilities ───────────────────────────────────────────
class EarlyStopping:
    """Early Stopping mit Patience und Best-Model-Checkpointing.

    Überwacht einen Monitor-Wert (z.B. validation loss).
    Wenn sich der Wert für `patience` Epochen nicht verbessert,
    wird das Training abgebrochen. Das beste Modell wird automatisch
    gespeichert.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0001,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.best_state_dict = None

        if mode == "min":
            self.is_better = lambda new, best: new < best - min_delta
            self.worse_sign = "↓"
        else:
            self.is_better = lambda new, best: new > best + min_delta
            self.worse_sign = "↑"

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Prüft einen neuen Score und speichert das Modell wenn besser.

        Returns
            True wenn Training abgebrochen werden soll (Early Stop)
        """
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            # Deep copy des state_dict
            self.best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            if self.verbose:
                print(f"  [✓] Neuer Bestwert: {score:.6f} (Epoche {epoch})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [{self.worse_sign}] Keine Verbesserung ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n[Early Stop] Keine Verbesserung seit {self.patience} Epochen.")
                    print(f"  Bester Wert: {self.best_score:.6f} (Epoche {self.best_epoch})")
        return self.early_stop


def restore_best_model(model: nn.Module, early_stopping: EarlyStopping) -> nn.Module:
    """Stellt das beste gespeicherte Modell wieder her."""
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)
    return model


def save_model(
    model: nn.Module,
    name: str,
    extra: dict = None,
) -> Path:
    """Speichert Model-State-Dict + optionale Metadaten.

    Parameters
        model : Das zu speichernde PyTorch-Modell
        name : Dateiname ohne Endung (z.B. "mlp", "lstm")
        extra : Optional, Dict mit Metadaten (loss, epoch, ...)

    Returns
        Pfad zur gespeicherten Datei
    """
    save_dir = PROJECT_ROOT / "artifacts" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{name}.pt"

    checkpoint = {"state_dict": model.state_dict()}
    if extra:
        checkpoint["metadata"] = extra

    torch.save(checkpoint, save_path)
    print(f"  Modell gespeichert: {save_path}")
    return save_path


def load_model(
    model: nn.Module,
    name: str,
    device: torch.device = None,
) -> nn.Module:
    """Lädt ein gespeichertes Modell (state_dict + optionale Metadaten).

    Parameters
        model : Instanz des Modells (mit gleicher Architektur)
        name : Dateiname ohne Endung
        device : Ziel-Gerät

    Returns
        Geladenes Modell + Metadaten als Tuple (model, metadata|None)
    """
    load_path = PROJECT_ROOT / "artifacts" / "models" / f"{name}.pt"
    if not load_path.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {load_path}")

    checkpoint = torch.load(load_path, map_location=device or "cpu",
                            weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        metadata = checkpoint.get("metadata", None)
    else:
        # Altes Format: direktes state_dict
        model.load_state_dict(checkpoint)
        metadata = None

    if device is not None:
        model = model.to(device)
    return model, metadata
