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


def create_dataloader(
    X,
    y = None,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> DataLoader:
    """Erstellt einen GPU-optimierten DataLoader.

    Akzeptiert entweder:
      - X, y als Tensoren → erstellt TensorDataset
      - X als Dataset (y=None) → verwendet das Dataset direkt

    Optimierungen:
      - num_workers > 0 → paralleles Laden im Hintergrund
      - prefetch_factor → nächster Batch wird geladen, während GPU rechnet
      - pin_memory → beschleunigt CPU→GPU Transfer um ~2x
      - persistent_workers → Workers bleiben über Epochen hinweg am Leben

    Auf Windows: num_workers=0 empfohlen bei großen Datasets (>50 GB RAM).
    """
    if y is not None:
        dataset = TensorDataset(X, y)
    else:
        dataset = X  # X ist bereits ein Dataset

    kw = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        kw["num_workers"] = num_workers
        kw["prefetch_factor"] = prefetch_factor
        if shuffle:
            kw["persistent_workers"] = persistent_workers

    return DataLoader(dataset, **kw)


def validate_batched(
    model: nn.Module,
    X,
    y=None,
    criterion: nn.Module = None,
    device: torch.device = None,
    batch_size: int = 2048,
    use_amp: bool = False,
    threshold: float = 0.5,
) -> dict:
    """Führt Validierung in Batches durch (verhindert GPU-OOM).

    Akzeptiert entweder:
      - X, y als Tensoren → erstellt TensorDataset
      - X als Dataset (y=None) → verwendet Dataset direkt

    Returns
        dict mit keys: loss, accuracy, all_probs, all_preds
    """
    was_training = model.training
    model.eval()

    if y is not None:
        dataset = TensorDataset(X, y)
    else:
        dataset = X  # bereits ein Dataset

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            if use_amp and criterion is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(X_batch).squeeze()
                    loss = criterion(logits, y_batch)
            elif criterion is not None:
                logits = model(X_batch).squeeze()
                loss = criterion(logits, y_batch)
            else:
                logits = model(X_batch).squeeze()

            if criterion is not None:
                total_loss += loss.item() * len(y_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            total_correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())

    if was_training:
        model.train()

    result = {
        "accuracy": total_correct / max(total_samples, 1),
        "all_probs": torch.cat(all_probs).numpy(),
        "all_preds": torch.cat(all_preds).numpy(),
    }
    if criterion is not None:
        result["loss"] = total_loss / max(total_samples, 1)
    return result


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
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    X_tensors, y_tensors = [], []
    skipped = 0
    total_seqs = 0

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
        n_rows = len(feature_array)

        # ── Vektorisierter Sliding-Window-Bau ─────────────────
        # Window [i-seq_len : i] → Label i  (kein Data Leakage)
        #
        # sliding_window_view legt die Window-Dimension ANS ENDE:
        #   feature_array (N, 82) → X_view (N-seq_len+1, 82, seq_len)
        #   → Transpose auf (N, seq_len, 82) nötig
        n_windows = n_rows - seq_len
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            X_view = sliding_window_view(feature_array, (seq_len,), axis=0)
            X_win = X_view[:n_windows]  # shape (n_windows, 82, seq_len)
            # Transpose → (n_windows, seq_len, n_features)
            X_chunk = np.ascontiguousarray(
                X_win.transpose(0, 2, 1), dtype=np.float32
            )
        except (ImportError, AttributeError):
            # Fallback: numpy < 1.20 → vorallozieren & in C schleife kopieren
            X_chunk = np.empty((n_windows, seq_len, len(features)), dtype=np.float32)
            for i in range(n_windows):
                X_chunk[i] = feature_array[i : i + seq_len]

        y_chunk = labels[seq_len:].astype(np.float32)

        # Direkt als Tensor speichern (spart np.stack am Ende)
        X_tensors.append(torch.from_numpy(X_chunk))
        y_tensors.append(torch.from_numpy(y_chunk))
        total_seqs += n_windows

        if (f_idx + 1) % 20 == 0:
            print(f"    [{f_idx + 1}/{len(files)}] {Path(fpath).stem} "
                  f"→ {total_seqs:,} Sequenzen (kumuliert)")

    if skipped:
        print(f"  {skipped} Dateien übersprungen (zu kurz)")

    if not X_tensors:
        raise RuntimeError("Keine gültigen Sequenzen gebaut – alle Dateien zu kurz?")

    # Einmalig caten (deutlich schneller als np.stack + torch.tensor)
    X = torch.cat(X_tensors, dim=0)  # (N, seq_len, n_features)
    y = torch.cat(y_tensors, dim=0)  # (N,)

    print(f"  Fertig: {X.shape[0]:,} Sequenzen, Shape={X.shape}")
    return X, y  # beide torch.Tensor (kompatibel mit TensorDataset)


def build_sequences_fast(
    file_glob: str,
    features: list[str],
    scaler,
    seq_len: int = 30,
    target_col: str = "breakout_30m",
    max_files: Optional[int] = None,
    sample_every: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


# ─── Lazy Sequence Dataset (speichereffizient) ────────────────────
class LazySequenceDataset(torch.utils.data.Dataset):
    """Dataset, das Sequenzen von Disk statt aus RAM lädt.

    Hält nur max_cached Dateien gleichzeitig im RAM (Default: 3).
    Bei 100 Dateien à ~1 GB → RAM-Verbrauch ~3 GB statt ~120 GB.
    Unterstützt Shuffling via DataLoader(shuffle=True).
    """

    def __init__(self, pt_files: list[str], max_cached: int = 3):
        import bisect
        self.pt_files = pt_files
        self.max_cached = max_cached

        # Index-Mapping: globale ID → (file_idx, offset)
        self._cumsum = []
        self._lengths = []  # Sample-Counts pro Datei (für ChunkedShuffleSampler)
        total = 0
        for pf in pt_files:
            data = torch.load(pf, map_location="cpu", weights_only=True)
            n = data["X"].size(0)
            self._lengths.append(n)
            total += n
            self._cumsum.append(total)
        self._total = total

        self._cache = {}          # file_idx → (X_tensor, y_tensor)
        self._cache_order = []    # LRU-Reihenfolge
        self._bisect = bisect

        print(f"  LazyDataset: {total:,} Samples, {len(pt_files)} Dateien "
              f"(max {max_cached} im RAM)")

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        # Binärsuche: welche Datei?
        file_idx = self._bisect.bisect_left(self._cumsum, idx + 1)
        offset = idx - (self._cumsum[file_idx - 1] if file_idx > 0 else 0)

        # Datei bei Bedarf laden
        if file_idx not in self._cache:
            while len(self._cache) >= self.max_cached and self._cache_order:
                oldest = self._cache_order.pop(0)
                self._cache.pop(oldest, None)
            data = torch.load(
                self.pt_files[file_idx],
                map_location="cpu",
                weights_only=True,
            )
            self._cache[file_idx] = (data["X"], data["y"])

        # LRU updaten
        if file_idx in self._cache_order:
            self._cache_order.remove(file_idx)
        self._cache_order.append(file_idx)

        X, y = self._cache[file_idx]
        return X[offset], y[offset]


def build_sequences_to_disk(
    file_glob: str,
    features: list[str],
    scaler,
    cache_dir: str,
    seq_len: int = 30,
    target_col: str = "breakout_30m",
    max_files: Optional[int] = None,
    sample_every: int = 1,
    max_cached: int = 3,
) -> LazySequenceDataset:
    """Baut Sequenzen pro Datei und speichert sie als .pt auf Disk.

    Im Gegensatz zu build_sequences() wird NICHT alles in RAM geladen,
    sondern pro Datei gebaut, gespeichert und wieder freigegeben.
    Gibt ein LazySequenceDataset zurück.

    Parameters
        cache_dir : Verzeichnis für .pt Dateien (wird erstellt falls nötig)
        max_cached : Maximal gleichzeitig im RAM gehaltene Dateien (Default: 3)
    """
    files = sorted(glob(file_glob))
    if not files:
        raise FileNotFoundError(f"Keine Dateien gefunden mit: {file_glob}")

    if max_files:
        files = files[:max_files]
        print(f"  [TEST-MODUS] Nur {len(files)} Dateien")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"  Baue Sequenzen ({len(files)} Dateien, seq_len={seq_len}) → {cache_path}")

    pt_files = []
    total_seqs = 0
    skipped = 0

    for f_idx, fpath in enumerate(files):
        out_file = cache_path / f"{Path(fpath).stem}_seq.pt"

        if out_file.exists():
            # Bereits gebaut → nur Länge peeken
            data = torch.load(out_file, map_location="cpu", weights_only=True)
            n = data["X"].size(0)
            total_seqs += n
            pt_files.append(str(out_file))
            if (f_idx + 1) % 20 == 0:
                print(f"    [{f_idx + 1}/{len(files)}] {Path(fpath).stem} "
                      f"(CACHE) → {total_seqs:,} Sequenzen kumuliert")
            continue

        # ── Datei einlesen & Sequenzen bauen ─────────────────
        df = pd.read_parquet(fpath)
        if len(df) <= seq_len:
            skipped += 1
            continue

        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        feature_array = scaler.transform(df[features].to_numpy(dtype="float64"))
        labels = df[target_col].to_numpy(dtype="float32")
        n_rows = len(feature_array)
        n_windows = n_rows - seq_len

        # Vektorisierter Sliding-Window-Bau
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            X_view = sliding_window_view(feature_array, (seq_len,), axis=0)
            X_win = X_view[:n_windows]
            X_chunk = np.ascontiguousarray(X_win.transpose(0, 2, 1), dtype=np.float32)
        except (ImportError, AttributeError):
            X_chunk = np.empty((n_windows, seq_len, len(features)), dtype=np.float32)
            for i in range(n_windows):
                X_chunk[i] = feature_array[i : i + seq_len]

        y_chunk = labels[seq_len:].astype(np.float32)

        if sample_every > 1:
            X_chunk = X_chunk[::sample_every]
            y_chunk = y_chunk[::sample_every]
            n_windows = len(X_chunk)

        # Auf Disk speichern & RAM sofort freigeben
        torch.save(
            {"X": torch.from_numpy(X_chunk), "y": torch.from_numpy(y_chunk)},
            out_file,
        )
        del X_chunk, y_chunk, feature_array, labels
        pt_files.append(str(out_file))
        total_seqs += n_windows

        if (f_idx + 1) % 20 == 0:
            print(f"    [{f_idx + 1}/{len(files)}] {Path(fpath).stem} "
                  f"→ {total_seqs:,} Sequenzen kumuliert")

    if skipped:
        print(f"  {skipped} Dateien übersprungen (zu kurz)")
    print(f"  Fertig: {total_seqs:,} Sequenzen in {len(pt_files)} .pt Dateien")

    return LazySequenceDataset(pt_files, max_cached=max_cached)


# ─── Chunked-Shuffle-Sampler ────────────────────────────────────
class ChunkedShuffleSampler(torch.utils.data.Sampler):
    """Sampler: shuffled Datei-Reihenfolge, sequentiell innerhalb jeder Datei.

    Verhindert Cache-Thrashing bei LazySequenceDataset:
      - Dateien werden pro Epoche neu geshuffelt
      - Innerhalb einer Datei wird sequentiell (oder geshuffelt) gelesen
      - → LazySequenceDataset muss nur 1–2 Dateien gleichzeitig laden
    """

    def __init__(self, dataset: "LazySequenceDataset", shuffle_within: bool = True):
        self.lengths = dataset._lengths   # Sample-Counts pro Datei
        self.total = dataset._total
        self.n_files = len(self.lengths)
        self.shuffle_within = shuffle_within

    def __len__(self) -> int:
        return self.total

    def __iter__(self):
        import random
        # Datei-Reihenfolge shuffeln
        file_order = list(range(self.n_files))
        random.shuffle(file_order)

        for file_idx in file_order:
            # Start-Index dieser Datei im globalen Index-Raum
            start = sum(self.lengths[:file_idx])
            n_samples = self.lengths[file_idx]

            if self.shuffle_within:
                # Samples innerhalb der Datei shuffeln
                indices = list(range(start, start + n_samples))
                random.shuffle(indices)
                yield from indices
            else:
                yield from range(start, start + n_samples)


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
