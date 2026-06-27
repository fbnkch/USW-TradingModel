"""
Training für sequenzielle Modelle (LSTM, GRU, CNN-1D).

Aufruf:
  python scripts/06_model_training/train_sequential.py --model lstm
  python scripts/06_model_training/train_sequential.py --model gru
  python scripts/06_model_training/train_sequential.py --model cnn

Besonderheiten:
  - Baut Sequenzen aus per-symbol Parquet-Dateien (nicht Shards!)
  - GPU-Training mit Automatic Mixed Precision (AMP)
  - Early Stopping, LR-Scheduling, Gradient-Clipping
  - Speichert bestes Modell + Trainings-Metadaten
  - Dauert ~20-60 Min auf GPU (RTX 3090), ~2-4h auf CPU
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

from model_sequential import LSTMBreakoutModel, GRUBreakoutModel, CNNBreakoutModel
from utils import (
    PROJECT_ROOT, params,
    get_device, enable_amp,
    load_features, load_scaler, load_class_balance,
    build_sequences, build_sequences_fast,
    EarlyStopping, restore_best_model, save_model,
)


# ═══════════════════════════════════════════════════════════════════
# KONFIGURATION
# ═══════════════════════════════════════════════════════════════════
# Per-Symbol Dateien (zeitlich sortiert – wichtig für Sequenzen!)
PRE_SPLIT_PATH = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]

# Mapping von Modellnamen zu Klassen
MODEL_CLASSES = {
    "lstm": LSTMBreakoutModel,
    "gru": GRUBreakoutModel,
    "cnn": CNNBreakoutModel,
}


def get_default_config(model_type: str) -> dict:
    """Gibt die empfohlene Default-Konfiguration pro Modelltyp zurück."""
    configs = {
        "lstm": {
            "seq_len": 30,
            "batch_size": 256,
            "epochs": 40,
            "lr": 0.001,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.35,
            "bidirectional": True,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "threshold": 0.5,
            "sample_every": 1,   # 1 = alle Sequenzen verwenden
            "max_files": None,   # None = alle Dateien
        },
        "gru": {
            "seq_len": 30,
            "batch_size": 256,
            "epochs": 40,
            "lr": 0.001,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.35,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "threshold": 0.5,
            "sample_every": 1,
            "max_files": None,
        },
        "cnn": {
            "seq_len": 30,
            "batch_size": 512,
            "epochs": 40,
            "lr": 0.001,
            "hidden_channels": 64,
            "kernel_sizes": (3, 5, 10),
            "dropout": 0.35,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "threshold": 0.5,
            "sample_every": 1,
            "max_files": None,
        },
    }
    if model_type not in configs:
        raise ValueError(f"Unbekannter Modelltyp: {model_type}. "
                         f"Erwartet: {list(configs.keys())}")
    return configs[model_type]


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════
def train_sequential(
    model_type: str = "lstm",
    config_overrides: dict = None,
) -> dict:
    """Führt das Training für ein sequenzielles Modell durch.

    Parameters
        model_type : "lstm", "gru", oder "cnn"
        config_overrides : Optional, Dict mit zu überschreibenden Config-Werten

    Returns
        Dict mit Trainings-Ergebnissen (best_loss, best_acc, history, ...)
    """
    cfg = get_default_config(model_type)
    if config_overrides:
        cfg.update(config_overrides)

    # ── GPU & AMP ──────────────────────────────────────────────
    device = get_device(verbose=True)
    use_amp = enable_amp(device)
    scaler_amp = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Features & Scaler ──────────────────────────────────────
    features = load_features()
    scaler = load_scaler()
    balance = load_class_balance()
    print(f"Features: {len(features)} | Klassen-Balance: "
          f"{balance['positive_ratio']:.1%} Breakout\n")

    # ── Sequenzen bauen ────────────────────────────────────────
    print("=" * 60)
    print("Baue Trainings-Sequenzen...")
    train_glob = str(PRE_SPLIT_PATH / "*_train.parquet")
    X_train, y_train = build_sequences_fast(
        train_glob, features, scaler,
        seq_len=cfg["seq_len"],
        max_files=cfg["max_files"],
        sample_every=cfg["sample_every"],
    )

    print("\nBaue Validierungs-Sequenzen...")
    val_glob = str(PRE_SPLIT_PATH / "*_validation.parquet")
    X_val, y_val = build_sequences_fast(
        val_glob, features, scaler,
        seq_len=cfg["seq_len"],
        max_files=cfg["max_files"],
        sample_every=cfg["sample_every"],
    )

    # ── DataLoader ─────────────────────────────────────────────
    train_dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,  # 0 = kein Multiprocessing (stabiler auf Windows)
    )
    print(f"\nDataLoader: {len(loader)} Batches à ≤{cfg['batch_size']}")

    # ── Modell ─────────────────────────────────────────────────
    model_cls = MODEL_CLASSES[model_type]

    # Baue Modell mit den richtigen Konstruktor-Parametern
    if model_type == "cnn":
        model = model_cls(
            input_size=len(features),
            hidden_channels=cfg["hidden_channels"],
            kernel_sizes=cfg["kernel_sizes"],
            dropout=cfg["dropout"],
        )
    elif model_type in ("lstm", "gru"):
        kwargs = dict(
            input_size=len(features),
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
        if model_type == "lstm":
            kwargs["bidirectional"] = cfg.get("bidirectional", True)
        model = model_cls(**kwargs)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{model_type.upper()} Modell: {total_params:,} Parameter "
          f"({trainable_params:,} trainierbar)")

    # ── Loss, Optimizer, Scheduler ─────────────────────────────
    # BCE-Loss mit leichter positiver Gewichtung (Breakouts minimal
    # seltener als Non-Breakouts → leichter Ausgleich)
    pos_weight = torch.tensor([balance.get("scale_pos_weight", 1.0)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-5),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.0001,
        mode="min",
        verbose=True,
    )

    # ── Validierungsdaten auf GPU ──────────────────────────────
    X_val_gpu = X_val.to(device)
    y_val_gpu = y_val.to(device)

    # ── Training-Loop ──────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    t_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"Training: {model_type.upper()} | {cfg['epochs']} Epochen "
          f"| Batch-Size {cfg['batch_size']} | LR {cfg['lr']}")
    print(f"AMP: {use_amp} | Gerät: {device}")
    print(f"{'=' * 60}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Training-Phase ─────────────────────────────────
        model.train()
        train_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(X_batch).squeeze()
                    loss = criterion(logits, y_batch)
                scaler_amp.scale(loss).backward()

                # Gradient Clipping
                if cfg.get("grad_clip"):
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["grad_clip"]
                    )

                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(X_batch).squeeze()
                loss = criterion(logits, y_batch)
                loss.backward()

                if cfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["grad_clip"]
                    )

                optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # ── Validation-Phase ───────────────────────────────
        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    val_logits = model(X_val_gpu).squeeze()
                    val_loss = criterion(val_logits, y_val_gpu).item()
            else:
                val_logits = model(X_val_gpu).squeeze()
                val_loss = criterion(val_logits, y_val_gpu).item()

            val_prob = torch.sigmoid(val_logits)
            val_pred = (val_prob > cfg["threshold"]).float()
            val_acc = (val_pred == y_val_gpu).float().mean().item()

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── Logging ────────────────────────────────────────
        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"Zeit: {elapsed:.0f}s"
        )

        # ── Scheduler & Early Stopping ─────────────────────
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != cfg["lr"] and epoch > 5:
            print(f"  LR reduziert auf {current_lr:.2e}")

        stop = early_stopping.step(val_loss, model, epoch)
        if stop:
            break

    # ── Bestes Modell wiederherstellen ──────────────────────────
    model = restore_best_model(model, early_stopping)
    total_time = time.time() - t_start

    # ── Finale Evaluation auf Validation ────────────────────────
    model.eval()
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda"):
                final_logits = model(X_val_gpu).squeeze()
        else:
            final_logits = model(X_val_gpu).squeeze()

        final_prob = torch.sigmoid(final_logits).cpu().numpy()
        final_pred = (final_prob > cfg["threshold"]).astype(int)
        final_acc = (final_pred == y_val.numpy()).mean()

    # ── Ergebnisse ──────────────────────────────────────────────
    results = {
        "model_type": model_type,
        "config": cfg,
        "best_val_loss": early_stopping.best_score,
        "best_epoch": early_stopping.best_epoch,
        "final_val_acc": final_acc,
        "total_epochs": epoch,
        "train_time_s": total_time,
        "n_features": len(features),
        "n_params": total_params,
        "history": history,
    }

    # ── Speichern ───────────────────────────────────────────────
    save_model(
        model,
        name=f"{model_type}_model",
        extra=results,
    )

    # ── Zusammenfassung ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Training abgeschlossen: {model_type.upper()}")
    print(f"  Epochen:          {results['total_epochs']} "
          f"(beste: {results['best_epoch']})")
    print(f"  Bester Val Loss:  {results['best_val_loss']:.6f}")
    print(f"  Final Val Acc:    {results['final_val_acc']:.2%}")
    print(f"  Trainingszeit:    {total_time/60:.1f} Min")
    print(f"  Modell-Parameter: {total_params:,}")
    print(f"{'=' * 60}")

    return results


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training für sequenzielle Breakout-Modelle"
    )
    parser.add_argument(
        "--model", type=str, default="lstm",
        choices=["lstm", "gru", "cnn"],
        help="Modelltyp (default: lstm)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Anzahl Epochen (überschreibt Default)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch-Größe (überschreibt Default)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning Rate (überschreibt Default)"
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Maximale Anzahl Dateien für schnelle Tests"
    )
    parser.add_argument(
        "--sample_every", type=int, default=None,
        help="Nur jede N-te Sequenz verwenden (Subsampling)"
    )
    args = parser.parse_args()

    # Config-Overrides aus CLI-Argumenten
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.max_files is not None:
        overrides["max_files"] = args.max_files
    if args.sample_every is not None:
        overrides["sample_every"] = args.sample_every

    train_sequential(args.model, overrides if overrides else None)
