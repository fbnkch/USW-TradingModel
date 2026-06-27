"""
Training für das verbesserte MLP (BreakoutModel V2).

Verbesserungen gegenüber V1:
  - GPU-Training mit Automatic Mixed Precision (AMP)
  - Global Scaler Integration
  - Early Stopping (Patience 10)
  - Learning Rate Scheduling (ReduceLROnPlateau)
  - Gradient Clipping
  - Speichert bestes Modell (nicht nur letztes)
  - Detaillierte Trainings-Metriken

Aufruf:
  python scripts/06_model_training/train.py
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import BreakoutModel
from utils import (
    PROJECT_ROOT, params,
    get_device, enable_amp,
    load_features, load_scaler, load_class_balance,
    load_shards, prepare_tensors,
    EarlyStopping, restore_best_model, save_model,
)

# ═══════════════════════════════════════════════════════════════════
# KONFIGURATION
# ═══════════════════════════════════════════════════════════════════
EPOCHS      = 50
BATCH_SIZE  = 1024
LR          = 0.001
THRESHOLD   = 0.5
WEIGHT_DECAY = 1e-5
GRAD_CLIP    = 1.0


def train_mlp() -> dict:
    """Führt das MLP-Training mit allen Verbesserungen durch."""
    t_start = time.time()

    # ── GPU & AMP ──────────────────────────────────────────────
    device = get_device(verbose=True)
    use_amp = enable_amp(device)
    scaler_amp = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Features, Scaler, Klassen-Balance ─────────────────────
    features = load_features()
    scaler = load_scaler()
    balance = load_class_balance()
    print(f"Features: {len(features)} | Klassen-Balance: "
          f"{balance['positive_ratio']:.1%} Breakout\n")

    # ── Daten laden (Shards) ──────────────────────────────────
    print("=" * 60)
    print("Lade Trainingsdaten (alle 100 Shards)...")
    train_df = load_shards("train", features)
    print(f"  Train: {len(train_df):,} Samples")

    print("Lade Validierungsdaten...")
    val_df = load_shards("validation", features)
    print(f"  Val:   {len(val_df):,} Samples")

    # ── Tensoren vorbereiten (mit Scaler, NOCH auf CPU) ──────
    # WICHTIG: Tensoren bleiben auf CPU. pin_memory im DataLoader
    # funktioniert NUR mit CPU-Tensoren. Der Transfer auf GPU
    # passiert explizit im Trainingsloop.
    print("\nWende GlobalScaler an & erstelle Tensoren (CPU)...")
    X_train, y_train = prepare_tensors(train_df, features, scaler, device=None)
    X_val, y_val = prepare_tensors(val_df, features, scaler, device=None)

    # Freigabe des DataFrame-Speichers
    del train_df, val_df

    # Validation-Tensoren direkt auf GPU (werden komplett auf einmal
    # evaluiert, kein Batching nötig)
    X_val_gpu = X_val.to(device)
    y_val_gpu = y_val.to(device)

    # ── DataLoader ─────────────────────────────────────────────
    # pin_memory=True: CPU-Tensoren werden in pinned memory vorgehalten,
    # was den Transfer auf die GPU beschleunigt (~2x schneller).
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    print(f"DataLoader: {len(loader)} Batches à {BATCH_SIZE}")

    # ── Modell ─────────────────────────────────────────────────
    model = BreakoutModel(
        input_size=len(features),
        hidden_sizes=(128, 64, 32, 16),
        dropout=0.4,
        use_batch_norm=True,
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nMLP V2 Modell: {total_params:,} Parameter")

    # ── Loss, Optimizer, Scheduler ─────────────────────────────
    # BCE-Loss mit leichter Gewichtung
    pos_weight = torch.tensor([balance.get("scale_pos_weight", 1.0)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
    )

    early_stopping = EarlyStopping(
        patience=12,
        min_delta=0.0001,
        mode="min",
        verbose=True,
    )

    # ── Training-Loop ──────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n{'=' * 60}")
    print(f"Training: MLP V2 | {EPOCHS} Epochen | Batch-Size {BATCH_SIZE} "
          f"| LR {LR} | AMP: {use_amp}")
    print(f"Gerät: {device}")
    print(f"{'=' * 60}\n")

    for epoch in range(1, EPOCHS + 1):
        # ── Training-Phase ─────────────────────────────────
        model.train()
        train_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            # Batch auf GPU transferieren (von pinned CPU memory)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(X_batch).squeeze()
                    loss = criterion(logits, y_batch)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(X_batch).squeeze()
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
            val_pred = (val_prob > THRESHOLD).float()
            val_acc = (val_pred == y_val_gpu).float().mean().item()

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── Logging ────────────────────────────────────────
        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"Zeit: {elapsed:.0f}s"
        )

        # ── Scheduler & Early Stopping ─────────────────────
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < LR * 0.9:
            print(f"  LR reduziert auf {current_lr:.2e}")

        stop = early_stopping.step(val_loss, model, epoch)
        if stop:
            break

    # ── Bestes Modell wiederherstellen ─────────────────────────
    model = restore_best_model(model, early_stopping)
    total_time = time.time() - t_start

    # ── Finale Evaluation ─────────────────────────────────────
    model.eval()
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda"):
                final_logits = model(X_val_gpu).squeeze()
        else:
            final_logits = model(X_val_gpu).squeeze()

        final_prob = torch.sigmoid(final_logits).cpu().numpy()
        final_pred = (final_prob > THRESHOLD).astype(int)
        final_acc = (final_pred == y_val_gpu.cpu().numpy()).mean()

    # ── Ergebnisse ─────────────────────────────────────────────
    results = {
        "model_type": "mlp",
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden_sizes": [128, 64, 32, 16],
            "dropout": 0.4,
        },
        "best_val_loss": early_stopping.best_score,
        "best_epoch": early_stopping.best_epoch,
        "final_val_acc": float(final_acc),
        "total_epochs": epoch,
        "train_time_s": total_time,
        "n_features": len(features),
        "n_params": total_params,
        "history": history,
    }

    # ── Speichern ──────────────────────────────────────────────
    save_model(model, name="mlp_model", extra=results)

    # ── Zusammenfassung ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Training abgeschlossen: MLP V2")
    print(f"  Epochen:          {results['total_epochs']} "
          f"(beste: {results['best_epoch']})")
    print(f"  Bester Val Loss:  {results['best_val_loss']:.6f}")
    print(f"  Final Val Acc:    {results['final_val_acc']:.2%}")
    print(f"  Trainingszeit:    {total_time/60:.1f} Min")
    print(f"  Modell-Parameter: {total_params:,}")
    print(f"{'=' * 60}")

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_mlp()
