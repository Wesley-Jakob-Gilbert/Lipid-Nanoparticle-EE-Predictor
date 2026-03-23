"""
train.py — Training loop for the LNP EE PINN.

Loss:
    L_total = L_data + alpha * L_physics

    L_data:    MSE between predicted and observed EE (supervised signal).
    L_physics: Weighted sum of physics residuals (R1 + R2 + R3) from physics.py.
               These residuals encode:
                 R1 — N/P monotonicity (EE rises with N/P in charge-limited regime)
                 R2 — Thermodynamic mixing consistency (lipid mole fraction gradient)
                 R3 — Boundary condition: EE → 0 at infinite particle size

Usage (from lnp_ml_phase1/):
    python -m pinn.train --data data/lnp_firstpaper_encapsulation.csv \
                         --epochs 200 --alpha 0.5 --lr 1e-3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import build_model
from .physics import total_physics_loss
from .preprocess import load_and_preprocess, FEATURE_COLS, FEATURE_INDEX


def parse_args():
    p = argparse.ArgumentParser(description="Train LNP EE PINN")
    p.add_argument("--data", type=str, required=True,
                   help="Path to EE CSV (lnp_firstpaper_encapsulation.csv or similar)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Weight on physics loss (0 = pure data-driven)")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-residual", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs/pinn",
                   help="Output directory for checkpoints and metrics")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto', 'cpu', or 'cuda'")
    return p.parse_args()


def get_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    alpha: float,
    device: str,
    n_features: int,
) -> dict:
    model.train()
    total_loss = total_data = total_phys = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device).requires_grad_(True)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # --- Data loss ---
        ee_pred = model(x_batch)
        loss_data = nn.functional.mse_loss(ee_pred.squeeze(-1), y_batch)

        # --- Physics loss ---
        loss_phys = total_physics_loss(
            model=model,
            x_batch=x_batch,
            np_col=FEATURE_INDEX["np_ratio"],
            x_il_col=FEATURE_INDEX["ionizable_lipid_mole_fraction"],
            size_col=FEATURE_INDEX["particle_size_nm"],
            n_features=n_features,
            device=device,
        )

        loss = loss_data + alpha * loss_phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_data += loss_data.item()
        total_phys += loss_phys.item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "loss_data": total_data / n,
        "loss_physics": total_phys / n,
    }


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    preds, targets = [], []
    for x_batch, y_batch in loader:
        ee_pred = model(x_batch.to(device)).squeeze(-1)
        preds.append(ee_pred.cpu())
        targets.append(y_batch)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mse = nn.functional.mse_loss(preds, targets).item()
    mae = (preds - targets).abs().mean().item()
    return {"val_mse": mse, "val_mae": mae}


def main():
    args = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[PINN] Device: {device}")
    print(f"[PINN] Loading data: {args.data}")

    X, y, scaler = load_and_preprocess(args.data)
    n_features = X.shape[1]
    print(f"[PINN] Samples: {len(y)} | Features: {n_features}")

    # Train / val split (80/20)
    split = int(0.8 * len(y))
    idx = np.random.permutation(len(y))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    X_val   = torch.tensor(X[val_idx],   dtype=torch.float32)
    y_val   = torch.tensor(y[val_idx],   dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size
    )

    model = build_model(n_features=n_features, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_mse = float("inf")

    print(f"[PINN] Training for {args.epochs} epochs (alpha={args.alpha})")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.alpha, device, n_features
        )
        val_metrics = eval_epoch(model, val_loader, device)
        scheduler.step()

        row = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(row)

        if val_metrics["val_mse"] < best_val_mse:
            best_val_mse = val_metrics["val_mse"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d} | "
                f"loss={train_metrics['loss']:.4f} "
                f"(data={train_metrics['loss_data']:.4f}, "
                f"phys={train_metrics['loss_physics']:.4f}) | "
                f"val_mse={val_metrics['val_mse']:.4f} "
                f"val_mae={val_metrics['val_mae']:.4f}"
            )

    # Save history
    with open(out_dir / "pinn_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[PINN] Best val MSE: {best_val_mse:.4f}")
    print(f"[PINN] Checkpoint: {out_dir / 'best_model.pt'}")
    print(f"[PINN] History:    {out_dir / 'pinn_training_history.json'}")


if __name__ == "__main__":
    main()
