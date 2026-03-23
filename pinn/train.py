"""
train.py — Training loop for the LNP EE PINN.

Loss:
    L_total = L_data + alpha * L_physics

    L_data:    MSE between predicted and observed EE (supervised signal).
    L_physics: Weighted sum of physics residuals (R1 + R2 + R3) from physics.py.
               These residuals encode:
                 R1 — N/P monotonicity (EE rises with N/P in charge-limited regime)
                 R2 — Thermodynamic mixing consistency (lipid mole fraction gradient)
                 R3 — Boundary condition: EE -> 0 at infinite particle size

Usage:
    # Random 80/20 split (default)
    python -m pinn.train --data data/lnp_atlas_export.csv --epochs 200 --alpha 0.3

    # GroupKFold on paper_doi (leakage-free evaluation)
    python -m pinn.train --data data/lnp_atlas_export.csv --cv groupkfold --epochs 200 --alpha 0.3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .model import build_model
from .physics import total_physics_loss
from .preprocess import (
    load_and_preprocess,
    load_and_preprocess_with_groups,
    FEATURE_COLS,
    FEATURE_INDEX,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train LNP EE PINN")
    p.add_argument("--data", type=str, required=True,
                   help="Path to LNP Atlas CSV")
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
    p.add_argument("--cv", type=str, default="random",
                   choices=["random", "groupkfold"],
                   help="Validation strategy: 'random' (80/20) or 'groupkfold' (paper_doi)")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Number of folds for GroupKFold (ignored if --cv=random)")
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


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    """Run inference and return predictions as numpy array."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    return model(X_t).squeeze(-1).cpu().numpy()


def run_groupkfold(X_raw, y, groups, args, device):
    """Run GroupKFold cross-validation on paper_doi groups.

    Trains a fresh model per fold with a per-fold scaler to prevent leakage.
    Returns aggregate and per-fold metrics in EE% space.
    """
    n_features = X_raw.shape[1]
    gkf = GroupKFold(n_splits=args.n_folds)
    oof_preds = np.zeros(len(y), dtype=np.float32)
    fold_metrics = []

    n_groups = len(np.unique(groups))
    print(f"[PINN] GroupKFold: {args.n_folds} folds | {len(y)} samples | {n_groups} paper groups")
    print(f"[PINN] Training {args.epochs} epochs per fold (alpha={args.alpha})\n")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups)):
        # Reproducible per-fold seeding
        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)

        # Fit scaler on training fold only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_raw[train_idx]).astype(np.float32)
        X_val = scaler.transform(X_raw[val_idx]).astype(np.float32)
        y_train = y[train_idx]
        y_val = y[val_idx]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=args.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=args.batch_size,
        )

        # Fresh model per fold
        model = build_model(n_features=n_features, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_mse = float("inf")
        best_state = None

        for epoch in range(1, args.epochs + 1):
            train_epoch(model, train_loader, optimizer, args.alpha, device, n_features)
            val_metrics = eval_epoch(model, val_loader, device)
            scheduler.step()

            if val_metrics["val_mse"] < best_val_mse:
                best_val_mse = val_metrics["val_mse"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Load best checkpoint, predict on validation set
        model.load_state_dict(best_state)
        val_preds = predict(model, X_val, device)
        oof_preds[val_idx] = val_preds

        # Per-fold metrics in EE% space
        y_val_pct = y_val * 100.0
        val_preds_pct = val_preds * 100.0
        fold_rmse = float(np.sqrt(mean_squared_error(y_val_pct, val_preds_pct)))
        fold_r2 = float(r2_score(y_val_pct, val_preds_pct))
        fold_mae = float(mean_absolute_error(y_val_pct, val_preds_pct))

        fold_metrics.append({
            "fold": fold + 1,
            "rmse_pct": round(fold_rmse, 2),
            "r2": round(fold_r2, 3),
            "mae_pct": round(fold_mae, 2),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })
        print(f"  Fold {fold+1}: RMSE={fold_rmse:.2f}% | R2={fold_r2:.3f} | "
              f"MAE={fold_mae:.2f}% | train={len(train_idx)} val={len(val_idx)}")

    # Aggregate OOF metrics
    y_pct = y * 100.0
    oof_pct = oof_preds * 100.0
    oof_rmse = float(np.sqrt(mean_squared_error(y_pct, oof_pct)))
    oof_r2 = float(r2_score(y_pct, oof_pct))
    oof_mae = float(mean_absolute_error(y_pct, oof_pct))

    print(f"\n  OOF RMSE: {oof_rmse:.2f}% | OOF R2: {oof_r2:.3f} | OOF MAE: {oof_mae:.2f}%")

    return {
        "cv_mode": "groupkfold",
        "n_folds": args.n_folds,
        "n_samples": len(y),
        "n_groups": int(n_groups),
        "epochs_per_fold": args.epochs,
        "alpha": args.alpha,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "n_residual": args.n_residual,
        "oof_metrics": {
            "rmse_pct": round(oof_rmse, 2),
            "r2": round(oof_r2, 3),
            "mae_pct": round(oof_mae, 2),
        },
        "fold_metrics": fold_metrics,
    }


def main():
    args = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[PINN] Device: {device}")
    print(f"[PINN] Loading data: {args.data}")

    if args.cv == "groupkfold":
        X_raw, y, groups = load_and_preprocess_with_groups(args.data)
        results = run_groupkfold(X_raw, y, groups, args, device)

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "groupkfold_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[PINN] Results saved to: {out_dir / 'groupkfold_results.json'}")
        return

    # --- Default: random 80/20 split ---
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
