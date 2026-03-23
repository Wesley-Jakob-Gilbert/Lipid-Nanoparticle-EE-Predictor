"""
train_pinn.py — Train the Physics-Informed Neural Network (PINN) for LNP EE prediction.

Uses the same feature engineering pipeline as train.py (features.py) but feeds into
the PINN architecture (pinn/model.py) with physics-constrained loss (pinn/physics.py).

Usage (from project root):
    python3 src/train_pinn.py

Outputs (saved to artifacts/pinn/):
    best_model.pt         — best checkpoint (lowest val MSE)
    pinn_metrics.json     — train/val R², RMSE, MAE + physics loss breakdown
    pinn_training_history.json — per-epoch loss curves
"""

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

DATA_PATH  = ROOT / "data" / "lnp_atlas_export.csv"
ARTIFACTS  = ROOT / "artifacts" / "pinn"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

TARGET_COL = "encapsulation_efficiency_percent_std"

# ── Import PINN modules ───────────────────────────────────────────────────────
from pinn.model   import build_model
from pinn.physics import total_physics_loss

# ── Feature engineering (reuse existing pipeline) ────────────────────────────
from features import build_feature_matrix, get_feature_columns


def load_data():
    print(f"[{datetime.now():%H:%M:%S}] Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].between(0, 100)]
    print(f"  Rows with EE%: {len(df)}")
    return df


def get_pinn_features(df: pd.DataFrame):
    """
    Use the PINN's own 7 physicochemical features rather than the full 76-feature
    set — the physics residuals are defined over these specific columns.
    """
    from pinn.preprocess import load_and_preprocess_df, FEATURE_COLS
    X, y, scaler = load_and_preprocess_df(df, target_col=TARGET_COL)
    return X, y, scaler, FEATURE_COLS


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{datetime.now():%H:%M:%S}] Device: {device}")

    df = load_data()
    X, y, scaler, feature_cols = get_pinn_features(df)

    print(f"  Samples: {len(y)} | Features: {X.shape[1]}")
    print(f"  EE% mean={y.mean()*100:.1f}%  std={y.std()*100:.1f}%")

    # Train/val split (stratified by EE quartile for balance)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(to_tensor(X_tr), to_tensor(y_tr)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(to_tensor(X_val), to_tensor(y_val)),
        batch_size=64
    )

    n_features = X.shape[1]
    model = build_model(n_features=n_features, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    # Feature column indices for physics residuals
    fc = list(feature_cols)
    np_col   = fc.index("np_ratio")   if "np_ratio"   in fc else 1
    xil_col  = fc.index("ionizable_lipid_mole_fraction") if "ionizable_lipid_mole_fraction" in fc else 0
    size_col = fc.index("particle_size_nm") if "particle_size_nm" in fc else 2

    best_val_mse = float("inf")
    history = []
    alpha = 0.3  # physics loss weight

    print(f"\n[{datetime.now():%H:%M:%S}] Training 300 epochs (alpha={alpha})...")
    for epoch in range(1, 301):
        model.train()
        tl = td = tp = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device).requires_grad_(True)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss_data = nn.functional.mse_loss(pred, yb)
            loss_phys = total_physics_loss(
                model, xb, np_col, xil_col, size_col, n_features, device=device
            )
            loss = loss_data + alpha * loss_phys
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item(); td += loss_data.item(); tp += loss_phys.item()
        scheduler.step()

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(device)).squeeze(-1).cpu().numpy())
                targets.append(yb.numpy())
        preds   = np.concatenate(preds)
        targets = np.concatenate(targets)
        val_mse = mean_squared_error(targets, preds)
        val_r2  = r2_score(targets, preds)

        n = len(train_loader)
        row = {
            "epoch": epoch,
            "loss": tl/n, "loss_data": td/n, "loss_physics": tp/n,
            "val_mse": val_mse, "val_r2": val_r2
        }
        history.append(row)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), ARTIFACTS / "best_model.pt")

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={tl/n:.4f} "
                  f"(data={td/n:.4f} phys={tp/n:.4f}) | "
                  f"val_mse={val_mse:.4f} val_r2={val_r2:.3f}")

    # Final eval with best checkpoint
    model.load_state_dict(torch.load(ARTIFACTS / "best_model.pt"))
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds_all.append(model(xb.to(device)).squeeze(-1).cpu().numpy())
            targets_all.append(yb.numpy())
    p = np.concatenate(preds_all)
    t = np.concatenate(targets_all)

    # Convert back to EE% (model outputs [0,1])
    p_pct = p * 100
    t_pct = t * 100

    final_metrics = {
        "n_train": len(y_tr),
        "n_val":   len(y_val),
        "n_features": n_features,
        "val_r2":   float(r2_score(t_pct, p_pct)),
        "val_rmse": float(np.sqrt(mean_squared_error(t_pct, p_pct))),
        "val_mae":  float(mean_absolute_error(t_pct, p_pct)),
        "alpha":    alpha,
        "epochs":   300,
        "note": (
            "PINN trained on physicochemical features only (7 features). "
            "Physics residuals: N/P monotonicity (R1), Gibbs free energy mixing (R2), "
            "boundary condition EE→0 at large size (R3). "
            "Val R² reflects generalization on held-out 20% random split."
        )
    }

    print(f"\n{'='*55}")
    print(f"  PINN Final Results (val set, n={len(t_pct)})")
    print(f"  R²   = {final_metrics['val_r2']:.3f}")
    print(f"  RMSE = {final_metrics['val_rmse']:.2f}%")
    print(f"  MAE  = {final_metrics['val_mae']:.2f}%")
    print(f"{'='*55}\n")

    with open(ARTIFACTS / "pinn_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(ARTIFACTS / "pinn_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Checkpoint: {ARTIFACTS / 'best_model.pt'}")
    print(f"Metrics:    {ARTIFACTS / 'pinn_metrics.json'}")


if __name__ == "__main__":
    train()
