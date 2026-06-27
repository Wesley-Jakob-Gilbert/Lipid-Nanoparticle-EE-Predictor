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
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from jax import random
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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
    p.add_argument("--cv", type=str, default="random",
                   choices=["random", "groupkfold"],
                   help="Validation strategy: 'random' (80/20) or 'groupkfold' (paper_doi)")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Number of folds for GroupKFold (ignored if --cv=random)")
    return p.parse_args()


def batch_iter(X, y, batch_size, key):
    """Yield shuffled mini-batches as JAX arrays."""
    n = len(y)
    perm = random.permutation(key, n)
    X_shuf, y_shuf = X[perm], y[perm]
    for i in range(0, n, batch_size):
        yield X_shuf[i:i + batch_size], y_shuf[i:i + batch_size]


def eval_epoch(model, params, X_val, y_val) -> dict:
    ee_pred = model.apply({'params': params}, X_val, training=False).squeeze(-1)
    mse = float(jnp.mean((ee_pred - y_val) ** 2))
    mae = float(jnp.mean(jnp.abs(ee_pred - y_val)))
    return {"val_mse": mse, "val_mae": mae}


def predict(model, params, X) -> jax.Array:
    return model.apply({'params': params}, X, training=False).squeeze(-1)


def run_groupkfold(X_raw, y, groups, args):
    """Run GroupKFold cross-validation on paper_doi groups.

    Trains a fresh model per fold with a per-fold scaler to prevent leakage.
    Returns aggregate and per-fold metrics in EE% space.
    """
    n_features = X_raw.shape[1]
    gkf = GroupKFold(n_splits=args.n_folds)
    model = build_model(n_features=n_features)

    np_col = FEATURE_INDEX["np_ratio"]
    x_il_col = FEATURE_INDEX["ionizable_lipid_mole_fraction"]
    size_col = FEATURE_INDEX["particle_size_nm"]

    n_groups = int(jnp.unique(groups).shape[0])
    print(f"[PINN] GroupKFold: {args.n_folds} folds | {len(y)} samples | {n_groups} paper groups")
    print(f"[PINN] Training {args.epochs} epochs per fold (alpha={args.alpha})\n")
    gkf_start = time.perf_counter()

    # Single optimizer shared across folds; opt_state is reset per fold
    approx_steps = args.epochs * max(1, len(y) // args.batch_size)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            optax.cosine_decay_schedule(args.lr, approx_steps),
            weight_decay=1e-4,
        ),
    )

    def loss_fn(params, x_batch, y_batch, key):
        dropout_key = random.fold_in(key, 0)
        ee_pred = model.apply(
            {'params': params}, x_batch, training=True,
            rngs={'dropout': dropout_key},
        )
        loss_data = jnp.mean((ee_pred.squeeze(-1) - y_batch) ** 2)
        loss_phys = total_physics_loss(
            model, params, x_batch,
            np_col=np_col, x_il_col=x_il_col,
            size_col=size_col, n_features=n_features,
        )
        return loss_data + args.alpha * loss_phys, (loss_data, loss_phys)

    @jax.jit
    def train_step(params, opt_state, x_batch, y_batch, key):
        (loss, (ld, lp)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, x_batch, y_batch, key)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss, ld, lp

    oof_preds = jnp.zeros(len(y), dtype=jnp.float32)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups)):
        fold_start = time.perf_counter()
        fold_key = random.PRNGKey(args.seed + fold)

        scaler = StandardScaler()
        X_train = jnp.array(scaler.fit_transform(jnp.array(X_raw[train_idx])), dtype=jnp.float32)
        X_val = jnp.array(scaler.transform(jnp.array(X_raw[val_idx])), dtype=jnp.float32)
        y_train = y[train_idx]
        y_val = y[val_idx]

        fold_key, init_key = random.split(fold_key)
        params = model.init(init_key, jnp.ones((1, n_features)))['params']
        opt_state = optimizer.init(params)

        best_val_mse = float("inf")
        best_params = params

        for epoch in range(1, args.epochs + 1):
            fold_key, epoch_key = random.split(fold_key)
            for x_batch, y_batch in batch_iter(X_train, y_train, args.batch_size, epoch_key):
                fold_key, step_key = random.split(fold_key)
                params, opt_state, _, _, _ = train_step(
                    params, opt_state, x_batch, y_batch, step_key
                )

            val_metrics = eval_epoch(model, params, X_val, y_val)
            if val_metrics["val_mse"] < best_val_mse:
                best_val_mse = val_metrics["val_mse"]
                best_params = params

        jax.block_until_ready(best_params)
        fold_elapsed = time.perf_counter() - fold_start
        val_preds = predict(model, best_params, X_val)
        oof_preds = oof_preds.at[val_idx].set(val_preds)

        y_val_pct = jnp.array(y_val) * 100.0
        val_preds_pct = val_preds * 100.0
        fold_rmse = float(jnp.sqrt(mean_squared_error(y_val_pct, val_preds_pct)))
        fold_r2 = float(r2_score(y_val_pct, val_preds_pct))
        fold_mae = float(mean_absolute_error(y_val_pct, val_preds_pct))

        fold_metrics.append({
            "fold": fold + 1,
            "fold_time_s": round(fold_elapsed, 2),
            "rmse_pct": round(fold_rmse, 2),
            "r2": round(fold_r2, 3),
            "mae_pct": round(fold_mae, 2),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
        })
        print(f"  Fold {fold+1}: RMSE={fold_rmse:.2f}% | R2={fold_r2:.3f} | "
              f"MAE={fold_mae:.2f}% | train={len(train_idx)} val={len(val_idx)}")

    y_pct = jnp.array(y) * 100.0
    oof_pct = oof_preds * 100.0
    oof_rmse = float(jnp.sqrt(mean_squared_error(y_pct, oof_pct)))
    oof_r2 = float(r2_score(y_pct, oof_pct))
    oof_mae = float(mean_absolute_error(y_pct, oof_pct))

    gkf_elapsed = time.perf_counter() - gkf_start
    print(f"\n  OOF RMSE: {oof_rmse:.2f}% | OOF R2: {oof_r2:.3f} | OOF MAE: {oof_mae:.2f}%")
    print(f"[PINN] GroupKFold complete: {gkf_elapsed:.2f}s total | {gkf_elapsed / args.n_folds:.2f}s/fold avg")

    return {
        "cv_mode": "groupkfold",
        "n_folds": args.n_folds,
        "n_samples": int(len(y)),
        "n_groups": n_groups,
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
        "training_time_s": round(gkf_elapsed, 2),
        "time_per_fold_s": round(gkf_elapsed / args.n_folds, 2),
    }


def main():
    args = parse_args()
    key = random.PRNGKey(args.seed)

    print(f"[PINN] JAX backend : {jax.default_backend()}")
    print(f"[PINN] JAX devices : {jax.devices()}")
    print(f"[PINN] Loading data: {args.data}")

    if args.cv == "groupkfold":
        X_raw, y, groups = load_and_preprocess_with_groups(args.data)
        results = run_groupkfold(X_raw, y, groups, args)

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

    key, split_key = random.split(key)
    perm = random.permutation(split_key, len(y))
    split = int(0.8 * len(y))
    train_idx, val_idx = perm[:split], perm[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = build_model(n_features=n_features)
    key, init_key = random.split(key)
    params = model.init(init_key, jnp.ones((1, n_features)))['params']

    n_steps = args.epochs * max(1, len(X_train) // args.batch_size)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            optax.cosine_decay_schedule(args.lr, n_steps),
            weight_decay=1e-4,
        ),
    )
    opt_state = optimizer.init(params)

    np_col = FEATURE_INDEX["np_ratio"]
    x_il_col = FEATURE_INDEX["ionizable_lipid_mole_fraction"]
    size_col = FEATURE_INDEX["particle_size_nm"]

    def loss_fn(params, x_batch, y_batch, key):
        dropout_key = random.fold_in(key, 0)
        ee_pred = model.apply(
            {'params': params}, x_batch, training=True,
            rngs={'dropout': dropout_key},
        )
        loss_data = jnp.mean((ee_pred.squeeze(-1) - y_batch) ** 2)
        loss_phys = total_physics_loss(
            model, params, x_batch,
            np_col=np_col, x_il_col=x_il_col,
            size_col=size_col, n_features=n_features,
        )
        return loss_data + args.alpha * loss_phys, (loss_data, loss_phys)

    @jax.jit
    def train_step(params, opt_state, x_batch, y_batch, key):
        (loss, (ld, lp)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, x_batch, y_batch, key)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss, ld, lp

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_mse = float("inf")
    best_params = params

    print(f"[PINN] Training for {args.epochs} epochs (alpha={args.alpha})")
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        key, epoch_key = random.split(key)
        total_loss = total_data = total_phys = 0.0
        n_batches = 0

        for x_batch, y_batch in batch_iter(X_train, y_train, args.batch_size, epoch_key):
            key, step_key = random.split(key)
            params, opt_state, loss, ld, lp = train_step(
                params, opt_state, x_batch, y_batch, step_key
            )
            total_loss += float(loss)
            total_data += float(ld)
            total_phys += float(lp)
            n_batches += 1

        jax.block_until_ready(params)
        epoch_elapsed = time.perf_counter() - epoch_start
        val_metrics = eval_epoch(model, params, X_val, y_val)

        if val_metrics["val_mse"] < best_val_mse:
            best_val_mse = val_metrics["val_mse"]
            best_params = params
            with open(out_dir / "best_params.pkl", "wb") as f:
                pickle.dump(best_params, f)

        row = {
            "epoch": epoch,
            "epoch_time_s": round(epoch_elapsed, 4),
            "loss": total_loss / max(1, n_batches),
            "loss_data": total_data / max(1, n_batches),
            "loss_physics": total_phys / max(1, n_batches),
            **val_metrics,
        }
        history.append(row)

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d} | "
                f"time={epoch_elapsed:.3f}s | "
                f"loss={row['loss']:.4f} "
                f"(data={row['loss_data']:.4f}, "
                f"phys={row['loss_physics']:.4f}) | "
                f"val_mse={val_metrics['val_mse']:.4f} "
                f"val_mae={val_metrics['val_mae']:.4f}"
            )

    train_elapsed = time.perf_counter() - train_start
    print(f"[PINN] Training complete: {train_elapsed:.2f}s total | {train_elapsed / args.epochs:.3f}s/epoch avg")

    with open(out_dir / "pinn_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[PINN] Best val MSE: {best_val_mse:.4f}")
    print(f"[PINN] Checkpoint: {out_dir / 'best_params.pkl'}")
    print(f"[PINN] History:    {out_dir / 'pinn_training_history.json'}")


if __name__ == "__main__":
    main()
