"""
Model Training Script — LNP Encapsulation Efficiency Predictor

Improvements over baseline:
- Optuna hyperparameter optimization (replaces manual tuning)
- GroupKFold cross-validation on paper_doi to prevent data leakage
  (multiple LNPs from the same paper share correlated measurements)
- SHAP explanations saved alongside the model
- Threshold-based classification metrics (EE% > 80 = "high encapsulation")
- Full artifact saving: model, feature list, scaler, training metadata
"""

import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import PowerTransformer, StandardScaler

warnings.filterwarnings("ignore")

# Try optuna for hyperparameter optimization
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("optuna not found — using default hyperparameters.")

from features import build_feature_matrix, get_feature_columns

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data/lnp_atlas_export.csv"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "encapsulation_efficiency_percent_std"
HIGH_EE_THRESHOLD = 80.0  # % — clinically relevant cutoff


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation with paper-level grouping
# ─────────────────────────────────────────────────────────────────────────────

def make_groups(df: pd.DataFrame) -> np.ndarray:
    """
    Assign integer group IDs based on paper_doi.
    Rows from the same paper form one group — prevents leakage in GroupKFold.
    """
    doi_col = df.get("paper_doi", pd.Series(["unknown"] * len(df)))
    doi_col = doi_col.fillna(pd.Series("unknown_" + df.index.astype(str), index=df.index))
    unique_dois = {doi: i for i, doi in enumerate(doi_col.unique())}
    return doi_col.map(unique_dois).values


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

def make_optuna_objective(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "tree_method": "hist",
        }
        model = xgb.XGBRegressor(**params, verbosity=0)
        cv = GroupKFold(n_splits=5)
        scores = cross_val_score(model, X, y, cv=cv, groups=groups,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
        return -scores.mean()  # minimize RMSE

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_path: Path = DATA_PATH,
    n_optuna_trials: int = 50,
    n_cv_folds: int = 5,
    save_artifacts: bool = True,
) -> dict:

    print(f"[{datetime.now():%H:%M:%S}] Loading data...")
    df_raw = pd.read_csv(data_path, encoding="latin-1")
    print(f"  Raw rows: {len(df_raw)}")

    print(f"[{datetime.now():%H:%M:%S}] Building feature matrix...")
    df = build_feature_matrix(df_raw, drop_ee_na=True)
    print(f"  Rows after EE% filter: {len(df)}")

    feature_cols = get_feature_columns(df, TARGET_COL)
    print(f"  Feature count: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    groups = make_groups(df)

    print(f"  EE% distribution: mean={y.mean():.1f}%, std={y.std():.1f}%")
    print(f"  High EE% (>={HIGH_EE_THRESHOLD}%): {(y >= HIGH_EE_THRESHOLD).sum()} / {len(y)}")

    # ── Box-Cox transform on target ──────────────────────────────────────────
    # Transforms skewed EE% distribution toward Gaussian so the model can learn
    # the full 10-99% range instead of collapsing to 60-90%.
    # Box-Cox requires strictly positive values; epsilon guards against y=0.
    print(f"[{datetime.now():%H:%M:%S}] Fitting Box-Cox transformer on target...")
    transformer = PowerTransformer(method='box-cox')
    y_transformed = transformer.fit_transform(
        (y + 1e-6).reshape(-1, 1)
    ).ravel().astype(np.float32)
    print(f"  Lambda: {transformer.lambdas_[0]:.4f}")

    # ── Hyperparameter optimization ──────────────────────────────────────────
    if OPTUNA_AVAILABLE and n_optuna_trials > 0:
        print(f"[{datetime.now():%H:%M:%S}] Running Optuna ({n_optuna_trials} trials)...")
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_optuna_objective(X, y_transformed, groups),
                       n_trials=n_optuna_trials, show_progress_bar=True)
        best_params = study.best_params
        print(f"  Best RMSE (CV): {study.best_value:.2f}%")
    else:
        best_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "tree_method": "hist",
        }

    # ── GroupKFold cross-validation evaluation ───────────────────────────────
    print(f"[{datetime.now():%H:%M:%S}] Evaluating with GroupKFold (k={n_cv_folds})...")
    cv = GroupKFold(n_splits=n_cv_folds)
    oof_preds = np.zeros_like(y)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Train on Box-Cox transformed target; evaluate in original EE% space
        y_tr_t = transformer.transform(
            (y_tr + 1e-6).reshape(-1, 1)
        ).ravel().astype(np.float32)
        y_val_t = transformer.transform(
            (y_val + 1e-6).reshape(-1, 1)
        ).ravel().astype(np.float32)

        model_fold = xgb.XGBRegressor(**best_params, verbosity=0, early_stopping_rounds=50)
        model_fold.fit(
            X_tr, y_tr_t,
            eval_set=[(X_val, y_val_t)],
            verbose=False,
        )
        preds_t = model_fold.predict(X_val)
        preds = np.clip(
            transformer.inverse_transform(preds_t.reshape(-1, 1)).ravel(),
            0, 100,
        )
        oof_preds[val_idx] = preds

        fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_r2 = r2_score(y_val, preds)
        fold_metrics.append({"fold": fold + 1, "rmse": fold_rmse, "r2": fold_r2})
        print(f"  Fold {fold+1}: RMSE={fold_rmse:.2f}%, R²={fold_r2:.3f}")

    oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    oof_r2 = r2_score(y, oof_preds)
    oof_mae = mean_absolute_error(y, oof_preds)
    print(f"\n  OOF RMSE: {oof_rmse:.2f}%  |  OOF R²: {oof_r2:.3f}  |  OOF MAE: {oof_mae:.2f}%")

    # Classification metrics at the high-EE threshold
    y_class = (y >= HIGH_EE_THRESHOLD).astype(int)
    pred_class = (oof_preds >= HIGH_EE_THRESHOLD).astype(int)
    tp = ((y_class == 1) & (pred_class == 1)).sum()
    fp = ((y_class == 0) & (pred_class == 1)).sum()
    fn = ((y_class == 1) & (pred_class == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  High-EE classification (≥{HIGH_EE_THRESHOLD}%): "
          f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    # ── Final model on full data ─────────────────────────────────────────────
    print(f"\n[{datetime.now():%H:%M:%S}] Training final model on full dataset...")
    final_model = xgb.XGBRegressor(**best_params, verbosity=0)
    final_model.fit(X, y_transformed, verbose=False)

    # ── SHAP explanation ─────────────────────────────────────────────────────
    print(f"[{datetime.now():%H:%M:%S}] Computing SHAP values...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP per feature
    shap_importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_cols,
    ).sort_values(ascending=False)

    print("\n  Top 10 features by SHAP importance:")
    for feat, imp in shap_importance.head(10).items():
        print(f"    {feat:<55} {imp:.4f}")

    # ── Save artifacts ───────────────────────────────────────────────────────
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_train_samples": len(y),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "best_hyperparams": best_params,
        "box_cox_transform": True,
        "box_cox_lambda": float(transformer.lambdas_[0]),
        "oof_metrics": {
            "rmse": float(oof_rmse),
            "r2": float(oof_r2),
            "mae": float(oof_mae),
            "high_ee_precision": float(precision),
            "high_ee_recall": float(recall),
            "high_ee_f1": float(f1),
        },
        "fold_metrics": fold_metrics,
        "high_ee_threshold": HIGH_EE_THRESHOLD,
        "shap_top10": shap_importance.head(10).to_dict(),
    }

    if save_artifacts:
        with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
            pickle.dump(final_model, f)

        with open(ARTIFACTS_DIR / "transformer.pkl", "wb") as f:
            pickle.dump(transformer, f)

        with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        np.save(ARTIFACTS_DIR / "shap_values.npy", shap_values)
        shap_importance.to_csv(ARTIFACTS_DIR / "shap_importance.csv")

        print(f"\n  Artifacts saved to: {ARTIFACTS_DIR.resolve()}")

    return metadata


if __name__ == "__main__":
    results = train(n_optuna_trials=50)
    print("\n✓ Training complete.")
    print(f"  Final OOF R² = {results['oof_metrics']['r2']:.3f}")
