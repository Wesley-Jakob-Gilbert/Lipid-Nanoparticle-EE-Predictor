"""
preprocess.py — Feature engineering for the LNP EE PINN.
Adapted for the LNP Atlas schema (lnp_atlas_export.csv).

Features derived:
    ionizable_lipid_mole_fraction  — IL fraction from lipid_molar_ratio
    np_ratio                       — N/P proxy (x_IL * 10)
    particle_size_nm               — from particle_size_nm_std
    pdi                            — from pdi_std
    zeta_mv                        — from zeta_potential_mv_std
    peg_fraction                   — from lipid_molar_ratio (2nd component)
    cholesterol_fraction           — from lipid_molar_ratio (3rd component)
"""

import re
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "ionizable_lipid_mole_fraction",
    "np_ratio",
    "particle_size_nm",
    "pdi",
    "zeta_mv",
    "peg_fraction",
    "cholesterol_fraction",
]

FEATURE_INDEX = {col: i for i, col in enumerate(FEATURE_COLS)}


def parse_molar_ratio(ratio_str):
    try:
        parts = [float(x) for x in str(ratio_str).split(":")]
        if len(parts) != 4:
            return (np.nan,) * 4
        total = sum(parts)
        if total == 0:
            return (np.nan,) * 4
        return tuple(p / total for p in parts)
    except (ValueError, AttributeError):
        return (np.nan,) * 4


def clean_numeric(series):
    """Strip ~ and other non-numeric chars, coerce to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[~<>≈]", "", regex=True).str.strip(),
        errors="coerce"
    )


def load_and_preprocess(
    csv_path: str,
    **kwargs  # ignored; kept for API compat
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    # Filter to rows with EE
    ee_raw = clean_numeric(df["encapsulation_efficiency_percent_std"])
    mask = ee_raw.notna()
    df = df[mask].copy()
    ee_vals = ee_raw[mask].values.astype(np.float32)

    # Normalize EE: values are in percent (0–100) → fraction [0,1]
    y = np.clip(ee_vals / 100.0, 0.0, 1.0)

    # Feature engineering
    ratios = df["lipid_molar_ratio"].apply(parse_molar_ratio)
    ratio_df = pd.DataFrame(
        ratios.tolist(),
        columns=["ionizable_lipid_mole_fraction", "peg_fraction",
                 "cholesterol_fraction", "helper_frac"],
        index=df.index,
    )

    feat = pd.DataFrame(index=df.index)
    feat["ionizable_lipid_mole_fraction"] = ratio_df["ionizable_lipid_mole_fraction"]
    feat["np_ratio"] = feat["ionizable_lipid_mole_fraction"] * 10.0
    feat["particle_size_nm"] = clean_numeric(df["particle_size_nm_std"])
    feat["pdi"] = clean_numeric(df["pdi_std"])
    feat["zeta_mv"] = clean_numeric(df["zeta_potential_mv_std"])
    feat["peg_fraction"] = ratio_df["peg_fraction"]
    feat["cholesterol_fraction"] = ratio_df["cholesterol_fraction"]
    feat["ee"] = y

    feat = feat[FEATURE_COLS + ["ee"]].dropna()

    if len(feat) < 10:
        raise ValueError(f"Only {len(feat)} valid rows after NaN-drop — check your CSV.")

    X_raw = feat[FEATURE_COLS].values.astype(np.float32)
    y_clean = feat["ee"].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    print(f"[preprocess] {len(y_clean)} valid rows | EE mean={y_clean.mean():.3f} std={y_clean.std():.3f}")
    return X, y_clean, scaler
