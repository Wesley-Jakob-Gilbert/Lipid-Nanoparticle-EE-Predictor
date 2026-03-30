"""
preprocess.py — Feature engineering for LNP EE PINN.

Supports two feature sets:
  - FEATURE_COLS (6 features, no zeta): 327 usable rows from cleaned LNP Atlas
  - FEATURE_COLS_WITH_ZETA (7 features): 101 usable rows

The no-zeta set is the default and recommended for training due to 3x more data.
Physics residuals R1, R2, R3 do not use zeta, so physics loss is unaffected.
"""

import re
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─── Feature sets ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "ionizable_lipid_mole_fraction",   # x_IL
    "np_ratio",                         # N/P proxy
    "particle_size_nm",                 # DLS Z-average
    "pdi",                              # polydispersity index
    "peg_fraction",                     # PEG-lipid mol fraction
    "cholesterol_fraction",             # cholesterol mol fraction
]

FEATURE_COLS_WITH_ZETA = [
    "ionizable_lipid_mole_fraction",
    "np_ratio",
    "particle_size_nm",
    "pdi",
    "zeta_mv",
    "peg_fraction",
    "cholesterol_fraction",
]

# Index maps used by physics residuals
FEATURE_INDEX = {col: i for i, col in enumerate(FEATURE_COLS)}
FEATURE_INDEX_WITH_ZETA = {col: i for i, col in enumerate(FEATURE_COLS_WITH_ZETA)}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_molar_ratio(ratio_str):
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


def _clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[~<>≈]", "", regex=True).str.strip(),
        errors="coerce"
    )


def _parse_ee_robust(val):
    """Robust EE parser handling ± noise, ranges, and dual-assay entries."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    try:
        return float(s)
    except:
        pass
    s2 = re.sub(r'^[><=~≈?¿]+', '', s).strip()
    try:
        return float(s2)
    except:
        pass
    # ± or ¡¾ encoded ±
    pm = re.match(r'^([\d.]+)\s*(?:¡¾|±|\+/-|--)\s*[\d.]+', s)
    if pm:
        try:
            return float(pm.group(1))
        except:
            pass
    # dual-assay: first number before '('
    semi = re.match(r'^([\d.]+)\s*\(', s)
    if semi:
        try:
            return float(semi.group(1))
        except:
            pass
    # range: midpoint
    rng = re.match(r'^([\d.]+)\s*[-–]\s*([\d.]+)$', s2)
    if rng:
        try:
            return (float(rng.group(1)) + float(rng.group(2))) / 2.0
        except:
            pass
    return None


# ─── Main loader ──────────────────────────────────────────────────────────────

def load_and_preprocess(
    csv_path: str,
    include_zeta: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Load CSV, engineer features, standardize.

    Args:
        csv_path:     Path to lnp_atlas_cleaned.csv or lnp_atlas_export.csv
        include_zeta: If True, include zeta_mv as a feature (fewer rows).
                      Default False (recommended — 3x more data).

    Returns:
        X:      Standardized feature array, shape (N, n_features).
        y:      EE targets in [0, 1], shape (N,).
        scaler: Fitted StandardScaler.
    """
    df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    # Use pre-cleaned EE column if available, else parse raw
    if "encapsulation_efficiency_clean" in df.columns:
        ee_vals = pd.to_numeric(df["encapsulation_efficiency_clean"], errors="coerce")
    else:
        ee_vals = df["encapsulation_efficiency_percent_std"].apply(_parse_ee_robust)
        ee_vals = pd.to_numeric(ee_vals, errors="coerce")

    # Parse molar ratios
    ratios = df["lipid_molar_ratio"].apply(_parse_molar_ratio)
    ratio_df = pd.DataFrame(
        ratios.tolist(),
        columns=["il_frac", "peg_frac", "chol_frac", "helper_frac"],
        index=df.index,
    )

    feat = pd.DataFrame(index=df.index)
    feat["ionizable_lipid_mole_fraction"] = ratio_df["il_frac"]
    feat["np_ratio"] = feat["ionizable_lipid_mole_fraction"] * 10.0
    feat["particle_size_nm"] = _clean_numeric(df["particle_size_nm_std"])
    feat["pdi"] = _clean_numeric(df["pdi_std"])
    if include_zeta:
        feat["zeta_mv"] = _clean_numeric(df["zeta_potential_mv_std"])
    feat["peg_fraction"] = ratio_df["peg_frac"]
    feat["cholesterol_fraction"] = ratio_df["chol_frac"]
    feat["ee"] = np.clip(ee_vals / 100.0, 0.0, 1.0)

    feat_cols = FEATURE_COLS_WITH_ZETA if include_zeta else FEATURE_COLS
    feat = feat[feat_cols + ["ee"]].dropna()

    if len(feat) < 10:
        raise ValueError(f"Only {len(feat)} valid rows — check CSV path and columns.")

    X_raw = feat[feat_cols].values.astype(np.float32)
    y = feat["ee"].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    zeta_note = "with zeta" if include_zeta else "no zeta"
    print(f"[preprocess] {len(y)} rows | {len(feat_cols)} features ({zeta_note}) | "
          f"EE mean={y.mean():.3f} std={y.std():.3f}")
    return X, y, scaler

