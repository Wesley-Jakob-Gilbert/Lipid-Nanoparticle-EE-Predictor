"""
preprocess.py — Feature engineering for the LNP EE PINN.

Derives physics-meaningful continuous features from the raw CSV schema
(defined in data_dictionary.md). All features are standardized (z-score)
before being passed to the model.

Feature derivations:
    ionizable_lipid_mole_fraction:
        Parsed from the `molar_ratio` field (e.g. "50:10:38.5:1.5").
        x_IL = IL_parts / total_parts

    np_ratio:
        Nitrogen-to-phosphate molar ratio. Approximated from x_IL and
        particle size as a proxy when direct N/P is not reported.
        N/P ≈ x_IL * (4π(d/2)³/3) * ρ_lipid / (M_lipid * n_phosphate)
        In practice: use x_IL * 10 as a data-driven N/P proxy.
        (Real N/P requires cargo concentration; flag for future work.)

    peg_fraction:
        PEG-lipid mole fraction, parsed from molar_ratio field (4th component).

    cholesterol_fraction:
        Cholesterol mole fraction (3rd component of molar_ratio).
"""

import re
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─── Feature column names (canonical order) ───────────────────────────────────
FEATURE_COLS = [
    "ionizable_lipid_mole_fraction",  # x_IL
    "np_ratio",                        # N/P proxy
    "particle_size_nm",                # DLS Z-average
    "pdi",                             # polydispersity index
    "zeta_mv",                         # surface charge
    "peg_fraction",                    # PEG-lipid mol fraction
    "cholesterol_fraction",            # cholesterol mol fraction
]

# Index map for physics residuals
FEATURE_INDEX = {col: i for i, col in enumerate(FEATURE_COLS)}


def parse_molar_ratio(ratio_str: str) -> Tuple[float, float, float, float]:
    """
    Parse a molar ratio string like '50:10:38.5:1.5' into
    (il_frac, helper_frac, chol_frac, peg_frac).

    Returns fractions (sum ≈ 1.0). Returns (NaN,)*4 on parse failure.
    """
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive physics-meaningful features from the raw extraction schema.

    Args:
        df: Raw DataFrame loaded from CSV.

    Returns:
        DataFrame with FEATURE_COLS as columns.
    """
    feat = pd.DataFrame()

    # Parse molar ratios
    ratios = df["molar_ratio"].apply(parse_molar_ratio)
    ratio_df = pd.DataFrame(
        ratios.tolist(),
        columns=["ionizable_lipid_mole_fraction", "helper_frac",
                 "cholesterol_fraction", "peg_fraction"]
    )

    feat["ionizable_lipid_mole_fraction"] = ratio_df["ionizable_lipid_mole_fraction"]
    feat["cholesterol_fraction"] = ratio_df["cholesterol_fraction"]
    feat["peg_fraction"] = ratio_df["peg_fraction"]

    # N/P proxy: x_IL * 10 (literature range ~4–12 for typical LNPs)
    feat["np_ratio"] = feat["ionizable_lipid_mole_fraction"] * 10.0

    # Physical characterization
    feat["particle_size_nm"] = pd.to_numeric(df["particle_size_nm"], errors="coerce")
    feat["pdi"] = pd.to_numeric(df["pdi"], errors="coerce")

    # Zeta potential: strip stray leading quote artifacts (e.g. "'-5.4" → -5.4)
    zeta_raw = df["zeta_mV"].astype(str).str.replace("'", "", regex=False)
    feat["zeta_mv"] = pd.to_numeric(zeta_raw, errors="coerce")

    return feat[FEATURE_COLS]


def load_and_preprocess(
    csv_path: str,
    ee_col: str = "uptake_value",
    ee_unit_col: str = "uptake_unit",
    ee_metric_col: str = "uptake_metric_name",
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Load CSV, filter to EE rows, engineer features, and standardize.

    Args:
        csv_path:      Path to the extraction CSV.
        ee_col:        Column holding the numeric EE value.
        ee_unit_col:   Column holding the unit (should be '%').
        ee_metric_col: Column holding the metric name.

    Returns:
        X:      Standardized feature array, shape (N, n_features).
        y:      EE targets normalized to [0, 1], shape (N,).
        scaler: Fitted StandardScaler (for inverse-transform at inference).
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Filter to encapsulation efficiency rows
    ee_mask = df[ee_metric_col].str.lower().str.contains(
        "encapsulation", na=False
    )
    df_ee = df[ee_mask].copy()

    if len(df_ee) == 0:
        # Fallback: treat all rows as EE data (e.g. sample_seed.csv)
        warnings.warn(
            f"No 'encapsulation efficiency' rows found in {csv_path}. "
            "Using all rows. Check uptake_metric_name column.",
            UserWarning,
        )
        df_ee = df.copy()

    # Parse EE values → [0, 1]
    ee_vals = pd.to_numeric(df_ee[ee_col], errors="coerce")
    unit_vals = df_ee[ee_unit_col].astype(str).str.strip()

    # Convert % → fraction
    is_percent = unit_vals.str.contains("%", na=False)
    ee_frac = np.where(is_percent, ee_vals / 100.0, ee_vals)

    # Feature engineering
    feat_df = engineer_features(df_ee)

    # Combine and drop rows with any NaN
    feat_df["ee"] = ee_frac
    feat_df = feat_df.dropna()

    if len(feat_df) == 0:
        raise ValueError(
            f"No valid rows remain after feature engineering and NaN-drop for {csv_path}."
        )

    X_raw = feat_df[FEATURE_COLS].values.astype(np.float32)
    y = feat_df["ee"].values.astype(np.float32)

    # Clip EE to [0, 1] (handle any >100% measurement artifacts)
    y = np.clip(y, 0.0, 1.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    return X, y, scaler


# ─────────────────────────────────────────────────────────────────────────────
# LNP Atlas adapter (for the OneDrive project / public repo)
# Maps LNP Atlas column names → PINN feature columns
# ─────────────────────────────────────────────────────────────────────────────

def _parse_molar_ratio_atlas(ratio_str: str) -> Tuple[float, float, float, float]:
    """Parse 'IL:helper:sterol:PEG' or any 4-component ratio from LNP Atlas."""
    try:
        parts = [float(x) for x in str(ratio_str).split(":")]
        if len(parts) < 2:
            return (np.nan,) * 4
        # Pad to 4 if shorter
        while len(parts) < 4:
            parts.append(0.0)
        total = sum(parts[:4])
        if total == 0:
            return (np.nan,) * 4
        fracs = [p / total for p in parts[:4]]
        return tuple(fracs)  # (il, helper, sterol/chol, peg)
    except (ValueError, AttributeError):
        return (np.nan,) * 4


def load_and_preprocess_df(
    df: pd.DataFrame,
    target_col: str = "encapsulation_efficiency_percent_std",
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Adapter for LNP Atlas DataFrames (column names differ from extraction schema).

    Args:
        df:         DataFrame loaded from lnp_atlas_export.csv (already filtered for EE rows).
        target_col: Column containing EE% values (0–100 scale).

    Returns:
        X:      Standardized feature array, shape (N, n_features).
        y:      EE targets normalized to [0, 1], shape (N,).
        scaler: Fitted StandardScaler.
    """
    feat = pd.DataFrame()

    # Parse molar ratios from LNP Atlas 'lipid_molar_ratio' column
    ratios = df["lipid_molar_ratio"].apply(_parse_molar_ratio_atlas)
    ratio_df = pd.DataFrame(
        ratios.tolist(),
        columns=["ionizable_lipid_mole_fraction", "helper_frac",
                 "cholesterol_fraction", "peg_fraction"],
        index=df.index,
    )

    feat["ionizable_lipid_mole_fraction"] = ratio_df["ionizable_lipid_mole_fraction"]
    feat["cholesterol_fraction"] = ratio_df["cholesterol_fraction"]
    feat["peg_fraction"] = ratio_df["peg_fraction"]

    # N/P proxy
    feat["np_ratio"] = feat["ionizable_lipid_mole_fraction"] * 10.0

    # Physical measurements
    feat["particle_size_nm"] = pd.to_numeric(df["particle_size_nm_std"], errors="coerce")
    feat["pdi"] = pd.to_numeric(df["pdi_std"], errors="coerce")
    feat["zeta_mv"] = pd.to_numeric(df["zeta_potential_mv_std"], errors="coerce")

    # Target: EE% → [0, 1]
    ee_vals = pd.to_numeric(df[target_col], errors="coerce") / 100.0
    feat["ee"] = ee_vals.values

    feat = feat.dropna()

    if len(feat) == 0:
        raise ValueError("No valid rows after feature engineering — check column names.")

    X_raw = feat[FEATURE_COLS].values.astype(np.float32)
    y = np.clip(feat["ee"].values.astype(np.float32), 0.0, 1.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    return X, y, scaler
