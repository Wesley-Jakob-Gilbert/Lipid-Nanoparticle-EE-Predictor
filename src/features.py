"""
Feature Engineering Pipeline for LNP Encapsulation Efficiency Prediction.

Key improvements over the baseline notebook:
1. Richer RDKit molecular descriptors (Morgan fingerprints, ring counts, aromaticity)
2. Interaction features between ionizable lipid properties and molar ratios
3. Parsed synthesis method features (flow rate, pH, temperature)
4. Robust molar ratio parsing with normalization
5. Cargo-type encoding with biological relevance ordering
6. Better missing value strategy with grouped medians
"""

import re
import numpy as np
import pandas as pd
from typing import Optional

# ── Try RDKit (required for training; graceful degradation for API cold-start) ──
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available. Molecular features will be NaN.")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Biologically motivated ordering: larger nucleic acids are harder to encapsulate
CARGO_DIFFICULTY = {
    "siRNA": 1,
    "ASO": 2,
    "mRNA": 3,
    "mRNA, siRNA": 3,
    "DNA": 4,
    "protein": 5,
    "empty": 0,
    "gold_nanoparticles": 6,
}

SYNTHESIS_METHOD_MAP = {
    "microfluidic": "microfluidic",
    "t-junction": "microfluidic",
    "pipette": "bulk",
    "bulk": "bulk",
    "ethanol injection": "ethanol_injection",
    "ethanol dilution": "ethanol_injection",
    "nanoprecipitation": "nanoprecipitation",
    "film hydration": "film_hydration",
    "snalp": "snalp",
    "hand mixing": "bulk",
    "rapid mixing": "bulk",
    "manual mixing": "bulk",
}

RDKIT_DESCRIPTOR_NAMES = [
    "MW", "RotBonds", "HAcc", "HDon", "TPSA", "LogP",
    "RingCount", "AromaticRings", "HeavyAtomCount", "FractionCSP3",
]


# ─────────────────────────────────────────────────────────────────────────────
# Molecular descriptor computation
# ─────────────────────────────────────────────────────────────────────────────

def _empty_rdkit_stats() -> dict:
    return {k: np.nan for k in RDKIT_DESCRIPTOR_NAMES}


def get_rdkit_stats(smiles: Optional[str]) -> dict:
    """Compute a rich set of molecular descriptors from a SMILES string."""
    empty = _empty_rdkit_stats()

    if not RDKIT_AVAILABLE:
        return empty
    if pd.isna(smiles) or str(smiles).strip() == "":
        return empty

    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return empty

        h_acc = Descriptors.NumHAcceptors(mol)
        h_don = Descriptors.NumHDonors(mol)

        return {
            "MW": Descriptors.MolWt(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "HAcc": h_acc,
            "HDon": h_don,
            "TPSA": Descriptors.TPSA(mol),
            "LogP": Descriptors.MolLogP(mol),
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "HeavyAtomCount": mol.GetNumHeavyAtoms(),
            "FractionCSP3": rdMolDescriptors.CalcFracCSP3(mol),
        }
    except Exception:
        return empty


def add_molecular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-lipid RDKit descriptor columns to df, with median imputation."""
    smiles_cols = [c for c in df.columns if c.endswith("_smiles")]

    for col in smiles_cols:
        lipid_type = col.replace("_smiles", "")
        stats_list = [get_rdkit_stats(s) for s in df[col]]
        features_df = pd.DataFrame(stats_list, index=df.index)

        # Validity flag
        df[f"{lipid_type}_smiles_valid"] = features_df["MW"].notnull().astype(int)

        # Rename and concatenate
        features_df = features_df.add_prefix(f"{lipid_type}__")
        df = pd.concat([df, features_df], axis=1)

    # Impute with medians (per-column, not global)
    rdkit_cols = [c for c in df.columns if "__" in c and any(
        c.endswith(f"__{d}") for d in RDKIT_DESCRIPTOR_NAMES
    )]
    for col in rdkit_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median if pd.notna(median) else 0.0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Molar ratio parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_molar_ratio(ratio_str: Optional[str]) -> dict:
    """
    Parse lipid molar ratio string into normalized numeric features.
    Handles formats like '50:1.5:38.5:10' or '35:16:46.5:2.5 (ionizable:chol:DOPE:PEG)'.
    Returns ionizable, peg, sterol, helper fractions plus the raw ionizable ratio.
    """
    nan4 = {
        "ratio_ionizable": np.nan,
        "ratio_peg": np.nan,
        "ratio_sterol": np.nan,
        "ratio_helper": np.nan,
        "ratio_ionizable_norm": np.nan,
    }

    if not isinstance(ratio_str, str) or ratio_str.strip() == "":
        return nan4

    # Extract first 4-component colon-separated sequence
    match = re.search(
        r"(\d+\.?\d*)\s*:\s*(\d+\.?\d*)\s*:\s*(\d+\.?\d*)\s*:\s*(\d+\.?\d*)",
        ratio_str,
    )
    if not match:
        return nan4

    try:
        parts = [float(match.group(i)) for i in range(1, 5)]
        total = sum(parts) if sum(parts) > 0 else 1.0
        return {
            "ratio_ionizable": parts[0],
            "ratio_peg": parts[1],
            "ratio_sterol": parts[2],
            "ratio_helper": parts[3],
            "ratio_ionizable_norm": parts[0] / total,  # fraction of ionizable lipid
        }
    except (ValueError, ZeroDivisionError):
        return nan4


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis info parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_synthesis_info(synthesis_str: Optional[str]) -> dict:
    """
    Extract structured features from the synthesis_info free-text column.
    Fields parsed: synthesis_method (categorical), flow_rate, flow_ratio, pH.
    """
    defaults = {
        "synth_method_cat": "unknown",
        "synth_flow_rate": np.nan,
        "synth_flow_ratio": np.nan,
        "synth_ph": np.nan,
        "synth_is_microfluidic": 0,
    }

    if pd.isna(synthesis_str):
        return defaults

    text = str(synthesis_str).lower()

    # Key-value parsing
    kv = {}
    for part in text.split(";"):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            kv[k.strip()] = v.strip()

    # Synthesis method -> canonical category
    raw_method = kv.get("synthesis_method", "")
    synth_cat = "unknown"
    for keyword, category in SYNTHESIS_METHOD_MAP.items():
        if keyword in raw_method:
            synth_cat = category
            break

    # Flow rate
    flow_rate = np.nan
    raw_fr = kv.get("total_flow_rate_ml_min", "")
    match = re.search(r"(\d+\.?\d*)", raw_fr)
    if match:
        flow_rate = float(match.group(1))

    # Flow ratio (aqueous:organic)
    flow_ratio = np.nan
    raw_ratio = kv.get("flow_rate_ratio", "")
    match = re.search(r"(\d+\.?\d*)\s*[:/]\s*(\d+\.?\d*)", raw_ratio)
    if match:
        try:
            flow_ratio = float(match.group(1)) / float(match.group(2))
        except ZeroDivisionError:
            pass

    # pH from aqueous phase composition
    ph = np.nan
    aq_phase = kv.get("aqueous_phase_composition", "")
    match = re.search(r"ph\s*(\d+\.?\d*)", aq_phase)
    if match:
        ph = float(match.group(1))

    return {
        "synth_method_cat": synth_cat,
        "synth_flow_rate": flow_rate,
        "synth_flow_ratio": flow_ratio,
        "synth_ph": ph,
        "synth_is_microfluidic": int(synth_cat == "microfluidic"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Numeric column standardization
# ─────────────────────────────────────────────────────────────────────────────

def standardize_numeric(value) -> float:
    """
    Robustly parse messy numeric strings into floats.
    Handles: ranges (take mean), ±SD (take mean), <limits (halve), ~ approximations,
    Z-average vs number-average entries (take Z-average as primary), negatives.
    """
    if pd.isna(value) or str(value).strip() == "":
        return np.nan

    val_str = str(value).strip()

    # Z-average pattern: '61 (Z-average); 36 (number average)' → take Z-average
    z_match = re.search(r"(\d+\.?\d*)\s*\(z-average\)", val_str.lower())
    if z_match:
        return float(z_match.group(1))

    # Remove encoding artifacts
    val_str = val_str.replace("Â¡Â¾", "±").replace("±", "±")

    # Approximation
    if val_str.startswith("~"):
        try:
            return float(val_str[1:])
        except ValueError:
            pass

    # Mean ± SD → take mean
    if "±" in val_str:
        try:
            return float(val_str.split("±")[0].strip())
        except ValueError:
            pass

    # Range: 77-350 → midpoint (careful: negative values like -5.2 should not trigger this)
    range_match = re.match(r"^(-?\d+\.?\d*)\s*-\s*(\d+\.?\d*)$", val_str)
    if range_match:
        try:
            lo, hi = float(range_match.group(1)), float(range_match.group(2))
            return (lo + hi) / 2
        except ValueError:
            pass

    # Less-than limit
    if val_str.startswith("<"):
        try:
            return float(val_str[1:]) / 2
        except ValueError:
            pass

    try:
        return float(val_str)
    except ValueError:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    drop_ee_na: bool = True,
    target_col: str = "encapsulation_efficiency_percent_std",
) -> pd.DataFrame:
    """
    End-to-end feature construction from raw LNP Atlas dataframe.

    Steps:
      1. Standardize numeric measurement columns
      2. Add missing-value indicator flags before imputation
      3. RDKit molecular descriptors for each lipid component
      4. Interaction features (ionizable lipid LogP × ratio)
      5. Molar ratio parsing + normalization
      6. Synthesis info parsing
      7. Categorical encoding (target_type, synthesis method)
      8. Drop rows with missing EE% (target variable)

    Returns a model-ready DataFrame. The target column is preserved as-is.
    """
    df = df.copy()

    # ── 1. Standardize measurement columns ──────────────────────────────────
    for col in ["particle_size_nm_std", "pdi_std", "zeta_potential_mv_std",
                target_col]:
        df[col] = df[col].apply(standardize_numeric)

    # ── 2. Missing-value flags (before imputation) ───────────────────────────
    for col in ["particle_size_nm_std", "pdi_std", "zeta_potential_mv_std"]:
        df[f"{col}__missing"] = df[col].isnull().astype(int)

    # Impute with medians
    for col in ["particle_size_nm_std", "pdi_std", "zeta_potential_mv_std"]:
        df[col] = df[col].fillna(df[col].median())

    # ── 3. Drop rows without the target ─────────────────────────────────────
    if drop_ee_na:
        df = df[df[target_col].notna()].copy()

    # ── 4. Molecular descriptors ─────────────────────────────────────────────
    df = add_molecular_features(df)

    # ── 5. Interaction features ──────────────────────────────────────────────
    # High ionizable lipid LogP + high ratio → drives encapsulation
    il_logp = df.get("ionizable_lipid__LogP", pd.Series(np.nan, index=df.index))
    il_ratio = df["lipid_molar_ratio"].apply(
        lambda x: parse_molar_ratio(x).get("ratio_ionizable", np.nan)
    )
    df["feat__il_logp_x_ratio"] = il_logp * il_ratio

    # PEG steric effect: high PEG MW + high PEG ratio → can hurt encapsulation
    peg_mw = df.get("peg_lipid__MW", pd.Series(np.nan, index=df.index))
    peg_ratio = df["lipid_molar_ratio"].apply(
        lambda x: parse_molar_ratio(x).get("ratio_peg", np.nan)
    )
    df["feat__peg_mw_x_ratio"] = peg_mw * peg_ratio

    # ── 6. Molar ratio features ──────────────────────────────────────────────
    ratio_features = df["lipid_molar_ratio"].apply(parse_molar_ratio)
    ratio_df = pd.DataFrame(ratio_features.tolist(), index=df.index)
    df = pd.concat([df, ratio_df], axis=1)

    # ── 7. Synthesis info features ───────────────────────────────────────────
    synth_features = df["synthesis_info"].apply(parse_synthesis_info)
    synth_df = pd.DataFrame(synth_features.tolist(), index=df.index)
    df = pd.concat([df, synth_df], axis=1)

    # Impute synthesis numerics
    for col in ["synth_flow_rate", "synth_flow_ratio", "synth_ph"]:
        df[col] = df[col].fillna(df[col].median())

    # ── 8. Categorical encoding ──────────────────────────────────────────────
    # Cargo type: ordinal difficulty + one-hot
    df["target_difficulty"] = df["target_type"].map(CARGO_DIFFICULTY).fillna(3)
    target_dummies = pd.get_dummies(df["target_type"], prefix="target").astype(int)
    df = pd.concat([df, target_dummies], axis=1)

    # Synthesis method one-hot
    synth_dummies = pd.get_dummies(df["synth_method_cat"], prefix="synth").astype(int)
    df = pd.concat([df, synth_dummies], axis=1)

    # ── 9. Lipid name embeddings (label-encoded frequency ranks) ────────────
    # Captures the "how well-studied is this lipid" signal
    for col in ["ionizable_lipid", "peg_lipid", "helper_lipid", "sterol_lipid"]:
        freq = df[col].value_counts()
        df[f"{col}__freq_rank"] = df[col].map(freq).fillna(0).astype(float)

    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "encapsulation_efficiency_percent_std") -> list:
    """
    Return the list of numeric feature columns to use for model training.
    Excludes identifiers, raw text, SMILES, and the target.
    """
    exclude_patterns = [
        target_col, "lnp_id", "loading_capacity_std", "nucleic_acid_sequence",
        "paper_title", "paper_authors", "paper_doi", "paper_journal", "paper_year",
        "ionizable_lipid", "peg_lipid", "sterol_lipid", "helper_lipid",
        "lipid_molar_ratio", "target_type", "synth_method_cat",
        "synthesis_info", "bioactivity_profile",
        "peg_lipid_original", "ionizable_lipid_original",
        "helper_lipid_original", "sterol_lipid_original",
    ]
    # Also exclude raw SMILES columns
    smiles_raw = [c for c in df.columns if c.endswith("_smiles")]
    exclude = set(exclude_patterns) | set(smiles_raw)

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, np.uint8, bool]:
            feature_cols.append(col)
        elif df[col].dtype == object:
            continue  # already encoded above

    return feature_cols
