"""
Shared utilities for the Streamlit multipage app.
Cached data loaders, path constants, and plotly styling helpers.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
ARTIFACTS = _ROOT / "artifacts"
DATA_PATH = _ROOT / "data" / "lnp_atlas_cleaned.csv"

# Ensure src/ and project root are importable
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

# ── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#4A90D9",
    "secondary": "#E8735A",
    "accent": "#5CB85C",
    "muted": "#95A5A6",
    "dark": "#2C3E50",
}


# ── Plotly styling ───────────────────────────────────────────────────────────
def clean_plotly_layout(fig, title="", xaxis_title="", yaxis_title=""):
    """Apply consistent, publication-quality styling to a plotly figure."""
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(size=13),
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ddd",
            borderwidth=1,
        ),
    )
    return fig


def format_feature_name(name: str) -> str:
    """Clean up raw feature column names for display."""
    name = name.replace("__", " : ").replace("_std", "")
    name = name.replace("_", " ").title()
    return name


# ── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, encoding="latin-1", low_memory=False)


@st.cache_data
def load_metadata() -> dict:
    with open(ARTIFACTS / "metadata.json") as f:
        return json.load(f)


@st.cache_data
def load_shap_importance() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "shap_importance.csv")


@st.cache_data
def load_shap_values() -> np.ndarray:
    return np.load(ARTIFACTS / "shap_values.npy")


@st.cache_data
def load_pinn_history() -> list:
    with open(ARTIFACTS / "pinn" / "pinn_training_history.json") as f:
        return json.load(f)


@st.cache_data
def load_pinn_metrics() -> dict:
    with open(ARTIFACTS / "pinn" / "pinn_metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_pinn_groupkfold() -> dict:
    with open(ARTIFACTS / "pinn" / "groupkfold_results.json") as f:
        return json.load(f)


@st.cache_resource
def load_xgboost_model():
    """Load the trained XGBoost model, Box-Cox transformer, and metadata."""
    with open(ARTIFACTS / "model.pkl", "rb") as f:
        model = pickle.load(f)
    transformer = None
    transformer_path = ARTIFACTS / "transformer.pkl"
    if transformer_path.exists():
        with open(transformer_path, "rb") as f:
            transformer = pickle.load(f)
    metadata = load_metadata()
    return model, transformer, metadata


@st.cache_resource
def load_pinn_model():
    """Load the trained PINN model (CPU inference)."""
    import torch
    from pinn.model import build_model
    model = build_model(n_features=7, device="cpu")
    state = torch.load(
        ARTIFACTS / "pinn" / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def load_pinn_scaler():
    """Re-derive the StandardScaler by running the PINN preprocessing pipeline."""
    from pinn.preprocess import load_and_preprocess
    _, _, scaler = load_and_preprocess(str(DATA_PATH))
    return scaler
