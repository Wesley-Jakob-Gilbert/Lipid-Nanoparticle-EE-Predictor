import streamlit as st
import sys
from pathlib import Path

# Robustly find project root regardless of whether CWD is project root or pages/
_HERE = Path(__file__).resolve().parent
_ROOT = next(
    (p for p in [_HERE, _HERE.parent, _HERE.parent.parent] if (p / 'src').exists()),
    _HERE,
)
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from features import build_feature_matrix, get_feature_columns

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)
ARTIFACTS = _ROOT / 'artifacts'
DATA_PATH = _ROOT / 'data' / 'lnp_atlas_export.csv'

TARGET_COL = 'encapsulation_efficiency_percent_std'

st.set_page_config(page_title="LNP EE Predictor", layout="wide")
st.title("LNP Encapsulation Efficiency Predictor")

# -----------------------------------------------------------
# Load and featurize data (cached so it only runs once)
# -----------------------------------------------------------

@st.cache_data
def load_data():
    df_raw = pd.read_csv(DATA_PATH, encoding='latin-1')
    df = build_feature_matrix(df_raw, drop_ee_na=True)
    return df


st.subheader("Load Data from LNP Atlas")

with st.status("Loading & Featurizing Data", expanded=False) as status:
    df = load_data()
    feat_cols = get_feature_columns(df, TARGET_COL)
    X = df[feat_cols].values
    y = df[TARGET_COL].values
    st.write(df)
    st.write(f'Usable rows: {len(df)} | Features: {len(feat_cols)}')
    status.update(label="Featurization complete", state="complete", expanded=True)


# -----------------------------------------------------------
# Train XGBoost model
# -----------------------------------------------------------
st.subheader("Train XGBoost Model")

if st.button("Train XGBoost"):
    with st.spinner("Training XGBoost with Optuna (this may take a few minutes)..."):
        import train as train_xg
        xg_results = train_xg.train(n_optuna_trials=50)
    st.success("XGBoost training complete!")
    st.json(xg_results)


# -----------------------------------------------------------
# Train PINN model (via subprocess — pinn.train uses argparse)
# -----------------------------------------------------------
st.subheader("Train PINN Model")

if st.button("Train PINN"):
    import subprocess
    with st.spinner("Training PINN (this may take a few minutes)..."):
        result = subprocess.run(
            [sys.executable, "-m", "pinn.train",
             "--data", str(DATA_PATH),
             "--epochs", "200",
             "--alpha", "0.3",
             "--out", str(ARTIFACTS / "pinn")],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
    if result.returncode == 0:
        st.success("PINN training complete!")
        st.text(result.stdout)
        # Load saved metrics if available
        pinn_metrics_path = ARTIFACTS / "pinn" / "pinn_training_history.json"
        if pinn_metrics_path.exists():
            with open(pinn_metrics_path) as f:
                st.json(json.load(f)[-1])  # show final epoch
    else:
        st.error("PINN training failed")
        st.text(result.stderr)


# -----------------------------------------------------------
# TODO: See basic model evaluations
# -----------------------------------------------------------

# -----------------------------------------------------------
# TODO: See SHAP values
# -----------------------------------------------------------

# -----------------------------------------------------------
# TODO: Selectbox among multiple existing rows that shows
# predicted from XGBoost and PINN and actual EE%
# -----------------------------------------------------------
