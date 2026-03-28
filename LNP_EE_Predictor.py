import streamlit as st

st.set_page_config(page_title="LNP EE Predictor", page_icon="\U0001f9ec", layout="wide")

from streamlit_utils import load_metadata, load_pinn_metrics

# ── Landing page ─────────────────────────────────────────────────────────────

st.title("LNP Encapsulation Efficiency Predictor")
st.caption(
    "Machine learning models for predicting lipid nanoparticle encapsulation "
    "efficiency, trained on the LNP Atlas (1,092 formulations)"
)

# Metric cards
meta = load_metadata()
pinn = load_pinn_metrics()

c1, c2, c3 = st.columns(3)
c1.metric("Dataset", "1,092 formulations", delta=f"{meta['n_train_samples']} with EE%")
c2.metric("XGBoost Features", str(meta["n_features"]),
          delta=f"OOF R\u00b2 = {meta['oof_metrics']['r2']:.3f}")
c3.metric("PINN Features", f"{pinn['n_features']} physicochemical",
          delta=f"Val R\u00b2 = {pinn['val_r2']:.3f}")

# Project overview
st.markdown("""
### About This Project

This project tackles a real problem in nanomedicine: **predicting how efficiently
lipid nanoparticles (LNPs) encapsulate therapeutic cargo** like mRNA and siRNA.

Two complementary modeling approaches are compared:

- **XGBoost** uses 76 engineered features (molecular descriptors, molar ratios,
  synthesis conditions) with Optuna hyperparameter optimization and Box-Cox
  target transformation.
- **Physics-Informed Neural Network (PINN)** uses 7 physicochemical features
  with three physics-derived loss terms enforcing thermodynamic and electrostatic
  priors.

Both models are evaluated with **GroupKFold cross-validation** on `paper_doi` to
prevent data leakage from correlated measurements within the same study. Current
performance is modest (XGBoost OOF R\u00b2 = 0.14, PINN OOF R\u00b2 = 0.44) \u2014 an honest
reflection of the difficulty of predicting EE% from formulation parameters alone.
""")

# Navigation cards
st.markdown("### Explore")
n1, n2, n3 = st.columns(3)
with n1:
    st.page_link("pages/1_Data_Exploration.py", label="Data Exploration",
                 icon="\U0001f4ca")
    st.caption("Interactive exploration of the LNP Atlas dataset")
with n2:
    st.page_link("pages/2_XGBoost_Model.py", label="XGBoost Model",
                 icon="\U0001f332")
    st.caption("Feature importance, SHAP analysis, and live predictions")
with n3:
    st.page_link("pages/3_PINN_Model.py", label="PINN Model",
                 icon="\U0001f9e0")
    st.caption("Architecture, physics losses, training replay, and predictions")

# Data flow diagram
st.markdown("### Pipeline Architecture")
st.graphviz_chart("""
digraph pipeline {
    rankdir=LR
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11]
    edge [color="#95A5A6"]

    csv  [label="LNP Atlas\\n(1,092 rows)", fillcolor="#F5F7FA"]
    feat [label="Feature\\nEngineering", fillcolor="#F5F7FA"]

    xgb  [label="XGBoost\\n76 features", fillcolor="#D6EAF8"]
    pinn [label="PINN\\n7 features", fillcolor="#FADBD8"]

    ee   [label="EE %\\nPrediction", fillcolor="#D5F5E3"]

    csv  -> feat
    feat -> xgb
    feat -> pinn
    xgb  -> ee
    pinn -> ee
}
""")
