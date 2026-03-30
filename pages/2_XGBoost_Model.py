import streamlit as st

st.set_page_config(page_title="XGBoost Model", page_icon="🌲", layout="wide")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_utils import (
    load_metadata, load_shap_importance,
    clean_plotly_layout, format_feature_name, COLORS,
)

meta     = load_metadata()
shap_imp = load_shap_importance()
shap_imp.columns = ["feature", "importance"]

st.title("🌲 XGBoost Model")
st.caption("🚧 Active development — performance metrics reflect current dataset limitations")

# ── Development status banner ─────────────────────────────────────────────────
oof = meta["oof_metrics"]
st.info(
    f"**Current OOF R² = {oof['r2']:.3f}, RMSE = {oof['rmse']:.1f}%** under GroupKFold "
    f"(paper_doi, k=5). See the landing page for a full explanation of why cross-paper "
    f"generalization is hard with this dataset. The evaluation methodology is rigorous — "
    f"the data structure is the bottleneck, not the model.",
    icon="📊"
)

# ── Model summary ─────────────────────────────────────────────────────────────
st.subheader("Model Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training Samples", meta["n_train_samples"])
c2.metric("Features", meta["n_features"])
c3.metric("OOF R²", f"{oof['r2']:.3f}")
c4.metric("OOF RMSE", f"{oof['rmse']:.1f}%")

with st.expander("Hyperparameters"):
    hp_data = {
        "n_estimators": 500, "learning_rate": 0.05, "max_depth": 4,
        "subsample": 0.80, "colsample_bytree": 0.80,
        "min_child_weight": 5, "reg_alpha": 0.1, "reg_lambda": 1.0,
    }
    st.dataframe(
        pd.DataFrame(hp_data.items(), columns=["Parameter", "Value"]),
        use_container_width=True, hide_index=True
    )

st.markdown("""
**Feature groups (88 total):**
- **Molecular descriptors** — MW, LogP, TPSA, RotBonds, HBAcc, HDon, RingCount, FractionCSP3 for each of 4 lipid components (ionizable, PEG, sterol, helper) via RDKit
- **Molar ratios** — ionizable, PEG, cholesterol, helper fractions + normalized variants
- **Synthesis conditions** — flow rate, flow ratio, buffer pH, synthesis method (microfluidic/bulk/ethanol injection)
- **Cargo & context** — target type difficulty, mRNA/siRNA binary flags, N:P ratio (23% coverage)
- **Publication metadata** — paper year, journal tier (Nature/Science=3, Adv/ACS=2, other=1)
- **Interaction terms** — IL LogP × ionizable fraction, PEG MW × PEG fraction
""")

# ── GroupKFold CV ──────────────────────────────────────────────────────────────
st.subheader("GroupKFold Cross-Validation")
st.markdown("""
Papers are held out entirely — no formulations from the same paper appear in both train and
validation. This is the correct evaluation for deployment to new labs/papers, and is why
performance is lower than a naive random split would show.
""")

fold_df = pd.DataFrame(oof["fold_metrics"])

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["R² per Fold", "RMSE per Fold (%)"])

r2_colors = [COLORS["accent"] if r >= 0 else COLORS["secondary"] for r in fold_df["r2"]]
fig.add_trace(go.Bar(x=fold_df["fold"].astype(str), y=fold_df["r2"],
                     marker_color=r2_colors, name="R²"), row=1, col=1)
fig.add_hline(y=oof["r2"], line_dash="dash", line_color=COLORS["muted"],
              annotation_text=f"OOF R²={oof['r2']:.3f}", row=1, col=1)
fig.add_hline(y=0, line_color="#cccccc", line_width=1, row=1, col=1)

fig.add_trace(go.Bar(x=fold_df["fold"].astype(str), y=fold_df["rmse"],
                     marker_color=COLORS["primary"], name="RMSE"), row=1, col=2)
fig.add_hline(y=oof["rmse"], line_dash="dash", line_color=COLORS["muted"],
              annotation_text=f"OOF RMSE={oof['rmse']:.1f}%", row=1, col=2)

fig.update_layout(template="plotly_white", height=380, showlegend=False,
                  margin=dict(l=50, r=30, t=50, b=30))
fig.update_xaxes(title_text="Fold")
fig.update_yaxes(title_text="R²", row=1, col=1)
fig.update_yaxes(title_text="RMSE (%)", row=1, col=2)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Negative R² (folds 1, 2, 4) means predictions are worse than predicting the mean — "
    "expected when test papers differ substantially from training papers in unmeasured ways. "
    "Fold 5 generalizes well (R²=0.28), suggesting some papers are more 'typical'."
)

# ── Feature importance ─────────────────────────────────────────────────────────
st.subheader("Feature Importance")
st.caption("XGBoost gain-based importance (higher = more predictive splits)")

n_show = st.slider("Features to display", 10, min(50, len(shap_imp)), 20, step=5)
top = shap_imp.head(n_show).copy()
top["display"] = top["feature"].apply(
    lambda f: format_feature_name(f) if callable(format_feature_name) else f.replace("_", " ").title()
)

fig2 = go.Figure(go.Bar(
    x=top["importance"][::-1],
    y=top["display"][::-1],
    orientation="h",
    marker_color=COLORS["primary"],
))
fig2.update_layout(template="plotly_white", height=max(400, n_show * 22),
                   margin=dict(l=200, r=30, t=30, b=30))
fig2.update_xaxes(title_text="Gain Importance")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
**Notable top features:**
- **paper_year_feat** and **peg_lipid__freq_rank** rank highly — capturing systematic
  differences across papers (protocol era, lipid identity prevalence)
- **ratio_helper / ratio_ionizable** — molar composition remains the strongest formulation signal
- **synth_flow_ratio** — microfluidic mixing ratio influences particle formation kinetics
""")

# ── What's next ────────────────────────────────────────────────────────────────
with st.expander("🔭 Development Roadmap"):
    st.markdown("""
**Completed:**
- ✅ XGBoost baseline with GroupKFold + leakage-free evaluation
- ✅ Data cleaning: 523 → 636 rows (EE parser, cKK-E12 SMILES fix)
- ✅ Feature expansion: N:P ratio, publication year, journal tier, cargo flags
- ✅ Leakage bug fix (removed target from feature matrix)
- ✅ PINN with physics residuals R1–R3

**In progress:**
- 🔄 LNPDB integration (thousands of additional rows)
- 🔄 Mixed-effects model to explicitly account for paper-level confounding
- 🔄 Standardized single-lab validation dataset

**Planned:**
- 📋 Assay method extraction when source data provides it
- 📋 Transformer/attention architecture (inspired by COMET)
- 📋 API endpoint for batch formulation screening
""")
