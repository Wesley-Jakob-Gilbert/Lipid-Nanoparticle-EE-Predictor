import streamlit as st

st.set_page_config(page_title="PINN Model", page_icon="\U0001f9e0", layout="wide")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_utils import (
    load_metadata, load_pinn_history, load_pinn_metrics,
    load_pinn_groupkfold, load_pinn_model, load_pinn_scaler,
    clean_plotly_layout, COLORS,
)

# ── Load artifacts ───────────────────────────────────────────────────────────
pinn_hist = load_pinn_history()
pinn_metrics = load_pinn_metrics()
pinn_gkf = load_pinn_groupkfold()
xgb_meta = load_metadata()

st.title("Physics-Informed Neural Network")

# ── Architecture diagram ─────────────────────────────────────────────────────
st.subheader("Architecture")

st.graphviz_chart("""
digraph PINN {
    rankdir=LR
    node [fontname="Helvetica", fontsize=10]
    edge [color="#95A5A6"]

    subgraph cluster_inputs {
        label="Input Features (7)"
        style=rounded
        color="#4A90D9"
        fontcolor="#4A90D9"
        node [shape=ellipse, style=filled, fillcolor="#D6EAF8"]
        il  [label="ionizable lipid\\nmole fraction"]
        np  [label="N/P ratio"]
        sz  [label="particle\\nsize (nm)"]
        pdi [label="PDI"]
        z   [label="zeta\\npotential (mV)"]
        peg [label="PEG\\nfraction"]
        ch  [label="cholesterol\\nfraction"]
    }

    subgraph cluster_network {
        label="Neural Network"
        style=rounded
        color="#2C3E50"
        fontcolor="#2C3E50"
        node [shape=box, style="rounded,filled", fillcolor="#F5F7FA"]
        proj [label="Linear(7 → 128)\\nLayerNorm + GELU"]
        rb1  [label="ResidualBlock(128)"]
        rb2  [label="ResidualBlock(128)"]
        rb3  [label="ResidualBlock(128)"]
        head [label="Linear(128 → 32)\\nGELU\\nLinear(32 → 1)\\nSigmoid"]
    }

    ee [label="EE ∈ [0,1]", shape=box, style="rounded,filled",
        fillcolor="#D5F5E3", fontcolor="#1A1A2E"]

    subgraph cluster_physics {
        label="Physics Loss (α=0.3)"
        style=rounded
        color="#E8735A"
        fontcolor="#E8735A"
        node [shape=box, style="rounded,filled", fillcolor="#FADBD8"]
        r1 [label="R1: N/P\\nMonotonicity"]
        r2 [label="R2: Gibbs\\nFree Energy"]
        r3 [label="R3: Boundary\\nEE→0 as d→∞"]
    }

    il -> proj; np -> proj; sz -> proj; pdi -> proj
    z -> proj; peg -> proj; ch -> proj

    proj -> rb1 -> rb2 -> rb3 -> head -> ee

    head -> r1 [style=dashed, color="#E8735A"]
    head -> r2 [style=dashed, color="#E8735A"]
    head -> r3 [style=dashed, color="#E8735A"]
}
""", use_container_width=True)

# ── Physics losses explained ─────────────────────────────────────────────────
st.subheader("Physics-Informed Loss Functions")
st.caption(
    "The total loss is L = L_data + 0.3 \u00d7 L_physics, where L_physics "
    "encodes three domain-knowledge constraints via autograd."
)

tab1, tab2, tab3 = st.tabs([
    "R1: N/P Monotonicity",
    "R2: Gibbs Free Energy",
    "R3: Boundary Condition",
])

with tab1:
    st.latex(r"\frac{\partial \, \text{EE}}{\partial \, \text{NP}} > 0 "
             r"\quad \text{for } \text{NP} < \text{NP}_{\text{opt}} = 6")
    st.markdown(
        "Below the optimal nitrogen-to-phosphate ratio, increasing N/P improves "
        "charge neutralization between cationic lipid headgroups and anionic cargo "
        "phosphate backbone, so EE should monotonically increase. The penalty is "
        "applied via autograd: if \u2202EE/\u2202NP < 0 for samples where NP < 6, "
        "the squared negative gradient is added to the loss."
    )
    # Illustrative chart
    x_np = np.linspace(0, 12, 200)
    y_ee_expected = 1 / (1 + np.exp(-1.5 * (x_np - 4)))
    fig_r1 = px.line(x=x_np, y=y_ee_expected * 100,
                     labels={"x": "N/P Ratio", "y": "Expected EE%"})
    fig_r1.add_vline(x=6, line_dash="dash", line_color=COLORS["secondary"],
                     annotation_text="NP_opt = 6")
    clean_plotly_layout(fig_r1, title="Expected: EE increases with N/P below optimum")
    fig_r1.update_layout(height=300)
    st.plotly_chart(fig_r1, use_container_width=True)

with tab2:
    st.latex(r"\Delta G_{\text{mix}} = RT\left[x \ln x + (1-x) \ln(1-x)\right]")
    st.markdown(
        "The sign of \u2202EE/\u2202x\u2097\u2097 should agree with the "
        "thermodynamic driving force. Where \u0394G_mix is decreasing "
        "(d\u0394G/dx < 0), higher ionizable lipid fraction lowers free energy "
        "and favors encapsulation. A sign mismatch between the model gradient "
        "and the analytical derivative is penalized."
    )
    x_il = np.linspace(0.01, 0.99, 200)
    dg = x_il * np.log(x_il) + (1 - x_il) * np.log(1 - x_il)
    fig_r2 = px.line(x=x_il, y=dg,
                     labels={"x": "Ionizable Lipid Mole Fraction (x)",
                             "y": "\u0394G_mix / RT"})
    fig_r2.add_vline(x=0.5, line_dash="dash", line_color=COLORS["muted"],
                     annotation_text="minimum at x=0.5")
    clean_plotly_layout(fig_r2, title="Gibbs Free Energy of Mixing")
    fig_r2.update_layout(height=300)
    st.plotly_chart(fig_r2, use_container_width=True)

with tab3:
    st.latex(r"\text{EE} \to 0 \quad \text{as } d \to \infty")
    st.markdown(
        "As particle size approaches infinity, the lipid assembly becomes too "
        "dilute to effectively encapsulate cargo. This boundary condition prevents "
        "the model from predicting high EE for unrealistically large particles. "
        "A synthetic test point at a large normalized size penalizes nonzero EE."
    )
    x_sz = np.linspace(50, 2000, 200)
    y_bc = 90 * np.exp(-0.003 * (x_sz - 50))
    fig_r3 = px.line(x=x_sz, y=y_bc,
                     labels={"x": "Particle Size (nm)", "y": "Expected EE%"})
    clean_plotly_layout(fig_r3, title="Expected: EE decays toward 0 at large sizes")
    fig_r3.update_layout(height=300)
    st.plotly_chart(fig_r3, use_container_width=True)

# ── Training replay ──────────────────────────────────────────────────────────
st.subheader("Training Replay")
st.caption(
    f"{pinn_metrics['epochs']} epochs on {pinn_metrics['n_train']} samples "
    f"({pinn_metrics['n_val']} held out). Physics loss weight \u03b1 = "
    f"{pinn_metrics['alpha']}."
)

max_epoch = len(pinn_hist)
epoch = st.slider("Epoch", min_value=1, max_value=max_epoch, value=max_epoch)

hist_slice = pinn_hist[:epoch]
epochs_arr = [h["epoch"] for h in hist_slice]
loss_data = [h["loss_data"] for h in hist_slice]
loss_phys = [h["loss_physics"] for h in hist_slice]
val_r2 = [h["val_r2"] for h in hist_slice]

chart_col, metric_col = st.columns([3, 1])

with chart_col:
    fig_train = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Loss Components", "Validation R\u00b2"],
        vertical_spacing=0.12,
    )
    fig_train.add_trace(go.Scatter(
        x=epochs_arr, y=loss_data, name="Data Loss (MSE)",
        line=dict(color=COLORS["primary"]),
    ), row=1, col=1)
    fig_train.add_trace(go.Scatter(
        x=epochs_arr, y=loss_phys, name="Physics Loss",
        line=dict(color=COLORS["secondary"]),
    ), row=1, col=1)
    fig_train.add_trace(go.Scatter(
        x=epochs_arr, y=val_r2, name="Val R\u00b2",
        line=dict(color=COLORS["accent"]),
        showlegend=False,
    ), row=2, col=1)

    fig_train.update_layout(
        template="plotly_white", height=450,
        margin=dict(l=60, r=30, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_train.update_xaxes(title_text="Epoch", row=2, col=1)
    fig_train.update_yaxes(title_text="Loss", row=1, col=1)
    fig_train.update_yaxes(title_text="R\u00b2", row=2, col=1)
    st.plotly_chart(fig_train, use_container_width=True)

with metric_col:
    cur = hist_slice[-1]
    st.metric("Epoch", cur["epoch"])
    st.metric("Data Loss", f"{cur['loss_data']:.4f}")
    st.metric("Physics Loss", f"{cur['loss_physics']:.4f}")
    st.metric("Val R\u00b2", f"{cur['val_r2']:.3f}")
    st.metric("Val MSE", f"{cur['val_mse']:.4f}")

# ── GroupKFold cross-validation ──────────────────────────────────────────────
st.subheader("GroupKFold Cross-Validation (by paper DOI)")
st.info(
    f"Only **{pinn_gkf['n_samples']} samples** from "
    f"**{pinn_gkf['n_groups']} papers** have all 7 PINN features. "
    "Small fold sizes (as few as 8 samples) cause high R\u00b2 variance. "
    "Negative R\u00b2 means the fold's predictions are worse than predicting "
    "the mean \u2014 expected when a held-out paper's formulations differ "
    "substantially from the training set."
)

gkf_folds = pd.DataFrame(pinn_gkf["fold_metrics"])
oof = pinn_gkf["oof_metrics"]

fig_gkf = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["R\u00b2 per Fold", "RMSE per Fold"],
                        vertical_spacing=0.15)

r2_colors = [COLORS["accent"] if r >= 0 else COLORS["secondary"]
             for r in gkf_folds["r2"]]
fig_gkf.add_trace(go.Bar(
    x=gkf_folds["fold"], y=gkf_folds["r2"], marker_color=r2_colors,
    name="R\u00b2", showlegend=False,
    text=[f"n={n}" for n in gkf_folds["n_val"]],
    textposition="outside",
), row=1, col=1)
fig_gkf.add_hline(y=oof["r2"], line_dash="dash", line_color=COLORS["muted"],
                  annotation_text=f"OOF R\u00b2 = {oof['r2']:.3f}", row=1, col=1)

fig_gkf.add_trace(go.Bar(
    x=gkf_folds["fold"], y=gkf_folds["rmse_pct"], marker_color=COLORS["primary"],
    name="RMSE", showlegend=False,
    text=[f"n={n}" for n in gkf_folds["n_val"]],
    textposition="outside",
), row=2, col=1)
fig_gkf.add_hline(y=oof["rmse_pct"], line_dash="dash", line_color=COLORS["muted"],
                  annotation_text=f"OOF RMSE = {oof['rmse_pct']:.1f}%", row=2, col=1)

fig_gkf.update_layout(template="plotly_white", height=450,
                      margin=dict(l=60, r=30, t=50, b=30))
fig_gkf.update_xaxes(title_text="Fold", row=2, col=1)
fig_gkf.update_yaxes(title_text="R\u00b2", row=1, col=1)
fig_gkf.update_yaxes(title_text="RMSE (%)", row=2, col=1)
st.plotly_chart(fig_gkf, use_container_width=True)

# ── Model comparison ─────────────────────────────────────────────────────────
st.subheader("XGBoost vs PINN Comparison")

comparison = pd.DataFrame([
    {
        "Model": "XGBoost",
        "Samples": xgb_meta["n_train_samples"],
        "Features": xgb_meta["n_features"],
        "OOF R\u00b2": f"{xgb_meta['oof_metrics']['r2']:.3f}",
        "OOF RMSE": f"{xgb_meta['oof_metrics']['rmse']:.1f}%",
        "Approach": "Gradient boosting + 76 engineered features",
    },
    {
        "Model": "PINN",
        "Samples": pinn_gkf["n_samples"],
        "Features": 7,
        "OOF R\u00b2": f"{oof['r2']:.3f}",
        "OOF RMSE": f"{oof['rmse_pct']:.1f}%",
        "Approach": "Neural net + physics loss (7 features)",
    },
])
st.dataframe(comparison, use_container_width=True, hide_index=True)
st.caption(
    "The two models are trained on different subsets (523 vs 93 samples) and "
    "different feature sets (76 engineered vs 7 physicochemical), so direct "
    "comparison requires nuance."
)

# ── Interactive predictor ────────────────────────────────────────────────────
st.subheader("Predict EE%")
st.caption("Adjust the 7 physicochemical features to see the PINN's prediction.")

import torch

SLIDER_CONFIGS = {
    "ionizable_lipid_mole_fraction": ("Ionizable Lipid Mole Fraction", 0.0, 1.0, 0.5, 0.01),
    "np_ratio": ("N/P Ratio", 0.0, 20.0, 5.0, 0.5),
    "particle_size_nm": ("Particle Size (nm)", 30.0, 500.0, 100.0, 5.0),
    "pdi": ("PDI", 0.0, 1.0, 0.15, 0.01),
    "zeta_mv": ("Zeta Potential (mV)", -50.0, 50.0, 0.0, 1.0),
    "peg_fraction": ("PEG Fraction", 0.0, 0.3, 0.025, 0.005),
    "cholesterol_fraction": ("Cholesterol Fraction", 0.0, 0.6, 0.385, 0.01),
}

slider_values = []
cols_row1 = st.columns(3)
cols_row2 = st.columns(4)
all_cols = cols_row1 + cols_row2

for i, (feat, (label, mn, mx, default, step)) in enumerate(SLIDER_CONFIGS.items()):
    with all_cols[i]:
        val = st.slider(label, min_value=mn, max_value=mx, value=default, step=step)
        slider_values.append(val)

# Scale and predict
pinn_model = load_pinn_model()
scaler = load_pinn_scaler()

x_raw = np.array([slider_values], dtype=np.float32)
x_scaled = scaler.transform(x_raw).astype(np.float32)

with torch.no_grad():
    pred = pinn_model(torch.tensor(x_scaled)).item()

pred_ee = pred * 100.0  # fraction -> percent
pred_ee = float(np.clip(pred_ee, 0, 100))

pc1, pc2 = st.columns([1, 2])
with pc1:
    st.metric("Predicted EE%", f"{pred_ee:.1f}%")

with pc2:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_ee,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [0, 50], "color": "#FADBD8"},
                {"range": [50, 80], "color": "#FCF3CF"},
                {"range": [80, 100], "color": "#D5F5E3"},
            ],
        },
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=30, r=30, t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)
