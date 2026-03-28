import streamlit as st

st.set_page_config(page_title="XGBoost Model", page_icon="\U0001f332", layout="wide")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_utils import (
    load_metadata, load_shap_importance, load_shap_values,
    load_xgboost_model, clean_plotly_layout, format_feature_name, COLORS,
)

# ── Load artifacts ───────────────────────────────────────────────────────────
meta = load_metadata()
shap_imp = load_shap_importance()
shap_imp.columns = ["feature", "mean_abs_shap"]
shap_vals = load_shap_values()
model, transformer, _ = load_xgboost_model()
feat_cols = meta["feature_columns"]

st.title("XGBoost Model")

# ── Model card ───────────────────────────────────────────────────────────────
st.subheader("Model Card")

left, right = st.columns(2)

with left:
    st.markdown("**Hyperparameters** (Optuna-tuned, 50 trials)")
    hp = meta["best_hyperparams"]
    hp_df = pd.DataFrame([
        ("n_estimators", f"{hp['n_estimators']}"),
        ("learning_rate", f"{hp['learning_rate']:.4f}"),
        ("max_depth", f"{hp['max_depth']}"),
        ("subsample", f"{hp['subsample']:.3f}"),
        ("colsample_bytree", f"{hp['colsample_bytree']:.3f}"),
        ("min_child_weight", f"{hp['min_child_weight']}"),
        ("reg_alpha", f"{hp['reg_alpha']:.2e}"),
        ("reg_lambda", f"{hp['reg_lambda']:.4f}"),
    ], columns=["Parameter", "Value"])
    st.dataframe(hp_df, use_container_width=True, hide_index=True)

with right:
    st.markdown("**Training Summary**")
    oof = meta["oof_metrics"]
    r1, r2 = st.columns(2)
    r1.metric("Training Samples", meta["n_train_samples"])
    r2.metric("Features", meta["n_features"])
    r3, r4 = st.columns(2)
    r3.metric("OOF R\u00b2", f"{oof['r2']:.3f}")
    r4.metric("OOF RMSE", f"{oof['rmse']:.1f}%")
    r5, r6 = st.columns(2)
    r5.metric("OOF MAE", f"{oof['mae']:.1f}%")
    r6.metric("Box-Cox \u03bb", f"{meta['box_cox_lambda']:.3f}")

# ── GroupKFold cross-validation ──────────────────────────────────────────────
st.subheader("GroupKFold Cross-Validation (by paper DOI)")
st.caption(
    "Papers are never split across folds, preventing data leakage from "
    "correlated measurements within the same study."
)

folds = meta["fold_metrics"]
fold_df = pd.DataFrame(folds)

fig_cv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=["R\u00b2 per Fold", "RMSE per Fold"],
                       vertical_spacing=0.15)

r2_colors = [COLORS["accent"] if r >= 0 else COLORS["secondary"]
             for r in fold_df["r2"]]
fig_cv.add_trace(go.Bar(
    x=fold_df["fold"], y=fold_df["r2"], marker_color=r2_colors,
    name="R\u00b2", showlegend=False,
), row=1, col=1)
fig_cv.add_hline(y=oof["r2"], line_dash="dash", line_color=COLORS["muted"],
                 annotation_text=f"OOF R\u00b2 = {oof['r2']:.3f}", row=1, col=1)

fig_cv.add_trace(go.Bar(
    x=fold_df["fold"], y=fold_df["rmse"], marker_color=COLORS["primary"],
    name="RMSE", showlegend=False,
), row=2, col=1)
fig_cv.add_hline(y=oof["rmse"], line_dash="dash", line_color=COLORS["muted"],
                 annotation_text=f"OOF RMSE = {oof['rmse']:.1f}%", row=2, col=1)

fig_cv.update_layout(template="plotly_white", height=450,
                     margin=dict(l=60, r=30, t=50, b=30))
fig_cv.update_xaxes(title_text="Fold", row=2, col=1)
fig_cv.update_yaxes(title_text="R\u00b2", row=1, col=1)
fig_cv.update_yaxes(title_text="RMSE (%)", row=2, col=1)
st.plotly_chart(fig_cv, use_container_width=True)

# ── SHAP feature importance ─────────────────────────────────────────────────
st.subheader("SHAP Feature Importance")

n_show = st.slider("Number of features to show", 5, 76, 15, step=5)
top_shap = shap_imp.head(n_show).copy()
top_shap["display_name"] = top_shap["feature"].apply(format_feature_name)
top_shap = top_shap.sort_values("mean_abs_shap")  # ascending for horizontal bar

fig_shap = px.bar(
    top_shap, x="mean_abs_shap", y="display_name", orientation="h",
    color="mean_abs_shap", color_continuous_scale="Blues",
    labels={"mean_abs_shap": "Mean |SHAP|", "display_name": ""},
)
clean_plotly_layout(fig_shap, title=f"Top {n_show} Features by SHAP Importance")
fig_shap.update_layout(height=max(350, n_show * 25), coloraxis_showscale=False)
st.plotly_chart(fig_shap, use_container_width=True)

# ── SHAP dependence ──────────────────────────────────────────────────────────
st.subheader("SHAP Dependence")
st.info(
    "SHAP values are in Box-Cox transformed space (\u03bb = "
    f"{meta['box_cox_lambda']:.2f}). Positive values push the prediction "
    "toward higher EE%."
)

top20 = shap_imp.head(20)["feature"].tolist()
dep_feature = st.selectbox("Select feature", options=top20,
                           format_func=format_feature_name)

feat_idx = feat_cols.index(dep_feature)

# Get feature values for the dependence plot (need actual X, not just SHAP)
from streamlit_utils import load_raw_data
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import build_feature_matrix, get_feature_columns


@st.cache_data
def get_featurized_data():
    df_raw = load_raw_data()
    df = build_feature_matrix(df_raw, drop_ee_na=True)
    target = "encapsulation_efficiency_percent_std"
    cols = get_feature_columns(df, target)
    return df[cols].values, cols


X_feat, feat_col_list = get_featurized_data()

if dep_feature in feat_col_list:
    col_idx = feat_col_list.index(dep_feature)
    feature_values = X_feat[:, col_idx]
    shap_for_feature = shap_vals[:, feat_idx]

    fig_dep = px.scatter(
        x=feature_values, y=shap_for_feature,
        opacity=0.5,
        color_discrete_sequence=[COLORS["primary"]],
        labels={"x": format_feature_name(dep_feature), "y": "SHAP Value (Box-Cox space)"},
    )
    clean_plotly_layout(fig_dep, title=f"SHAP Dependence: {format_feature_name(dep_feature)}",
                        xaxis_title=format_feature_name(dep_feature),
                        yaxis_title="SHAP Value")
    st.plotly_chart(fig_dep, use_container_width=True)

# ── Interactive predictor ────────────────────────────────────────────────────
st.subheader("Predict EE%")
st.caption(
    "Adjust the top features below. All other features are held at "
    "training-set median values."
)

medians = meta["training_medians"]

# Top 6 features with reasonable slider ranges
SLIDER_CONFIGS = {
    "peg_lipid__freq_rank": ("PEG Lipid Frequency Rank", 0.0, 500.0, 1.0),
    "ratio_helper": ("Helper Lipid Ratio", 0.0, 50.0, 0.5),
    "particle_size_nm_std": ("Particle Size (nm)", 30.0, 500.0, 5.0),
    "ratio_ionizable_norm": ("Ionizable Lipid Fraction", 0.1, 0.8, 0.01),
    "ratio_peg": ("PEG Ratio", 0.0, 20.0, 0.5),
    "ionizable_lipid__freq_rank": ("Ionizable Lipid Frequency Rank", 0.0, 500.0, 1.0),
}

slider_values = {}
cols = st.columns(3)
for i, (feat, (label, mn, mx, step)) in enumerate(SLIDER_CONFIGS.items()):
    med = float(medians.get(feat, (mn + mx) / 2))
    med = max(mn, min(mx, med))  # clamp to slider range
    with cols[i % 3]:
        slider_values[feat] = st.slider(label, min_value=mn, max_value=mx,
                                         value=med, step=step)

# Build feature vector
x_input = np.array([[float(medians.get(f, 0.0)) for f in feat_cols]])
for feat, val in slider_values.items():
    if feat in feat_cols:
        x_input[0, feat_cols.index(feat)] = val

# Predict
raw_pred = model.predict(x_input)
if transformer is not None:
    pred_ee = transformer.inverse_transform(raw_pred.reshape(-1, 1)).ravel()[0]
else:
    pred_ee = raw_pred[0]
pred_ee = float(np.clip(pred_ee, 0, 100))

# Display
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
