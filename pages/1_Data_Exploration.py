import streamlit as st

st.set_page_config(page_title="Data Exploration", page_icon="\U0001f4ca", layout="wide")

import pandas as pd
import plotly.express as px
from streamlit_utils import load_raw_data, clean_plotly_layout, COLORS

# ── Column definitions ───────────────────────────────────────────────────────
# Display name -> column name mapping for numeric distributions
NUMERIC_COLS = {
    "Encapsulation Efficiency (%)": "encapsulation_efficiency_percent_std",
    "Particle Size (nm)": "particle_size_nm_std",
    "PDI": "pdi_std",
    "Zeta Potential (mV)": "zeta_potential_mv_std",
}

CATEGORICAL_COLS = {
    "Cargo Type": "target_type",
}

ALL_PLOT_COLS = {**NUMERIC_COLS, **CATEGORICAL_COLS}

# ── Load data ────────────────────────────────────────────────────────────────
df = load_raw_data()

st.title("Data Exploration")
st.caption("Interactive exploration of the LNP Atlas dataset (1,092 formulations)")

# ── Dataset overview ─────────────────────────────────────────────────────────
st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Formulations", f"{len(df):,}")
c2.metric("Unique Papers", df["paper_doi"].nunique())
c3.metric("Unique Ionizable Lipids", df["ionizable_lipid"].nunique())
c4.metric("EE% Measured", int(df["encapsulation_efficiency_percent_std"].notna().sum()))

st.dataframe(df, use_container_width=True, height=300)

# ── Distribution explorer ────────────────────────────────────────────────────
st.subheader("Distribution Explorer")

selected = st.selectbox("Select variable", options=list(ALL_PLOT_COLS.keys()))
col = ALL_PLOT_COLS[selected]

if col in NUMERIC_COLS.values():
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    fig = px.histogram(
        values, x=col, marginal="box", nbins=40,
        color_discrete_sequence=[COLORS["primary"]],
        labels={col: selected},
    )
    clean_plotly_layout(fig, title=f"Distribution of {selected}",
                        xaxis_title=selected, yaxis_title="Count")
else:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(
        counts, x="count", y=col, orientation="h",
        color_discrete_sequence=[COLORS["primary"]],
        labels={col: selected},
    )
    clean_plotly_layout(fig, title=f"Distribution of {selected}",
                        xaxis_title="Count", yaxis_title=selected)

st.plotly_chart(fig, use_container_width=True)

# ── Scatter plot explorer ────────────────────────────────────────────────────
st.subheader("Scatter Plot Explorer")

numeric_options = list(NUMERIC_COLS.keys())
sc1, sc2 = st.columns(2)
x_label = sc1.selectbox("X axis", options=numeric_options, index=1)  # particle size
y_label = sc2.selectbox("Y axis", options=numeric_options, index=0)  # EE%

x_col = NUMERIC_COLS[x_label]
y_col = NUMERIC_COLS[y_label]

scatter_df = df[[x_col, y_col, "target_type"]].copy()
scatter_df[x_col] = pd.to_numeric(scatter_df[x_col], errors="coerce")
scatter_df[y_col] = pd.to_numeric(scatter_df[y_col], errors="coerce")
scatter_df = scatter_df.dropna()

show_trend = st.checkbox("Show trendline")
fig2 = px.scatter(
    scatter_df, x=x_col, y=y_col, color="target_type",
    opacity=0.6,
    trendline="ols" if show_trend else None,
    labels={x_col: x_label, y_col: y_label, "target_type": "Cargo Type"},
)
clean_plotly_layout(fig2, title=f"{y_label} vs {x_label}",
                    xaxis_title=x_label, yaxis_title=y_label)
st.plotly_chart(fig2, use_container_width=True)

# ── Missing data ─────────────────────────────────────────────────────────────
st.subheader("Missing Data")

missing = df.isnull().sum().sort_values(ascending=True)
missing = missing[missing > 0]

if len(missing) > 0:
    fig3 = px.bar(
        x=missing.values, y=missing.index, orientation="h",
        color_discrete_sequence=[COLORS["secondary"]],
        labels={"x": "Missing Count", "y": "Column"},
    )
    fig3.add_vline(x=len(df), line_dash="dash", line_color=COLORS["muted"],
                   annotation_text=f"Total rows ({len(df)})")
    clean_plotly_layout(fig3, title="Missing Values by Column",
                        xaxis_title="Missing Count", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.success("No missing values in the dataset!")
