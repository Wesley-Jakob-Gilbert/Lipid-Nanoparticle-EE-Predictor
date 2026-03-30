import streamlit as st

st.set_page_config(page_title="LNP EE Predictor", page_icon="🧬", layout="wide")

from streamlit_utils import load_metadata, load_pinn_metrics

meta = load_metadata()
pinn = load_pinn_metrics()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("LNP Encapsulation Efficiency Predictor 🧬")
st.caption(
    "Machine learning + physics-informed models for predicting lipid nanoparticle "
    "encapsulation efficiency · Trained on LNP Atlas (1,092 formulations) · "
    "**🚧 Active development**"
)

st.info(
    "⚠️ **This project is under active development.** Current model performance is modest "
    f"(XGBoost OOF R²={meta['oof_metrics']['r2']:.3f}) due to a known data limitation — "
    "see *Why is R² low?* below for a full explanation. The pipeline, architecture, and "
    "honest evaluation methodology are the primary contributions at this stage.",
    icon="🔬"
)

# ── Metric cards ─────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Dataset", "1,092 formulations", delta=f"{meta['n_train_samples']} with EE%")
c2.metric(
    "XGBoost",
    f"OOF R² = {meta['oof_metrics']['r2']:.3f}",
    delta=f"{meta['n_features']} features · RMSE {meta['oof_metrics']['rmse']:.1f}%",
    delta_color="off"
)
c3.metric(
    "PINN",
    f"Val R² = {pinn['val_r2']:.3f}",
    delta=f"{pinn['n_features']} physics features · OOF R² {pinn['oof_r2']:.3f}",
    delta_color="off"
)

# ── About ─────────────────────────────────────────────────────────────────────
st.markdown("### About This Project")
st.markdown("""
LNP encapsulation efficiency (EE%) — the fraction of nucleic acid cargo successfully
packaged — is a critical quality attribute for mRNA therapeutics and CRISPR delivery.
This project builds predictive models for EE% from formulation parameters to accelerate
rational LNP design.

**Two complementary models are implemented and compared:**

- **XGBoost** — 88 engineered features including RDKit molecular descriptors for all four
  lipid components, molar ratios, synthesis conditions (flow rate, pH, method), cargo type,
  publication year, journal tier, and measured N:P ratio. Evaluated with GroupKFold on
  `paper_doi` to prevent data leakage across correlated measurements.

- **Physics-Informed Neural Network (PINN)** — 7 physicochemical features with three
  physics-derived loss residuals encoding N:P monotonicity (R1), Gibbs free energy of
  mixing (R2), and an infinite-dilution boundary condition (R3). The physics priors help
  constrain predictions in data-sparse regimes.
""")

# ── Why is R² low? ─────────────────────────────────────────────────────────
with st.expander("📊 Why is R² low? (Read this first)", expanded=True):
    st.markdown(f"""
**Short answer:** The data has a fundamental structure problem — not a model problem.

**Root cause: paper-level confounders dominate the signal.**

Within-paper EE variance and between-paper EE variance are approximately equal in the
LNP Atlas dataset. This means ~50% of EE variance is explained by unmeasured lab-level
factors that differ across publications:

| Confounder | Effect on EE | In dataset? |
|---|---|---|
| Assay method (RiboGreen vs SpectraMax) | ±5–15% systematic offset | ❌ Not recorded |
| Buffer pH during formulation | Strong effect on ionizable lipid protonation | ~40% coverage |
| Operator technique / equipment | Significant batch-to-batch noise | ❌ Not recorded |
| Lipid lot purity | Changes effective formulation | ❌ Not recorded |

**Under GroupKFold cross-validation** (which holds out entire papers at test time — the
realistic deployment scenario), the model must generalize to papers it has never seen.
Because ~50% of variance comes from unmeasured confounders that are paper-specific,
**no ML model trained on this data can achieve high R² under this CV strategy.**

The XGBoost OOF R²={meta['oof_metrics']['r2']:.3f} and PINN OOF R²={pinn['oof_r2']:.3f}
are honest reflections of this ceiling — not model failures.

**What would fix this:**
1. Standardized experimental data from a single lab with consistent protocol
2. Explicit assay method + buffer conditions recorded per row
3. ~5,000+ rows across 200+ papers to statistically overcome paper-level confounding

**What this project demonstrates despite the limitation:**
- Correct leakage-free evaluation methodology (GroupKFold on paper_doi)
- Physics-informed modeling with thermodynamically-grounded loss residuals
- Feature engineering pipeline for heterogeneous LNP literature data
- Honest scientific communication of model limitations
""")

# ── Navigate ──────────────────────────────────────────────────────────────────
st.markdown("### Explore")
n1, n2, n3 = st.columns(3)
with n1:
    st.page_link("pages/1_Data_Exploration.py", label="📊 Data Exploration")
    st.caption("EE distributions, feature correlations, missing data heatmap")
with n2:
    st.page_link("pages/2_XGBoost_Model.py", label="🌲 XGBoost Model")
    st.caption("Feature importance, CV results, live predictor")
with n3:
    st.page_link("pages/3_PINN_Model.py", label="🧠 PINN Model")
    st.caption("Physics architecture, training curves, live predictor")

# ── Pipeline diagram ──────────────────────────────────────────────────────────
st.markdown("### Pipeline Architecture")
st.graphviz_chart("""
digraph pipeline {
    rankdir=LR
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11]
    edge [color="#95A5A6"]

    raw  [label="LNP Atlas\\n1,092 rows", fillcolor="#F5F7FA"]
    clean [label="Data Cleaning\\n+174 rows rescued\\n(EE parser + SMILES fix)", fillcolor="#FEF9E7"]
    feat [label="Feature Engineering\\n88 features\\n(RDKit + synthesis + cargo)", fillcolor="#F5F7FA"]

    xgb  [label="XGBoost\\n88 features\\nOOF R²=0.082", fillcolor="#D6EAF8"]
    pinn [label="PINN\\n7 features\\nOOF R²=0.438", fillcolor="#FADBD8"]

    cv   [label="GroupKFold CV\\n(paper_doi)\\nleakage-free", fillcolor="#D5F5E3"]

    raw  -> clean -> feat
    feat -> xgb -> cv
    feat -> pinn -> cv
}
""")

st.markdown("---")
st.caption(
    "Data: [LNP Atlas](https://www.nature.com/articles/s41565-023-01511-6) · "
    "Code: [GitHub](https://github.com/Wesley-Jakob-Gilbert/Lipid-Nanoparticle-EE-Predictor) · "
    "Wesley Gilbert · BS Biophysics · [wesley-j-gilbert.com](https://www.wesley-j-gilbert.com)"
)
