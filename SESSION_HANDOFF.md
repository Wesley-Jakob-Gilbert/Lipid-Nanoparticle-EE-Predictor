# Session Handoff — LNP EE Predictor

**Date**: 2026-03-28
**Last commit**: `b30c4e5` — Streamlit multipage dashboard
**Branch**: `main` (2 commits ahead of origin — NOT yet pushed)

---

## Who is the user?

Wesley Gilbert, BS Biophysics (Scientific Computing). Currently a manufacturing engineer at Intel, applying to research assistant positions at Broad Institute, NY Genome Center, and HHMI. This project is his main portfolio piece for PIs reviewing his GitHub.

---

## What was just completed (this session)

### Streamlit Multipage Dashboard
Built a full interactive frontend to showcase the project. All 7 files are committed:

| File | Purpose |
|------|---------|
| `.streamlit/config.toml` | White + blue scientific theme |
| `streamlit_utils.py` | Shared cached loaders, plotly styling, path constants |
| `LNP_EE_Predictor.py` | Landing page: metrics, overview, navigation, pipeline graphviz diagram |
| `pages/1_Data_Exploration.py` | Distribution explorer (selectbox), scatter explorer, missing data |
| `pages/2_XGBoost_Model.py` | Model card, GroupKFold CV, SHAP importance/dependence, live predictor with gauge |
| `pages/3_PINN_Model.py` | Architecture graphviz, physics loss tabs (LaTeX + charts), training replay slider, GroupKFold, model comparison, live predictor |
| `requirements.txt` | Added `plotly>=5.18.0`, `graphviz>=0.20.0` |

**Status**: Committed but NOT pushed. NOT yet tested (user hasn't run `streamlit run` yet). May need `pip install plotly graphviz` and system-level graphviz binary.

---

## What needs to be done next (the interrupted task)

### PINN without Zeta Potential — THE MAIN NEXT TASK

The user explicitly requested: "make a PINN without the zeta potential dependency (in addition to the PINN that already exists) for more usable data."

**Why**: The current PINN only uses 93 rows because `zeta_potential_mv_std` is the data bottleneck (only ~326/1092 rows have it). Dropping zeta_mv gives ~323 usable rows (3.5x increase). None of the 3 physics residuals (R1, R2, R3) use zeta_mv, so the physics is unaffected.

**Research already done** (from an Explore agent in this session):
- `pinn/preprocess.py` FEATURE_COLS has zeta_mv at index 4
- Physics residuals only use indices 0 (x_IL for R2), 1 (np_ratio for R1), 2 (particle_size for R3)
- `pinn/model.py` EEPredictor takes `n_features` as a parameter — NOT hardcoded to 7
- `pinn/train.py` infers n_features from data shape
- Column indices for physics loss would remain correct if zeta_mv is simply removed (indices 0, 1, 2 stay the same: ionizable_lipid_mole_fraction, np_ratio, particle_size_nm)

**Implementation approach** (not yet started):
1. Make the feature set configurable in `pinn/preprocess.py` — add a `FEATURE_COLS_NO_ZETA` list (6 features) alongside existing `FEATURE_COLS` (7 features)
2. Add a CLI flag to `pinn/train.py` like `--no-zeta` that selects the 6-feature set
3. Save artifacts to a separate directory (e.g., `artifacts/pinn_no_zeta/`)
4. Train both variants and compare results
5. Update README, Streamlit dashboard, and CLAUDE.md with the new model
6. The user said "If it works, make sure to clean the project up with this additional model so that the project is presentation-worthy"

**Key files to modify**:
- `pinn/preprocess.py` — add FEATURE_COLS_NO_ZETA, update _load_core or add parameter
- `pinn/train.py` — add --no-zeta CLI flag, route to correct feature set
- `artifacts/pinn_no_zeta/` — new artifact directory
- `README.md` — add results row for PINN (no zeta)
- `pages/3_PINN_Model.py` — add toggle/tab to switch between 7-feature and 6-feature PINN
- `streamlit_utils.py` — add loaders for no-zeta artifacts
- `pinn/README.md` — update with new variant documentation

---

## Project Architecture (quick reference)

```
LNP Predictor/
├── data/lnp_atlas_export.csv       # 1,092 formulations (in repo, not gitignored)
├── src/
│   ├── features.py                 # XGBoost feature engineering (76 features)
│   └── train.py                    # XGBoost training (GroupKFold, Optuna, Box-Cox)
├── pinn/
│   ├── model.py                    # EEPredictor: 7→128→ResBlock×3→32→1+Sigmoid
│   ├── physics.py                  # R1 (N/P monotonicity), R2 (Gibbs), R3 (boundary)
│   ├── preprocess.py               # 7-feature extraction, _load_core(), StandardScaler
│   └── train.py                    # Training loop, GroupKFold + random split modes
├── api/main.py                     # FastAPI REST service (XGBoost only currently)
├── artifacts/
│   ├── model.pkl, transformer.pkl  # XGBoost model + Box-Cox
│   ├── metadata.json               # Hyperparams, fold metrics, SHAP top10, medians
│   ├── shap_values.npy             # (523, 76) SHAP matrix
│   ├── shap_importance.csv         # Feature importance ranking
│   └── pinn/
│       ├── best_model.pt           # PINN checkpoint (7 features)
│       ├── pinn_metrics.json       # Val R²=0.677, RMSE=8.71 (random split)
│       ├── pinn_training_history.json  # 300 epochs
│       └── groupkfold_results.json # OOF R²=0.438, RMSE=11.25%
├── LNP_EE_Predictor.py             # Streamlit landing page
├── streamlit_utils.py              # Shared Streamlit utilities
├── pages/
│   ├── 1_Data_Exploration.py
│   ├── 2_XGBoost_Model.py
│   └── 3_PINN_Model.py
└── tests/test_pipeline.py
```

## Current Model Results

| Model | Samples | Features | OOF R² | OOF RMSE | CV Method |
|-------|---------|----------|--------|----------|-----------|
| XGBoost | 523 | 76 | 0.136 | 21.24% | GroupKFold (paper_doi) |
| PINN (7-feat) | 93 | 7 | 0.438 | 11.25% | GroupKFold (paper_doi) |
| PINN (7-feat, random) | 90 (72+18) | 7 | 0.677 | 8.71% | 80/20 random split |

---

## Known Issues

- **Streamlit app not yet tested** — committed but user hasn't run it yet. May have runtime issues.
- **`src/train_pinn.py`** has a broken import (`from pinn.preprocess import load_and_preprocess_df` — that function doesn't exist). This file is NOT used by the Streamlit app (which uses subprocess to call `pinn.train` instead). Could be deleted or fixed.
- **PINN scaler not saved as artifact** — the Streamlit app works around this by re-deriving the scaler via `load_and_preprocess()` on every cold start. A better fix would be to save `scaler.pkl` during training.
- **XGBoost underfits** — predicts narrow 60-90% range while actual spans 10-99%. Known issue documented in README.
- **Graphviz system dependency** — `st.graphviz_chart` needs system-level graphviz installed. On Streamlit Cloud, would need a `packages.txt` with `graphviz`.

---

## Git State

- **Branch**: `main`
- **Remote**: 2 commits ahead of `origin/main` (unpushed):
  - `a24ca5e` — Added beginnings of streamlit front-end
  - `b30c4e5` — Add Streamlit multipage dashboard (the big one)
- **Previous pushed commits**: `7338003` (GroupKFold for PINN), `2893549` (public release polish)

---

## User Preferences (from memory)

- Prefers few interactive graphs per page over many static ones
- Wants honest scientific communication (disclose limitations)
- Uses Streamlit (familiar with it)
- Not a frontend engineer — Python-focused
- Wants the project to impress PIs reviewing his GitHub
- Target audience: biophysics PIs at Broad, NYGC, HHMI
