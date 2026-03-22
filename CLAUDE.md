# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LNP-EE Predictor: An XGBoost-based ML system predicting lipid nanoparticle (LNP) encapsulation efficiency (EE%) trained on the LNP Atlas dataset (1,092 formulations). Includes a FastAPI REST service, feature engineering pipeline, and Docker support.

**Known Issues (from README):**
- Import paths are broken due to a directory reorganization (post-reorganization imports not fixed)
- The XGBoost model underfits — predicts a narrow 60–90% range while actual data spans 10–99%
- The `Heteroscedasticity/` directory contains guides (not code) for fixing the distribution issue

## Commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Train model (requires `data/lnp_atlas_export.csv` and `artifacts/` directory):
```bash
python src/train.py
```

Run API server:
```bash
uvicorn api.main:app --reload --port 8000
```

Run tests:
```bash
pytest tests/test_pipeline.py -v
```

Run a single test class:
```bash
pytest tests/test_pipeline.py::TestParseMolarRatio -v
```

Docker:
```bash
docker-compose up --build
```

## Architecture

### Data Flow
Raw CSV → `src/features.py` (feature engineering) → `src/train.py` (XGBoost + Optuna) → `artifacts/` → `api/main.py` (inference)

### Key Directories
- `src/` — Feature engineering (`features.py`) and training (`train.py`)
- `api/` — FastAPI service (`main.py`) serving 4 endpoints: `POST /predict`, `POST /predict/batch`, `GET /model/info`, `GET /health`
- `tests/` — pytest suite; API tests are skipped if `artifacts/model.pkl` is not present
- `artifacts/` — Gitignored. Contains `model.pkl`, `metadata.json`, `shap_values.npy`, `shap_importance.csv` after training
- `data/` — Gitignored. Place `lnp_atlas_export.csv` here before training
- `Heteroscedasticity/` — Documentation only (no runnable scripts); guides for fixing distribution mismatch

### Feature Engineering (`src/features.py`)
Produces 70+ features grouped as:
- **Molecular descriptors**: RDKit-computed (MW, LogP, TPSA, etc.) for each of 4 lipid components (ionizable, PEG, sterol, helper)
- **Molar ratios**: Parsed from `"50:1.5:38.5:10"` format strings
- **Synthesis info**: Extracted from free-text (`synthesis_method`, `flow_rate`, `flow_ratio`, `pH`)
- **Interaction features**: `IL_LogP × ratio_ionizable`, `PEG_MW × ratio_peg`
- **Categorical encoding**: One-hot for `target_type` (8 types) and `synthesis_method` (7 canonical methods); frequency ranking for lipid names
- **Cargo difficulty**: Ordinal encoding defined in `CARGO_DIFFICULTY` dict

### Model Training (`src/train.py`)
- Target column: `encapsulation_efficiency_percent_std`
- Data leakage prevention: `GroupKFold(n_splits=5)` grouped by `paper_doi`
- Optuna runs 50 trials optimizing 8 XGBoost hyperparameters (RMSE objective)
- Saves SHAP values and top-10 feature importances in `metadata.json`

### API Inference (`api/main.py`)
- Artifacts loaded at startup; all fields in `LNPFormulation` are optional
- Missing features imputed with training medians stored in `metadata.json`
- Predictions clipped to `[0, 100]`; `confidence_note` reflects count of missing features
