"""
FastAPI service for LNP Encapsulation Efficiency prediction.

Endpoints:
  POST /predict          — predict EE% for a single LNP formulation
  POST /predict/batch    — predict EE% for up to 100 formulations
  GET  /model/info       — model metadata, training metrics, top SHAP features
  GET  /health           — liveness check
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings("ignore")

# Path resolution works whether run from /api or project root
_HERE = Path(__file__).parent
ARTIFACTS_DIR = _HERE / "artifacts"

# ─────────────────────────────────────────────────────────────────────────────
# Load artifacts at startup
# ─────────────────────────────────────────────────────────────────────────────

def _load_artifacts():
    model_path = ARTIFACTS_DIR / "model.pkl"
    meta_path = ARTIFACTS_DIR / "metadata.json"

    if not model_path.exists():
        raise RuntimeError(
            f"Model artifact not found at {model_path}. "
            "Run `python src/train.py` first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(meta_path) as f:
        metadata = json.load(f)

    transformer = None
    transformer_path = ARTIFACTS_DIR / "transformer.pkl"
    if transformer_path.exists():
        with open(transformer_path, "rb") as f:
            transformer = pickle.load(f)

    return model, metadata, transformer


try:
    MODEL, METADATA, TRANSFORMER = _load_artifacts()
    FEATURE_COLS = METADATA["feature_columns"]
    MODEL_READY = True
except Exception as e:
    print(f"WARNING: Could not load model — {e}")
    MODEL, METADATA, TRANSFORMER, FEATURE_COLS = None, {}, None, []
    MODEL_READY = False


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class LNPFormulation(BaseModel):
    """
    A single LNP formulation for prediction.
    All fields are optional — missing values are imputed using training-set medians.
    Provide at least ionizable_lipid_smiles for a meaningful prediction.
    """

    # Lipid SMILES (provide at least the ionizable lipid)
    ionizable_lipid_smiles: Optional[str] = Field(
        None, description="SMILES string for the ionizable lipid",
        examples=["CC(C)N(CC(O)CCCCCCCCCCCC)CC(O)CCCCCCCCCCCC"]
    )
    peg_lipid_smiles: Optional[str] = Field(
        None, description="SMILES string for the PEG lipid"
    )
    sterol_lipid_smiles: Optional[str] = Field(
        None, description="SMILES string for the sterol lipid (usually cholesterol)"
    )
    helper_lipid_smiles: Optional[str] = Field(
        None, description="SMILES string for the helper lipid (e.g. DSPC, DOPE)"
    )

    # Lipid names (used for frequency-rank feature)
    ionizable_lipid: Optional[str] = Field(None, examples=["ALC-0315", "SM-102"])
    peg_lipid: Optional[str] = Field(None, examples=["ALC-0159", "PEG2000-DMG"])
    helper_lipid: Optional[str] = Field(None, examples=["DSPC", "DOPE"])
    sterol_lipid: Optional[str] = Field(None, examples=["cholesterol"])

    # Formulation parameters
    lipid_molar_ratio: Optional[str] = Field(
        None,
        description="Molar ratio as 'ionizable:peg:sterol:helper', e.g. '50:1.5:38.5:10'",
        examples=["50:1.5:38.5:10", "46.3:9.4:42.7:1.6"]
    )
    target_type: Optional[str] = Field(
        None,
        description="Nucleic acid cargo type",
        examples=["mRNA", "siRNA", "DNA"]
    )

    # Measured properties (if already characterized)
    particle_size_nm: Optional[float] = Field(None, ge=0, le=2000, description="Particle size in nm")
    pdi: Optional[float] = Field(None, ge=0, le=1, description="Polydispersity index")
    zeta_potential_mv: Optional[float] = Field(None, description="Zeta potential in mV")

    # Synthesis conditions
    synthesis_method: Optional[str] = Field(None, examples=["microfluidic", "bulk"])
    flow_rate_ml_min: Optional[float] = Field(None, ge=0)
    flow_ratio: Optional[float] = Field(None, ge=0)
    buffer_ph: Optional[float] = Field(None, ge=0, le=14)

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v):
        valid = {"mRNA", "siRNA", "DNA", "ASO", "protein", "empty", "mRNA, siRNA", "gold_nanoparticles"}
        if v is not None and v not in valid:
            raise ValueError(f"target_type must be one of {sorted(valid)}")
        return v


class PredictionResponse(BaseModel):
    predicted_ee_percent: float = Field(description="Predicted encapsulation efficiency (%)")
    is_high_ee: bool = Field(description=f"Whether predicted EE% ≥ 80% (high encapsulation)")
    confidence_note: str
    input_summary: dict


class BatchPredictionRequest(BaseModel):
    formulations: List[LNPFormulation] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    n_formulations: int
    mean_predicted_ee: float
    n_high_ee: int


# ─────────────────────────────────────────────────────────────────────────────
# Feature construction for inference
# ─────────────────────────────────────────────────────────────────────────────

# Import feature functions (same code as training — no skew)
import sys
sys.path.insert(0, str(_HERE.parent / "src"))
from features import (
    get_rdkit_stats, parse_molar_ratio, parse_synthesis_info,
    CARGO_DIFFICULTY, RDKIT_DESCRIPTOR_NAMES,
)


def formulation_to_feature_row(form: LNPFormulation, training_medians: dict) -> dict:
    """
    Convert a LNPFormulation pydantic object into a flat feature dict
    aligned to FEATURE_COLS from the trained model.
    """
    row = {}

    # ── Molecular descriptors ─────────────────────────────────────────────
    lipid_smiles_map = {
        "ionizable_lipid": form.ionizable_lipid_smiles,
        "peg_lipid": form.peg_lipid_smiles,
        "sterol_lipid": form.sterol_lipid_smiles,
        "helper_lipid": form.helper_lipid_smiles,
    }
    for lipid_type, smiles in lipid_smiles_map.items():
        stats = get_rdkit_stats(smiles)
        row[f"{lipid_type}_smiles_valid"] = int(smiles is not None and stats["MW"] is not np.nan)
        for desc_name, val in stats.items():
            col = f"{lipid_type}__{desc_name}"
            row[col] = val if val is not None and not (isinstance(val, float) and np.isnan(val)) \
                         else training_medians.get(col, 0.0)

    # ── Molar ratios ──────────────────────────────────────────────────────
    ratios = parse_molar_ratio(form.lipid_molar_ratio)
    row.update(ratios)

    # ── Interaction features ──────────────────────────────────────────────
    il_logp = row.get("ionizable_lipid__LogP", 0.0)
    il_ratio = ratios.get("ratio_ionizable", 0.0) or 0.0
    peg_mw = row.get("peg_lipid__MW", 0.0)
    peg_ratio = ratios.get("ratio_peg", 0.0) or 0.0
    row["feat__il_logp_x_ratio"] = il_logp * il_ratio
    row["feat__peg_mw_x_ratio"] = peg_mw * peg_ratio

    # ── Measurement features ──────────────────────────────────────────────
    row["particle_size_nm_std"] = form.particle_size_nm \
        if form.particle_size_nm is not None else training_medians.get("particle_size_nm_std", 100.0)
    row["particle_size_nm_std__missing"] = int(form.particle_size_nm is None)

    row["pdi_std"] = form.pdi \
        if form.pdi is not None else training_medians.get("pdi_std", 0.15)
    row["pdi_std__missing"] = int(form.pdi is None)

    row["zeta_potential_mv_std"] = form.zeta_potential_mv \
        if form.zeta_potential_mv is not None else training_medians.get("zeta_potential_mv_std", -5.0)
    row["zeta_potential_mv_std__missing"] = int(form.zeta_potential_mv is None)

    # ── Synthesis features ────────────────────────────────────────────────
    # If synthesis_info string is provided use parser; otherwise use direct fields
    if form.synthesis_method:
        synth_cat = form.synthesis_method.lower()
        is_micro = int("microfluidic" in synth_cat or "t-junction" in synth_cat)
    else:
        synth_cat = "unknown"
        is_micro = 0

    row["synth_is_microfluidic"] = is_micro
    row["synth_flow_rate"] = form.flow_rate_ml_min \
        if form.flow_rate_ml_min is not None else training_medians.get("synth_flow_rate", 12.0)
    row["synth_flow_ratio"] = form.flow_ratio \
        if form.flow_ratio is not None else training_medians.get("synth_flow_ratio", 3.0)
    row["synth_ph"] = form.buffer_ph \
        if form.buffer_ph is not None else training_medians.get("synth_ph", 4.0)

    # Synthesis method one-hot (must match training dummy columns)
    for method in ["bulk", "ethanol_injection", "film_hydration", "microfluidic",
                   "nanoprecipitation", "snalp", "unknown"]:
        row[f"synth_{method}"] = int(synth_cat == method)

    # ── Cargo type ────────────────────────────────────────────────────────
    cargo = form.target_type or "mRNA"
    row["target_difficulty"] = CARGO_DIFFICULTY.get(cargo, 3)
    for t in ["ASO", "DNA", "empty", "gold_nanoparticles", "mRNA", "mRNA, siRNA",
              "protein", "siRNA"]:
        row[f"target_{t}"] = int(cargo == t)

    # ── Lipid frequency ranks (use 0 = unseen lipid) ─────────────────────
    freq_ranks = METADATA.get("lipid_freq_ranks", {})
    for col_prefix in ["ionizable_lipid", "peg_lipid", "helper_lipid", "sterol_lipid"]:
        lipid_name = getattr(form, col_prefix, None) or ""
        rank_key = f"{col_prefix}__freq_rank"
        row[rank_key] = freq_ranks.get(f"{col_prefix}:{lipid_name}", 0.0)

    return row


def predict_single(form: LNPFormulation) -> PredictionResponse:
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    training_medians = METADATA.get("training_medians", {})
    row = formulation_to_feature_row(form, training_medians)

    # Align to training feature columns
    X = np.array([row.get(col, 0.0) for col in FEATURE_COLS], dtype=np.float32).reshape(1, -1)
    raw_pred = MODEL.predict(X)
    if TRANSFORMER is not None:
        raw_pred = TRANSFORMER.inverse_transform(raw_pred.reshape(-1, 1)).ravel()
    pred = float(np.clip(raw_pred[0], 0, 100))

    n_missing = sum(1 for v in row.values() if v is None or (isinstance(v, float) and np.isnan(v)))
    confidence = (
        "High confidence — most features provided"
        if n_missing < 5
        else f"Moderate confidence — {n_missing} features imputed from training medians"
    )

    return PredictionResponse(
        predicted_ee_percent=round(pred, 2),
        is_high_ee=pred >= METADATA.get("high_ee_threshold", 80.0),
        confidence_note=confidence,
        input_summary={
            "ionizable_lipid": form.ionizable_lipid or "(SMILES provided)" if form.ionizable_lipid_smiles else "unknown",
            "target_type": form.target_type or "not specified",
            "lipid_molar_ratio": form.lipid_molar_ratio or "not specified",
            "particle_size_nm": form.particle_size_nm,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LNP-EE Predictor",
    description=(
        "Predict lipid nanoparticle encapsulation efficiency (EE%) "
        "from formulation parameters and molecular descriptors. "
        "Trained on the LNP Atlas open-source dataset."
    ),
    version="1.0.0",
    contact={"name": "LNP-EE Predictor"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_READY}


@app.get("/model/info")
def model_info():
    if not METADATA:
        raise HTTPException(status_code=503, detail="Model metadata not available.")
    return {
        "trained_at": METADATA.get("trained_at"),
        "n_training_samples": METADATA.get("n_train_samples"),
        "n_features": METADATA.get("n_features"),
        "oof_metrics": METADATA.get("oof_metrics"),
        "high_ee_threshold_percent": METADATA.get("high_ee_threshold"),
        "top10_shap_features": METADATA.get("shap_top10"),
        "hyperparameters": METADATA.get("best_hyperparams"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(formulation: LNPFormulation):
    """
    Predict encapsulation efficiency for a single LNP formulation.
    Returns EE% prediction and whether it exceeds the high-EE threshold (80%).
    """
    return predict_single(formulation)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Predict EE% for up to 100 LNP formulations in one request.
    """
    predictions = [predict_single(f) for f in request.formulations]
    preds = [p.predicted_ee_percent for p in predictions]
    return BatchPredictionResponse(
        predictions=predictions,
        n_formulations=len(predictions),
        mean_predicted_ee=round(float(np.mean(preds)), 2),
        n_high_ee=sum(p.is_high_ee for p in predictions),
    )
