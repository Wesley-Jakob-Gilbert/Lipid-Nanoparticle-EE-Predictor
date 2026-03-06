"""
Tests for LNP-EE Predictor — covers feature engineering and API endpoints.
Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import (
    standardize_numeric,
    parse_molar_ratio,
    parse_synthesis_info,
    get_rdkit_stats,
    CARGO_DIFFICULTY,
)


# ─────────────────────────────────────────────────────────────────────────────
# standardize_numeric
# ─────────────────────────────────────────────────────────────────────────────

class TestStandardizeNumeric:
    def test_plain_float(self):
        assert standardize_numeric("85.5") == pytest.approx(85.5)

    def test_approximation(self):
        assert standardize_numeric("~100") == pytest.approx(100.0)

    def test_plus_minus(self):
        # 183.3 ± 1.5 → 183.3
        assert standardize_numeric("183.3 ± 1.5") == pytest.approx(183.3)

    def test_range(self):
        # 77-350 → midpoint
        assert standardize_numeric("77-350") == pytest.approx(213.5)

    def test_less_than(self):
        # <0.02 → 0.01
        assert standardize_numeric("<0.02") == pytest.approx(0.01)

    def test_z_average(self):
        # '61 (Z-average); 36 (number average)' → 61
        assert standardize_numeric("61 (Z-average); 36 (number average)") == pytest.approx(61.0)

    def test_nan_passthrough(self):
        assert np.isnan(standardize_numeric(None))
        assert np.isnan(standardize_numeric(""))
        assert np.isnan(standardize_numeric(float("nan")))

    def test_negative(self):
        assert standardize_numeric("-5.2") == pytest.approx(-5.2)


# ─────────────────────────────────────────────────────────────────────────────
# parse_molar_ratio
# ─────────────────────────────────────────────────────────────────────────────

class TestParseMolarRatio:
    def test_standard(self):
        result = parse_molar_ratio("50:1.5:38.5:10")
        assert result["ratio_ionizable"] == pytest.approx(50.0)
        assert result["ratio_peg"] == pytest.approx(1.5)
        assert result["ratio_sterol"] == pytest.approx(38.5)
        assert result["ratio_helper"] == pytest.approx(10.0)

    def test_annotated(self):
        result = parse_molar_ratio("35:16:46.5:2.5 (ionizable:chol:DOPE:PEG)")
        assert result["ratio_ionizable"] == pytest.approx(35.0)

    def test_normalization(self):
        result = parse_molar_ratio("50:1.5:38.5:10")
        total = 50 + 1.5 + 38.5 + 10
        assert result["ratio_ionizable_norm"] == pytest.approx(50.0 / total)

    def test_none_input(self):
        result = parse_molar_ratio(None)
        assert np.isnan(result["ratio_ionizable"])

    def test_malformed(self):
        result = parse_molar_ratio("not a ratio")
        assert np.isnan(result["ratio_ionizable"])


# ─────────────────────────────────────────────────────────────────────────────
# parse_synthesis_info
# ─────────────────────────────────────────────────────────────────────────────

class TestParseSynthesisInfo:
    _sample = (
        "synthesis_method: Microfluidic; device_used: T-junction device; "
        "total_flow_rate_ml_min: 12; flow_rate_ratio: 3:1; "
        "aqueous_phase_composition: 10 mM citrate buffer, pH 4.0; "
        "organic_phase_composition: ethanol"
    )

    def test_microfluidic_detection(self):
        result = parse_synthesis_info(self._sample)
        assert result["synth_method_cat"] == "microfluidic"
        assert result["synth_is_microfluidic"] == 1

    def test_flow_rate(self):
        result = parse_synthesis_info(self._sample)
        assert result["synth_flow_rate"] == pytest.approx(12.0)

    def test_ph_parsing(self):
        result = parse_synthesis_info(self._sample)
        assert result["synth_ph"] == pytest.approx(4.0)

    def test_nan_input(self):
        result = parse_synthesis_info(None)
        assert result["synth_method_cat"] == "unknown"
        assert np.isnan(result["synth_flow_rate"])

    def test_bulk_method(self):
        result = parse_synthesis_info("synthesis_method: Pipette mixing")
        assert result["synth_method_cat"] == "bulk"
        assert result["synth_is_microfluidic"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# get_rdkit_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRdkitStats:
    # Cholesterol SMILES
    _cholesterol = "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"

    def test_cholesterol_mw(self):
        stats = get_rdkit_stats(self._cholesterol)
        if stats["MW"] is not np.nan:  # only if RDKit installed
            assert 380 < stats["MW"] < 390  # Cholesterol MW ≈ 386.65

    def test_none_input(self):
        stats = get_rdkit_stats(None)
        assert np.isnan(stats["MW"])

    def test_invalid_smiles(self):
        stats = get_rdkit_stats("NOT_A_SMILES_XYZ")
        assert np.isnan(stats["MW"])

    def test_empty_string(self):
        stats = get_rdkit_stats("")
        assert np.isnan(stats["MW"])


# ─────────────────────────────────────────────────────────────────────────────
# CARGO_DIFFICULTY ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestCargoDifficulty:
    def test_ordering(self):
        # siRNA should be easier than mRNA which should be easier than DNA
        assert CARGO_DIFFICULTY["siRNA"] < CARGO_DIFFICULTY["mRNA"]
        assert CARGO_DIFFICULTY["mRNA"] < CARGO_DIFFICULTY["DNA"]

    def test_empty_is_zero(self):
        assert CARGO_DIFFICULTY["empty"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# API integration tests (requires model artifacts to be built)
# ─────────────────────────────────────────────────────────────────────────────

def _api_available():
    """Skip API tests if model artifacts haven't been built yet."""
    artifacts = Path(__file__).parent.parent / "artifacts" / "model.pkl"
    return artifacts.exists()


@pytest.mark.skipif(not _api_available(), reason="Model artifacts not built yet")
class TestAPI:
    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
        from main import app
        self.client = TestClient(app)

    def test_health(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_predict_minimal(self):
        payload = {
            "ionizable_lipid": "ALC-0315",
            "target_type": "mRNA",
            "lipid_molar_ratio": "46.3:9.4:42.7:1.6",
        }
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["predicted_ee_percent"] <= 100
        assert isinstance(data["is_high_ee"], bool)

    def test_predict_with_smiles(self):
        payload = {
            "ionizable_lipid_smiles": "CC(C)N(CC(O)CCCCCCCCCCCC)CC(O)CCCCCCCCCCCC",
            "target_type": "siRNA",
            "lipid_molar_ratio": "50:1.5:38.5:10",
            "particle_size_nm": 90.0,
            "pdi": 0.12,
        }
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert 0 <= resp.json()["predicted_ee_percent"] <= 100

    def test_batch_predict(self):
        payload = {
            "formulations": [
                {"ionizable_lipid": "ALC-0315", "target_type": "mRNA"},
                {"ionizable_lipid": "SM-102", "target_type": "mRNA"},
                {"ionizable_lipid": "DLin-MC3-DMA", "target_type": "siRNA"},
            ]
        }
        resp = self.client.post("/predict/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_formulations"] == 3
        assert len(data["predictions"]) == 3

    def test_invalid_target_type(self):
        payload = {"ionizable_lipid": "ALC-0315", "target_type": "INVALID"}
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_model_info(self):
        resp = self.client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "oof_metrics" in data
        assert "top10_shap_features" in data
