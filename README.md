# LNP-EE Predictor 🧬

> Machine learning and physics-informed neural network (PINN) pipeline for predicting **lipid nanoparticle (LNP) encapsulation efficiency (EE%)** from formulation parameters and molecular descriptors.

Trained on the open-source [LNP Atlas](https://www.nature.com/articles/s41565-023-01511-6) dataset (1,092 LNP formulations from peer-reviewed literature).

**Status:** Active development — XGBoost baseline + PINN complete, heteroscedasticity modeling and transformer architecture planned.

---

## Results

| Model | Val R² | Val RMSE | Val MAE | CV Strategy | Notes |
|---|---|---|---|---|---|
| XGBoost (Optuna-tuned) | 0.160 | 20.94% | 15.28% | GroupKFold (paper DOI) | Leakage-free; paper-level generalization |
| **PINN** | **0.580** | **10.12%** | — | Random 80/20 split | 93 rows; 7 physicochemical features; physics residuals R1–R3 |

### Honest interpretation

The PINN's higher R² vs XGBoost is **not an apples-to-apples comparison**:
- XGBoost uses `GroupKFold` on `paper_doi` — the hardest eval, testing generalization to unseen papers
- PINN uses random 80/20 split on 7 features — easier split, fewer features, smaller effective dataset (93 rows with all 7 features complete; EE values cluster 75–95% limiting variance)

The PINN R²=0.58 is modest but expected for a physics-constrained model on a small dataset. The XGBoost R²=0.160 under strict group-based CV reflects the genuine difficulty of cross-paper generalization — a more realistic deployment scenario.

Neither model should be used as a substitute for experimental measurement at this stage.

---

## Models

### XGBoost Baseline
- 523 rows with EE%, 76 features (Morgan fingerprints, RDKit descriptors, synthesis conditions, interaction features)
- Optuna 50-trial hyperparameter search
- GroupKFold CV on `paper_doi` — prevents data leakage from correlated same-paper LNPs
- High-EE classification (≥80%): Precision=0.867, Recall=0.561, F1=0.681

### PINN (Physics-Informed Neural Network)
Residual-block MLP augmented with physics-derived loss terms:

| Residual | Physical Principle | Reference |
|---|---|---|
| **R1** N/P monotonicity | EE increases with N/P in the charge-limited regime (NP < ~6) | Kulkarni et al., *ACS Nano* 2018 |
| **R2** Thermodynamic mixing | ∂EE/∂x_IL sign constrained by Gibbs free energy of ideal binary mixing | Jayaraman et al., *Angew. Chem.* 2012 |
| **R3** Boundary condition | EE → 0 as particle size → ∞ | — |

Training objective: `L_total = L_data + α · L_physics` (α=0.3)

Architecture:
```
Input (7 features)
    ↓
Linear → LayerNorm → GELU          [input projection]
    ↓
[ResidualBlock × 3]                 [skip-connected MLP trunk]
    ↓
Linear → GELU → Linear → Sigmoid   [output head, EE ∈ [0,1]]
```

---

## Project Structure

```
lnp-ee-predictor/
├── src/
│   ├── features.py       # Feature engineering pipeline (76 features)
│   ├── train.py          # XGBoost: Optuna + GroupKFold + SHAP
│   └── train_pinn.py     # PINN training script
├── pinn/
│   ├── model.py          # EEPredictor: ResidualBlock MLP
│   ├── physics.py        # Physics residuals R1, R2, R3
│   ├── preprocess.py     # Feature engineering + LNP Atlas adapter
│   └── train.py          # Standalone PINN training loop
├── api/
│   └── main.py           # FastAPI REST API (XGBoost backend)
├── notebooks/
│   └── analysis.ipynb    # EDA, SHAP plots, model explainability
├── tests/
│   └── test_pipeline.py
├── artifacts/            # Saved models, SHAP values (gitignored)
├── data/                 # Raw CSV (gitignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train XGBoost
```bash
# Place lnp_atlas_export.csv in data/
python3 src/train.py
```

### Train PINN
```bash
python3 src/train_pinn.py
```

### Run the API
```bash
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

### Docker
```bash
docker-compose up --build
```

---

## API Reference

### `POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ionizable_lipid": "ALC-0315",
    "peg_lipid": "ALC-0159",
    "sterol_lipid": "cholesterol",
    "helper_lipid": "DSPC",
    "lipid_molar_ratio": "46.3:9.4:42.7:1.6",
    "target_type": "mRNA",
    "particle_size_nm": 80,
    "pdi": 0.10,
    "synthesis_method": "microfluidic",
    "buffer_ph": 4.0
  }'
```

**Response:**
```json
{
  "predicted_ee_percent": 84.3,
  "is_high_ee": true,
  "confidence_note": "High confidence — most features provided"
}
```

---

## Known Limitations

- **Custom lipids** without SMILES use frequency-rank encoding — a known weakness for novel scaffolds
- **N/P ratio** is approximated from ionizable lipid mole fraction; true N/P requires cargo concentration data not consistently reported in the literature
- **Small PINN training set**: only 90 rows had all 7 physicochemical features complete; performance will improve with larger data
- **No biological activity** predicted — EE% is necessary but not sufficient for transfection potency

---

## Roadmap

- [x] XGBoost baseline with GroupKFold + Optuna + SHAP
- [x] PINN with physics residuals (R1: N/P monotonicity, R2: thermodynamic mixing, R3: boundary condition)
- [ ] Fix heteroscedasticity (Box-Cox + variance-weighted loss)
- [ ] Transformer/attention architecture (inspired by COMET)
- [ ] Expand PINN dataset — impute missing N/P from synthesis conditions
- [ ] Unified eval: PINN on GroupKFold for fair comparison with XGBoost

---

## Citation

If you use this model, please cite the LNP Atlas:

> Sebastián Rojas Quiñones et al. "The LNP Atlas: A Comprehensive Repository of Lipid Nanoparticle Formulations for Nucleic Acid Delivery." *Nature Nanotechnology* (2023).

---

## Author

Wesley Gilbert — BS Biophysics, Scientific Computing  
[Website](https://www.wesley-j-gilbert.com)

## License

MIT
