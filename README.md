# LNP-EE Predictor

Machine learning and physics-informed neural network pipeline for predicting **lipid nanoparticle (LNP) encapsulation efficiency (EE%)** from formulation parameters and molecular descriptors.

Trained on the open-source [LNP Atlas](https://www.nature.com/articles/s41565-023-01511-6) dataset — 1,092 LNP formulations curated from peer-reviewed literature.

**Status:** Active development. XGBoost baseline and PINN are complete. Heteroscedasticity correction and unified evaluation planned.

---

## Motivation

Encapsulation efficiency is one of the first go/no-go metrics in LNP development, yet it is measured empirically for nearly every new formulation. Predictive models could accelerate screening by rank-ordering candidate formulations before synthesis, reducing the number of wet-lab iterations needed to reach a high-EE lead.

The central challenge is data. Published LNP literature is heterogeneous — different labs report different subsets of formulation parameters, assay conditions vary, and EE values can reflect different measurement protocols. This project explores two complementary strategies for learning from that limited, noisy data: a high-capacity gradient-boosted tree model that extracts signal from rich feature engineering, and a physics-informed neural network that uses first-principles constraints to compensate for small labeled datasets.

---

## Results

| Model | Val R² | Val RMSE | Training Rows | Features | CV Strategy |
|---|---|---|---|---|---|
| XGBoost (Optuna-tuned) | 0.160 | 20.94% | 523 | 76 | GroupKFold on paper DOI |
| PINN (ResidualBlock MLP) | 0.580 | 10.12% | 93 | 7 | Random 80/20 split |

### Honest interpretation

The PINN's higher R² is **not an apples-to-apples comparison** and should not be read as "PINN beats XGBoost."

- The **XGBoost** evaluation uses `GroupKFold` on `paper_doi`, ensuring no formulations from the same paper appear in both training and validation. This is the hardest and most realistic split — it tests generalization to unseen labs and experimental protocols, which is what deployment actually requires.
- The **PINN** uses a random 80/20 split on 93 rows. This is a much easier evaluation: samples from the same paper can appear on both sides of the split, and the effective variance in EE is compressed because values in this subset cluster between 75–95%.

The PINN subset is also smaller because only 97 of 527 EE-labeled rows in the LNP Atlas have reported zeta potential — a required PINN input. That restriction to 93 complete rows limits generalizability.

The XGBoost R²=0.160 under strict group-based CV reflects the genuine difficulty of cross-paper generalization, not a failure of the model per se. The PINN R²=0.58 is encouraging for a physics-constrained model trained on 93 points, but the train R²=0.993 signals overfitting that the physics priors only partially mitigate at this dataset size.

**Neither model should be used as a substitute for experimental measurement.** The intended use is rank-ordering candidate formulations to reduce experimental screening burden.

---

## Models

### XGBoost Baseline

The baseline model applies gradient boosting to a rich feature representation of the LNP formulation.

- **523 rows** with EE% from the LNP Atlas; **76 features** spanning:
  - RDKit molecular descriptors (MW, LogP, TPSA, rotatable bonds, H-bond donors/acceptors) for each of four lipid components
  - Molar ratio parsing from strings such as `"50:1.5:38.5:10"`
  - Synthesis conditions extracted from free text (flow rate, flow ratio, buffer pH)
  - Interaction features: `IL_LogP × ratio_ionizable`, `PEG_MW × ratio_peg`
  - One-hot encoding for target cargo type and synthesis method; frequency-rank encoding for lipid names
  - Ordinal cargo difficulty score
- **GroupKFold(n_splits=5) on `paper_doi`** prevents data leakage from correlated same-paper formulations
- **Optuna 50-trial hyperparameter search** over 8 XGBoost hyperparameters (RMSE objective)
- **SHAP** values computed for all training samples; top-10 feature importances saved to `metadata.json`
- High-EE classification (threshold ≥80%): Precision=0.867, Recall=0.561, F1=0.681

### PINN (Physics-Informed Neural Network)

The PINN is the primary active development focus. It augments a supervised MSE loss with physics residuals derived from lipid self-assembly thermodynamics, with the goal of improving generalization from the small labeled datasets typical in LNP literature.

#### Input features

| Feature | Derivation |
|---|---|
| `ionizable_lipid_mole_fraction` | IL / total lipid, parsed from molar ratio string |
| `np_ratio` | N/P proxy = x_IL × 10 |
| `particle_size_nm` | DLS Z-average diameter |
| `pdi` | Polydispersity index |
| `zeta_mv` | Zeta potential (mV) |
| `peg_fraction` | PEG-lipid mole fraction |
| `cholesterol_fraction` | Cholesterol mole fraction |

#### Physics residuals

Three physics-informed loss terms constrain the model to physically plausible predictions, enforced via automatic differentiation through the network:

| Residual | Physical Principle | Reference |
|---|---|---|
| **R1** N/P monotonicity | EE increases with N/P ratio in the charge-limited regime (N/P < ~6); penalizes negative dEE/dNP below this threshold | Kulkarni et al., *ACS Nano* 2018; Kulkarni et al., *Nanoscale* 2019 |
| **R2** Thermodynamic mixing | Sign of dEE/dx_IL constrained by Gibbs free energy of ideal binary mixing: ΔG_mix = RT[x·ln(x) + (1−x)·ln(1−x)]; penalty applied when predicted gradient disagrees with thermodynamic expectation | Jayaraman et al., *Angew. Chem.* 2012 |
| **R3** Boundary condition | EE → 0 as particle size → ∞ (infinite dilution / failure of self-assembly); enforced via a synthetic boundary point during each training step | — |

Training objective: `L_total = L_data + 0.3 · (0.5·R1 + 0.5·R2 + 0.1·R3)`

#### Architecture

```
Input (7 features)
    |
Linear -> LayerNorm -> GELU          [input projection]
    |
[ResidualBlock x 3]                  [skip-connected MLP trunk]
    |
Linear -> GELU -> Linear -> Sigmoid  [output head, EE in [0, 1]]
```

Each ResidualBlock: `LayerNorm -> Linear -> GELU -> Linear`, with a skip connection summed before the next block. The sigmoid output enforces EE in [0, 1] by construction.

---

## Project Structure

```
lnp-ee-predictor/
|-- src/
|   |-- features.py       # Feature engineering pipeline (76 features)
|   |-- train.py          # XGBoost: Optuna + GroupKFold + SHAP
|   `-- train_pinn.py     # Entry point for PINN training
|-- pinn/
|   |-- model.py          # EEPredictor: ResidualBlock MLP, sigmoid output
|   |-- physics.py        # Physics residuals R1, R2, R3 and total_physics_loss()
|   |-- preprocess.py     # Feature engineering and LNP Atlas adapter
|   `-- train.py          # PINN training loop
|-- api/
|   `-- main.py           # FastAPI REST service (XGBoost backend)
|-- notebooks/
|   `-- analysis.ipynb    # EDA, SHAP plots, model explainability
|-- tests/
|   `-- test_pipeline.py
|-- artifacts/            # Saved models, SHAP values (gitignored)
|-- data/                 # Raw CSV (gitignored)
|-- Dockerfile
|-- docker-compose.yml
`-- requirements.txt
```

---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train XGBoost

```bash
python src/train.py
```

### Train PINN

```bash
python src/train_pinn.py
# or directly:
python -m pinn.train --data data/lnp_atlas_export.csv --epochs 300 --alpha 0.3 --out artifacts/pinn
```

### Run the API

```bash
uvicorn api.main:app --reload --port 8000
# Interactive docs: http://localhost:8000/docs
```

### Docker

```bash
docker-compose up --build
```

---

## API Reference

The REST API serves the XGBoost model. All fields are optional; missing features are imputed with training-set medians stored in `artifacts/metadata.json`.

### POST /predict

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

### Other endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict/batch` | POST | Batch predictions for multiple formulations |
| `/model/info` | GET | Model metadata and top feature importances |
| `/health` | GET | Liveness check |

---

## Known Limitations

- **PINN dataset size**: Only 93 of 1,092 LNP Atlas records have all 7 required features (zeta potential is the binding constraint — reported in fewer than 20% of EE-labeled rows). Performance is expected to improve substantially with >300 labeled, complete records.
- **Evaluation mismatch**: XGBoost and PINN use different CV strategies and different subsets of the data. A fair comparison requires running PINN under `GroupKFold` on a matched dataset — this is planned but not yet implemented.
- **Custom lipids**: Lipids without SMILES strings fall back to frequency-rank encoding, which carries no structural information. Novel scaffolds are a known weakness.
- **N/P ratio approximation**: True N/P requires cargo concentration, which is inconsistently reported. The proxy (x_IL × 10) introduces noise into R1 enforcement.
- **No biological activity**: EE% is necessary but not sufficient for transfection potency. Delivery efficacy in cells is not predicted.
- **Heteroscedasticity**: EE variance is higher in the mid-range (40–70%) than at the extremes. The current MSE objective does not account for this; Box-Cox transformation and variance-weighted loss are planned.

---

## Roadmap

- [x] XGBoost baseline: GroupKFold + Optuna + SHAP
- [x] PINN: ResidualBlock MLP with physics residuals R1 (N/P monotonicity), R2 (thermodynamic mixing), R3 (boundary condition)
- [ ] Heteroscedasticity correction: Box-Cox target transform + variance-weighted loss
- [ ] Unified evaluation: PINN under GroupKFold for direct comparison with XGBoost
- [ ] Expand PINN dataset: impute missing zeta potential or relax the feature requirement
- [ ] REST API endpoint for PINN inference (currently XGBoost only)
- [ ] Transformer / attention architecture for formulation-level sequence modeling

---

## Citation

If you use this code or the underlying dataset, please cite the LNP Atlas:

> Sebastián Rojas Quiñones et al. "The LNP Atlas: A Comprehensive Repository of Lipid Nanoparticle Formulations for Nucleic Acid Delivery." *Nature Nanotechnology* (2023). https://doi.org/10.1038/s41565-023-01511-6

Physics residual references:

> Kulkarni, J. A. et al. "On the formation and morphology of lipid nanoparticles containing ionizable cationic lipids and siRNA." *ACS Nano* 12, 4787–4795 (2018).

> Kulkarni, J. A. et al. "Lipid Nanoparticle Technology for Clinical Translation of siRNA Therapeutics." *Nanoscale* 11, 21733–21739 (2019).

> Jayaraman, M. et al. "Maximizing the Potency of siRNA Lipid Nanoparticles for Hepatic Gene Silencing In Vivo." *Angewandte Chemie International Edition* 51, 8529–8533 (2012).

---

## Author

Wesley Gilbert — BS Biophysics, Scientific Computing
[wesley-j-gilbert.com](https://www.wesley-j-gilbert.com)

## License

MIT
