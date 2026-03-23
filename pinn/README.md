# PINN — Physics-Informed Neural Network for LNP Encapsulation Efficiency

## Overview

This module implements a **Physics-Informed Neural Network (PINN)** that predicts
the encapsulation efficiency (EE) of lipid nanoparticles (LNPs) from their
physicochemical formulation parameters.

Unlike a purely data-driven model, the PINN augments the supervised MSE loss
with **physics residuals** derived from first-principles knowledge of lipid
self-assembly and electrostatic cargo encapsulation. This improves generalization
from the small labeled datasets typical in LNP literature.

## Physics Priors Encoded

| Residual | Physical Principle | Reference |
|----------|-------------------|-----------|
| **R1** — N/P monotonicity | EE increases with N/P ratio in the charge-limited regime (NP < ~6) | Kulkarni et al., *ACS Nano* 2018 |
| **R2** — Thermodynamic mixing | Sign of ∂EE/∂x_IL matches sign of −d(ΔG_mix)/dx_IL (ideal binary mixing) | Jayaraman et al., *Angew. Chem.* 2012 |
| **R3** — Boundary condition | EE → 0 as particle size → ∞ (infinite dilution / no self-assembly) | — |

## Architecture

```
Input (7 features)
    ↓
Linear → LayerNorm → GELU          [input projection]
    ↓
[ResidualBlock × 3]                 [skip-connected MLP trunk]
    ↓
Linear → GELU → Linear → Sigmoid   [output head, EE ∈ [0,1]]
```

## Input Features

| Feature | Derivation |
|---------|-----------|
| `ionizable_lipid_mole_fraction` | Parsed from molar ratio string (IL / total) |
| `np_ratio` | N/P proxy derived from x_IL (direct N/P pending cargo concentration data) |
| `particle_size_nm` | DLS Z-average diameter |
| `pdi` | Polydispersity index |
| `zeta_mv` | Surface zeta potential (mV) |
| `peg_fraction` | PEG-lipid mole fraction |
| `cholesterol_fraction` | Cholesterol mole fraction |

## Files

```
pinn/
├── __init__.py       — module init
├── model.py          — EEPredictor (MLP with residual blocks)
├── physics.py        — Physics residuals R1, R2, R3 + total_physics_loss()
├── preprocess.py     — Feature engineering + StandardScaler pipeline
├── train.py          — Training loop (data loss + physics loss)
└── README.md         — This file
```

## Training

```bash
# From lnp_ml_phase1/
python -m pinn.train \
    --data data/lnp_firstpaper_encapsulation.csv \
    --epochs 200 \
    --alpha 0.5 \
    --lr 1e-3 \
    --out outputs/pinn
```

`--alpha` controls the physics loss weight. `alpha=0` reduces to pure MSE regression.

## Current Status

- ✅ Model architecture implemented and tested
- ✅ Physics residuals R1, R2, R3 implemented with autograd
- ✅ Preprocessing pipeline parses real LNP Atlas EE data (48 records)
- ⏳ Dataset expansion ongoing (targeting >200 EE records from literature)
- ⏳ Hyperparameter sweep (alpha, hidden_dim, n_residual)
- ⏳ Comparison vs. pure-data XGBoost baseline

## Limitations & Honest Caveats

- N/P ratio is currently approximated from x_IL; direct N/P requires cargo
  concentration data not yet consistently reported in extraction records.
- Current EE dataset (LNP Atlas) has limited variance in EE values (clustered
  near 95%); more diverse experimental data is needed for meaningful generalization.
- Model is at MVP/proof-of-concept stage. Performance metrics pending larger dataset.
