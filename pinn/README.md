# PINN — Physics-Informed Neural Network for LNP Encapsulation Efficiency

## Overview

A **Physics-Informed Neural Network (PINN)** that predicts LNP encapsulation efficiency (EE%)
from physicochemical formulation parameters. The model augments supervised MSE loss with
**physics residuals** derived from first-principles lipid self-assembly thermodynamics,
improving generalization from the small labeled datasets typical in LNP literature.

## Results (LNP Atlas, 93 rows with all 7 features complete)

| Split | R² | RMSE |
|---|---|---|
| Train (74 samples) | 0.993 | 1.28% |
| Val (19 samples) | 0.580 | 10.12% |

**Honest interpretation:** The model fits training data well but shows expected overfitting on the
small validation set (R²=0.58). This is largely a **data limitation** — only 93 of 1,092 LNP Atlas
records have all 7 required features present, and EE variance is compressed (most values cluster
75–95%). The physics priors (R1–R3) help constrain predictions to physically plausible regions
even where data is sparse. Model is best used for **rank-ordering candidate formulations**, not as
a precision measurement tool. Performance is expected to improve substantially with >300 labeled
EE records.

## Why Only 93 Rows?

The LNP Atlas contains 1,092 formulation records, of which ~527 have a quantitative EE% value.
The PINN requires all 7 input features to be present (no imputation). Per-feature availability
among the 527 EE rows:

| Feature | Available rows |
|---|---|
| `ionizable_lipid_mole_fraction` | 484 |
| `np_ratio` | 484 |
| `particle_size_nm` | 489 |
| `pdi` | 386 |
| `zeta_mv` | **97** ← bottleneck |
| `peg_fraction` | 484 |
| `cholesterol_fraction` | 484 |

`zeta_potential` is the binding constraint: only 97 of the ~527 EE rows have a reported zeta
value, which is why `dropna()` across all 7 features yields 93 complete rows.

**Future improvement:** Dropping `zeta_mv` from the feature set (6 features) would give
approximately **323 usable rows** — roughly 3.5× more training data. Alternatively, imputing
zeta from other formulation parameters (e.g., ionizable lipid pKa, mole fraction) could recover
more rows while retaining the physical signal. Either path is expected to substantially improve
validation performance.

## Physics Priors Encoded

| Residual | Physical Principle | Reference |
|---|---|---|
| **R1** N/P monotonicity | EE increases with N/P ratio in the charge-limited regime (NP < ~6) | Kulkarni et al., *ACS Nano* 2018 |
| **R2** Thermodynamic mixing | ∂EE/∂x_IL sign constrained by Gibbs free energy of ideal binary mixing: ΔG_mix = RT[x·ln(x)+(1−x)·ln(1−x)] | Jayaraman et al., *Angew. Chem.* 2012 |
| **R3** Boundary condition | EE → 0 as particle size → ∞ (infinite dilution / no self-assembly) | — |

Training loss: `L_total = L_data + 0.3 · L_physics`

## Architecture

```
Input (7 features)
    ↓
Linear → LayerNorm → GELU          [input projection]
    ↓
[ResidualBlock × 3]                 [skip-connected MLP trunk]
    ↓
Linear → GELU → Linear → Sigmoid   [output: EE ∈ [0,1]]
```

## Input Features

| Feature | Derivation |
|---|---|
| `ionizable_lipid_mole_fraction` | IL / total lipid from molar ratio string |
| `np_ratio` | N/P proxy = x_IL × 10 |
| `particle_size_nm` | DLS Z-average diameter |
| `pdi` | Polydispersity index |
| `zeta_mv` | Zeta potential (mV) |
| `peg_fraction` | PEG-lipid mole fraction |
| `cholesterol_fraction` | Cholesterol mole fraction |

## Training

```bash
# From project root
python -m pinn.train --data data/lnp_atlas_export.csv --epochs 300 --alpha 0.3 --out artifacts/pinn
```

## Files

```
pinn/
├── model.py       — EEPredictor (ResidualBlock MLP, sigmoid output)
├── physics.py     — Physics residuals R1, R2, R3 + total_physics_loss()
├── preprocess.py  — LNP Atlas feature engineering + StandardScaler
├── train.py       — Training loop
└── README.md      — This file
```
