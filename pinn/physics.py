"""
physics.py — Physics residuals for the LNP EE PINN.

The encapsulation efficiency (EE) of an LNP is governed by the interplay of:

1. Electrostatic attraction between the cationic/ionizable lipid headgroup
   and the anionic cargo (mRNA phosphate backbone).
2. Lipid mixing thermodynamics: the free energy of mixing across the
   ionizable lipid mole fraction (x_IL) determines the energetic favorability
   of forming a stable core.
3. N/P ratio (nitrogen-to-phosphate ratio): controls charge neutralization;
   suboptimal N/P leaves cargo un-encapsulated.

Physics-informed residual losses enforce:

    R1 (monotonic EE–N/P): dEE/dNP > 0 for NP < NP_opt  (charge-limited regime)
    R2 (thermodynamic mixing): ΔG_mix = RT[x·ln(x) + (1-x)·ln(1-x)] drives
        an optimal x_IL that maximizes EE; penalty when predicted EE gradient
        w.r.t. x_IL disagrees with the sign of d(ΔG_mix)/dx_IL.
    R3 (boundary condition): EE → 0 as particle_size → ∞ (infinite dilution limit)
        enforced via a soft penalty on size-extrapolated predictions.

References:
    - Kulkarni et al., ACS Nano 2018 (ionizable lipid pKa and EE)
    - Kulkarni et al., Nanoscale 2019 (N/P ratio effects on mRNA EE)
    - Jayaraman et al., Angew. Chem. 2012 (lipid mixing and phase behavior)
"""

import jax
import jax.numpy as jnp


# ─── Physical constants ────────────────────────────────────────────────────────
R = 8.314e-3  # kJ / (mol·K)
T = 310.15    # K (physiological temperature, 37°C)


def delta_g_mixing(x_il: jax.Array) -> jax.Array:
    """
    Ideal Gibbs free energy of mixing for a binary lipid mixture.

    ΔG_mix / (RT) = x·ln(x) + (1−x)·ln(1−x)

    Args:
        x_il: Ionizable lipid mole fraction, shape (...,). Values in (0, 1).

    Returns:
        Dimensionless ΔG_mix / RT, same shape as x_il.
    """
    eps = 1e-7
    x = jnp.clip(x_il, eps, 1.0 - eps)
    return x * jnp.log(x) + (1.0 - x) * jnp.log(1.0 - x)


def d_delta_g_mixing_dx(x_il: jax.Array) -> jax.Array:
    """
    Analytical derivative of ΔG_mix w.r.t. x_il.

    d(ΔG/RT)/dx = ln(x) - ln(1-x)  (i.e., ln(x / (1-x)))

    Sign tells us whether increasing x_il lowers (negative) or
    raises (positive) mixing free energy.
    """
    eps = 1e-7
    x = jnp.clip(x_il, eps, 1.0 - eps)
    return jnp.log(x) - jnp.log(1.0 - x)


def residual_np_monotonicity(
    model,
    params,
    x_batch: jax.Array,
    np_col: int,
    np_opt: float = 6.0,
) -> jax.Array:
    """
    R1: EE should increase with N/P ratio below the optimal N/P.

    Penalizes negative ∂EE/∂NP for samples where NP < np_opt.

    Args:
        model:    The PINN (EEPredictor).
        params:   Model parameters pytree.
        x_batch:  Feature batch, shape (B, n_features).
        np_col:   Column index of the N/P ratio feature.
        np_opt:   Optimal N/P threshold (default 6.0 per literature).

    Returns:
        Scalar residual loss.
    """
    def ee_sum(x):
        return model.apply({'params': params}, x, training=False).sum()

    grad = jax.grad(ee_sum)(x_batch)          # (B, n_features)
    d_ee_d_np = grad[:, np_col]               # (B,)

    np_vals = x_batch[:, np_col]
    mask = (np_vals < np_opt).astype(jnp.float32)
    return (jax.nn.relu(-d_ee_d_np) * mask).mean()


def residual_thermodynamic_mixing(
    model,
    params,
    x_batch: jax.Array,
    x_il_col: int,
) -> jax.Array:
    """
    R2: Sign of ∂EE/∂x_il should agree with -d(ΔG_mix)/dx_il.

    Rationale: where ΔG_mix is decreasing (d/dx < 0, x_il < 0.5),
    higher x_il lowers free energy → favors encapsulation → EE should rise.
    The PINN is penalized when its gradient disagrees with this physical sign.

    Args:
        model:     The PINN (EEPredictor).
        params:    Model parameters pytree.
        x_batch:   Feature batch, shape (B, n_features).
        x_il_col:  Column index of the ionizable lipid mole fraction feature.

    Returns:
        Scalar residual loss.
    """
    def ee_sum(x):
        return model.apply({'params': params}, x, training=False).sum()

    grad = jax.grad(ee_sum)(x_batch)
    d_ee_d_xil = grad[:, x_il_col]

    x_il_vals = x_batch[:, x_il_col]
    phys_sign = -d_delta_g_mixing_dx(x_il_vals)
    return jax.nn.relu(-d_ee_d_xil * phys_sign).mean()


def residual_boundary_size(
    model,
    params,
    size_col: int,
    n_features: int,
    large_size: float = 10.0,
) -> jax.Array:
    """
    R3: EE → 0 as particle size → ∞ (infinite dilution / poor self-assembly).

    Constructs a synthetic boundary point with all features at their
    "neutral" value (0 after normalization) except size, which is set
    to `large_size`. Penalizes high EE predictions at this boundary.

    Args:
        model:      The PINN.
        params:     Model parameters pytree.
        size_col:   Column index of the particle size feature (normalized).
        n_features: Total number of input features.
        large_size: Normalized value representing an unrealistically large particle.

    Returns:
        Scalar residual loss.
    """
    x_boundary = jnp.zeros((1, n_features))
    x_boundary = x_boundary.at[0, size_col].set(large_size)
    ee_boundary = model.apply({'params': params}, x_boundary, training=False)
    return (ee_boundary ** 2).mean()


def total_physics_loss(
    model,
    params,
    x_batch: jax.Array,
    np_col: int,
    x_il_col: int,
    size_col: int,
    n_features: int,
    lambda_np: float = 0.5,
    lambda_mix: float = 0.5,
    lambda_bc: float = 0.1,
) -> jax.Array:
    """
    Weighted sum of all physics residuals.

    Loss_phys = λ_np·R1 + λ_mix·R2 + λ_bc·R3

    Args:
        model:      EEPredictor instance.
        params:     Model parameters pytree.
        x_batch:    Feature tensor (B, n_features).
        np_col:     Index of N/P ratio feature.
        x_il_col:   Index of ionizable lipid mole fraction feature.
        size_col:   Index of particle size feature.
        n_features: Number of input features.
        lambda_np:  Weight for N/P monotonicity residual.
        lambda_mix: Weight for thermodynamic mixing residual.
        lambda_bc:  Weight for boundary condition residual.

    Returns:
        Scalar total physics loss.
    """
    r1 = residual_np_monotonicity(model, params, x_batch, np_col)
    r2 = residual_thermodynamic_mixing(model, params, x_batch, x_il_col)
    r3 = residual_boundary_size(model, params, size_col, n_features)
    return lambda_np * r1 + lambda_mix * r2 + lambda_bc * r3
