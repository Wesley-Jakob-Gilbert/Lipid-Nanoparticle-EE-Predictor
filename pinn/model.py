"""
model.py — EEPredictor: Physics-Informed Neural Network for LNP
           Encapsulation Efficiency (EE) prediction.

Architecture:
    - Fully-connected MLP with residual skip connections every 2 layers.
    - Sigmoid output layer: EE ∈ [0, 1] (fraction, not percent).
    - Training objective: L_total = L_data + α·L_physics
        where L_data is MSE on labeled EE observations and
        L_physics is the weighted physics residual from physics.py.

Input features (normalized, continuous):
    - ionizable_lipid_mole_fraction  (x_IL, derived from molar_ratio)
    - np_ratio                       (N/P, derived from lipid/cargo charge)
    - particle_size_nm               (DLS Z-average)
    - pdi                            (polydispersity index)
    - peg_fraction                   (PEG-lipid mole fraction)
    - cholesterol_fraction           (cholesterol mole fraction)

Output:
    - ee_pred: predicted encapsulation efficiency ∈ [0, 1]
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class ResidualBlock(nn.Module):
    """Two-layer residual block with LayerNorm and GELU activation."""
    hidden_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.LayerNorm()(h)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout)(h, deterministic=not training)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.LayerNorm()(h)
        return nn.gelu(x + h)


class EEPredictor(nn.Module):
    """
    Physics-Informed Neural Network for LNP Encapsulation Efficiency.

    Args:
        n_features:  Number of input physicochemical features.
        hidden_dim:  Width of hidden layers (default 128).
        n_residual:  Number of residual blocks (default 3).
        dropout:     Dropout probability (default 0.1).

    Example:
        >>> model = EEPredictor(n_features=6)
        >>> key = jax.random.PRNGKey(0)
        >>> x = jax.random.normal(key, (16, 6))
        >>> params = model.init(key, x)['params']
        >>> ee = model.apply({'params': params}, x)   # shape (16, 1), values in [0, 1]
    """
    n_features: int = 6
    hidden_dim: int = 128
    n_residual: int = 3
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        # Input projection
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.LayerNorm()(h)
        h = nn.gelu(h)

        # Residual trunk
        for _ in range(self.n_residual):
            h = ResidualBlock(self.hidden_dim, self.dropout)(h, training)

        # Output head: predict EE ∈ [0, 1]
        h = nn.Dense(32)(h)
        h = nn.gelu(h)
        h = nn.Dense(1)(h)
        return nn.sigmoid(h)


def build_model(n_features: int = 6) -> EEPredictor:
    """Convenience constructor."""
    return EEPredictor(n_features=n_features)
