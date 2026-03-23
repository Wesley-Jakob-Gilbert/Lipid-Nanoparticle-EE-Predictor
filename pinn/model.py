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
    - zeta_mv                        (surface charge, mV)
    - peg_fraction                   (PEG-lipid mole fraction)
    - cholesterol_fraction           (cholesterol mole fraction)

Output:
    - ee_pred: predicted encapsulation efficiency ∈ [0, 1]
"""

import torch
import torch.nn as nn
from typing import List


class ResidualBlock(nn.Module):
    """Two-layer residual block with LayerNorm and GELU activation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class EEPredictor(nn.Module):
    """
    Physics-Informed Neural Network for LNP Encapsulation Efficiency.

    Args:
        n_features:  Number of input physicochemical features.
        hidden_dim:  Width of hidden layers (default 128).
        n_residual:  Number of residual blocks (default 3).
        dropout:     Dropout probability (default 0.1).

    Example:
        >>> model = EEPredictor(n_features=7)
        >>> x = torch.randn(16, 7)
        >>> ee = model(x)   # shape (16, 1), values in [0, 1]
    """

    def __init__(
        self,
        n_features: int = 7,
        hidden_dim: int = 128,
        n_residual: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual trunk
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_residual)]
        )

        # Output head: predict EE ∈ [0, 1]
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (B, n_features).

        Returns:
            ee_pred: Predicted EE, shape (B, 1), values in [0, 1].
        """
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        return self.output_head(h)


def build_model(n_features: int = 7, device: str = "cpu") -> EEPredictor:
    """Convenience constructor. Returns model on specified device."""
    model = EEPredictor(n_features=n_features)
    return model.to(device)
