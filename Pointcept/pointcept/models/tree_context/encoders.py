"""
Per-source context encoders for tree species classification.

Each encoder projects its raw input to a unified 256-dim latent vector
with LayerNorm. All sources produce the same output dimensionality,
making them interchangeable in the fusion classifier.

Phase 1: One encoder is active at a time (single-source experiments).
Phase 2: All four encoders feed into a CLS-attention aggregator.

Input preprocessing notes:
- AlphaEarth (64-dim): raw satellite embeddings, passed as-is.
- Topo (6-dim): raw topographic variables (elevation, slope, northness,
  eastness, TRI, TPI) — the small MLP handles scale differences.
- SINR (256-dim): ReLU-activated backbone features, passed as-is.
- GeoPlantNet (18-dim): species logit scores; the encoder applies
  log(clamp(x, min=eps)) to compress the range (matching Phase 0
  preprocessing). NaN values should be replaced with 0 in the dataset
  before reaching the encoder, which maps to log(eps) ≈ -13.8 (a
  consistent "no signal" sentinel).
"""

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS


@MODELS.register_module("AEEncoder")
class AlphaEarthEncoder(nn.Module):
    """Projects 64-dim AlphaEarth satellite embedding to output_dim."""

    def __init__(self, input_dim: int = 64, output_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


@MODELS.register_module("TopoEncoder")
class TopoEncoder(nn.Module):
    """Encodes 6 topographic variables to output_dim via a 2-layer MLP."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mlp(x))


@MODELS.register_module("SINREncoder")
class SINREncoder(nn.Module):
    """Projects 256-dim SINR habitat embedding to output_dim."""

    def __init__(self, input_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


@MODELS.register_module("GPNEncoder")
class GeoPlantNetEncoder(nn.Module):
    """Encodes GeoPlantNet species logit scores to output_dim.

    Applies log(clamp(x, min=eps)) before the linear projection to
    compress the score range. Missing/NaN values should be zeroed in the
    dataset; they arrive here as 0 and are mapped to log(eps) ≈ -13.8,
    a consistent 'no signal' sentinel that the network can learn.
    """

    def __init__(
        self,
        input_dim: int = 18,
        output_dim: int = 256,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log(x.clamp(min=self.eps))
        return self.norm(self.proj(x))
