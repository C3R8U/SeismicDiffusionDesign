"""Joint seismic response and limit-exceedance estimation for structural frames."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class ExceedanceNetwork(nn.Module):
    """Use shared design features to learn response magnitude and rare exceedance events."""
    def __init__(self, width=192, dropout=.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(57 * 3 + 8 + 4, width),
            nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width // 2), nn.GELU(),
        )
        self.regression = nn.Linear(width // 2, 6)
        self.exceedance = nn.Linear(width // 2, 3)

    def forward(self, continuous, mask, geometry, ground_motion):
        batch = continuous.shape[0]
        active = continuous * mask.unsqueeze(-1)
        features = torch.cat((active.reshape(batch, -1), mask.float(), geometry, ground_motion), -1)
        hidden = self.encoder(features)
        estimate = self.regression(hidden)
        return estimate[:, :3], estimate[:, 3:].clamp(-8, 4), self.exceedance(hidden)

    @staticmethod
    def objective(mean, logvar, logits, target, classes, positives, lambda_exceedance=.75):
        # Rare events get additional weight while all response magnitudes remain supervised.
        regression = .5 * ((target - mean).square() * (-logvar).exp() + logvar).mean()
        rare = F.binary_cross_entropy_with_logits(logits, classes, pos_weight=positives)
        return regression + lambda_exceedance * rare, regression, rare
