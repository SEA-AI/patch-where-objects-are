"""Neck of the uniform detector: the last block's stride-16 feature map goes
to the RT-DETR head; there is no feature pyramid.
"""

from __future__ import annotations

import torch
from torch import nn


class SingleScaleNeck(nn.Module):
    """Parameter-free neck passing one backbone feature map to the head. The
    projection from ``embed_dim`` to the head's ``hidden_dim`` is in
    ``RTDETRHead.input_proj``.

    Args:
        intermediate_layers: The single block index whose output feeds the head.
    """

    def __init__(self, intermediate_layers: list[int]) -> None:
        super().__init__()
        if len(intermediate_layers) != 1:
            raise ValueError(
                "SingleScaleNeck expects exactly one layer index, "
                f"got {intermediate_layers}"
            )
        self.intermediate_layers = list(intermediate_layers)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Check for one ``(B, D, Hp, Wp)`` feature map and return it unchanged."""
        if len(features) != 1:
            raise ValueError(f"Expected 1 feature map, got {len(features)}")
        f = features[0]
        if f.ndim != 4:
            raise ValueError(
                f"Expected (B, D, Hp, Wp) feature map, got shape {tuple(f.shape)}"
            )
        return [f]
