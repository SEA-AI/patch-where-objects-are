# Adapted from DINOv3ViTRopePositionEmbedding in transformers
# (https://github.com/huggingface/transformers),
# src/transformers/models/dinov3_vit/modeling_dinov3_vit.py.
# Copyright 2025 Meta AI and The HuggingFace Inc. team. Licensed under the
# Apache License 2.0.
#
# Modifications: RoPE computed at explicit token centers instead of the regular
# patch grid.

"""RoPE of the transformers DINOv3 ViT at explicit token centers, for
mixed-resolution token sets.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTRopePositionEmbedding,
    augment_patches_center_coordinates,
)


def rope_at_coords(
    rope_embed: DINOv3ViTRopePositionEmbedding, coords: Tensor, *, HW: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """RoPE ``(cos, sin)`` at explicit token centers. On the regular grid
    centers ``(i + 0.5, j + 0.5)`` this equals ``rope_embed(pixel_values)``.

    Args:
        rope_embed: The ViT's ``DINOv3ViTRopePositionEmbedding``.
        coords: ``(..., 2)`` token centers in (row, col) order, in the units
            of ``HW`` (base-grid units with ``HW=(Hp, Wp)``). Leading dims
            pass through: ``(B, N, 2)`` gives ``(B, N, D_head)``.
        HW: Grid extent the coordinates are normalized by.

    Returns:
        ``(cos, sin)``, each ``(..., D_head)``.
    """
    H, W = HW
    coords = coords.to(device=rope_embed.inv_freq.device, dtype=torch.float32)
    coords = 2.0 * (coords / coords.new_tensor([H, W])) - 1.0
    # Same RoPE coordinate augmentation as rope_embed.forward (see dinov3.py).
    if rope_embed.training:
        cfg = rope_embed.config
        coords = augment_patches_center_coordinates(
            coords,
            shift=cfg.pos_embed_shift,
            jitter=cfg.pos_embed_jitter,
            rescale=cfg.pos_embed_rescale,
        )
    # (..., 2, D_head / 4) -> (..., D_head)
    angles = 2 * math.pi * coords[..., :, None] * rope_embed.inv_freq
    angles = angles.flatten(-2)
    angles = torch.cat([angles, angles], -1)
    return torch.cos(angles), torch.sin(angles)
