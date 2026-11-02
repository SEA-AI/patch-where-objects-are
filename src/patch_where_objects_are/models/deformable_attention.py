# Adapted from RT-DETR (https://github.com/lyuwenyu/RT-DETR),
# rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py (MSDeformableAttention) and
# rtdetr_pytorch/src/zoo/rtdetr/utils.py (deformable_attention_core_func).
# Copyright (c) 2023 lyuwenyu.
# Licensed under the Apache License 2.0.
#
# These follow Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR),
# models/ops/modules/ms_deform_attn.py and
# models/ops/functions/ms_deform_attn_func.py (ms_deform_attn_core_pytorch).
# Copyright (c) 2020 SenseTime.
# Licensed under the Apache License 2.0.
#
# Modifications: sparse_deformable_attention_core_func and the cell_to_token
# argument added (sampling of mixed-resolution tokens through a base-cell ->
# token index), value_mask handling fixed, code reformatted and documented.

"""Multi-scale deformable attention in pure PyTorch (``F.grid_sample``), after
RT-DETR (Zhao et al. 2024); no CUDA extension. The sparse head samples the
mixed-resolution tokens through a finest-level cell -> token index.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


def deformable_attention_core_func(
    value: torch.Tensor,
    value_spatial_shapes: list,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Core deformable attention using F.grid_sample (no CUDA extension).

    Args:
        value:                ``(B, Lv, n_heads, head_dim)``
        value_spatial_shapes: list of ``(H_l, W_l)`` for each level
        sampling_locations:   ``(B, Lq, n_heads, n_levels, n_points, 2)``
                              coordinates in [0, 1]
        attention_weights:    ``(B, Lq, n_heads, n_levels, n_points)``

    Returns:
        ``(B, Lq, embed_dim)``
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)

    # Map [0, 1] → [-1, 1] for grid_sample.
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # (B, H*W, n_head, head_dim) → (B*n_head, head_dim, H, W)
        value_l_ = (
            value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        )
        # (B, Lq, n_head, n_points, 2) → (B*n_head, Lq, n_points, 2)
        sampling_grid_l_ = (
            sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        )
        # Sampled values: (B*n_head, head_dim, Lq, n_points)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    # (B, Lq, n_head, n_levels, n_points) → (B*n_head, 1, Lq, n_levels*n_points)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points
    )

    # Weighted sum over sampling points.
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .reshape(bs, n_head * c, Len_q)
    )
    return output.permute(0, 2, 1)  # (B, Lq, embed_dim)


def sparse_deformable_attention_core_func(
    value: torch.Tensor,
    value_spatial_shape: list,
    cell_to_token: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Deformable attention on the mixed-resolution tokens: bilinear sampling
    on the base grid, where each cell's value is its token's (``cell_to_token``).
    The token values are written to the base grid with one gather and sampled
    as in the dense head.
    """
    bs, _, n_head, c = value.shape
    n_levels = sampling_locations.shape[3]
    if n_levels != 1 or len(value_spatial_shape) != 1:
        raise ValueError("Sparse deformable attention supports one feature level")

    h, w = value_spatial_shape[0]
    if cell_to_token.shape != (bs, h * w):
        raise ValueError(
            f"Expected cell_to_token {(bs, h * w)}, got {tuple(cell_to_token.shape)}"
        )

    grid_value = torch.gather(
        value, 1, cell_to_token[:, :, None, None].expand(-1, -1, n_head, c)
    )  # (B, h * w, n_head, head_dim)
    return deformable_attention_core_func(
        grid_value, value_spatial_shape, sampling_locations, attention_weights
    )


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile(
            [1, self.num_levels, self.num_points, 1]
        )
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(
            1, 1, -1, 1
        )
        grid_init = grid_init * scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: list,
        value_mask: torch.Tensor = None,
        cell_to_token: torch.Tensor = None,
    ) -> torch.Tensor:
        """Attend from each query to sampled points of the value feature maps.

        Args:
            query:               ``(B, Lq, C)``
            reference_points:    ``(B, Lq, n_levels, 2)`` in [0, 1]
            value:               ``(B, Lv, C)``
            value_spatial_shapes: list of ``(H_l, W_l)``
            value_mask:          ``(B, Lv)`` boolean (True = valid)
            cell_to_token:       ``(B, H * W)`` token index of each base cell;
                                 None for dense (non-token) features
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = value.new_tensor(value_spatial_shapes).flip([1])
            offset_normalizer = offset_normalizer.reshape(
                1, 1, 1, self.num_levels, 1, 2
            )
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "reference_points last dim must be 2 or 4, "
                f"got {reference_points.shape[-1]}"
            )

        if cell_to_token is None:
            output = deformable_attention_core_func(
                value, value_spatial_shapes, sampling_locations, attention_weights
            )
        else:
            output = sparse_deformable_attention_core_func(
                value,
                value_spatial_shapes,
                cell_to_token,
                sampling_locations,
                attention_weights,
            )
        return self.output_proj(output)
