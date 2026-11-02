"""Tokens and the base grid, for the detection head.

Dense head: ``tokens_to_base_grid`` writes each token's features to every base
cell its patch covers (an ``m = 1`` token fills one cell, an ``m = 4`` token
its 4x4 block), giving the ``(D, Hp, Wp)`` feature map the single-scale
RT-DETR head reads. Sparse head: ``base_grid_token_indices`` gives the index
of the token covering each base cell, through which the proposals and the
decoder's cross-attention read the tokens. Tokens of a valid
``Patchification`` never overlap; if they do, the finest scale wins (coarser
scales are written first).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from patch_where_objects_are.adaptive_patch.patchification import Patchification


def base_grid_token_indices(pf: Patchification, device: torch.device) -> Tensor:
    """``(Hp * Wp,)`` index of the token covering each base cell, row-major. For
    the sparse detection head.
    """
    h, w = pf.grid_hw
    grid = torch.zeros(h, w, dtype=torch.long, device=device)
    scales = pf.scales.to(device)
    base_indices = pf.base_indices.to(device)

    # Match tokens_to_base_grid's coarse-to-fine overlap policy.
    for m in sorted(torch.unique(scales).tolist(), reverse=True):
        (idx,) = torch.where(scales == m)
        gy = base_indices[idx, 0]
        gx = base_indices[idx, 1]
        ar = torch.arange(m, device=device)
        ys = (gy[:, None, None] + ar[None, :, None]).expand(-1, m, m).reshape(-1)
        xs = (gx[:, None, None] + ar[None, None, :]).expand(-1, m, m).reshape(-1)
        grid[ys, xs] = idx.repeat_interleave(m * m)
    return grid.flatten()


def tokens_to_base_grid(tokens: Tensor, pf: Patchification) -> Tensor:
    """Duplicate one image's ``(N, D)`` tokens, in ``pf``'s token order, onto
    the base grid: ``(D, Hp, Wp)``. For the dense detection head.
    """
    if tokens.shape[0] != pf.num_tokens:
        raise ValueError(
            f"{tokens.shape[0]} tokens for patchification of {pf.num_tokens}"
        )
    h, w = pf.grid_hw
    d = tokens.shape[1]
    device = tokens.device
    grid = tokens.new_zeros(d, h, w)
    scales = pf.scales.to(device)
    base_indices = pf.base_indices.to(device)

    # Coarse-to-fine so the finest scale wins on (invalid) overlaps.
    for m in sorted(torch.unique(scales).tolist(), reverse=True):
        (idx,) = torch.where(scales == m)
        gy = base_indices[idx, 0]
        gx = base_indices[idx, 1]
        if m == 1:
            grid[:, gy, gx] = tokens[idx].transpose(0, 1)
        else:
            ar = torch.arange(m, device=gy.device)
            ys = (gy[:, None, None] + ar[None, :, None]).expand(-1, m, m).reshape(-1)
            xs = (gx[:, None, None] + ar[None, None, :]).expand(-1, m, m).reshape(-1)
            vals = tokens[idx].repeat_interleave(m * m, dim=0)  # (K*m*m, D)
            grid[:, ys, xs] = vals.transpose(0, 1)
    return grid


def batch_tokens_to_base_grid(
    tokens: Tensor | Sequence[Tensor],
    patchifications: Patchification | Sequence[Patchification],
) -> Tensor:
    """Batched ``tokens_to_base_grid``: ``(B, N, D)`` or a list of per-image
    ``(N_b, D)`` tensors (token counts may differ) -> ``(B, D, Hp, Wp)``.
    """
    token_list = list(tokens)
    if isinstance(patchifications, Patchification):
        pfs = [patchifications] * len(token_list)
    else:
        pfs = list(patchifications)
    if len(pfs) != len(token_list):
        raise ValueError(f"{len(pfs)} patchifications for {len(token_list)} token sets")
    grids = {pf.grid_hw for pf in pfs}
    if len(grids) != 1:
        raise ValueError(f"mixed base-grid sizes in batch: {sorted(grids)}")
    return torch.stack(
        [tokens_to_base_grid(t, pf) for t, pf in zip(token_list, pfs, strict=True)]
    )
