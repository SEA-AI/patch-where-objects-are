# Entropy criterion of APT (Choudhury et al. 2026,
# https://github.com/rccchoudhury/apt, commit 707337e). The pixel binning
# follows compute_patch_entropy_batched in src/models/entropy_utils.py. The APT
# repository states no license.
#
# Modifications: single scale, no padding, scatter_add histogram, 256 bins by
# default.

"""Entropy splitting criterion (APT, Choudhury et al. 2026).

The score of a finest-level cell is the Shannon entropy, in bits, of the
intensity histogram of its ``patch_size`` x ``patch_size`` grayscale pixels.
Follows APT's ``compute_patch_entropy_batched``
(https://github.com/rccchoudhury/apt, ``src/models/entropy_utils.py``):
grayscale ``0.2989 R + 0.5870 G + 0.1140 B`` on [0, 255], ``bins`` histogram
bins over [0, 256), probabilities normalized by the cell area, ``eps = 1e-10``.
A pixel falls into bin ``floor(value * bins / 256)``, which equals APT's
one-hot encoding for ``bins = 256``.
"""

from __future__ import annotations

import torch

_EPS = 1e-10
_RGB_WEIGHTS = (0.2989, 0.5870, 0.1140)


def to_grayscale_255(images: torch.Tensor) -> torch.Tensor:
    """``(B, 3, H, W)`` RGB in [0, 1] -> ``(B, H, W)`` grayscale in [0, 255]."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected (B, 3, H, W), got {tuple(images.shape)}")
    r, g, b = images.unbind(dim=1)
    gray = _RGB_WEIGHTS[0] * r + _RGB_WEIGHTS[1] * g + _RGB_WEIGHTS[2] * b
    return gray * 255.0


def compute_cell_entropy(
    gray: torch.Tensor,
    patch_size: int = 16,
    bins: int = 256,  # APT uses 512, which does not change our results
) -> torch.Tensor:
    """Entropy in bits of each ``patch_size`` x ``patch_size`` cell.

    Args:
        gray: ``(B, H, W)`` grayscale in [0, 255]; ``H`` and ``W`` divisible by
            ``patch_size``.
        patch_size: Cell side in pixels.
        bins: Histogram bins over [0, 256).

    Returns:
        ``(B, H / patch_size, W / patch_size)`` float32 score map.
    """
    if gray.ndim != 3:
        raise ValueError(f"Expected (B, H, W), got {tuple(gray.shape)}")
    b, h, w = gray.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"H={h}, W={w} not divisible by patch_size={patch_size}")

    gh, gw = h // patch_size, w // patch_size
    cells = (
        gray.reshape(b, gh, patch_size, gw, patch_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(b * gh * gw, patch_size * patch_size)
    )  # (b * gh * gw, patch_size ** 2)
    idx = (cells * (bins / 256.0)).long().clamp_(0, bins - 1)  # bin per pixel
    hist = torch.zeros(cells.shape[0], bins, device=gray.device, dtype=torch.float32)
    hist.scatter_add_(
        1, idx, torch.ones_like(cells, dtype=torch.float32)
    )  # histogram per cell
    probs = hist / float(patch_size * patch_size)
    entropy = -(probs * torch.log2(probs + _EPS)).sum(dim=1)
    return entropy.reshape(b, gh, gw)
