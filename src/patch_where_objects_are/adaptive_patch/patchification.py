"""Mixed-resolution patch arrangement of one image.

The image is divided into the finest-level grid of ``patch_size`` cells, the
base grid. A ``Patchification`` lists, per token, its patch-size multiplier
``m`` (patch side ``m * patch_size`` pixels) and the top-left base cell it
covers. ``MixedPatchEmbed`` embeds the patches from it, RoPE reads
``rope_coords()``, and ``grid_broadcast`` duplicates the tokens back onto the
base grid for the detection head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Patchification:
    """Patch arrangement of one image on the base grid.

    Attributes:
        patch_size:   Finest patch side ``p`` in pixels (16 for DINOv3 ViT-B/16).
        grid_hw:      Base grid size ``(Hp, Wp)`` = image size // ``patch_size``.
        scales:       ``(N,)`` patch-size multiplier ``m`` per token.
        base_indices: ``(N, 2)`` top-left base cell ``(gy, gx)`` per token; the
                      token covers the cells ``[gy, gy+m) x [gx, gx+m)``.
    """

    patch_size: int
    grid_hw: tuple[int, int]
    scales: Tensor
    base_indices: Tensor

    @property
    def num_tokens(self) -> int:
        """Number of tokens ``N``."""
        return int(self.scales.shape[0])

    @classmethod
    def uniform(cls, grid_h: int, grid_w: int, patch_size: int) -> Patchification:
        """Uniform grid of finest-level patches: one ``m = 1`` token per base cell.

        Token order is row-major (``gy * Wp + gx``), as in DINOv3's patch
        embedding.
        """
        gy, gx = torch.meshgrid(
            torch.arange(grid_h), torch.arange(grid_w), indexing="ij"
        )
        base_indices = torch.stack([gy.flatten(), gx.flatten()], dim=1)
        return cls(
            patch_size=patch_size,
            grid_hw=(grid_h, grid_w),
            scales=torch.ones(grid_h * grid_w, dtype=torch.long),
            base_indices=base_indices,
        )

    def rope_coords(self) -> Tensor:
        """``(N, 2)`` float32 token centers ``(gy + m/2, gx + m/2)`` in base-grid
        units: an ``m = 1`` token at ``(0, 0)`` has center ``(0.5, 0.5)``.
        """
        m = self.scales.to(torch.float32)
        return self.base_indices.to(torch.float32) + (m / 2.0).unsqueeze(1)
