"""Mixed-resolution patch embedding with DINOv3's pretrained patch conv
(APT's Resize and Conv2D + ZeroMLP).

With ``E`` the pretrained embedding of ``p x p`` patches, a patch ``P`` of
size ``m p`` is embedded as

  - ``m == 1``: ``E(P)``,
  - ``m > 1``:  ``E(resize_p(P)) + ZeroMLP(fuse(E(subpatches(P))))``, where
    ``fuse`` pools the ``m x m`` embeddings of the finest-level sub-patches
    with stride-2 convolutions.

The ZeroMLP's output layer is zero-initialized, so at the start of the
adaptation a coarse token equals ``E(resize_p(P))``, the embedding of a
downsized patch as the pretrained ViT saw it; fine-tuning learns how much
sub-patch detail to add. The pretrained patch conv is shared by
reference, not copied.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from patch_where_objects_are.adaptive_patch.patchification import Patchification


class ZeroInitMLP(nn.Module):
    """Linear -> GELU -> Linear with a zero-initialized output layer: the
    output is 0 at init, and since only the output layer is zero, gradients
    still flow.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """``(..., D)`` -> ``(..., D)``."""
        return self.fc2(self.act(self.fc1(x)))


class _ScaleFusion(nn.Module):
    """Fuse the ``m x m`` sub-patch embeddings of one patch into its residual:
    ``log2 m`` stride-2 convolutions pool the grid to 1x1, then a ``ZeroInitMLP``.
    """

    def __init__(self, dim: int, m: int) -> None:
        super().__init__()
        if m < 2 or (m & (m - 1)) != 0:
            raise ValueError(f"scale multiplier must be a power of 2 >= 2, got {m}")
        self.m = m
        layers: list[nn.Module] = []
        for _ in range(int(math.log2(m))):
            layers.append(nn.Conv2d(dim, dim, kernel_size=2, stride=2))
            layers.append(nn.GELU())
        self.convs = nn.Sequential(*layers)
        self.mlp = ZeroInitMLP(dim)

    def forward(self, sub_grid: Tensor) -> Tensor:
        """``(K, D, m, m)`` sub-embedding grids → ``(K, D)`` residuals."""
        x = self.convs(sub_grid)  # (K, D, 1, 1)
        return self.mlp(x.flatten(1))


class MixedPatchEmbed(nn.Module):
    """Embed the patches of a ``Patchification`` with the pretrained patch conv.

    Args:
        patch_embed: The backbone's pretrained ``p x p`` patch conv (shared reference).
        scales:      Allowed patch-size multipliers, e.g. ``(1, 2, 4)``; one
                     fusion module per ``m > 1``.
    """

    def __init__(
        self, patch_embed: nn.Conv2d, scales: Sequence[int] = (1, 2, 4)
    ) -> None:
        super().__init__()
        self._patch_embed_ref = [patch_embed]  # not registered as a submodule
        self.scales = tuple(sorted({int(m) for m in scales}))
        if 1 not in self.scales:
            raise ValueError(f"scales must include the base scale 1, got {scales}")
        dim = patch_embed.out_channels
        self.fusion = nn.ModuleDict(
            {str(m): _ScaleFusion(dim, m) for m in self.scales if m > 1}
        )

    @property
    def patch_embed(self) -> nn.Conv2d:
        """The shared pretrained ``p x p`` patch conv."""
        return self._patch_embed_ref[0]

    @property
    def base_patch_size(self) -> int:
        """Finest patch side ``p`` in pixels: the patch conv's kernel size."""
        return self.patch_embed.kernel_size[0]

    def forward(
        self,
        img: Tensor,
        patchifications: Patchification | Sequence[Patchification],
    ) -> list[Tensor]:
        """Args:
            img: ``(B, 3, H, W)`` normalized images; ``H``, ``W`` multiples of
                the finest patch size.
            patchifications: One ``Patchification`` for all images, or one per
                image (token counts may differ).

        Returns:
            Per-image tokens ``(N_b, D)`` in each patchification's token order.
        """
        B, _, H, W = img.shape
        p = self.base_patch_size
        if H % p or W % p:
            raise ValueError(f"image size {(H, W)} not divisible by patch size {p}")
        if isinstance(patchifications, Patchification):
            pfs = [patchifications] * B
        else:
            pfs = list(patchifications)
        if len(pfs) != B:
            raise ValueError(f"{len(pfs)} patchifications for batch of {B}")
        grid_hw = (H // p, W // p)
        for pf in pfs:
            if pf.grid_hw != grid_hw or pf.patch_size != p:
                raise ValueError(
                    f"patchification grid {pf.grid_hw}/p={pf.patch_size} does not "
                    f"match image grid {grid_hw}/p={p}"
                )

        # One conv pass embeds every finest-level patch.
        base_grid = self.patch_embed(img)  # (B, D, Hp, Wp)

        out: list[Tensor] = []
        # for each image, embed the image a.t. patchification and return the tokens
        for b, pf in enumerate(pfs):
            out.append(self._embed_one(img[b], base_grid[b], pf))
        return out

    def _embed_one(self, img: Tensor, base_grid: Tensor, pf: Patchification) -> Tensor:
        """K is the number of tokens within scale m, D is the dimension of the tokens.

        -> (K, D, m, m) = K patches, each with an mxm grid of sub-embeddings
        """
        p = self.base_patch_size
        device = img.device
        chunks: list[Tensor] = []
        orders: list[Tensor] = []
        base_indices = pf.base_indices.to(device)
        for m in torch.unique(pf.scales).tolist():
            if m not in self.scales:
                raise ValueError(
                    f"patchification uses scale {m}, allowed {self.scales}"
                )
            (idx,) = torch.where(
                pf.scales.to(device) == m
            )  # where scale multiplier per token is m
            gy = base_indices[idx, 0]  # y coordinate of the token
            gx = base_indices[idx, 1]  # x coordinate of the token
            if m == 1:
                vals = base_grid[:, gy, gx].transpose(0, 1)  # (K, D)
            else:
                # Sub-patch path: m x m block of precomputed sub-embeddings.
                ar = torch.arange(m, device=gy.device)
                ys = gy[:, None, None] + ar[None, :, None]  # (K, m, 1)
                xs = gx[:, None, None] + ar[None, None, :]  # (K, 1, m)
                sub = base_grid[:, ys, xs].permute(1, 0, 2, 3)  # (K, D, m, m)
                e_sub = self.fusion[str(m)](sub)  # (K, D)

                # Resize path: patch pixels → p x p → pretrained conv.
                arp = torch.arange(m * p, device=gy.device)
                pys = (gy * p)[:, None, None] + arp[None, :, None]
                pxs = (gx * p)[:, None, None] + arp[None, None, :]
                pix = img[:, pys, pxs].permute(1, 0, 2, 3)  # (K, 3, m*p, m*p)
                pix = F.interpolate(
                    pix,
                    size=(p, p),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                e_res = self.patch_embed(pix).flatten(1)  # (K, D)
                vals = e_res + e_sub
            chunks.append(vals)
            orders.append(idx)
        order = torch.cat(orders)
        tokens = torch.cat(chunks, dim=0)
        # Restore the patchification's token order.
        return tokens[torch.argsort(order)]
