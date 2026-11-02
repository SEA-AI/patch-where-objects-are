"""Closed-form backbone GFLOPs of the DINOv3 ViT with the mixed-resolution
patch embedding. 1 FLOP = 1 multiply-accumulate, as in fvcore.

Counted:
  - the patch conv over the full base grid (the mixed-resolution embedding
    embeds every finest-level patch once, whatever the token count),
  - per block: QKV, projection and feed-forward (``12 N D^2`` at
    ``ffn_ratio`` 4) and the attention matmuls (``2 N^2 D``),
  - per coarse token: the resize-path conv, the stride-2 fusion convs and
    the fusion MLP (``2 D^2``).

Not counted: LayerNorms and activations, the detection head and NMS.
"""

from __future__ import annotations

from collections.abc import Mapping


def dinov3_backbone_gflops(
    scale_counts: Mapping[int, float],
    grid_cells: int,
    embed_dim: int,
    depth: int,
    patch_size: int = 16,
    n_prefix: int = 5,
    ffn_ratio: float = 4.0,
) -> float:
    """Backbone GFLOPs of one image.

    Args:
        scale_counts: Tokens per patch-size multiplier, e.g.
            ``{1: 226, 2: 24, 4: 160}`` (floats allowed, e.g. dataset means);
            uniform patches: ``{1: grid_cells}``.
        grid_cells:  Number of base cells ``Hp * Wp``.
        embed_dim:   ViT width ``D``.
        depth:       Number of transformer blocks.
        patch_size:  Finest patch size ``p``.
        n_prefix:    cls + register tokens (DINOv3: 1 + 4).
        ffn_ratio:   Feed-forward expansion ratio.
    """
    d = float(embed_dim)
    p = float(patch_size)
    n_patch = float(sum(scale_counts.values()))
    n = n_patch + n_prefix

    # Dense stride-p patch conv over the full grid.
    macs = grid_cells * 3.0 * p * p * d

    # Per-coarse-token extras (MixedPatchEmbed).
    for m, count in scale_counts.items():
        if m <= 1 or count == 0:
            continue
        resize_conv = 3.0 * p * p * d
        # Stride-2 conv chain m -> m/2 -> ... -> 1 (kernel 2): each level with
        # output side s has s^2 * 4 * D^2 MACs.
        fusion_convs = 0.0
        s = m // 2
        while s >= 1:
            fusion_convs += s * s * 4.0 * d * d
            s //= 2
        mlp = 2.0 * d * d
        macs += count * (resize_conv + fusion_convs + mlp)

    # Transformer blocks: (3 + 1 + 2*ffn_ratio) N D^2 linear + 2 N^2 D attention.
    per_block = (4.0 + 2.0 * ffn_ratio) * n * d * d + 2.0 * n * n * d
    macs += depth * per_block

    return macs / 1e9
