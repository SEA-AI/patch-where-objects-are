# Feature-distortion criterion of Quadformer (Ronen et al. 2023,
# https://github.com/TomerRonen34/mixed-resolution-vit, commit 09c9a7d).
# _extract_features follows _extract_features_shufflenet in
# mixed_res/patch_scorers/feature_based_patch_scorer.py and _to_node_grid the
# map-to-grid reduction of PatchScorer._calculate_scores in
# mixed_res/patch_scorers/patch_scorer.py.
#
# Modifications: batched scoring at the native resolution, two-axis resize,
# per-scale scores, bilinear instead of torchvision resize in _to_node_grid,
# selector class.

"""Feature-distortion splitting criterion (Quadformer, Ronen et al. 2023).

The score of a cell is the mean squared error between the stride-32 features
of the image and of its blurred copy (bilinear downsampling by the blur
factor and upsampling back), computed with an ImageNet-pretrained
ShuffleNetV2x0.5 as in Quadformer's ``FeatureBasedPatchScorer``
(github.com/TomerRonen34/mixed-resolution-vit). Each stride-32 score is
assigned to the four finest-level cells it covers.

Finest-level scoring uses the blur-x2 map. Per-scale scoring
scores a level-``m`` node with the blur-x``m`` map on that level's grid.
An image smaller than the input resolution is scored at ``min(native, input)``
per axis and its map is bilinearly resized to the finest-level grid, which
avoids upsampling artifacts in the scores.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor, nn

from patch_where_objects_are.adaptive_patch.patchification import Patchification
from patch_where_objects_are.adaptive_patch.selectors import score_quadtree


def _resize_hw(x: Tensor, h: int, w: int) -> Tensor:
    # Two-axis resize for the native-resolution path.
    return TF.resize(x, [h, w], antialias=False)


def _extract_features(net: nn.Module, x: Tensor, downscale: int) -> Tensor:
    """Stride (16 or 32) -``downscale`` features, as Quadformer's
    ``_extract_features_shufflenet``.
    """
    x = net.conv1(x)
    x = net.maxpool(x)
    x = net.stage2(x)
    x = net.stage3(x)
    if downscale == 32:
        x = net.stage4(x)
        x = net.conv5(x)
    return x


def _feature_distortion(f_hi: Tensor, f_lo: Tensor) -> Tensor:
    """Per-channel MSE between features of an image and its blurred copy."""
    return ((f_lo - f_hi) ** 2).mean(dim=1)


def _scoring_side(native: int, size: int, downscale: int) -> int:
    """Scoring side ``min(native, size)``, rounded to a multiple of ``downscale``."""
    s = min(int(native), int(size))
    return max(downscale, round(s / downscale) * downscale)


@torch.no_grad()
def _candidate_grid_maps(
    net: nn.Module,
    img: Tensor,
    downscale: int,
    blur_factors: Sequence[int],
    native_hw: Sequence[tuple[int, int]],
) -> dict[int, Tensor]:
    """``{blur_factor: (B, size/downscale, size/downscale)}`` feature-distortion
    maps on the stride-``downscale`` grid of the input (26x26 at 832/32), each
    image scored at ``min(native, size)`` per axis.
    """
    size = img.shape[-1]
    if len(native_hw) != img.shape[0]:
        raise ValueError(
            f"native_hw has {len(native_hw)} entries for a batch of "
            f"{img.shape[0]} images"
        )
    cand = size // downscale
    out = {
        bf: img.new_empty((img.shape[0], cand, cand), dtype=torch.float32)
        for bf in blur_factors
    }
    # Batch the images of the same scoring size.
    groups: dict[tuple[int, int], list[int]] = {}
    for b, (nh, nw) in enumerate(native_hw):
        key = (_scoring_side(nh, size, downscale), _scoring_side(nw, size, downscale))
        groups.setdefault(key, []).append(b)
    for (sh, sw), idx in groups.items():
        x = img[idx] if (sh, sw) == (size, size) else _resize_hw(img[idx], sh, sw)
        f_hi = _extract_features(net, x, downscale)
        for (
            bf
        ) in blur_factors:  # for each blur factor, calculate the feature distortion
            blurry = _resize_hw(
                _resize_hw(x, max(sh // bf, 1), max(sw // bf, 1)), sh, sw
            )
            m = _feature_distortion(
                f_hi, _extract_features(net, blurry, downscale)
            ).float()
            if m.shape[-2:] != (cand, cand):
                m = F.interpolate(
                    m.unsqueeze(1),
                    size=(cand, cand),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            out[bf][idx] = m
    return out


def _to_node_grid(imp: Tensor, node_grid: int) -> Tensor:
    """Reduce a score map to the ``node_grid`` x ``node_grid`` grid of one level,
    as Quadformer's ``patch_scorer.py``: an integer ratio is mean-pooled, any
    other is bilinearly resized.
    """
    map_size = imp.shape[-1]
    if map_size < node_grid:
        raise ValueError(f"map_size ({map_size}) < node_grid ({node_grid})")
    if map_size == node_grid:
        return imp
    if map_size % node_grid == 0:
        k = map_size // node_grid
        return F.avg_pool2d(imp.unsqueeze(1), kernel_size=k, stride=k).squeeze(1)
    return F.interpolate(
        imp.unsqueeze(1),
        size=(node_grid, node_grid),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


@torch.no_grad()
def quadformer_scale_scores(
    net: nn.Module,
    img: Tensor,
    patch_size: int,
    downscale: int = 32,
    max_scale: int = 4,
    *,
    native_hw: Sequence[tuple[int, int]],
) -> dict[int, Tensor]:
    """Per-scale score maps ``{m: (B, H/(m p), W/(m p))}`` for ``m`` in
    ``{2, ..., max_scale}``: a level-``m`` node is scored by the blur-x``m``
    map reduced to its grid. Level 1 is never split, so also never scored.
    """
    size = img.shape[-1]
    scales: list[int] = []
    m = 2
    while m <= max_scale:
        scales.append(m)
        m *= 2
    maps = _candidate_grid_maps(net, img, downscale, scales, native_hw)
    return {m: _to_node_grid(maps[m], size // (m * patch_size)).float() for m in scales}


@torch.no_grad()
def quadformer_importance_map(
    net: nn.Module,
    img: Tensor,
    patch_size: int,
    downscale: int = 32,
    blur_factor: int = 2,
    *,
    native_hw: Sequence[tuple[int, int]],
) -> Tensor:
    """Finest-level feature-distortion map ``(B, H/patch_size, W/patch_size)``
    of an ImageNet-normalized batch: the blur-x``blur_factor`` map, each
    stride-``downscale`` score shared by the finest-level cells it covers.

    ``native_hw``: per-image ``(height, width)`` before the resize to the
    input resolution; an image is scored at ``min(native, input)`` per axis.
    """
    size = img.shape[-1]
    imp = _candidate_grid_maps(net, img, downscale, [blur_factor], native_hw)[
        blur_factor
    ]
    rep = (size // patch_size) // imp.shape[-1]
    if rep > 1:
        imp = imp.repeat_interleave(rep, dim=-1).repeat_interleave(rep, dim=-2)
    return imp


def build_quadformer_net(pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained ShuffleNetV2x0.5 as feature extractor."""
    from torchvision.models.shufflenetv2 import (
        ShuffleNet_V2_X0_5_Weights,
        shufflenet_v2_x0_5,
    )

    net = shufflenet_v2_x0_5(
        weights=ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
    )
    del net.fc
    return net


def _build_stride16_trunk(pretrained: bool, freeze: bool = True) -> nn.Module:
    """ShuffleNetV2x0.5 up to stage3 (stride 16); frozen unless ``freeze=False``."""
    net = build_quadformer_net(pretrained=pretrained)
    del net.stage4, net.conv5
    if freeze:
        net.requires_grad_(False)
        net.eval()
    return net


def _orig_sizes(targets: list[dict] | None, batch: int) -> list[tuple[int, int]]:
    """Per-image native ``(h, w)`` from the targets' ``orig_size``."""
    if targets is None or len(targets) != batch:
        raise ValueError(
            "feature distortion needs per-image 'orig_size' in the targets, but "
            f"got targets={'None' if targets is None else len(targets)} for a "
            f"batch of {batch}. Evaluation must pass the targets when the "
            "selector reports needs_orig_size."
        )
    sizes = []
    for t in targets:
        if "orig_size" not in t:
            raise ValueError(
                "feature distortion needs 'orig_size' in every target dict"
            )
        h, w = (int(v) for v in t["orig_size"].tolist())
        sizes.append((h, w))
    return sizes


class FeatureDistortionSelector(nn.Module):
    """Feature-distortion selector: frozen ShuffleNetV2x0.5 scorer + budgeted
    quadtree. Images are scored at ``min(native, input)`` resolution per axis,
    so the targets must carry each image's ``orig_size``.

    ``scoring="finest"`` scores the finest-level cells with the blur-x2 map and
    propagates the scores to coarser nodes by ``pool``; ``scoring="per_scale"``
    scores a level-``m`` node with the blur-x``m`` map on its own grid. The
    scorer is a submodule, so it follows the detector's device and is stored
    in its checkpoints, but is never trained.
    """

    needs_orig_size = True

    def __init__(
        self,
        patch_size: int,
        token_budget: int,
        downscale: int = 32,
        blur_factor: int = 2,
        max_scale: int = 4,
        pretrained: bool = True,
        scoring: str = "finest",
        pool: str = "max",
    ) -> None:
        super().__init__()
        self.net = build_quadformer_net(pretrained=pretrained)
        self.net.requires_grad_(False)
        self.net.eval()
        self.patch_size = patch_size
        self.token_budget = int(token_budget)
        self.downscale = downscale
        self.blur_factor = blur_factor
        self.max_scale = max_scale
        if scoring not in {"finest", "per_scale"}:
            raise ValueError(
                f"scoring must be 'finest' or 'per_scale', got {scoring!r}"
            )
        if pool not in {"max", "mean"}:
            raise ValueError(f"pool must be 'max' or 'mean', got {pool!r}")
        if scoring == "per_scale" and pool != "max":
            raise ValueError("pool='mean' applies to finest-cell scoring only")
        self.scoring = scoring
        self.pool = pool

    def train(self, mode: bool = True) -> FeatureDistortionSelector:
        """Keep the frozen scorer in eval mode regardless of detector mode."""
        super().train(mode)
        self.net.eval()
        return self

    def forward(
        self, img: Tensor, targets: list[dict] | None = None
    ) -> list[Patchification]:
        """Patchifications of an ImageNet-normalized batch; ``targets`` is read
        only for ``orig_size``.
        """
        native_hw = _orig_sizes(targets, img.shape[0])
        per_scale = None
        if self.scoring == "per_scale":
            per_scale = {
                m: g.cpu()
                for m, g in quadformer_scale_scores(
                    self.net,
                    img,
                    self.patch_size,
                    self.downscale,
                    self.max_scale,
                    native_hw=native_hw,
                ).items()
            }
            maps = (
                per_scale[2].repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            )
        else:
            maps = (
                quadformer_importance_map(
                    self.net,
                    img,
                    self.patch_size,
                    self.downscale,
                    self.blur_factor,
                    native_hw=native_hw,
                )
                .float()
                .cpu()
            )
        return [
            score_quadtree(
                maps[b],
                self.patch_size,
                self.token_budget,
                self.max_scale,
                scale_scores=None
                if per_scale is None
                else {m: g[b] for m, g in per_scale.items()},
                pool=self.pool,
            )
            for b in range(img.shape[0])
        ]
