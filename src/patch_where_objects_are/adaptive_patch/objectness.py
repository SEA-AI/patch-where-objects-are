"""Objectness splitting criterion.

The objectness scorer maps an image to an objectness map with one score per
finest-level cell. It is trained against the ground-truth objectness map
``o*`` (``labels.centernet_heatmap``) with the weighted BCE below, then
frozen. The objectness selector runs it and splits the quadtree under the
token budget.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from patch_where_objects_are.adaptive_patch.feature_distortion import (
    _build_stride16_trunk,
    _extract_features,
)
from patch_where_objects_are.adaptive_patch.patchification import Patchification
from patch_where_objects_are.adaptive_patch.selectors import score_quadtree


def heatmap_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 10.0,
) -> torch.Tensor:
    """Weighted BCE between the predicted and the ground-truth objectness map.

    Per-cell weight ``w = 1 + (pos_weight - 1) * o*`` and loss
    ``sum(w * BCE) / sum(w)``; ``pos_weight > 1`` favors recall.

    Args:
        logits: ``(B, gh, gw)`` scorer output before the sigmoid.
        target: ``(B, gh, gw)`` ground-truth objectness map in [0, 1].
        pos_weight: Weight at ``o* = 1`` (``p_w``; 1.0 = unweighted).
    """
    per_cell = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = 1.0 + (pos_weight - 1.0) * target
    return (per_cell * weight).sum() / weight.sum()


class ObjectnessScorer(nn.Module):
    """Objectness scorer: a frozen ImageNet-pretrained ShuffleNetV2x0.5 up to
    its stride-16 feature map (the backbone feature distortion also uses) and
    a trained head Conv2D-GroupNorm-SiLU-Conv2D with one output channel.

    ``forward``: ``(B, 3, H, W)`` ImageNet-normalized -> ``(B, H/16, W/16)``
    objectness logits.
    """

    def __init__(
        self,
        patch_size: int = 16,
        head_channels: int = 96,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if patch_size != 16:
            raise ValueError("ObjectnessScorer is stride-16 (patch_size=16) only")
        self.patch_size = patch_size
        self.head_channels = head_channels  # hidden dim of trainable head
        self.trunk = _build_stride16_trunk(pretrained)
        trunk_dim = 96  # ShuffleNetV2x0.5 stage3 output channels
        self.head = nn.Sequential(
            nn.Conv2d(trunk_dim, head_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, head_channels), num_channels=head_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels, 1, kernel_size=1),
        )

    def train(self, mode: bool = True) -> ObjectnessScorer:
        """Keep the frozen trunk in eval mode."""
        super().train(mode)
        self.trunk.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        """``(B, 3, H, W)`` ImageNet-normalized -> ``(B, H/16, W/16)`` logits."""
        with torch.no_grad():
            f = _extract_features(self.trunk, images, downscale=16)
        return self.head(f).squeeze(1)

    def gflops_per_image(self, h: int, w: int) -> float:
        """GFLOPs of one forward (multiply-accumulates, as in ``utils.flops``)."""
        from torch.utils.flop_counter import FlopCounterMode

        dummy = torch.zeros(1, 3, h, w, device=next(self.parameters()).device)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            self(dummy)
        return fc.get_total_flops() / 2 / 1e9  # torch counts mul+add; MACs = /2


class ObjectnessSelector(nn.Module):
    """Objectness selector: frozen objectness scorer + budgeted quadtree.

    Loads a scorer checkpoint written by ``scripts/train_objectness.py``,
    scores each image and calls ``score_quadtree`` per image. The scorer is
    the submodule ``stem`` (the key under which detector checkpoints store
    it) and is never trained with the detector. Evaluation needs no targets.
    """

    def __init__(
        self,
        checkpoint_path: str,
        token_budget: int,
        max_scale: int = 4,
        pool: str = "max",
    ) -> None:
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.stem = ObjectnessScorer(
            patch_size=ckpt["patch_size"],
            head_channels=ckpt["head_channels"],
            pretrained=False,  # trunk weights come from the checkpoint
        )
        self.stem.load_state_dict(ckpt["model"])
        self.stem.requires_grad_(False)
        self.stem.eval()
        self.token_budget = int(token_budget)
        self.max_scale = max_scale
        if pool not in {"max", "mean"}:
            raise ValueError(f"pool must be 'max' or 'mean', got {pool!r}")
        self.pool = pool

    def train(self, mode: bool = True) -> ObjectnessSelector:
        """Keep the frozen scorer in eval mode."""
        super().train(mode)
        self.stem.eval()
        return self

    def forward(
        self, img: torch.Tensor, targets: list[dict] | None = None
    ) -> list[Patchification]:
        """Patchifications of an ImageNet-normalized batch; ``targets`` is ignored."""
        with torch.no_grad():
            maps = self.stem(img).float().cpu()
        if self.pool == "mean":
            maps = (
                maps.sigmoid()
            )  # max pool is monotonic anyway, mean pool not -> explicitly use sigmoid
        return [
            score_quadtree(
                maps[b],
                self.stem.patch_size,
                self.token_budget,
                self.max_scale,
                pool=self.pool,
            )
            for b in range(img.shape[0])
        ]
