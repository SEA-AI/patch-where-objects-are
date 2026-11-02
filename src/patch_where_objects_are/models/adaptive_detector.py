"""Mixed-resolution detector.

    image -> patch selector -> Patchification
          -> AdaptiveDinoV3Backbone.forward_tokens (mixed-resolution patch
             embedding, RoPE on the token centers)
          -> GridBroadcastNeck (tokens duplicated onto the base grid) or the
             sparse head
          -> RTDETRHead

With a uniform ``Patchification`` the output equals the uniform detector's.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from patch_where_objects_are.adaptive_patch.grid_broadcast import (
    batch_tokens_to_base_grid,
)
from patch_where_objects_are.adaptive_patch.patchification import Patchification
from patch_where_objects_are.models.backbones.dinov3_adaptive import (
    AdaptiveDinoV3Backbone,
)
from patch_where_objects_are.models.criterion import SetCriterion
from patch_where_objects_are.models.detection_head import RTDETRHead
from patch_where_objects_are.models.detector import Detector

# selector(img (B,3,H,W), targets or None) -> per-image Patchification list.
PatchSelector = Callable[[torch.Tensor, list[dict] | None], Sequence[Patchification]]


class GridBroadcastNeck(nn.Module):
    """Parameter-free neck: duplicates the tokens onto the base grid. The
    projection to the head's width is in ``RTDETRHead.input_proj``.
    """

    def __init__(self, intermediate_layers: list[int]) -> None:
        super().__init__()
        if len(intermediate_layers) != 1:
            raise ValueError(
                "GridBroadcastNeck expects exactly one layer index, "
                f"got {intermediate_layers}"
            )
        self.intermediate_layers = list(intermediate_layers)

    def forward(
        self,
        token_layers: Sequence[torch.Tensor],
        patchifications: Sequence[Patchification],
    ) -> list[torch.Tensor]:
        """Broadcast the single token layer to a ``(B, D, Hp, Wp)`` base-grid map."""
        if len(token_layers) != 1:
            raise ValueError(f"Expected 1 token layer, got {len(token_layers)}")
        return [batch_tokens_to_base_grid(token_layers[0], patchifications)]


class AdaptiveDINOv3Detector(Detector):
    """Mixed-resolution DINOv3 detector.

    Args:
        backbone:  ``AdaptiveDinoV3Backbone``.
        neck:      ``GridBroadcastNeck``.
        head:      Single-scale ``RTDETRHead``.
        criterion: Training criterion (None for inference).
        selector:  Patch selector; ``None`` = uniform patches.
    """

    def __init__(
        self,
        backbone: AdaptiveDinoV3Backbone,
        neck: GridBroadcastNeck,
        head: RTDETRHead,
        criterion: SetCriterion | None = None,
        confidence_threshold: float = 0.001,
        nms_iou_threshold: float = 0.5,
        selector: PatchSelector | None = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            fpn=neck,
            head=head,
            criterion=criterion,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        self.selector = selector

    @property
    def eval_needs_orig_size(self) -> bool:
        """True if evaluation must pass the targets for their ``orig_size``:
        feature distortion at native resolution needs each image's size, not
        its boxes, so unannotated test splits work.
        """
        return self.selector is not None and getattr(
            self.selector, "needs_orig_size", False
        )

    def _select_patchifications(
        self, img: torch.Tensor, targets: list[dict] | None
    ) -> list[Patchification]:
        if self.selector is None:
            B, _, H, W = img.shape
            return [self.backbone.uniform_patchification(H, W)] * B
        return list(self.selector(img, targets))

    def forward(
        self,
        img: torch.Tensor,
        targets: list[dict] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict]:
        """As ``Detector.forward``: loss dict in training, per-image detections
        at evaluation.
        """
        pfs = self._select_patchifications(img, targets)

        # Token counts per image and per level, read by the evaluation engine.
        self.last_token_counts = [pf.num_tokens for pf in pfs]
        self.last_grid_cells = pfs[0].grid_hw[0] * pfs[0].grid_hw[1]
        hist: dict[int, int] = {}
        per_image: list[dict[int, int]] = []
        for pf in pfs:
            img_hist: dict[int, int] = {}
            for m, c in zip(*torch.unique(pf.scales, return_counts=True), strict=True):
                img_hist[int(m)] = int(c)
                hist[int(m)] = hist.get(int(m), 0) + int(c)
            per_image.append(img_hist)
        self.last_scale_counts = hist
        self.last_scale_counts_per_image = per_image

        token_layers = self.backbone.forward_tokens(
            img, pfs, n=self.fpn.intermediate_layers
        )

        if self.head.full_sparse:
            head_out = self.head.forward_sparse(token_layers[0], pfs, targets=targets)
            return self._finalize(head_out, img, targets)

        # Dense head: the tokens are duplicated onto the base grid.
        feats = self.fpn(token_layers, pfs)

        head_out = self.head(feats, targets=targets)

        return self._finalize(head_out, img, targets)
