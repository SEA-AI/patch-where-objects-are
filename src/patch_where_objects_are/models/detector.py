"""Detector: backbone -> neck -> RT-DETR head.

Used as is by the uniform detector; the mixed-resolution detector subclasses
it.
"""

from __future__ import annotations

import torch
from torch import nn

from patch_where_objects_are.models.criterion import SetCriterion
from patch_where_objects_are.models.detection_head import RTDETRHead
from patch_where_objects_are.models.detector_base import DetectionModel
from patch_where_objects_are.utils.box_ops import (
    box_cxcywh_to_xyxy,
    nms,
)


class Detector(DetectionModel):
    """Detection model: backbone + neck + ``RTDETRHead``.

    Args:
        backbone:   Feature extractor with ``get_intermediate_layers``.
        fpn:        Neck turning the backbone output into feature maps.
        head:       ``RTDETRHead``.
        criterion:  ``SetCriterion`` for the training loss (None for inference).
        confidence_threshold: Score threshold of the detections at inference.
        nms_iou_threshold: NMS IoU threshold at inference.
    """

    def __init__(
        self,
        backbone: nn.Module,
        fpn: nn.Module,
        head: RTDETRHead,
        criterion: SetCriterion | None = None,
        confidence_threshold: float = 0.001,
        nms_iou_threshold: float = 0.5,
    ) -> None:
        super().__init__(backbone=backbone, fpn=fpn, head=head)
        self.criterion = criterion
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict]:
        """Args:
            images:  ``(B, C, H, W)`` normalized images.
            targets: Per-image target dicts (training) or None (inference).

        Returns:
            Training: dict with the scalar ``"loss"`` and its terms.
            Inference: per-image dicts with ``"boxes"`` (xyxy, pixels),
            ``"scores"`` and ``"labels"`` (class indices).
        """
        intermediate = self.backbone.get_intermediate_layers(
            images, n=self.fpn.intermediate_layers
        )  # [Tensor(B, D, Hp, Wp)]

        fpn_feats = self.fpn(intermediate)  # [P4]

        head_out = self.head(fpn_feats, targets=targets)
        return self._finalize(head_out, images, targets)

    def _finalize(
        self,
        head_out: dict,
        images: torch.Tensor,
        targets: list[dict] | None,
    ) -> dict[str, torch.Tensor] | list[dict]:
        """Loss in training, postprocessed detections at evaluation."""
        if self.training:
            assert targets is not None
            assert self.criterion is not None
            return self.criterion(head_out, targets)
        H, W = images.shape[-2:]
        return self._postprocess(head_out, H, W)

    def _postprocess(
        self,
        head_out: dict,
        H: int,
        W: int,
    ) -> list[dict]:
        """Per-image detections with boxes in xyxy pixel coordinates."""
        pred_logits = head_out["pred_logits"]  # (B, nq, C)
        pred_boxes = head_out["pred_boxes"]  # (B, nq, 4) cxcywh norm

        results = []
        B = pred_logits.shape[0]
        for b in range(B):
            scores = pred_logits[b].sigmoid()  # (nq, C)
            max_scores, labels = scores.max(-1)  # (nq,), (nq,)

            # Filter by confidence.
            keep = max_scores > self.confidence_threshold
            scores_b = max_scores[keep]
            labels_b = labels[keep]
            boxes_cxcywh = pred_boxes[b][keep]

            # Normalized cxcywh -> xyxy in pixels.
            boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
            scale = boxes_cxcywh.new_tensor([W, H, W, H])
            boxes_xyxy = boxes_xyxy * scale

            # Class-agnostic NMS.
            if len(boxes_xyxy) > 0:
                keep_nms = nms(boxes_xyxy, scores_b, self.nms_iou_threshold)
                scores_b = scores_b[keep_nms]
                labels_b = labels_b[keep_nms]
                boxes_xyxy = boxes_xyxy[keep_nms]

            results.append(
                {
                    "boxes": boxes_xyxy,
                    "scores": scores_b,
                    "labels": labels_b,
                }
            )
        return results

    def freeze_backbone(self, num_layers: int) -> None:
        """Freeze the patch embedding and the first ``num_layers`` transformer
        blocks; with all blocks frozen, also the cls, register and mask tokens
        and the norms.
        """
        if hasattr(self.backbone, "patch_embed"):
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is not None:
            for i in range(min(num_layers, len(blocks))):
                for p in blocks[i].parameters():
                    p.requires_grad = False
            if num_layers >= len(blocks):
                self._freeze_dinov3_base_extras()

    def _freeze_dinov3_base_extras(self) -> None:
        base = getattr(self.backbone, "model", None)
        emb = getattr(base, "embeddings", None)
        if emb is None:
            return
        for p in (emb.cls_token, emb.register_tokens, emb.mask_token):
            p.requires_grad = False
        for p in base.norm.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_param_groups(
        self, layer_lr_decay: float, base_lr: float, weight_decay: float
    ) -> list[dict]:
        """AdamW parameter groups with layer-wise learning-rate decay.

        Weights get ``weight_decay``; biases, norms and other 1-D parameters get 0.
        """
        groups = []
        depth = self.backbone.depth
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            lr = base_lr
            if layer_lr_decay < 1.0:
                for i in range(depth):
                    if (
                        f"backbone.blocks.{i}." in name
                        or f"backbone.model.model.layer.{i}." in name
                    ):
                        distance = depth - 1 - i
                        lr = base_lr * (layer_lr_decay**distance)
                        break
                else:
                    if (
                        "backbone.patch_embed" in name
                        or "backbone.model.embeddings.patch_embeddings" in name
                    ):
                        lr = base_lr * (layer_lr_decay**depth)
            no_decay = param.ndim == 1 or "bias" in name or "norm" in name
            wd = 0.0 if no_decay else weight_decay
            lr_scale = lr / base_lr if base_lr else 1.0
            groups.append(
                {
                    "params": [param],
                    "lr": lr,
                    "lr_scale": lr_scale,
                    "weight_decay": wd,
                }
            )
        return groups


# Patch selectors (``backbone.patch_selector``):
#   uniform             uniform patches: the uniform detector
#   uniform_mixed       uniform patches through the mixed-resolution path (sanity check)
#   objectness          objectness scorer (``backbone.scorer_checkpoint``)
#   feature_distortion  feature distortion, scored at native resolution
#   entropy             entropy
#   random              random score maps
# ``selector_pool`` (max | mean) and ``selector_scoring``
# (finest | per_scale).
