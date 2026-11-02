# loss_labels is adapted from Deformable DETR
# (https://github.com/fundamentalvision/Deformable-DETR),
# models/deformable_detr.py (SetCriterion), Copyright (c) 2020 SenseTime.
# Licensed under the Apache License 2.0.
#
# loss_boxes and the matched-index helper are adapted from DETR
# (https://github.com/facebookresearch/detr),
# models/detr.py, Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License 2.0.
# via RT-DETR (https://github.com/lyuwenyu/RT-DETR),
# rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py. Copyright (c) 2023 lyuwenyu.
# Licensed under the Apache License 2.0.
#
# Modifications: only the focal classification, L1 and GIoU losses; loss
# weighting inside forward with per-layer and denoising loss keys;
# reimplemented denoising indices; detection_loss_weight_dict and
# build_criterion added.

"""RT-DETR training criterion: focal classification, L1 and GIoU losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from patch_where_objects_are.models.detection_head import sigmoid_focal_loss
from patch_where_objects_are.models.matcher import HungarianMatcher
from patch_where_objects_are.utils.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)


def detection_loss_weight_dict(
    num_decoder_layers: int,
    *,
    num_denoising: int = 0,
    aux_loss: bool = True,
    loss_ce: float = 1.0,
    loss_bbox: float = 5.0,
    loss_giou: float = 2.0,
) -> dict[str, float]:
    """Explicit RT-DETR-style weights for main, per-layer aux, and denoising losses.

    Keys match ``SetCriterion`` outputs: ``loss_{ce,bbox,giou}`` on the final layer,
    ``loss_*_{i}`` for ``aux_outputs`` (one index per decoder depth, including
    encoder aux), and ``dn_loss_*_{i}`` when denoising is enabled (same as
    ``RTDETRHead`` wiring).
    """
    wd: dict[str, float] = {
        "loss_ce": loss_ce,
        "loss_bbox": loss_bbox,
        "loss_giou": loss_giou,
    }
    if not aux_loss:
        return wd
    for i in range(num_decoder_layers):
        wd[f"loss_ce_{i}"] = loss_ce
        wd[f"loss_bbox_{i}"] = loss_bbox
        wd[f"loss_giou_{i}"] = loss_giou
    if num_denoising > 0:
        for i in range(num_decoder_layers):
            wd[f"dn_loss_ce_{i}"] = loss_ce
            wd[f"dn_loss_bbox_{i}"] = loss_bbox
            wd[f"dn_loss_giou_{i}"] = loss_giou
    return wd


class SetCriterion(nn.Module):
    """RT-DETR training criterion (focal classification + L1 + GIoU).

    Also handles auxiliary decoder losses and contrastive denoising losses.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        focal_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha

    def _get_src_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat(
            [torch.full_like(s, i) for i, (s, _) in enumerate(indices)]
        )
        src_idx = torch.cat([s for s, _ in indices])
        return batch_idx, src_idx

    def loss_labels(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> dict[str, torch.Tensor]:
        """Sigmoid focal classification loss ``loss_ce`` over all queries.

        Unmatched queries get an all-zero (background) target.
        """
        src_logits = outputs["pred_logits"]  # (B, nq, C)
        idx = self._get_src_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices, strict=True)]
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_onehot = torch.zeros(
            [*src_logits.shape[:2], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            device=src_logits.device,
        )
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_onehot = target_onehot[:, :, :-1]

        loss_ce = (
            sigmoid_focal_loss(
                src_logits, target_onehot, num_boxes, alpha=self.focal_alpha
            )
            * src_logits.shape[1]
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> dict[str, torch.Tensor]:
        """L1 (``loss_bbox``) and GIoU (``loss_giou``) losses of the matched boxes.

        Boxes are normalized cxcywh; both sums are divided by ``num_boxes``.
        """
        idx = self._get_src_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][J] for t, (_, J) in zip(targets, indices, strict=True)]
        )

        loss_l1 = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes

        giou_mat = generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes),
        )
        loss_giou = (1 - torch.diag(giou_mat)).sum() / num_boxes

        return {"loss_bbox": loss_l1, "loss_giou": loss_giou}

    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, torch.Tensor]:
        """Match the head outputs to ``targets`` and compute all loss terms.

        Returns the unweighted terms of the final layer (``loss_*``), the
        auxiliary outputs (``loss_*_{i}``) and the denoising outputs
        (``dn_loss_*_{i}``), plus ``"loss"``: the ``weight_dict``-weighted sum
        of the terms that require grad.
        """
        indices = self.matcher(outputs, targets)
        num_boxes = max(1, sum(len(t["labels"]) for t in targets))

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        # Auxiliary decoder outputs.
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            aux_indices = self.matcher(aux, targets)
            for k, v in {
                **self.loss_labels(aux, targets, aux_indices, num_boxes),
                **self.loss_boxes(aux, targets, aux_indices, num_boxes),
            }.items():
                losses[f"{k}_{i}"] = v

        # Denoising auxiliary outputs.
        for i, dn_aux in enumerate(outputs.get("dn_aux_outputs", [])):
            dn_meta = outputs.get("dn_meta", {})
            dn_idx = self._get_dn_indices(dn_meta, targets)
            for k, v in {
                **self.loss_labels(dn_aux, targets, dn_idx, num_boxes),
                **self.loss_boxes(dn_aux, targets, dn_idx, num_boxes),
            }.items():
                losses[f"dn_{k}_{i}"] = v

        losses["loss"] = sum(
            self.weight_dict[k] * v
            for k, v in losses.items()
            if isinstance(v, torch.Tensor) and v.requires_grad and k in self.weight_dict
        )
        return losses

    def _get_dn_indices(
        self, dn_meta: dict, targets: list[dict]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build indices for denoising loss from dn_meta."""
        if not dn_meta:
            return [
                (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
            ]
        dn_positive_idx = dn_meta["dn_positive_idx"]
        dn_num_group = dn_meta["dn_num_group"]
        indices = []
        for _i, (t, pos_idx) in enumerate(zip(targets, dn_positive_idx, strict=True)):
            n_gt = len(t["labels"])
            if n_gt == 0 or len(pos_idx) == 0:
                indices.append(
                    (
                        torch.zeros(0, dtype=torch.long, device=t["labels"].device),
                        torch.zeros(0, dtype=torch.long, device=t["labels"].device),
                    )
                )
                continue
            tgt_idx = torch.arange(n_gt, device=pos_idx.device).tile(dn_num_group)
            indices.append((pos_idx, tgt_idx))
        return indices


def build_criterion(
    num_classes: int,
    *,
    num_decoder_layers: int = 6,
    num_denoising: int = 100,
    aux_loss: bool = True,
    loss_ce: float = 1.0,
    loss_bbox: float = 5.0,
    loss_giou: float = 2.0,
) -> SetCriterion:
    """Build a ``SetCriterion`` with a ``HungarianMatcher`` and RT-DETR loss weights."""
    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = detection_loss_weight_dict(
        num_decoder_layers,
        num_denoising=num_denoising,
        aux_loss=aux_loss,
        loss_ce=loss_ce,
        loss_bbox=loss_bbox,
        loss_giou=loss_giou,
    )
    return SetCriterion(num_classes, matcher, weight_dict)
