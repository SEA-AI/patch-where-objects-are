# Adapted from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR),
# models/matcher.py, Copyright (c) 2020 SenseTime.
# Licensed under the Apache License 2.0.
# That file is modified from DETR (https://github.com/facebookresearch/detr),
# models/matcher.py, Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License 2.0.
#
# Default costs as in RT-DETR (https://github.com/lyuwenyu/RT-DETR),
# rtdetr_pytorch/src/zoo/rtdetr/matcher.py. Copyright (c) 2023 lyuwenyu.
# Licensed under the Apache License 2.0.
#
# Modifications: focal-loss class cost only, costs as constructor arguments,
# cost matrix computed in fp32 with NaN/inf replaced.

"""Hungarian matching of predictions to targets (RT-DETR)."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from patch_where_objects_are.utils.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)


class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground-truth targets."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: list[dict],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Match each image's queries to its targets (Hungarian algorithm).

        The cost sums the weighted focal classification, L1 and negative GIoU
        terms between the ``pred_logits`` / ``pred_boxes`` (cxcywh) and the
        target ``labels`` / ``boxes``.

        Returns:
            Per image, ``(query_indices, target_indices)`` as int64 tensors.
        """
        bs, nq = outputs["pred_logits"].shape[:2]

        # Run matching in float32 outside autocast so cost matrices stay finite for
        # scipy.
        dev = outputs["pred_logits"].device
        amp_off = (
            torch.autocast(device_type="cuda", enabled=False)
            if dev.type == "cuda"
            else nullcontext()
        )
        with amp_off:
            logits = outputs["pred_logits"].flatten(0, 1).float()
            out_prob = logits.sigmoid()  # (B*nq, C)
            out_bbox = outputs["pred_boxes"].flatten(0, 1).float()  # (B*nq, 4)

            tgt_ids = torch.cat([t["labels"] for t in targets])
            tgt_bbox = torch.cat([t["boxes"] for t in targets]).float()

            # Focal-style classification cost.
            alpha, gamma = 0.25, 2.0
            neg_cost = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_cls = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]

            cost_box = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox),
            )

            C = (
                self.cost_class * cost_cls
                + self.cost_bbox * cost_box
                + self.cost_giou * cost_giou
            )
            C = torch.nan_to_num(C, nan=1e8, posinf=1e8, neginf=-1e8)
            C = C.reshape(bs, nq, -1).cpu()

        sizes = [len(t["labels"]) for t in targets]
        indices = [
            linear_sum_assignment(c[i].numpy())
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long))
            for i, j in indices
        ]
