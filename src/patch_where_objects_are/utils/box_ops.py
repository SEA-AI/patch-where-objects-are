# Follows DETR (https://github.com/facebookresearch/detr), util/box_ops.py.
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License 2.0.
#
# Modifications: functions rewritten (inline union, no degenerate-box asserts,
# clamped union), nms wrapper around torchvision added.

"""Bounding box utilities."""

from __future__ import annotations

import torch
from torchvision import ops


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) → (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute area of boxes in (x1, y1, x2, y2) format."""
    return (boxes[:, 2] - boxes[:, 0]).clamp(0) * (boxes[:, 3] - boxes[:, 1]).clamp(0)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU between all pairs of boxes.

    Args:
        boxes1: ``(N, 4)`` in xyxy format.
        boxes2: ``(M, 4)`` in xyxy format.

    Returns:
        ``(N, M)`` GIoU matrix.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    # Enclosing box.
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]

    return iou - (area_enc - union) / area_enc.clamp(min=1e-6)


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Non-maximum suppression wrapper.

    Args:
        boxes:         ``(N, 4)`` in xyxy format.
        scores:        ``(N,)``.
        iou_threshold: IoU threshold.

    Returns:
        Indices of kept boxes.
    """
    return ops.nms(boxes, scores, iou_threshold)
