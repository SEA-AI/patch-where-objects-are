"""Objectness scorer: its target map, its loss, its training and its recall."""

from __future__ import annotations

import numpy as np
import torch

from patch_where_objects_are.adaptive_patch.labels import centernet_heatmap
from patch_where_objects_are.adaptive_patch.objectness import (
    ObjectnessScorer,
    heatmap_bce_loss,
)
from patch_where_objects_are.adaptive_patch.selector_metrics import (
    object_min_ranks,
    recall_at_budget,
)


def test_target_map_peaks_at_the_box_center():
    boxes = torch.tensor([[16.0, 16.0, 48.0, 48.0]])  # xyxy px, center (32, 32)
    heat = centernet_heatmap(boxes, 16, (4, 4))
    assert heat.shape == (4, 4)
    assert heat.max() <= 1.0
    assert heat[1:3, 1:3].min() == heat.max()  # the 4 cells around the center
    assert heat[0, 3] < heat.max()


def test_loss_weights_missed_objects():
    target = torch.zeros(1, 4, 4)
    target[0, 1, 1] = 1.0
    logits = torch.full((1, 4, 4), -4.0)  # predicts no object anywhere
    assert heatmap_bce_loss(logits, target, pos_weight=10.0) > heatmap_bce_loss(
        logits, target, pos_weight=1.0
    )


def test_only_the_head_trains():
    model = ObjectnessScorer(patch_size=16, head_channels=8, pretrained=False).train()
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 8, 8)  # one score per 16 px cell
    out.sum().backward()
    assert not model.trunk.training
    assert all(p.grad is not None for p in model.head.parameters())
    assert all(p.grad is None for p in model.trunk.parameters())


def test_recall_at_budget():
    # One image with 2 x 2 cells; the object covers cell (0, 0), ranked 2nd.
    scores = np.array([[[0.5, 0.9], [0.1, 0.2]]])
    ranks = object_min_ranks(scores, np.array([0]), np.array([[0, 0, 0, 0]]))
    assert ranks.tolist() == [1]
    recall = recall_at_budget(ranks, 4, np.array([0.25, 0.5]))
    assert recall.tolist() == [0.0, 1.0]
