"""Patch selection: the budgeted quadtree and the selectors built on it."""

from __future__ import annotations

import pytest
import torch

from patch_where_objects_are.adaptive_patch.entropy import compute_cell_entropy
from patch_where_objects_are.adaptive_patch.feature_distortion import (
    FeatureDistortionSelector,
)
from patch_where_objects_are.adaptive_patch.objectness import ObjectnessSelector
from patch_where_objects_are.adaptive_patch.selectors import (
    CriterionMapSelector,
    score_quadtree,
)
from tests.helpers import covers_grid, scorer_checkpoint


def test_quadtree_respects_budget_and_covers_grid():
    scores = torch.rand(8, 8)  # 128 x 128 px image, 4 root nodes of 64 px
    for budget in (0, 10, 40, 64):
        pf = score_quadtree(scores, patch_size=16, token_budget=budget)
        assert pf.num_tokens <= max(budget, 4)
        assert covers_grid(pf)
    assert score_quadtree(scores, 16, token_budget=0).scales.unique().tolist() == [4]
    assert score_quadtree(scores, 16, token_budget=64).scales.unique().tolist() == [1]


def test_quadtree_splits_the_highest_scores_first():
    scores = torch.zeros(8, 8)
    scores[5, 6] = 1.0
    # 4 roots + one 64 -> 32 px split + one 32 -> 16 px split = 10 tokens.
    pf = score_quadtree(scores, patch_size=16, token_budget=10)
    fine = {
        tuple(idx)
        for m, idx in zip(pf.scales.tolist(), pf.base_indices.tolist(), strict=True)
        if m == 1
    }
    assert (5, 6) in fine
    assert len(fine) == 4


def test_cell_entropy():
    gray = torch.full((1, 16, 32), 128.0)  # two 16 px cells
    gray[:, :, 16:24] = 10.0  # second cell: half 128, half 10
    entropy = compute_cell_entropy(gray, patch_size=16)
    torch.testing.assert_close(entropy, torch.tensor([[[0.0, 1.0]]]))


@pytest.mark.parametrize(
    "name", ["objectness", "feature_distortion", "entropy", "random"]
)
def test_selectors_return_budgeted_partitions(name, tmp_path):
    budget = 20
    if name == "objectness":
        selector = ObjectnessSelector(
            scorer_checkpoint(tmp_path / "scorer.pth"), token_budget=budget
        )
    elif name == "feature_distortion":
        # Random weights instead of the ImageNet download; the partition
        # geometry does not depend on them.
        selector = FeatureDistortionSelector(
            patch_size=16, token_budget=budget, pretrained=False
        )
    else:
        selector = CriterionMapSelector(name, patch_size=16, token_budget=budget)
    img = torch.randn(2, 3, 128, 128)
    # Feature distortion scores at the native resolution, read from the targets.
    targets = [{"orig_size": torch.tensor([128, 128])} for _ in range(2)]

    pfs = selector(img, targets)

    assert len(pfs) == 2
    for pf in pfs:
        assert pf.grid_hw == (8, 8)
        assert pf.num_tokens <= budget
        assert covers_grid(pf)
