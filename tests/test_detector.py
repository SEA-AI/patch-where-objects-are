"""Uniform and mixed-resolution detectors: backbone -> neck -> RT-DETR head."""

from __future__ import annotations

import torch

from patch_where_objects_are.adaptive_patch.grid_broadcast import (
    batch_tokens_to_base_grid,
    tokens_to_base_grid,
)
from patch_where_objects_are.adaptive_patch.patchification import Patchification
from patch_where_objects_are.adaptive_patch.selectors import score_quadtree
from tests.helpers import (
    mixed_patchification_4x4,
    random_targets,
    tiny_detector,
    unit_layer_scale,
)


def test_tokens_fill_the_cells_they_cover():
    pf = mixed_patchification_4x4()
    tokens = torch.arange(13.0)[:, None]  # token i has value i
    grid = tokens_to_base_grid(tokens, pf)[0]
    assert grid.shape == (4, 4)
    assert (grid[:2, :2] == 0).all()  # the 32 px token covers 2 x 2 cells
    for i, (gy, gx) in enumerate(pf.base_indices.tolist()[1:], start=1):
        assert grid[gy, gx] == i


def test_param_groups_use_the_configured_weight_decay():
    groups = tiny_detector("uniform").get_param_groups(0.75, 1e-4, 0.05)
    assert {g["weight_decay"] for g in groups} == {0.0, 0.05}
    for g in groups:
        assert g["weight_decay"] == (0.0 if g["params"][0].ndim == 1 else 0.05)


def test_uniform_detector_trains_and_detects():
    model = tiny_detector("uniform").train()
    losses = model(torch.randn(2, 3, 64, 64), targets=random_targets(2))
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()

    model.eval()
    with torch.no_grad():
        detections = model(torch.randn(2, 3, 64, 64))
    assert len(detections) == 2
    assert all(set(d) == {"boxes", "scores", "labels"} for d in detections)


def test_mixed_resolution_detector_stays_within_the_token_budget():
    model = tiny_detector(
        "random", adaptive_token_budget=10, head_full_sparse=True
    ).train()
    losses = model(torch.randn(2, 3, 64, 64), targets=random_targets(2))
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()

    model.eval()
    with torch.no_grad():
        detections = model(torch.randn(2, 3, 64, 64))
    assert len(detections) == 2
    assert max(model.last_token_counts) <= 10


@torch.no_grad()
def test_mixed_resolution_detector_on_uniform_patches_equals_uniform_detector():
    mixed = tiny_detector("uniform_mixed").eval()
    uniform = tiny_detector("uniform").eval()
    # Same weights; the fusion modules are unused on uniform patches.
    uniform.load_state_dict(
        {
            k: v
            for k, v in mixed.state_dict().items()
            if not k.startswith("backbone.mixed_patch_embed")
        }
    )
    img = torch.randn(2, 3, 64, 64)
    for a, b in zip(mixed(img), uniform(img), strict=True):
        torch.testing.assert_close(a["boxes"], b["boxes"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(a["scores"], b["scores"], rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_sparse_head_equals_dense_head_on_the_broadcast_grid():
    """The sparse head reads the tokens directly; decoding the encoded tokens
    broadcast to the dense base grid gives the same predictions.
    """
    head = tiny_detector("uniform_mixed", head_full_sparse=True).head.eval()
    pf = score_quadtree(torch.rand(52, 52), 16, token_budget=256)
    tokens = [torch.randn(pf.num_tokens, 384)]

    dense = batch_tokens_to_base_grid(head.encode_sparse(tokens, [pf]), [pf])
    expected = head._decode_projected_features(
        [dense], [list(dense.shape[-2:])], [0], None
    )
    actual = head.forward_sparse(tokens, [pf])

    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_padded_tokens_do_not_change_the_detections():
    """Images with different token counts get the same detections in one
    padded batch as alone.
    """
    model = tiny_detector("uniform_mixed", head_full_sparse=True).eval()
    unit_layer_scale(model.backbone)
    img = torch.randn(2, 3, 64, 64)
    pfs = [mixed_patchification_4x4(), Patchification.uniform(4, 4, 16)]

    model.selector = lambda img, targets: pfs
    padded = model(img)

    for b, pf in enumerate(pfs):
        model.selector = lambda img, targets, pf=pf: [pf]
        alone = model(img[b : b + 1])[0]
        for key in ("boxes", "scores"):
            torch.testing.assert_close(padded[b][key], alone[key], rtol=1e-5, atol=1e-5)
        assert torch.equal(padded[b]["labels"], alone["labels"])


@torch.no_grad()
def test_sparse_head_batch_norm_ignores_the_padding():
    """In training, the input projection's BatchNorm updates its running
    statistics from the real tokens only.
    """
    head = tiny_detector("uniform_mixed", head_full_sparse=True).head.train()
    pfs = [mixed_patchification_4x4(), Patchification.uniform(4, 4, 16)]
    tokens = [torch.randn(pf.num_tokens, 384) for pf in pfs]
    conv, bn = head.input_proj[0]

    head.encode_sparse(tokens, pfs)

    proj = torch.cat(tokens) @ conv.weight[:, :, 0, 0].T  # (13 + 16, hidden_dim)
    m = bn.momentum
    torch.testing.assert_close(bn.running_mean, m * proj.mean(0))
    torch.testing.assert_close(bn.running_var, (1 - m) + m * proj.var(0))
