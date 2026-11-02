"""Small builders shared by the tests (CPU, random weights, no data)."""

from __future__ import annotations

import torch

from patch_where_objects_are.adaptive_patch.objectness import ObjectnessScorer
from patch_where_objects_are.adaptive_patch.patchification import Patchification
from patch_where_objects_are.models.factory import build_detector


def covers_grid(pf: Patchification) -> bool:
    """Every base cell is covered by exactly one token (valid quadtree partition)."""
    gh, gw = pf.grid_hw
    count = torch.zeros(gh, gw, dtype=torch.long)
    for m, (gy, gx) in zip(pf.scales.tolist(), pf.base_indices.tolist(), strict=True):
        count[gy : gy + m, gx : gx + m] += 1
    return bool((count == 1).all())


def mixed_patchification_4x4() -> Patchification:
    """4x4 base grid: one 32 px token at (0, 0) and 12 16 px tokens (13 tokens)."""
    indices = [(0, 0)] + [
        (gy, gx) for gy in range(4) for gx in range(4) if gy >= 2 or gx >= 2
    ]
    return Patchification(
        patch_size=16,
        grid_hw=(4, 4),
        scales=torch.tensor([2] + [1] * 12),
        base_indices=torch.tensor(indices),
    )


@torch.no_grad()
def unit_layer_scale(backbone) -> None:
    """Set the blocks' LayerScale to 1. At its init value (1e-5) attention barely
    changes the tokens, so a wrong attention mask would go unnoticed.
    """
    for block in backbone.blocks:
        block.layer_scale1.lambda1.fill_(1.0)
        block.layer_scale2.lambda1.fill_(1.0)


def tiny_detector(selector: str = "uniform", **kw):
    """ViT-S/16 on 64 x 64 images (4 x 4 base grid) with a one-layer RT-DETR head."""
    args = {
        "backbone_type": "dinov3_adaptive",
        "adaptive_patch_selector": selector,
        "dinov3_variant": "vits16",
        "dinov3_weights_path": "",
        "img_size": 64,
        "patch_size": 16,
        "fpn_hidden_dim": 64,
        "num_classes": 7,
        "num_queries": 10,
        "num_decoder_layers": 1,
        "num_denoising": 0,
        "aux_loss": False,
        "head_feat_strides": [16],
        "head_use_encoder_idx": [0],
    }
    args.update(kw)
    return build_detector(**args)


def random_targets(batch_size: int, num_classes: int = 7) -> list[dict]:
    """Two random boxes (normalized cxcywh) per image."""
    targets = []
    for _ in range(batch_size):
        boxes = torch.rand(2, 4)
        boxes[:, :2] = boxes[:, :2] * 0.7 + 0.1
        boxes[:, 2:] = boxes[:, 2:] * 0.3 + 0.05
        targets.append({"boxes": boxes, "labels": torch.randint(0, num_classes, (2,))})
    return targets


def scorer_checkpoint(path) -> str:
    """Objectness scorer checkpoint as ``scripts/train_objectness.py`` writes it."""
    model = ObjectnessScorer(patch_size=16, head_channels=8, pretrained=False)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 0,
            "head_channels": 8,
            "patch_size": 16,
            "img_size": 128,
        },
        path,
    )
    return str(path)
