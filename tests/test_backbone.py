"""Mixed-resolution patch embedding and DINOv3 backbone."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from patch_where_objects_are.adaptive_patch.mixed_patch_embed import MixedPatchEmbed
from patch_where_objects_are.adaptive_patch.patchification import Patchification
from patch_where_objects_are.models.backbones.dinov3 import DinoV3Config
from patch_where_objects_are.models.backbones.dinov3_adaptive import (
    AdaptiveDinoV3Backbone,
)
from patch_where_objects_are.utils.flops import dinov3_backbone_gflops
from tests.helpers import mixed_patchification_4x4, unit_layer_scale


def test_patch_embed_on_uniform_patches_equals_the_patch_conv():
    conv = nn.Conv2d(3, 32, kernel_size=16, stride=16)
    embed = MixedPatchEmbed(conv, scales=(1, 2, 4)).eval()
    img = torch.randn(2, 3, 64, 96)  # 4 x 6 grid

    tokens = embed(img, Patchification.uniform(4, 6, 16))

    expected = conv(img).flatten(2).transpose(1, 2)
    for b in range(2):
        torch.testing.assert_close(tokens[b], expected[b], rtol=0.0, atol=0.0)


def test_coarse_patch_embeds_its_resized_pixels_at_init():
    """The fusion MLP is zero-initialized, so a 32 px token starts as the patch
    conv applied to its pixels resized to 16 px.
    """
    conv = nn.Conv2d(3, 32, kernel_size=16, stride=16)
    embed = MixedPatchEmbed(conv, scales=(1, 2, 4)).eval()
    img = torch.randn(1, 3, 64, 64)

    tokens = embed(img, mixed_patchification_4x4())[0]

    resized = F.interpolate(
        img[:, :, :32, :32], size=(16, 16), mode="bilinear", antialias=True
    )
    torch.testing.assert_close(tokens[0], conv(resized).flatten(), rtol=1e-6, atol=1e-6)


def test_backbone_on_uniform_patches_equals_dinov3():
    cfg = DinoV3Config(variant="vits16", patch_size=16, img_size=64, weights_path="")
    backbone = AdaptiveDinoV3Backbone(cfg).eval()
    img = torch.randn(2, 3, 64, 96)

    mixed = backbone.forward_tokens(img, patchifications=None, n=[11])[0]
    stock = backbone.get_intermediate_layers(img, n=[11])[0].flatten(2).transpose(1, 2)

    torch.testing.assert_close(mixed, stock, rtol=1e-6, atol=1e-6)


def test_backbone_takes_a_different_token_count_per_image():
    cfg = DinoV3Config(variant="vits16", patch_size=16, img_size=64, weights_path="")
    backbone = AdaptiveDinoV3Backbone(cfg).eval()
    pfs = [mixed_patchification_4x4(), Patchification.uniform(4, 4, 16)]

    out = backbone.forward_tokens(torch.randn(2, 3, 64, 64), patchifications=pfs)[0]

    assert [t.shape for t in out] == [(13, 384), (16, 384)]


def test_backbone_gflops_grow_with_tokens():
    uniform_832 = dinov3_backbone_gflops({1: 2704}, 2704, 768, 12)
    uniform_448 = dinov3_backbone_gflops({1: 784}, 784, 768, 12)
    mixed_784 = dinov3_backbone_gflops({1: 400, 2: 200, 4: 184}, 2704, 768, 12)
    assert uniform_832 > mixed_784 > uniform_448


@torch.no_grad()
def test_padded_tokens_do_not_change_the_backbone_output():
    """Each image's tokens are the same whether the batch pads them or not."""
    cfg = DinoV3Config(variant="vits16", patch_size=16, img_size=64, weights_path="")
    backbone = AdaptiveDinoV3Backbone(cfg).eval()
    unit_layer_scale(backbone)
    img = torch.randn(2, 3, 64, 64)
    pfs = [mixed_patchification_4x4(), Patchification.uniform(4, 4, 16)]

    padded = backbone.forward_tokens(img, patchifications=pfs, n=[5, 11])

    for b, pf in enumerate(pfs):
        alone = backbone.forward_tokens(img[b : b + 1], patchifications=[pf], n=[5, 11])
        for padded_layer, alone_layer in zip(padded, alone, strict=True):
            torch.testing.assert_close(
                padded_layer[b], alone_layer[0], rtol=1e-5, atol=1e-5
            )


def test_backbone_pads_tokens_only_under_sdpa():
    """Eager attention adds the boolean padding mask instead of masking."""
    cfg = DinoV3Config(variant="vits16", patch_size=16, img_size=64, weights_path="")
    backbone = AdaptiveDinoV3Backbone(cfg).eval()
    backbone.model.set_attn_implementation("eager")
    pfs = [mixed_patchification_4x4(), Patchification.uniform(4, 4, 16)]

    with pytest.raises(RuntimeError, match="sdpa"):
        backbone.forward_tokens(torch.randn(2, 3, 64, 64), patchifications=pfs)
