# The DINOv3 ViT runs on DINOv3ViTModel of transformers
# (https://github.com/huggingface/transformers).
# The Meta -> transformers parameter mapping follows
# src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py.
# Copyright 2025 The HuggingFace Team. Licensed under the Apache License 2.0.
#
# No code of the Meta DINOv3 repository is used. The pretrained DINOv3 weights
# are not part of this repository and are subject to the DINOv3 License.

"""DINOv3 ViT backbone of the uniform detector, and the conversion of Meta
DINOv3 state dicts to ``DINOv3ViTModel`` parameter names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from transformers import DINOv3ViTConfig, DINOv3ViTModel

# (hidden size, depth, heads) of the LVD1689M ViTs.
_VIT_SIZES = {
    "vits16": (384, 12, 6),
    "vitb16": (768, 12, 12),
    "vitl16": (1024, 24, 16),
}


def _lvd1689m_vit_config(
    variant: str, patch_size: int, img_size: int
) -> DINOv3ViTConfig:
    """``DINOv3ViTConfig`` of the LVD1689M ViT checkpoints. Their q, k and v
    biases are masked to zero (``qkv.bias_mask``), so the projections have none.
    """
    hidden, depth, heads = _VIT_SIZES[variant]
    return DINOv3ViTConfig(
        patch_size=patch_size,
        image_size=img_size,
        hidden_size=hidden,
        intermediate_size=4 * hidden,
        num_hidden_layers=depth,
        num_attention_heads=heads,
        num_register_tokens=4,
        layerscale_value=1e-5,
        rope_theta=100.0,
        pos_embed_shift=None,
        pos_embed_jitter=None,
        # Meta's pretraining augmentation: in .train() (also with all blocks frozen)
        # RoPE coords are scaled by a log-uniform factor in [0.5, 2]; not in eval.
        pos_embed_rescale=2.0,
        query_bias=False,
        key_bias=False,
        value_bias=False,
        proj_bias=True,
        mlp_bias=True,
        use_gated_mlp=False,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        drop_path_rate=0.0,
        # AdaptiveDinoV3Backbone's boolean padding mask needs SDPA.
        attn_implementation="sdpa",
    )


# Meta DINOv3 parameter names -> ``DINOv3ViTModel`` parameter names.
_META_TOP_KEYS = {
    "cls_token": "embeddings.cls_token",
    "mask_token": "embeddings.mask_token",
    "storage_tokens": "embeddings.register_tokens",
    "patch_embed.proj.weight": "embeddings.patch_embeddings.weight",
    "patch_embed.proj.bias": "embeddings.patch_embeddings.bias",
    "norm.weight": "norm.weight",
    "norm.bias": "norm.bias",
}
_META_BLOCK_KEYS = {
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "attn.proj.weight": "attention.o_proj.weight",
    "attn.proj.bias": "attention.o_proj.bias",
    "ls1.gamma": "layer_scale1.lambda1",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
    "mlp.fc1.weight": "mlp.up_proj.weight",
    "mlp.fc1.bias": "mlp.up_proj.bias",
    "mlp.fc2.weight": "mlp.down_proj.weight",
    "mlp.fc2.bias": "mlp.down_proj.bias",
    "ls2.gamma": "layer_scale2.lambda1",
}


def _convert_meta_block(
    sd: dict[str, torch.Tensor], src: str, dst: str
) -> dict[str, torch.Tensor]:
    """One block's Meta parameters under ``src`` as ``DINOv3ViTLayer``
    parameters under ``dst``; pops them from ``sd``.
    """
    out = {
        dst + v: sd.pop(src + k) for k, v in _META_BLOCK_KEYS.items() if src + k in sd
    }
    qkv = sd.pop(src + "attn.qkv.weight", None)
    if qkv is not None:
        for name, w in zip(
            ("q_proj", "k_proj", "v_proj"), qkv.chunk(3, dim=0), strict=True
        ):
            out[f"{dst}attention.{name}.weight"] = w
    bias = sd.pop(src + "attn.qkv.bias", None)
    mask = sd.pop(src + "attn.qkv.bias_mask", None)
    if bias is not None:
        effective = bias if mask is None else bias * mask
        if effective.abs().max() > 0:
            raise ValueError(f"{src}attn.qkv: nonzero q/k/v bias is not supported")
    return out


def convert_meta_state_dict(
    sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename the Meta DINOv3 ViT parameters to ``DINOv3ViTModel`` names
    (``rope_embed.periods`` is dropped: transformers computes the periods).
    Other keys pass through.
    """
    sd = dict(sd)
    out: dict[str, torch.Tensor] = {}
    for k, v in _META_TOP_KEYS.items():
        if k in sd:
            t = sd.pop(k)
            out[v] = t.view(1, 1, -1) if k == "mask_token" else t
    sd.pop("rope_embed.periods", None)
    blocks = sorted({int(k.split(".")[1]) for k in sd if k.startswith("blocks.")})
    for i in blocks:
        out.update(_convert_meta_block(sd, f"blocks.{i}.", f"model.layer.{i}."))
    out.update(sd)
    return out


@dataclass(frozen=True)
class DinoV3Config:
    """Configuration of :class:`DinoV3Backbone`."""

    variant: str = "vitb16"  # vits16 | vitb16 | vitl16
    patch_size: int = 16
    img_size: int = 448
    weights_path: str = ""  # local checkpoint path
    intermediate_layers: Sequence[int] = (2, 5, 8, 11)


class DinoV3Backbone(nn.Module):
    """DINOv3 ViT backbone of the uniform detector. ``get_intermediate_layers``
    returns the normalized ``(B, D, Hp, Wp)`` feature maps of the given blocks.
    Meta DINOv3 pretrained weights load through :func:`convert_meta_state_dict`.
    """

    def __init__(self, cfg: DinoV3Config) -> None:
        super().__init__()
        self.cfg = cfg

        variant = {"vits": "vits16", "vitb": "vitb16", "vitl": "vitl16"}.get(
            cfg.variant.lower(), cfg.variant.lower()
        )
        if variant not in _VIT_SIZES:
            raise ValueError(f"Unsupported DINOv3 variant: {cfg.variant}")
        self.model = DINOv3ViTModel(
            _lvd1689m_vit_config(variant, cfg.patch_size, cfg.img_size)
        )
        self.embed_dim = self.model.config.hidden_size
        self.num_register_tokens = self.model.config.num_register_tokens

        # Pretrained weights: Meta's release, a flat state dict with Meta's names.
        if cfg.weights_path:
            state = torch.load(cfg.weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(convert_meta_state_dict(state))

        # Read by ``Detector.freeze_backbone`` and the layer-wise LR decay.
        self.patch_embed = self.model.embeddings.patch_embeddings
        self.blocks = self.model.model.layer
        self.depth = len(self.blocks)

    def prefix_tokens(self, B: int) -> torch.Tensor:
        """``(B, 1 + num_register_tokens, D)`` cls and register tokens."""
        emb = self.model.embeddings
        return torch.cat(
            [emb.cls_token.expand(B, -1, -1), emb.register_tokens.expand(B, -1, -1)],
            dim=1,
        )

    def get_intermediate_layers(
        self, img: torch.Tensor, n: list[int]
    ) -> list[torch.Tensor]:
        """Normalized ``(B, D, Hp, Wp)`` patch-token feature maps of the blocks
        ``n`` (cls and register tokens dropped).
        """
        if any(not 0 <= i < self.depth for i in n):
            raise ValueError(
                f"block indices {list(n)} out of range for depth {self.depth}"
            )
        B, _, H, W = img.shape
        p = self.cfg.patch_size
        # hidden_states[0] is the embedding output, hidden_states[i + 1] the output
        # of block i.
        hidden = self.model(pixel_values=img, output_hidden_states=True).hidden_states
        start = self.num_register_tokens + 1
        return [
            self.model.norm(hidden[i + 1])[:, start:]
            .reshape(B, H // p, W // p, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for i in n
        ]
