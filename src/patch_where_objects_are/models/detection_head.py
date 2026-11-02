# Adapted from RT-DETR (https://github.com/lyuwenyu/RT-DETR),
# rtdetr_pytorch/src/zoo/rtdetr/: rtdetr_decoder.py (decoder, query selection,
# anchors), hybrid_encoder.py (AIFI encoder layer,
# 2D sin-cos position embedding), denoising.py (contrastive
# denoising group) and utils.py (bias_init_with_prob, get_activation, MLP).
# Copyright (c) 2023 lyuwenyu.
# Licensed under the Apache License 2.0.
#
# sigmoid_focal_loss is adapted from DETR (https://github.com/facebookresearch/detr),
# models/segmentation.py. Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License 2.0.
#
# Modifications: single-scale encoder and head (CCFF / cross-scale fusion and
# the extra input levels removed), AIFI and decoder merged into RTDETRHead,
# sparse path added (encode_sparse, forward_sparse, _get_sparse_decoder_input,
# sparse_scale_embed, sin-cos embedding at explicit coordinates, key padding
# mask in AIFI), invalid anchors excluded from the top-k query selection,
# rewritten denoising attention mask, code restructured and documented.

"""Single-scale RT-DETR detection head (Zhao et al. 2024).

The AIFI encoder runs on the single stride-16 feature map (RT-DETR's CCFF
module is removed), and the deformable-attention decoder refines
``num_queries`` proposals. The dense path (``forward``) reads the feature map
on the base grid; the sparse path (``forward_sparse``) runs AIFI on the
mixed-resolution tokens and lets the proposals and the decoder read them
through a base-cell -> token index.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence

from patch_where_objects_are.adaptive_patch.grid_broadcast import (
    base_grid_token_indices,
)
from patch_where_objects_are.models.deformable_attention import MSDeformableAttention
from patch_where_objects_are.utils.misc import inverse_sigmoid

if TYPE_CHECKING:
    from patch_where_objects_are.adaptive_patch.patchification import Patchification


# ── Helpers ───────────────────────────────────────────────────────────────────


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """Return the logit bias whose sigmoid equals ``prior_prob``."""
    return float(-math.log((1 - prior_prob) / prior_prob))


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss for dense detection (Lin et al. 2017).

    Args:
        inputs:   Raw logits of shape ``(B, N, C)``.
        targets:  Binary targets of shape ``(B, N, C)``.
        num_boxes: Normalization factor (number of matched ground-truth boxes).
        alpha:    Focal loss alpha.
        gamma:    Focal loss gamma.

    Returns:
        Scalar loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce_loss
    return loss.mean(1).sum() / num_boxes


def get_activation(name: str) -> nn.Module:
    """Return the activation module for ``name`` (case-insensitive)."""
    name = name.lower()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """Simple multi-layer perceptron (feed-forward network)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act: str = "relu",
    ) -> None:
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim, *h], [*h, output_dim], strict=True)
        )
        self.act = get_activation(act)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(..., input_dim)`` to ``(..., output_dim)``."""
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ── AIFI (Attention in Feature Interaction) encoder ─────────────────────────


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer of AIFI."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def _with_pos(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: torch.Tensor,
        pos_embed: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Post-norm self-attention and FFN over ``(B, L, D)`` tokens.

        ``pos_embed`` is added to the queries and keys only;
        ``key_padding_mask`` is ``(B, L)``, True at padding positions.
        """
        q = k = self._with_pos(src, pos_embed)
        src2, _ = self.self_attn(q, k, value=src, key_padding_mask=key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class TransformerEncoder(nn.Module):
    """AIFI encoder: ``num_layers`` copies of a ``TransformerEncoderLayer``."""

    def __init__(self, layer: TransformerEncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(
        self,
        src: torch.Tensor,
        pos_embed: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run ``(B, L, D)`` tokens through all layers; returns the same shape."""
        for layer in self.layers:
            src = layer(src, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        return src

    @staticmethod
    def build_2d_sincos_pos_embed(
        w: int,
        h: int,
        embed_dim: int = 256,
        temperature: float = 10000.0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """``(1, h * w, embed_dim)`` sine-cosine embedding of the grid positions
        in row-major order; the first half encodes the row index.
        """
        device = device or torch.device("cpu")
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[
            None
        ]

    @staticmethod
    def build_2d_sincos_pos_embed_from_coords(
        coords: torch.Tensor,
        embed_dim: int = 256,
        temperature: float = 10000.0,
    ) -> torch.Tensor:
        """Encode ``(..., 2)`` coordinates in ``(row, col)`` order.

        Row half first, as :meth:`build_2d_sincos_pos_embed` does for the
        row-major token order of a square grid, so a uniform partition gets
        exactly the dense AIFI embedding.
        """
        if embed_dim % 4:
            raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
        pos_dim = embed_dim // 4
        omega = (
            torch.arange(pos_dim, dtype=torch.float32, device=coords.device) / pos_dim
        )
        omega = 1.0 / (temperature**omega)
        y, x = coords.to(torch.float32).unbind(dim=-1)
        out_x = x[..., None] * omega
        out_y = y[..., None] * omega
        return torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=-1)


# ── RT-DETR decoder ───────────────────────────────────────────────────────────


class TransformerDecoderLayer(nn.Module):
    """RT-DETR decoder layer: self-attention, deformable cross-attention, FFN."""

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 4,
        n_points: int = 4,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def _with_pos(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: list,
        memory_level_start_index: list,
        attn_mask: torch.Tensor | None = None,
        query_pos_embed: torch.Tensor | None = None,
        cell_to_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update the queries ``tgt`` ``(B, nq, D)``; returns the same shape.

        ``reference_points`` are the normalized cxcywh boxes ``(B, nq, 1, 4)``
        around which the cross-attention samples ``memory``.
        """
        # Self-attention.
        q = k = self._with_pos(tgt, query_pos_embed)
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # Deformable cross-attention.
        tgt2 = self.cross_attn(
            self._with_pos(tgt, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            cell_to_token=cell_to_token,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # FFN.
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        return self.norm3((tgt + self.dropout4(tgt2)).clamp(min=-65504, max=65504))


class TransformerDecoder(nn.Module):
    """RT-DETR decoder with iterative box refinement across its layers."""

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        eval_idx: int = -1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        tgt: torch.Tensor,
        ref_points_unact: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: list,
        memory_level_start_index: list,
        bbox_head: nn.ModuleList,
        score_head: nn.ModuleList,
        query_pos_head: nn.Module,
        attn_mask: torch.Tensor | None = None,
        cell_to_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the queries and refine the reference boxes layer by layer.

        Returns:
            Stacked boxes ``(L, B, nq, 4)`` (normalized cxcywh) and logits
            ``(L, B, nq, C)``: one entry per layer in training, only the
            ``eval_idx`` layer (``L = 1``) at evaluation.
        """
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        ref_points = None

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=attn_mask,
                query_pos_embed=query_pos_embed,
                cell_to_token=cell_to_token,
            )

            inter_ref_bbox = F.sigmoid(
                bbox_head[i](output) + inverse_sigmoid(ref_points_detach)
            )

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points))
                    )
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = (
                inter_ref_bbox.detach() if self.training else inter_ref_bbox
            )

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


# ── RTDETRHead ────────────────────────────────────────────────────────────────


def get_contrastive_denoising_training_group(  # noqa: PLR0915
    targets: list[dict],
    num_classes: int,
    num_queries: int,
    class_embed: nn.Embedding,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
) -> tuple:
    """Contrastive denoising group (DN-DETR style)."""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device
    max_gt = max(num_gts)
    if max_gt == 0:
        return None, None, None, None

    num_group = max(1, num_denoising // max_gt)
    bs = len(num_gts)

    input_class = torch.full(
        [bs, max_gt], num_classes, dtype=torch.int32, device=device
    )
    input_bbox = torch.zeros([bs, max_gt, 4], device=device)
    pad_mask = torch.zeros([bs, max_gt], dtype=torch.bool, device=device)

    for i in range(bs):
        n = num_gts[i]
        if n > 0:
            input_class[i, :n] = targets[i]["labels"]
            input_bbox[i, :n] = targets[i]["boxes"]
            pad_mask[i, :n] = True

    input_class = input_class.tile([1, 2 * num_group])
    input_bbox = input_bbox.tile([1, 2 * num_group, 1])
    pad_mask = pad_mask.tile([1, 2 * num_group])

    neg_mask = torch.zeros([bs, max_gt * 2, 1], device=device)
    neg_mask[:, max_gt:] = 1
    neg_mask = neg_mask.tile([1, num_group, 1])
    pos_mask = (1 - neg_mask).squeeze(-1) * pad_mask

    dn_positive_idx = torch.nonzero(pos_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])

    num_denoising_total = max_gt * 2 * num_group

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_class.float()) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(input_class, 0, num_classes)
        input_class = torch.where(mask & pad_mask, new_label, input_class)

    if box_noise_scale > 0:
        from patch_where_objects_are.utils.box_ops import (
            box_cxcywh_to_xyxy,
            box_xyxy_to_cxcywh,
        )

        known_bbox = box_cxcywh_to_xyxy(input_bbox)
        diff = input_bbox[..., 2:].repeat(1, 1, 2) * 0.5 * box_noise_scale
        rand_sign = torch.randint_like(input_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_bbox)
        rand_part = (rand_part + 1.0) * neg_mask.squeeze(-1).unsqueeze(
            -1
        ) + rand_part * (1 - neg_mask.squeeze(-1).unsqueeze(-1))
        rand_part *= rand_sign
        known_bbox = (known_bbox + rand_part * diff).clamp(0.0, 1.0)
        input_bbox = inverse_sigmoid(box_xyxy_to_cxcywh(known_bbox))

    input_class = class_embed(input_class)

    tgt_size = num_denoising_total + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool, device=device)
    attn_mask[num_denoising_total:, :num_denoising_total] = True
    for i in range(num_group):
        start = max_gt * 2 * i
        end = max_gt * 2 * (i + 1)
        if i > 0:
            attn_mask[start:end, :start] = True
        if i < num_group - 1:
            attn_mask[start:end, end:num_denoising_total] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising_total, num_queries],
    }
    return input_class, input_bbox, attn_mask, dn_meta


class RTDETRHead(nn.Module):
    """Single-scale RT-DETR head: AIFI encoder + deformable-attention decoder.

    Args:
        num_classes:        Number of classes.
        hidden_dim:         Width of the head.
        num_queries:        Number of decoder queries.
        feat_channels:      Channels of the input feature maps.
        feat_strides:       Strides of the input feature maps.
        num_levels:         Number of feature maps (``len(feat_channels)``).
        nhead:              Attention heads of encoder and decoder.
        num_encoder_layers: AIFI layers.
        use_encoder_idx:    Index of the feature map AIFI runs on.
        num_decoder_layers: Decoder layers.
        dim_feedforward:    Feed-forward width.
        dropout:            Dropout rate.
        num_denoising:      Number of denoising queries (0 = off).
        label_noise_ratio:  Label noise of the denoising queries.
        box_noise_scale:    Box noise of the denoising queries.
        learnt_init_query:  Learned query embeddings instead of proposals.
        aux_loss:           Auxiliary losses on every decoder layer.
        full_sparse:        Sparse head: AIFI runs on the mixed-resolution
                            tokens, and the proposals and the decoder's
                            cross-attention read them through a base-cell ->
                            token index instead of a feature map on the base
                            grid.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        num_classes: int = 7,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels: list[int] = (256, 256, 256, 256),
        feat_strides: list[int] = (8, 16, 32, 64),
        num_levels: int = 4,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        use_encoder_idx: list[int] = (1,),
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        aux_loss: bool = True,
        eps: float = 1e-2,
        full_sparse: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.feat_strides = list(feat_strides)
        self.num_levels = num_levels
        self.use_encoder_idx = list(use_encoder_idx)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learnt_init_query = learnt_init_query
        self.aux_loss = aux_loss
        self.eps = eps
        self.full_sparse = full_sparse

        # Input projection of each feature map to hidden_dim.
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
                for ch in feat_channels
            ]
        )

        # AIFI encoder.
        enc_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation="gelu"
        )
        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(enc_layer), num_encoder_layers)
                for _ in use_encoder_idx
            ]
        )
        if self.full_sparse:
            if num_levels != 1 or list(use_encoder_idx) != [0]:
                raise ValueError(
                    "full_sparse requires a single feature level with "
                    "use_encoder_idx=[0]"
                )
            # Scale embedding of log2(m), zero-initialized so that the tokens
            # are unchanged at init.
            self.sparse_scale_embed = nn.Linear(1, hidden_dim)
            init.zeros_(self.sparse_scale_embed.weight)
            init.zeros_(self.sparse_scale_embed.bias)

        # Denoising class embed.
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )

        # Decoder.
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation="relu",
            n_levels=num_levels,
            n_points=4,
        )
        self.decoder = TransformerDecoder(
            hidden_dim, decoder_layer, num_decoder_layers, eval_idx=-1
        )

        # Query / reference init.
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # Proposal head on the encoder output.
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # Per-layer decoder heads.
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(hidden_dim, hidden_dim, 4, num_layers=3)
                for _ in range(num_decoder_layers)
            ]
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head, strict=True):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)

    # ── Encoder input flattening ──────────────────────────────────────────────

    def _get_encoder_input(
        self, feats: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list, list, list[torch.Tensor]]:
        proj = [self.input_proj[i](f) for i, f in enumerate(feats)]

        feat_flat = []
        spatial_shapes = []
        level_start = [0]
        for f in proj:
            _, _, h, w = f.shape
            feat_flat.append(f.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
            level_start.append(h * w + level_start[-1])
        feat_flat = torch.cat(feat_flat, dim=1)
        level_start.pop()
        return feat_flat, spatial_shapes, level_start, proj

    # ── Anchor generation ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_anchors(
        self,
        spatial_shapes: list,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        grid_size: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate anchors from which reference points are sampled."""
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            gy, gx = torch.meshgrid(
                torch.arange(h, dtype=dtype, device=device),
                torch.arange(w, dtype=dtype, device=device),
                indexing="ij",
            )
            gxy = torch.stack([gx, gy], -1)
            gxy = (gxy + 0.5) / gxy.new_tensor([w, h])
            wh = torch.ones_like(gxy) * grid_size * (2.0**lvl)
            anchors.append(torch.cat([gxy, wh], -1).reshape(-1, h * w, 4))
        anchors = torch.cat(anchors, 1)
        valid = ((anchors > self.eps) & (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors).clamp(min=1e-9))
        anchors = torch.where(valid, anchors, torch.full_like(anchors, float("inf")))
        return anchors, valid

    # ── Decoder input ─────────────────────────────────────────────────────────

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes: list,
        denoising_class: torch.Tensor | None,
        denoising_bbox_unact: torch.Tensor | None,
    ) -> tuple:
        bs = memory.shape[0]
        anchors, valid_mask = self._generate_anchors(
            spatial_shapes, memory.device, memory.dtype
        )
        memory = valid_mask.to(memory.dtype) * memory

        # Every cell is a proposal candidate.
        out_mem = self.enc_output(memory)
        enc_cls = self.enc_score_head(out_mem)
        enc_box = self.enc_bbox_head(out_mem) + anchors

        # Cells with invalid anchors (at the border) are excluded from the top-k.
        scores_for_topk = enc_cls.max(-1).values.masked_fill(
            ~valid_mask.squeeze(-1), float("-inf")
        )
        _, topk = torch.topk(scores_for_topk, self.num_queries, dim=1)
        ref_unact = enc_box.gather(
            1, topk.unsqueeze(-1).expand(-1, -1, 4)
        )  # proposal boxes (logit space)
        enc_topk_boxes = F.sigmoid(ref_unact)  # proposal boxes (normalized)
        enc_topk_logits = enc_cls.gather(
            1, topk.unsqueeze(-1).expand(-1, -1, self.num_classes)
        )  # proposal logits

        if denoising_bbox_unact is not None:
            ref_unact = torch.cat([denoising_bbox_unact, ref_unact], dim=1)

        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).expand(bs, -1, -1)
        else:
            target = out_mem.gather(
                1, topk.unsqueeze(-1).expand(-1, -1, out_mem.shape[-1])
            ).detach()  # (B, num_queries, D)

        if denoising_class is not None:
            target = torch.cat([denoising_class, target], dim=1)

        return target, ref_unact.detach(), enc_topk_boxes, enc_topk_logits

    def _get_sparse_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes: list,
        cell_to_token: torch.Tensor,
        denoising_class: torch.Tensor | None,
        denoising_bbox_unact: torch.Tensor | None,
    ) -> tuple:
        """Proposals from the tokens: each base cell takes the prediction of its
        token, so a token covering ``m x m`` cells can become several proposals
        with different anchors.
        """
        bs = memory.shape[0]
        anchors, valid_mask = self._generate_anchors(
            spatial_shapes, memory.device, memory.dtype
        )

        # Per-token predictions.
        out_mem_sparse = self.enc_output(memory)
        enc_cls_sparse = self.enc_score_head(out_mem_sparse)
        enc_box_sparse = self.enc_bbox_head(out_mem_sparse)

        # Prediction of each base cell = that of its token.
        enc_cls = torch.gather(
            enc_cls_sparse,
            1,
            cell_to_token.unsqueeze(-1).expand(-1, -1, self.num_classes),
        )
        enc_box = (
            torch.gather(
                enc_box_sparse,
                1,
                cell_to_token.unsqueeze(-1).expand(-1, -1, 4),
            )
            + anchors
        )

        scores_for_topk = enc_cls.max(-1).values.masked_fill(
            ~valid_mask.squeeze(-1), float("-inf")
        )
        _, topk = torch.topk(scores_for_topk, self.num_queries, dim=1)
        ref_unact = enc_box.gather(1, topk.unsqueeze(-1).expand(-1, -1, 4))
        enc_topk_boxes = F.sigmoid(ref_unact)
        enc_topk_logits = enc_cls.gather(
            1, topk.unsqueeze(-1).expand(-1, -1, self.num_classes)
        )

        if denoising_bbox_unact is not None:
            ref_unact = torch.cat([denoising_bbox_unact, ref_unact], dim=1)

        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).expand(bs, -1, -1)
        else:
            topk_tokens = cell_to_token.gather(1, topk)
            target = torch.gather(
                out_mem_sparse,
                1,
                topk_tokens.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
            ).detach()

        if denoising_class is not None:
            target = torch.cat([denoising_class, target], dim=1)

        return (
            target,
            ref_unact.detach(),
            enc_topk_boxes,
            enc_topk_logits,
            cell_to_token,
        )

    def _decode_projected_features(
        self,
        proj_feats: list[torch.Tensor],
        spatial_shapes: list,
        level_start: list,
        targets: list[dict] | None,
    ) -> dict:
        memory = torch.cat([f.flatten(2).permute(0, 2, 1) for f in proj_feats], dim=1)
        return self._decode_memory(memory, spatial_shapes, level_start, targets)

    def _decode_memory(
        self,
        memory: torch.Tensor,
        spatial_shapes: list,
        level_start: list,
        targets: list[dict] | None,
        cell_to_token: torch.Tensor | None = None,
    ) -> dict:
        if self.training and self.num_denoising > 0 and targets is not None:
            dn_class, dn_bbox, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    self.num_denoising,
                    self.label_noise_ratio,
                    self.box_noise_scale,
                )
            )
        else:
            dn_class, dn_bbox, attn_mask, dn_meta = None, None, None, None

        if cell_to_token is None:
            # dense decoder input
            target, init_ref_unact, enc_topk_boxes, enc_topk_logits = (
                self._get_decoder_input(memory, spatial_shapes, dn_class, dn_bbox)
            )
            decoder_index = None
        else:
            # sparse decoder input
            (
                target,  # (B, num_queries, D)
                init_ref_unact,
                enc_topk_boxes,
                enc_topk_logits,
                decoder_index,
            ) = self._get_sparse_decoder_input(
                memory, spatial_shapes, cell_to_token, dn_class, dn_bbox
            )

        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_unact,
            memory,
            spatial_shapes,
            level_start,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            cell_to_token=decoder_index,
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )

        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        if self.training and self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(out_logits[:-1], out_bboxes[:-1], strict=True)
            ]
            out["aux_outputs"].append(
                {"pred_logits": enc_topk_logits, "pred_boxes": enc_topk_boxes}
            )
            if dn_meta is not None:
                out["dn_aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(dn_out_logits, dn_out_bboxes, strict=True)
                ]
                out["dn_meta"] = dn_meta

        return out

    def encode_sparse(
        self,
        tokens: torch.Tensor | Sequence[torch.Tensor],
        patchifications: Sequence[Patchification],
    ) -> torch.Tensor:
        """Project the mixed-resolution tokens, add their scale embedding and
        run AIFI on them with sine-cosine positions at the token centers.
        Token counts may differ: each image is padded to the longest, and the
        padded tokens are masked out as AIFI keys (no base cell points at them,
        so proposals and the decoder never read them) and left out of the
        input projection's BatchNorm statistics.

        Args:
            tokens: Per-image tokens ``(N, D)`` from the backbone.
            patchifications: Patchification of each image.

        Returns:
            ``(B, N_max, hidden_dim)`` encoded tokens.
        """
        if not self.full_sparse:
            raise RuntimeError("encode_sparse requires full_sparse=True")

        token_list = list(tokens)
        if len(token_list) != len(patchifications):
            raise ValueError(
                f"{len(token_list)} token sets for "
                f"{len(patchifications)} patchifications"
            )
        for token, patchification in zip(token_list, patchifications, strict=True):
            if token.shape[0] != patchification.num_tokens:
                raise ValueError(
                    f"{token.shape[0]} tokens for patchification of "
                    f"{patchification.num_tokens}"
                )

        counts = [token.shape[0] for token in token_list]
        if len(set(counts)) == 1:
            key_padding_mask = None
        else:
            n = torch.arange(max(counts), device=token_list[0].device)
            key_padding_mask = n[None] >= n.new_tensor(counts)[:, None]  # True: padding

        # Input projection on the concatenated real tokens, so that the
        # BatchNorm statistics do not count the padding.
        src = torch.cat(token_list)  # (sum N, D)
        src = self.input_proj[0](src.T[None, :, :, None])[0, :, :, 0].T
        src = pad_sequence(src.split(counts), batch_first=True)

        # Scale embedding.
        # Padding gets scale 1 and center 0, so its values stay finite.
        scales = pad_sequence(
            [patchification.scales for patchification in patchifications],
            batch_first=True,
            padding_value=1,
        ).to(device=src.device, dtype=src.dtype)
        src += self.sparse_scale_embed(torch.log2(scales).unsqueeze(-1))

        # The dense AIFI uses integer cell coordinates, so the token centers
        # are offset by 0.5.
        coords = pad_sequence(
            [patchification.rope_coords() for patchification in patchifications],
            batch_first=True,
        ).to(src.device)
        pos = TransformerEncoder.build_2d_sincos_pos_embed_from_coords(
            coords - 0.5, self.hidden_dim
        ).to(src.dtype)
        return self.encoder[0](src, pos_embed=pos, key_padding_mask=key_padding_mask)

    def forward_sparse(
        self,
        tokens: torch.Tensor | Sequence[torch.Tensor],
        patchifications: Sequence[Patchification],
        targets: list[dict] | None = None,
    ) -> dict:
        """Sparse head on the mixed-resolution tokens: AIFI, proposals and the
        decoder's cross-attention all read the tokens; no feature map on the
        base grid is built.

        Args:
            tokens: Per-image tokens ``(N, D)`` from the backbone.
            patchifications: Patchification of each image.
            targets: Per-image target dicts (training only).

        Returns:
            Dict with ``pred_logits``, ``pred_boxes`` and optionally
            ``aux_outputs``, ``dn_aux_outputs``.
        """
        enhanced = self.encode_sparse(tokens, patchifications)
        grids = {patchification.grid_hw for patchification in patchifications}
        if len(grids) != 1:
            raise ValueError(f"mixed base-grid sizes in batch: {sorted(grids)}")
        h, w = next(iter(grids))
        cell_to_token = torch.stack(
            [
                base_grid_token_indices(patchification, enhanced.device)
                for patchification in patchifications
            ]
        )  # token of each base cell
        return self._decode_memory(enhanced, [[h, w]], [0], targets, cell_to_token)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        feats: list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> dict:
        """Dense head on feature maps.

        Args:
            feats:   Feature maps ``(B, C, H_l, W_l)``, one per level.
            targets: Per-image target dicts (training only).

        Returns:
            Dict with ``pred_logits``, ``pred_boxes`` and optionally
            ``aux_outputs``, ``dn_aux_outputs``.
        """
        # Project and flatten the feature maps.
        _, spatial_shapes, level_start, proj_feats = self._get_encoder_input(feats)

        # AIFI on the selected feature map.
        for i, enc_idx in enumerate(self.use_encoder_idx):
            h, w = spatial_shapes[enc_idx]
            src = proj_feats[enc_idx].flatten(2).permute(0, 2, 1)
            pos = TransformerEncoder.build_2d_sincos_pos_embed(
                w, h, self.hidden_dim, device=src.device
            ).to(src.dtype)
            enhanced = self.encoder[i](src, pos_embed=pos)
            proj_feats[enc_idx] = enhanced.permute(0, 2, 1).reshape(
                -1, self.hidden_dim, h, w
            )

        return self._decode_projected_features(
            proj_feats, spatial_shapes, level_start, targets
        )
