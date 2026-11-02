"""DINOv3 over a mixed-resolution ``Patchification``.

The same pretrained ``DINOv3ViTModel`` as ``DinoV3Backbone``, but the
tokens come from ``MixedPatchEmbed`` and RoPE from the token centers
(``rope_at_coords``) instead of the regular grid. With a uniform
``Patchification`` the output equals ``DinoV3Backbone``'s.
The images of a batch may have different token counts: each is padded to the
longest, padded tokens are masked out as keys and get the identity rotation,
and the padding is dropped from the output. The boolean key mask requires
SDPA attention.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from patch_where_objects_are.adaptive_patch.mixed_patch_embed import MixedPatchEmbed
from patch_where_objects_are.adaptive_patch.patchification import Patchification
from patch_where_objects_are.models.backbones.dinov3 import DinoV3Backbone, DinoV3Config
from patch_where_objects_are.models.backbones.dinov3_ops import rope_at_coords


class AdaptiveDinoV3Backbone(DinoV3Backbone):
    """DINOv3 backbone over mixed-resolution patches.

    Args:
        cfg:             ``DinoV3Config``.
        adaptive_scales: Allowed patch-size multipliers (must include 1).
    """

    def __init__(
        self, cfg: DinoV3Config, adaptive_scales: Sequence[int] = (1, 2, 4)
    ) -> None:
        super().__init__(cfg)
        self.mixed_patch_embed = MixedPatchEmbed(
            self.patch_embed, scales=adaptive_scales
        )

    def uniform_patchification(self, H: int, W: int) -> Patchification:
        """Uniform patches for an ``H x W`` image."""
        p = self.cfg.patch_size
        return Patchification.uniform(H // p, W // p, p)

    def forward_tokens(
        self,
        img: Tensor,
        patchifications: Patchification | Sequence[Patchification] | None = None,
        n: list[int] | None = None,
    ) -> list[Tensor]:
        """ViT forward over the tokens of a mixed-resolution ``Patchification``.

        Args:
            img: ``(B, 3, H, W)`` normalized images.
            patchifications: One ``Patchification`` for all images or one per
                image (``None``: uniform patches). Token counts may differ.
            n: Indices of the blocks whose normalized outputs are returned
               (``None``: the last block).

        Returns:
            Per block index, without cls and register tokens and in each
            patchification's token order: one ``(B, N, D)`` tensor if all
            token counts are equal, else a list of per-image ``(N_b, D)``
            tensors.
        """
        B, _, H, W = img.shape
        p = self.cfg.patch_size
        Hp, Wp = H // p, W // p
        if patchifications is None:
            patchifications = self.uniform_patchification(H, W)
        if isinstance(patchifications, Patchification):
            pfs = [patchifications] * B
        else:
            pfs = list(patchifications)

        token_list = self.mixed_patch_embed(
            img, pfs
        )  # patchify each image and get list of tokens
        counts = [t.shape[0] for t in token_list]
        n_max = max(counts)
        ragged = len(set(counts)) != 1  # token counts differ

        model = self.model
        patch_start = self.num_register_tokens + 1

        if ragged:  # padding the tokens to the maximum token count
            # Only SDPA reads the boolean mask as "may attend"; eager attention
            # adds it to the scores and would mask nothing.
            attn_impl = model.config._attn_implementation
            if attn_impl != "sdpa":
                raise RuntimeError(
                    f"padded token batches need attn_implementation='sdpa', "
                    f"got {attn_impl!r}"
                )
            x: list[Tensor] = token_list[0].new_zeros(B, n_max, token_list[0].shape[1])
            coords: list[Patchification] = token_list[0].new_zeros(B, n_max, 2)
            valid = torch.zeros(B, n_max, dtype=torch.bool, device=img.device)
            for b, (t, pf) in enumerate(zip(token_list, pfs, strict=True)):
                x[b, : counts[b]] = t
                coords[b, : counts[b]] = pf.rope_coords().to(img.device)
                valid[b, : counts[b]] = True
        else:
            x = torch.stack(token_list)  # (B, N, D)
            coords = torch.stack([pf.rope_coords() for pf in pfs]).to(img.device)
            valid = None

        x = torch.cat([self.prefix_tokens(B), x], dim=1)

        # RoPE on the token centers.
        cos, sin = rope_at_coords(model.rope_embeddings, coords, HW=(Hp, Wp))
        if valid is not None:
            # Padded tokens get the identity rotation (sin 0, cos 1).
            v = valid.unsqueeze(-1)
            sin = sin * v
            cos = torch.where(v, cos, torch.ones_like(cos))
            # Keys that may be attended; cls and register tokens always.
            key_valid = torch.cat([valid.new_ones(B, patch_start), valid], dim=1)
            attn_mask = key_valid[:, None, None, :]  # (B, 1, 1, P+N_max)
        else:
            attn_mask = None
        rope = (cos.unsqueeze(1), sin.unsqueeze(1))  # (B, 1, N, D_head)

        n = list(n) if n is not None else [self.depth - 1]
        if any(not 0 <= i < self.depth for i in n):
            raise ValueError(f"block indices {n} out of range for depth {self.depth}")
        # hidden_states[i + 1] is the output of block i.
        hidden = model.model(
            x,
            position_embeddings=rope,
            attention_mask=attn_mask,
            output_hidden_states=True,
        ).hidden_states
        outputs = [model.norm(hidden[i + 1]) for i in n]
        if ragged:
            return [
                # Drop the cls, register and padded tokens.
                [out[b, patch_start : patch_start + counts[b]] for b in range(B)]
                for out in outputs
            ]
        return [out[:, patch_start:] for out in outputs]
