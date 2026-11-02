"""Quadtree patch splitting and the selectors built on it.

A selector maps an image batch to one ``Patchification`` per image. The
training-free criteria (entropy, random) score the finest-level cells and
split under a token budget.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import torch
from torch import Tensor

from patch_where_objects_are.adaptive_patch.patchification import Patchification

_TokenList = list[tuple[int, int, int]]  # (m, gy, gx)


def _tokens_to_patchification(
    tokens: _TokenList, grid_hw: tuple[int, int], patch_size: int
) -> Patchification:
    """``Patchification`` from ``(m, gy, gx)`` tokens."""
    scales = torch.tensor([t[0] for t in tokens], dtype=torch.long)
    indices = torch.tensor([[t[1], t[2]] for t in tokens], dtype=torch.long)
    return Patchification(
        patch_size=patch_size, grid_hw=grid_hw, scales=scales, base_indices=indices
    )


def score_quadtree(  # noqa: C901
    scores: Tensor,
    patch_size: int,
    token_budget: int,
    max_scale: int = 4,
    scale_scores: dict[int, Tensor] | None = None,
    scale_thresholds: dict[int, float] | None = None,
    pool: str = "max",
) -> Patchification:
    """Budgeted quadtree patch splitting of one image.

    All nodes start at the coarsest level ``max_scale``. The node with the
    highest score is split into its four children until the next split would
    exceed ``token_budget``. A node's score is the ``pool`` (max or
    mean) of its finest-level cell scores, or its own entry in
    ``scale_scores`` under per-scale scoring.

    Args:
        scores: ``(Hp, Wp)`` finest-level score map.
        patch_size: Finest patch size ``p``.
        token_budget: Upper bound on the number of tokens.
        max_scale: Coarsest patch-size multiplier ``M``.
        scale_scores: ``{m: (Hp/m, Wp/m)}`` per-scale score maps; a level
            listed here is scored from its own map instead of by pooling.
        scale_thresholds: ``{m: tau_m}``; a level-``m`` node scoring below
            ``tau_m`` is not split (APT's rule). The token count then varies
            per image, with ``token_budget`` as the maximum.
        pool: ``"max"`` or ``"mean"``.

    Returns:
        ``Patchification`` with at most ``token_budget`` tokens, unless the
        coarsest-level grid alone has more.
    """
    if pool not in {"max", "mean"}:
        raise ValueError(f"pool must be 'max' or 'mean', got {pool!r}")
    h, w = scores.shape
    s = scores.detach().to(torch.float64).cpu().numpy()
    tokens: _TokenList = []
    heap: list[tuple[float, int, int, int]] = []  # (-node score, gy, gx, m)

    grids: dict[int, np.ndarray] = {}
    if scale_scores:
        grids = {
            int(m): g.detach().to(torch.float64).cpu().numpy()
            for m, g in scale_scores.items()
        }

    def node_score(gy: int, gx: int, m: int) -> float:
        g = grids.get(m)
        if g is None:
            block = s[
                gy : gy + m, gx : gx + m
            ]  # take finest-scores and extract patch region
            if pool == "mean":
                return float(block.mean())
            return float(block.max())
        # multi-scale scoring
        return float(g[gy // m, gx // m])  # per-scale scoring

    def push(gy: int, gx: int, m: int) -> None:
        sc = node_score(gy, gx, m)  # get node score according to the scoring method
        if scale_thresholds is not None and sc < scale_thresholds.get(m, float("-inf")):
            tokens.append((m, gy, gx))
            return
        heapq.heappush(heap, (-sc, gy, gx, m))  # min-heap on the negated score

    def seed(gy: int, gx: int, m: int) -> int:
        if gy >= h or gx >= w:
            return 0  # outside the image
        if m == 1:
            tokens.append((1, gy, gx))  # finest-level cell: a leaf
            return 1
        if gy + m > h or gx + m > w:
            c = m // 2
            return sum(seed(gy + dy, gx + dx, c) for dy in (0, c) for dx in (0, c))
        push(gy, gx, m)  # potential split node, push to heap
        return 1

    # Start from the coarsest-level grid.
    n_tokens = 0
    for gy in range(0, h, max_scale):
        for gx in range(0, w, max_scale):
            n_tokens += seed(gy, gx, max_scale)

    while heap and n_tokens + 3 <= token_budget:
        _, gy, gx, m = heapq.heappop(heap)
        n_tokens += 3
        c = m // 2
        for dy in (0, c):
            for dx in (0, c):
                if c == 1:
                    tokens.append((1, gy + dy, gx + dx))  # leaf
                else:
                    push(gy + dy, gx + dx, c)  # potential split node, push to heap

    # Unsplit nodes become tokens.
    for _, gy, gx, m in heap:
        tokens.append((m, gy, gx))

    return _tokens_to_patchification(tokens, (h, w), patch_size)


class CriterionMapSelector:
    """Training-free selector: ``entropy`` (APT) or ``random`` score maps.

    ``entropy`` scores the finest-level cells and propagates the scores to
    coarser nodes by ``pool`` (max or mean). With
    ``scoring="per_scale"`` it scores a node by the entropy of its whole
    ``m * p`` region and splits it only if that entropy reaches APT's
    object-detection threshold (2.0 bits at ``m = 2`` and ``m = 4``), so the
    token count varies per image up to ``token_budget``.
    ``random`` draws uniform random scores. Images arrive
    ImageNet-normalized; the normalization is inverted, since entropy is
    defined on [0, 255] grayscale.
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)
    # APT's object-detection thresholds (patch sizes 128/64/32: 0.3/2/2 bits),
    # applied to the 32 px and 64 px nodes of the 16/32/64 quadtree.
    _APT_SCALE_THRESHOLDS: ClassVar[dict[int, float]] = {2: 2.0, 4: 2.0}
    _MODES: ClassVar[set[str]] = {"entropy", "random"}

    def __init__(
        self,
        mode: str,
        patch_size: int,
        token_budget: int,
        seed: int = 0,
        max_scale: int = 4,
        pool: str = "max",
        scoring: str = "finest",
    ) -> None:
        if mode not in self._MODES:
            raise ValueError(f"Unknown criterion selector mode: {mode!r}")
        if pool not in {"max", "mean"}:
            raise ValueError(f"pool must be 'max' or 'mean', got {pool!r}")
        if scoring not in {"finest", "per_scale"}:
            raise ValueError(
                f"scoring must be 'finest' or 'per_scale', got {scoring!r}"
            )
        if mode == "random" and (pool, scoring) != ("max", "finest"):
            raise ValueError("the random selector has no pool / scoring variants")
        if scoring == "per_scale" and pool != "max":
            raise ValueError("pool='mean' applies to finest-cell scoring only")
        self.mode = mode
        self.pool = pool
        self.scoring = scoring
        self.patch_size = patch_size
        self.token_budget = int(token_budget)
        self.max_scale = max_scale
        self.generator = torch.Generator().manual_seed(seed)

    def _gray(self, img: Tensor) -> Tensor:
        from patch_where_objects_are.adaptive_patch.entropy import to_grayscale_255

        mean = img.new_tensor(self._MEAN).view(1, 3, 1, 1)
        std = img.new_tensor(self._STD).view(1, 3, 1, 1)
        return to_grayscale_255((img * std + mean).clamp(0.0, 1.0))

    def _score_maps(self, img: Tensor) -> Tensor:
        B, _, H, W = img.shape
        p = self.patch_size
        if self.mode == "random":
            return torch.rand(B, H // p, W // p, generator=self.generator)
        from patch_where_objects_are.adaptive_patch.entropy import compute_cell_entropy

        return compute_cell_entropy(self._gray(img), patch_size=p)

    def _entropy_scale_maps(self, img: Tensor) -> dict[int, Tensor]:
        """``{m: (B, Hp/m, Wp/m)}`` entropy of the whole ``m * p`` region per level."""
        from patch_where_objects_are.adaptive_patch.entropy import compute_cell_entropy

        gray = self._gray(img)
        grids: dict[int, Tensor] = {}
        m = 2
        while m <= self.max_scale:
            grids[m] = (
                compute_cell_entropy(gray, patch_size=m * self.patch_size).float().cpu()
            )
            m *= 2
        return grids

    def __call__(
        self, img: Tensor, targets: Sequence[dict] | None = None
    ) -> list[Patchification]:
        """One ``Patchification`` per image of the ``(B, 3, H, W)`` batch;
        ``targets`` is unused.
        """
        per_scale_scoring = self.scoring == "per_scale"
        with torch.no_grad():
            maps = self._score_maps(img).float().cpu()
            per_scale = self._entropy_scale_maps(img) if per_scale_scoring else None

        scale_thresholds = self._APT_SCALE_THRESHOLDS if per_scale_scoring else None
        return [
            score_quadtree(
                maps[b],
                self.patch_size,
                self.token_budget,
                self.max_scale,
                scale_scores=None
                if per_scale is None
                else {m: g[b] for m, g in per_scale.items()},
                scale_thresholds=scale_thresholds,
                pool=self.pool,
            )
            for b in range(img.shape[0])
        ]
