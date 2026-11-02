"""Recall@k of a score map against the ground-truth boxes."""

from __future__ import annotations

import numpy as np


def object_min_ranks(
    scores: np.ndarray,
    obj_img_idx: np.ndarray,
    obj_ranges: np.ndarray,
) -> np.ndarray:
    """Lowest per-image score rank inside each object's cell rectangle.

    Rank 0 is the image's highest-scoring cell. An object is recalled with
    ``K`` selected cells per image iff its min rank is ``< K``.

    Args:
        scores: ``(N, gh, gw)`` per-image score maps.
        obj_img_idx: ``(M,)`` image index of each object.
        obj_ranges: ``(M, 4)`` inclusive cell rectangles ``(gy0, gy1, gx0, gx1)``
            (see ``adaptive_patch.labels.object_cell_ranges``).

    Returns:
        ``(M,)`` int64 min ranks.
    """
    n, gh, gw = scores.shape
    flat = scores.reshape(n, gh * gw).astype(np.float64)
    order = np.argsort(-flat, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(gh * gw)[None, :], axis=1)
    ranks = ranks.reshape(n, gh, gw)
    out = np.empty(obj_img_idx.size, dtype=np.int64)
    for m in range(obj_img_idx.size):
        gy0, gy1, gx0, gx1 = obj_ranges[m]
        out[m] = ranks[obj_img_idx[m], gy0 : gy1 + 1, gx0 : gx1 + 1].min()
    return out


def recall_at_budget(
    min_ranks: np.ndarray,
    num_cells: int,
    fine_fractions: np.ndarray,
) -> np.ndarray:
    """Recall@k: the fraction of objects with at least one selected cell when
    the top ``fine_fractions`` of each image's cells are selected.

    Args:
        min_ranks: ``(M,)`` output of :func:`object_min_ranks`.
        num_cells: Cells per image (``gh * gw``).
        fine_fractions: ``(F,)`` selected fractions ``k``.

    Returns:
        ``(F,)`` recall per fraction.
    """
    ks = np.ceil(np.asarray(fine_fractions) * num_cells).astype(np.int64)
    return np.array([(min_ranks < k).mean() for k in ks])
