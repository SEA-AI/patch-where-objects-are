"""Per-cell targets from the ground-truth boxes on the base grid.

- ``centernet_heatmap``: the ground-truth objectness map ``o*``, supervision
  of the objectness scorer,
- ``object_cell_ranges``: the cells each box covers (recall@k of the scorer).
"""

from __future__ import annotations

import torch


def object_cell_ranges(
    boxes_xyxy: torch.Tensor,
    patch_size: int,
    grid_hw: tuple[int, int],
) -> torch.Tensor:
    """Inclusive base-cell rectangle ``(gy0, gy1, gx0, gx1)`` covered by each box.

    Args:
        boxes_xyxy: ``(K, 4)`` boxes ``(x0, y0, x1, y1)`` in pixels of the
            input resolution.
        patch_size: Finest patch size ``p``.
        grid_hw: ``(gh, gw)`` base grid size, for clamping.

    Returns:
        ``(K, 4)`` long tensor; a box overlaps cell ``g`` iff ``g`` lies in
        its rectangle.
    """
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4:
        raise ValueError(f"Expected (K, 4) boxes, got {tuple(boxes_xyxy.shape)}")
    gh, gw = grid_hw
    x0, y0, x1, y1 = boxes_xyxy.unbind(dim=1)
    gx0 = torch.div(x0, patch_size, rounding_mode="floor")
    gy0 = torch.div(y0, patch_size, rounding_mode="floor")
    gx1 = torch.ceil(x1 / patch_size) - 1
    gy1 = torch.ceil(y1 / patch_size) - 1
    ranges = torch.stack([gy0, gy1, gx0, gx1], dim=1).long()  # box coords in grid cells
    ranges[:, 0].clamp_(0, gh - 1)
    ranges[:, 1].clamp_(0, gh - 1)
    ranges[:, 2].clamp_(0, gw - 1)
    ranges[:, 3].clamp_(0, gw - 1)
    return ranges


def centernet_heatmap(
    boxes_xyxy: torch.Tensor,
    patch_size: int,
    grid_hw: tuple[int, int],
    min_sigma: float = 0.5,
    sigma_divisor: float = 6.0,
) -> torch.Tensor:
    """Ground-truth objectness map ``o*``.

    Per box: an anisotropic Gaussian at the box center with
    ``sigma_x = max(w / (sigma_divisor * p), min_sigma)`` and ``sigma_y``
    likewise, in cell units. Cells take the max over boxes.

    Args:
        boxes_xyxy: ``(K, 4)`` boxes in pixels.
        patch_size: Finest patch size ``p``.
        grid_hw: ``(gh, gw)`` base grid size.
        min_sigma: Lower bound in cells; keeps small objects from collapsing
            into a single cell.
        sigma_divisor: 6 places the box boundary at three standard deviations.

    Returns:
        ``(gh, gw)`` float32 map in [0, 1].
    """
    gh, gw = grid_hw
    heat = torch.zeros(gh, gw, dtype=torch.float32)
    if boxes_xyxy.shape[0] == 0:
        return heat
    ys = torch.arange(gh, dtype=torch.float32).unsqueeze(1) + 0.5
    xs = torch.arange(gw, dtype=torch.float32).unsqueeze(0) + 0.5
    for k in range(boxes_xyxy.shape[0]):
        x0, y0, x1, y1 = boxes_xyxy[k].tolist()
        cx = (x0 + x1) / 2.0 / patch_size
        cy = (y0 + y1) / 2.0 / patch_size
        sigma_x = max((x1 - x0) / (sigma_divisor * patch_size), min_sigma)
        sigma_y = max((y1 - y0) / (sigma_divisor * patch_size), min_sigma)
        g = torch.exp(
            -((xs - cx) ** 2 / (2.0 * sigma_x**2) + (ys - cy) ** 2 / (2.0 * sigma_y**2))
        )
        heat = torch.maximum(heat, g)
    return heat
