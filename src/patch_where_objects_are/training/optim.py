"""Optimizer and learning-rate schedule utilities for detection fine-tuning."""

from __future__ import annotations

import torch
from torch import nn


def extend_detection_optimizer(
    optimizer: torch.optim.AdamW,
    model: nn.Module,
    layer_lr_decay: float,
    base_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Add newly trainable parameters without resetting Adam state for existing ones.

    At backbone unfreeze, head/FPN/already-unfrozen blocks keep momentum and
    variance; only newly unfrozen parameters are registered with fresh state.
    """
    existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
    get_param_groups = getattr(model, "get_param_groups", None)
    if not callable(get_param_groups):
        raise TypeError(f"{type(model).__name__} must implement get_param_groups")

    for group in get_param_groups(layer_lr_decay, base_lr, weight_decay):
        params = [p for p in group.get("params", []) if id(p) not in existing]
        if not params:
            continue
        existing.update(id(p) for p in params)
        new_group = dict(group)
        new_group["params"] = params
        optimizer.add_param_group(new_group)
    return optimizer


def apply_detection_lr_schedule(
    optimizer: torch.optim.Optimizer, scheduled_base_lr: float
) -> None:
    """Set each group's LR to the scheduled base LR times its ``lr_scale``
    (layer-wise LR decay, see ``Detector.get_param_groups``).
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = scheduled_base_lr * param_group["lr_scale"]
