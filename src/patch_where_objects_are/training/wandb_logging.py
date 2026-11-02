"""Compact W&B helpers for detection training and evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from torch import nn

from patch_where_objects_are.models.detector_base import (
    DETECTION_MODULE_GROUPS,
    detection_parameter_group,
)

if TYPE_CHECKING:
    import torch


def compute_grad_norms(
    model: nn.Module,
    groups: Iterable[str] = DETECTION_MODULE_GROUPS,
) -> dict[str, float]:
    """Return the L2 gradient norm for each coarse module group."""
    norms = dict.fromkeys(groups, 0.0)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        group = detection_parameter_group(name)
        if group is None or group not in norms:
            continue
        norms[group] += param.grad.detach().data.pow(2).sum().item()
    return {group: value**0.5 for group, value in norms.items()}


def get_lr_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Summarize learning rates across optimizer param groups."""
    lrs = [float(param_group["lr"]) for param_group in optimizer.param_groups]
    if not lrs:
        return {}
    return {
        "train/lr": max(lrs),
        "train/lr_min": min(lrs),
        "train/lr_max": max(lrs),
    }


def log_training_metrics(
    run: Any,
    *,
    step: int,
    losses: dict[str, float],
    optimizer: torch.optim.Optimizer,
    grad_norms: dict[str, float],
) -> None:
    """Log compact training scalars to W&B."""
    payload = {**losses, **get_lr_metrics(optimizer)}
    for group, value in grad_norms.items():
        payload[f"grad_norm/{group}"] = value
    run.log(payload, step=step)
