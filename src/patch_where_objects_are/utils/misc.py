# inverse_sigmoid is taken from Deformable DETR
# (https://github.com/fundamentalvision/Deformable-DETR),
# util/misc.py. Copyright (c) 2020 SenseTime.
# Licensed under the Apache License 2.0.
#
# SmoothedValue, MetricLogger and the distributed helpers are adapted from
# DETR (https://github.com/facebookresearch/detr), util/misc.py.
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License 2.0.
#
# cosine_scheduler is adapted from DINO (https://github.com/facebookresearch/dino),
# utils.py. Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License 2.0.
#
# Modifications: SmoothedValue and MetricLogger reduced and rewritten,
# all_gather restricted to tensors, cosine_scheduler ported from NumPy to
# Python lists.

"""Miscellaneous utilities shared across training scripts."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute logit(x) = log(x / (1 - x)), clamped for numerical stability."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def is_dist_avail_and_initialized() -> bool:
    """Return whether ``torch.distributed`` is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Return the number of distributed processes (1 when not distributed)."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Return the rank of this process (0 when not distributed)."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def all_gather(data: torch.Tensor) -> list[torch.Tensor]:
    """Gather a tensor from all processes (non-distributed: a one-element list)."""
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    tensor_list = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(tensor_list, data)
    return tensor_list


class SmoothedValue:
    """Track a series of values and provide a smoothed summary (window average)."""

    def __init__(
        self,
        window_size: int = 20,
        fmt: str = "{median:.4f} ({global_avg:.4f})",
    ) -> None:
        from collections import deque

        self.deque: deque = deque(maxlen=window_size)
        self.total: float = 0.0
        self.count: int = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> None:
        """Append ``value`` to the window and add it ``n`` times to the global total."""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """Reduce ``total`` and ``count`` across all distributed ranks in-place.

        After this call ``global_avg`` reflects the true average over the full
        dataset rather than rank-0's local shard.  No-op when not running under
        ``torch.distributed``.
        """
        if not is_dist_avail_and_initialized():
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t_total = torch.tensor([self.total], dtype=torch.float64, device=device)
        t_count = torch.tensor([self.count], dtype=torch.int64, device=device)
        dist.all_reduce(t_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_count, op=dist.ReduceOp.SUM)
        self.total = float(t_total[0].item())
        self.count = int(t_count[0].item())

    @property
    def median(self) -> float:
        """Median of the values in the window (0.0 if empty)."""
        import statistics

        d = sorted(self.deque)
        return statistics.median(d) if d else 0.0

    @property
    def avg(self) -> float:
        """Mean of the values in the window (0.0 if empty)."""
        return sum(self.deque) / len(self.deque) if self.deque else 0.0

    @property
    def global_avg(self) -> float:
        """Weighted mean over all updates (0.0 if ``count`` is 0)."""
        return self.total / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        value = self.deque[-1] if self.deque else 0.0
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=value,
        )


class MetricLogger:
    """Lightweight metric logger for training loops."""

    def __init__(self, delimiter: str = "  ") -> None:
        from collections import defaultdict

        self.meters: dict = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs: float) -> None:
        """Update the named meters; tensor values are converted with ``.item()``."""
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def synchronize_between_processes(self) -> None:
        """Call ``synchronize_between_processes`` on every tracked meter."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def __str__(self) -> str:
        return self.delimiter.join(f"{k}: {v}" for k, v in self.meters.items())


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
) -> list[float]:
    """Build a cosine annealing schedule with optional linear warm-up.

    Returns a flat list of length ``epochs * niter_per_ep``.
    """
    warmup_schedule: list[float] = []
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = list(
            torch.linspace(start_warmup_value, base_value, warmup_iters).tolist()
        )

    iters = torch.arange(epochs * niter_per_ep - warmup_iters)
    schedule = [
        final_value
        + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / len(iters)))
        for i in iters.tolist()
    ]
    full_schedule = warmup_schedule + schedule
    assert len(full_schedule) == epochs * niter_per_ep
    return full_schedule
