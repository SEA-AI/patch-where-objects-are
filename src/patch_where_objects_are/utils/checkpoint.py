"""Checkpoint save/load utilities (model + optimizer + scaler + epoch + config)."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda import amp

logger = logging.getLogger(__name__)


def _config_for_checkpoint(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    return config


def refuse_overwrite(path: str | Path) -> None:
    """Raise if ``path`` exists or is a symlink (dangling or not): ``torch.save``
    and ``np.savez`` write through symlinks and would replace the linked file.
    """
    p = Path(path)
    if p.is_symlink() or p.exists():
        raise FileExistsError(f"{p} already exists; refusing to overwrite it")


def check_fresh_run_dir(output_dir: str | Path) -> None:
    """Raise if ``output_dir`` already holds checkpoints (``*.pth``, including
    symlinks), so a training run never replaces existing ones.
    """
    out = Path(output_dir)
    existing = sorted(out.glob("*.pth")) if out.is_dir() else []
    if existing:
        raise FileExistsError(
            f"{out} already holds {len(existing)} checkpoint(s), e.g. "
            f"{existing[0].name}; training would overwrite them. Use another "
            "output root or seed."
        )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    epoch: int,
    config: Any,
    output_dir: str,
    suffix: str = "",
) -> None:
    """Save a training checkpoint.

    Saves ``checkpoint_epoch{epoch}{suffix}.pth`` and a symlink
    ``checkpoint_latest.pth`` → the new file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fname = f"checkpoint_epoch{epoch:04d}{suffix}.pth"
    path = out / fname
    refuse_overwrite(path)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "config": _config_for_checkpoint(config),
    }
    torch.save(state, path)
    logger.info(f"Saved checkpoint: {path}")

    # Update symlink to latest.
    latest = out / "checkpoint_latest.pth"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(fname)


def save_final_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    epoch: int,
    config: Any,
    output_dir: str,
) -> None:
    """Save the final checkpoint (``save_checkpoint`` with ``suffix="_final"``)."""
    save_checkpoint(
        model, optimizer, scaler, epoch, config, output_dir, suffix="_final"
    )


def best_checkpoint(output_dir: str | Path, kind: str = "best") -> Path:
    """Return the latest ``checkpoint_epoch*_<kind>.pth`` in ``output_dir``.

    ``kind`` is ``"best"`` (selected on AP@0.5) or ``"best_small"`` (selected
    on mean(AP_allsmall@0.3, AP_allsmall@0.5)).
    """
    out = Path(output_dir)
    pattern = f"checkpoint_epoch*_{kind}.pth"
    candidates = sorted(out.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No {kind} checkpoint ({pattern}) in {out}")
    return candidates[-1]


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: amp.GradScaler | None,
    path: str,
    allow_missing_prefixes: tuple = (),
) -> int:
    """Load a checkpoint.

    Args:
        model:     Model to load weights into.
        optimizer: Optimizer to restore state for (pass None to skip).
        scaler:    GradScaler to restore (pass None to skip).
        path:      Path to ``.pth`` file.
        allow_missing_prefixes: Parameter-name prefixes that may be absent from
            the checkpoint (they keep their initialization): the state a
            mixed-resolution model adds to the uniform checkpoint it
            warm-starts from (``checkpoint_allow_missing_prefixes``). Any other
            mismatch raises.

    Returns:
        ``epoch + 1`` (the epoch to start from next).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt["model"]
    if allow_missing_prefixes:
        result = model.load_state_dict(state, strict=False)
        bad_missing = [
            k
            for k in result.missing_keys
            if not k.startswith(tuple(allow_missing_prefixes))
        ]
        if bad_missing or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch beyond {allow_missing_prefixes}: "
                f"missing {bad_missing[:5]}, unexpected {result.unexpected_keys[:5]}"
            )
        if result.missing_keys:
            logger.info(
                "Checkpoint %s: %d key(s) kept at init (%s...)",
                path,
                len(result.missing_keys),
                result.missing_keys[0],
            )
    else:
        model.load_state_dict(state)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt.get("epoch", 0)
    logger.info(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch + 1
