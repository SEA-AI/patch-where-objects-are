# The structure of train_one_epoch / evaluate follows DETR
# (https://github.com/facebookresearch/detr),
# engine.py, Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License 2.0.

"""Detection training and evaluation engine."""

from __future__ import annotations

import math
import statistics
import sys
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import amp
from torch.utils.data import DataLoader

from patch_where_objects_are.models.backbones.dinov3 import DinoV3Backbone
from patch_where_objects_are.training.optim import apply_detection_lr_schedule
from patch_where_objects_are.training.wandb_logging import (
    compute_grad_norms,
    log_training_metrics,
)
from patch_where_objects_are.utils.flops import dinov3_backbone_gflops
from patch_where_objects_are.utils.misc import MetricLogger

if TYPE_CHECKING:
    from patch_where_objects_are.evaluators.coco_evaluator import CocoEvaluator


def train_one_epoch(  # noqa: PLR0913, PLR0917
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    epoch: int,
    lr_schedule: Sequence[float],
    max_norm: float = 0.1,
    use_amp: bool = False,
    log_freq: int = 100,
    global_step: int = 0,
    wandb_run: Any = None,
) -> dict[str, float]:
    """Train the detector for one epoch.

    Args:
        model:      ``Detector`` in train mode.
        dataloader: Yields ``(images, targets)`` batches.
        optimizer:  AdamW.
        scaler:     AMP gradient scaler.
        device:     Training device.
        epoch:      Current epoch index.
        lr_schedule: Flat list of base LRs, one per training iteration.
        max_norm:   Gradient clip norm (0 = disabled).
        use_amp:    Whether to use fp16 AMP.
        log_freq:   Log every N iterations.
        global_step: Global iteration counter at the start of this epoch.
        wandb_run:  Optional active W&B run for compact training logs.

    Returns:
        Dict with averaged loss values and updated ``global_step``.
    """
    model.train()
    metric_logger = MetricLogger()
    n_iter_per_epoch = len(dataloader)
    start_iter = epoch * n_iter_per_epoch

    for step, (images, targets) in enumerate(dataloader):
        it = start_iter + step
        scheduled_lr = lr_schedule[it] if it < len(lr_schedule) else lr_schedule[-1]
        apply_detection_lr_schedule(optimizer, scheduled_lr)

        images = images.to(device, non_blocking=True)
        targets = [
            {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        with amp.autocast("cuda", enabled=use_amp):
            losses = model(images, targets=targets)

        loss = losses.get(
            "loss",
            sum(
                v
                for v in losses.values()
                if isinstance(v, torch.Tensor) and v.requires_grad
            ),
        )

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()} at epoch {epoch} step {step}. Stopping.")
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        grad_norms = compute_grad_norms(model)
        scaler.step(optimizer)
        scaler.update()

        log_dict = {
            k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)
        }
        metric_logger.update(**log_dict)

        current_step = global_step + step
        if step % log_freq == 0:
            print(f"Epoch [{epoch}] [{step}/{len(dataloader)}]  {metric_logger}")
            if wandb_run is not None:
                log_training_metrics(
                    wandb_run,
                    step=current_step,
                    losses=log_dict,
                    optimizer=optimizer,
                    grad_norms=grad_norms,
                )

    result = {k: v.global_avg for k, v in metric_logger.meters.items()}
    result["global_step"] = global_step + n_iter_per_epoch
    return result


@torch.no_grad()
def evaluate(  # noqa: C901, PLR0912, PLR0915
    model: torch.nn.Module,
    dataloader: DataLoader,
    evaluator: CocoEvaluator,
    device: torch.device,
    confidence_threshold: float = 0.001,
    nms_threshold: float = 0.5,
    coco_category_ids: Sequence[int] | None = None,
) -> dict[str, float]:
    """Run evaluation over the full dataset.

    Args:
        model:       ``Detector`` (will be put in eval mode).
        dataloader:  Yields ``(images, targets)`` batches.
        evaluator:   ``CocoEvaluator`` instance.
        device:      Evaluation device.
        confidence_threshold: Score threshold for keeping detections.
        nms_threshold:        NMS IoU threshold.
        coco_category_ids:    If set, map model class index ``i`` to COCO
            ``category_id`` ``coco_category_ids[i]`` (same order as
            ``DetectionDataset`` / sorted JSON ids).

    Returns:
        Dict of COCO metric statistics.
    """
    model.eval()
    model.confidence_threshold = confidence_threshold
    model.nms_iou_threshold = nms_threshold

    lut: torch.Tensor | None = None
    if coco_category_ids is not None:
        lut = torch.as_tensor(list(coco_category_ids), dtype=torch.long)

    # Feature distortion needs the targets' ``orig_size`` at evaluation.
    # Outside training the model reads them only for patch selection.
    eval_needs_targets = bool(getattr(model, "eval_needs_orig_size", False))

    # Tokens per image, GFLOPs and images per second.
    token_counts: list[int] = []
    scale_totals: dict[int, int] = {}
    scale_counts_per_image: list[dict[int, int]] = []
    grid_cells: int | None = None
    forward_time = 0.0
    n_images = 0
    in_h = in_w = 0

    for _idx, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        in_h, in_w = images.shape[2], images.shape[3]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        if eval_needs_targets:
            results = model(images, targets=targets)
        else:
            results = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_time += time.perf_counter() - t0
        n_images += images.shape[0]

        batch_counts = getattr(model, "last_token_counts", None)
        if batch_counts is not None:
            token_counts.extend(batch_counts)
            grid_cells = getattr(model, "last_grid_cells", None)
            for m, c in getattr(model, "last_scale_counts", {}).items():
                scale_totals[m] = scale_totals.get(m, 0) + c
            scale_counts_per_image.extend(
                getattr(model, "last_scale_counts_per_image", [])
            )

        # Predictions keyed by image_id, with the boxes rescaled from the input
        # resolution to the original image size (``orig_size``) of the COCO
        # ground truth.
        predictions = {}
        for res, target in zip(results, targets, strict=True):
            img_id = int(target["image_id"].item())
            labels = res["labels"].detach().cpu().long()
            if lut is not None:
                labels = lut[labels]
            orig_h, orig_w = target["orig_size"].tolist()
            scale = torch.tensor(
                [orig_w / in_w, orig_h / in_h, orig_w / in_w, orig_h / in_h],
                dtype=res["boxes"].dtype,
            )
            boxes = res["boxes"].cpu() * scale
            predictions[img_id] = {
                "boxes": boxes,
                "scores": res["scores"].cpu(),
                "labels": labels,
            }
        evaluator.update(predictions)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    stats = evaluator.summarize()

    # ── Token counts, GFLOPs, throughput ──────────────────────────────────────
    backbone = getattr(model, "backbone", None)
    if not token_counts and isinstance(backbone, DinoV3Backbone) and n_images > 0:
        # Uniform detector: constant token count.
        patch_size = backbone.cfg.patch_size
        grid_cells = (in_h // patch_size) * (in_w // patch_size)
        token_counts = [grid_cells] * n_images
        scale_totals = {1: grid_cells * n_images}

    if token_counts and grid_cells:
        stats["tokens_per_image_mean"] = float(statistics.fmean(token_counts))
        stats["tokens_per_image_std"] = float(
            statistics.pstdev(token_counts) if len(token_counts) > 1 else 0.0
        )
        stats["compression_ratio"] = stats["tokens_per_image_mean"] / grid_cells
        if isinstance(backbone, DinoV3Backbone):
            # Mean of the per-image GFLOPs where per-image counts exist:
            # attention is quadratic in the token count, so the GFLOPs at the
            # mean count would understate the mean GFLOPs.
            gflops_kwargs = {
                "grid_cells": grid_cells,
                "embed_dim": backbone.embed_dim,
                "depth": backbone.depth,
                "patch_size": backbone.cfg.patch_size,
                "n_prefix": backbone.num_register_tokens + 1,
            }
            if len(scale_counts_per_image) == n_images:
                stats["backbone_gflops"] = float(
                    statistics.fmean(
                        dinov3_backbone_gflops(sc, **gflops_kwargs)
                        for sc in scale_counts_per_image
                    )
                )
            else:
                mean_scale_counts = {m: c / n_images for m, c in scale_totals.items()}
                stats["backbone_gflops"] = dinov3_backbone_gflops(
                    mean_scale_counts, **gflops_kwargs
                )
    if n_images > 0 and forward_time > 0:
        stats["images_per_s"] = n_images / forward_time

    return stats
