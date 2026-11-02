#!/usr/bin/env python3
r"""Detection training / evaluation entry point.

Usage:
    python scripts/train.py --config configs/uniform/sea_832.yaml --train.gpu 4
    python scripts/train.py --config configs/adaptive/lars.yaml \
        --backbone.patch_selector entropy --backbone.token_budget 576 --train.seed 1
    python scripts/train.py --config configs/adaptive/lars.yaml \
        --backbone.patch_selector entropy --backbone.token_budget 576 --train.seed 1 \
        --eval_only --data.val_json <test json>

Any config field can be overridden as ``--<section>.<field> <value>``.
A training run writes checkpoints, ``config.yaml`` and ``metrics.json`` to
``<train.output_root>/<run name>/s<seed>/``. It keeps two best checkpoints:
``*_best.pth`` (highest AP@0.5) and ``*_best_small.pth`` (highest
mean(AP_allsmall@0.3, AP_allsmall@0.5)).
``--eval_only`` evaluates ``train.load_for_eval`` (default: that run's
checkpoint chosen by ``--eval_checkpoint``, ``best`` or ``best_small``) on
``data.val_json`` and writes ``<checkpoint stem>_eval_<json stem>.json`` next
to the checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

DEFAULT_DETECTION_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "uniform" / "sea_832.yaml"
).resolve()


def _is_torchrun_launched() -> bool:
    """Return True if launched via torchrun / torch.distributed.run."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _resolve_pre_torch_gpu(argv: list[str]) -> int:
    """Physical GPU index used for CUDA_VISIBLE_DEVICES before importing torch.

    Priority:
    1) CLI override `--train.gpu`
    2) YAML config ``train.gpu`` from ``--config``
    3) Default ``0`` (matches ``TrainConfig.gpu`` default when YAML omits it).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_DETECTION_CONFIG))
    parser.add_argument("--train.gpu", dest="train_gpu", type=int, default=None)
    args, _ = parser.parse_known_args(argv)

    if args.train_gpu is not None:
        return args.train_gpu

    config_path = args.config
    if config_path:
        try:
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            train_cfg = raw.get("train", {})
            cfg_gpu = train_cfg.get("gpu")
            if cfg_gpu is not None:
                return int(cfg_gpu)
        except (OSError, ValueError, TypeError, yaml.YAMLError):
            pass

    # Matches ``TrainConfig.gpu`` default when YAML omits ``train.gpu``.
    return 0


# Set visibility before torch import so CUDA device discovery is correct.
# Skip when launched via torchrun: user controls CUDA_VISIBLE_DEVICES externally.
if "CUDA_VISIBLE_DEVICES" not in os.environ and not _is_torchrun_launched():
    _pre_torch_gpu = _resolve_pre_torch_gpu(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_torch_gpu)

import torch  # noqa: E402
from torch import amp  # noqa: E402

from patch_where_objects_are.config.detection import (  # noqa: E402
    DetectionConfig,
    load_detection_config,
    run_dir,
    run_name,
)
from patch_where_objects_are.config.loader import parse_overrides  # noqa: E402
from patch_where_objects_are.data.detect_dataset import (  # noqa: E402
    DetectionDataset,
    json_split,
    sorted_coco_category_ids,
)
from patch_where_objects_are.data.transforms import (  # noqa: E402
    DetectionTransformPipeline,
)
from patch_where_objects_are.evaluators.coco_evaluator import (  # noqa: E402
    CocoEvaluator,
)
from patch_where_objects_are.models.factory import (  # noqa: E402
    build_detector_from_config,
    checkpoint_allow_missing_prefixes,
)
from patch_where_objects_are.training.detect_engine import (  # noqa: E402
    evaluate,
    train_one_epoch,
)
from patch_where_objects_are.training.optim import (  # noqa: E402
    extend_detection_optimizer,
)
from patch_where_objects_are.utils.checkpoint import (  # noqa: E402
    best_checkpoint,
    check_fresh_run_dir,
    load_checkpoint,
    save_checkpoint,
    save_final_checkpoint,
)
from patch_where_objects_are.utils.misc import cosine_scheduler  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from pycocotools.coco import COCO  # noqa: E402


def _maybe_init_wandb(cfg: DetectionConfig, *, eval_only: bool = False):
    if not cfg.train.wandb:
        return None
    if not WANDB_AVAILABLE:
        raise ImportError(
            "wandb is not installed; pass --train.wandb false "
            "or pip install -e '.[wandb]'"
        )
    exp = run_name(cfg)
    # Logged with the run name filled in (empty experiment_name = derived).
    logged = {
        **cfg.__dict__,
        "train": dataclasses.replace(cfg.train, experiment_name=exp),
    }
    # W&B files under <output_root>/wandb.
    Path(cfg.train.output_root).mkdir(parents=True, exist_ok=True)
    init_kwargs = {
        "project": cfg.train.wandb_project,
        "config": logged,
        "dir": cfg.train.output_root,
        "settings": wandb.Settings(_disable_stats=True),
        "group": exp,
    }
    if eval_only:
        init_kwargs["job_type"] = "eval"
        init_kwargs["tags"] = ["eval"]
        if cfg.train.load_for_eval:
            ckpt_name = Path(cfg.train.load_for_eval).stem
            init_kwargs["name"] = (
                f"eval/{exp}/{ckpt_name}/{Path(cfg.data.val_json).stem}"
            )
            init_kwargs["notes"] = cfg.train.load_for_eval
        else:
            init_kwargs["name"] = f"eval/{exp}"
    else:
        init_kwargs["job_type"] = "train"
        init_kwargs["name"] = exp
    return wandb.init(**init_kwargs)


def _json_split_prefix(json_path: str) -> str:
    """W&B key prefix from the evaluated json path (``val/`` vs ``test/``)."""
    return "test/" if json_split(json_path) == "test" else "val/"


def _log_eval(run, stats: dict, *, prefix: str) -> None:
    """Log split-prefixed COCO scalars (test/ or val/ only)."""
    if run is None:
        return
    payload = {f"{prefix}{k}": v for k, v in stats.items()} if prefix else stats
    run.log(payload)


def _ap_allsmall_mean(stats: dict) -> float:
    """mean(AP_allsmall@0.3, AP_allsmall@0.5): selects ``*_best_small.pth``."""
    return (stats["AP_0.3_allsmall"] + stats["AP_0.5_allsmall"]) / 2


def load_config(yaml_path: str, overrides: dict) -> DetectionConfig:
    return load_detection_config(yaml_path, overrides)


def _write_json(path: Path, payload: dict) -> None:
    """Write results as JSON (numpy / torch values converted to Python numbers)."""

    def _default(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"not JSON serializable: {type(value)}")

    path.write_text(json.dumps(payload, indent=1, default=_default))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_DETECTION_CONFIG))
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--eval_checkpoint",
        choices=("best", "best_small"),
        default="best",
        help="--eval_only without --train.load_for_eval: evaluate the run's best "
        "checkpoint by AP@0.5 (best) or by mean(AP_allsmall@0.3, "
        "AP_allsmall@0.5) (best_small)",
    )
    args, unknown = parser.parse_known_args()

    overrides = parse_overrides(unknown)

    cfg = load_config(args.config, overrides)
    t = cfg.train
    eval_include_empty = cfg.data.eval_include_empty
    if args.eval_only and not t.load_for_eval:
        t.load_for_eval = str(best_checkpoint(run_dir(cfg), args.eval_checkpoint))
    if not args.eval_only:
        if t.load_for_eval:
            raise ValueError("train.load_for_eval requires --eval_only")
        if cfg.backbone.patch_selector != "uniform" and not t.init_from:
            raise ValueError(
                "Adaptive training requires --train.init_from <uniform checkpoint>"
            )
        # Never train into a run dir that already holds checkpoints.
        check_fresh_run_dir(run_dir(cfg))

    # Single-GPU path: CUDA_VISIBLE_DEVICES is
    # set before importing torch when not already in the environment. The chosen GPU
    # is therefore always logical ``cuda:0``.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(t.seed)
    np.random.seed(t.seed)  # noqa: NPY002
    random.seed(t.seed)

    logger.info(f"Using device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    bb = cfg.backbone
    d = cfg.data
    if bb.type.lower() != "dinov3_adaptive":
        raise ValueError(f"Unknown backbone.type: {bb.type!r}")

    model = build_detector_from_config(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params / 1e6:.1f}M")

    ckpt_allow_missing = checkpoint_allow_missing_prefixes(cfg)

    # Warm-start (weights only): the adaptive models start from the
    # uniform 832 checkpoint. Not with --eval_only, which loads the evaluated
    # checkpoint.
    if t.init_from and not args.eval_only:
        load_checkpoint(
            model,
            None,
            None,
            t.init_from,
            allow_missing_prefixes=ckpt_allow_missing,
        )

    # ── Datasets ──────────────────────────────────────────────────────────────
    transform_train = DetectionTransformPipeline(img_size=bb.img_size, augment=True)
    transform_val = DetectionTransformPipeline(img_size=bb.img_size, augment=False)

    train_dataset = DetectionDataset(
        json_path=d.train_json,
        data_root=d.image_root,
        transform=transform_train,
        img_size=bb.img_size,
    )
    val_dataset = DetectionDataset(
        json_path=d.val_json,
        data_root=d.image_root,
        transform=transform_val,
        img_size=bb.img_size,
        include_empty=eval_include_empty,
    )

    def collate_fn(batch):
        images, targets = zip(*batch, strict=True)
        return torch.stack(images), list(targets)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=t.batch_size,
        shuffle=True,
        num_workers=t.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=t.batch_size,
        shuffle=False,
        num_workers=t.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── COCO evaluator ────────────────────────────────────────────────────────
    coco_gt_val = COCO(d.val_json)
    val_coco_cat_ids = sorted_coco_category_ids(d.val_json)
    if len(val_coco_cat_ids) != cfg.head.num_classes:
        raise ValueError(
            f"{d.val_json}: {len(val_coco_cat_ids)} categories in JSON vs "
            f"head.num_classes={cfg.head.num_classes}"
        )

    if args.eval_only:
        run = _maybe_init_wandb(cfg, eval_only=True)
        load_checkpoint(
            model,
            None,
            None,
            t.load_for_eval,
            allow_missing_prefixes=ckpt_allow_missing,
        )
        val_evaluator = CocoEvaluator(
            coco_gt_val, iou_types=["bbox"], iou_min=t.iou_min
        )
        stats = evaluate(
            model,
            val_loader,
            val_evaluator,
            device,
            t.confidence_threshold,
            t.nms_iou_threshold,
            coco_category_ids=val_coco_cat_ids,
        )
        logger.info(f"Eval stats: {stats}")
        ckpt_path = Path(t.load_for_eval)
        _write_json(
            ckpt_path.parent / f"{ckpt_path.stem}_eval_{Path(d.val_json).stem}.json",
            {
                "checkpoint": str(ckpt_path),
                "json": d.val_json,
                "eval_include_empty": eval_include_empty,
                "stats": stats,
            },
        )
        _log_eval(run, stats, prefix=_json_split_prefix(d.val_json))
        if run is not None:
            run.finish()
        return

    # ── Optimizer ─────────────────────────────────────────────────────────────
    model.freeze_backbone(t.freeze_backbone_layers)
    param_groups = model.get_param_groups(t.layer_lr_decay, t.base_lr, t.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups, lr=t.base_lr, weight_decay=t.weight_decay, betas=(0.9, 0.999)
    )
    scaler = amp.GradScaler("cuda", enabled=t.use_amp)

    lr_schedule = cosine_scheduler(
        t.base_lr, t.min_lr, t.epochs, len(train_loader), warmup_epochs=t.warmup_epochs
    )

    # ── wandb ─────────────────────────────────────────────────────────────────
    run = _maybe_init_wandb(cfg)

    best_ap = -1.0
    best_ap_small = -1.0
    global_step = 0
    out_dir = Path(run_dir(cfg))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False)
    )
    metrics = {"run_name": run_name(cfg), "seed": t.seed, "val": []}

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(t.epochs):
        # Unfreeze backbone after unfreeze_epoch.
        if epoch == t.unfreeze_epoch:
            logger.info("Unfreezing full backbone.")
            model.unfreeze_backbone()
            extend_detection_optimizer(
                optimizer, model, t.layer_lr_decay, t.base_lr, t.weight_decay
            )

        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            lr_schedule=lr_schedule,
            max_norm=t.clip_grad,
            use_amp=t.use_amp,
            global_step=global_step,
            wandb_run=run,
        )
        global_step = int(
            train_stats.pop("global_step", global_step + len(train_loader))
        )
        logger.info(f"Epoch {epoch} train: {train_stats}")

        # Validation.
        val_evaluator = CocoEvaluator(
            coco_gt_val, iou_types=["bbox"], iou_min=t.iou_min
        )
        val_stats = evaluate(
            model,
            val_loader,
            val_evaluator,
            device,
            t.confidence_threshold,
            t.nms_iou_threshold,
            coco_category_ids=val_coco_cat_ids,
        )
        logger.info(f"Epoch {epoch} val: {val_stats}")

        if run is not None:
            run.log({"epoch": epoch, **train_stats, **val_stats}, step=global_step)

        # Save best (AP@0.5) and best_small (mean(AP_allsmall@0.3, @0.5)).
        ap = val_stats.get("AP_0.5_all", 0.0)
        if ap > best_ap:
            best_ap = ap
            save_checkpoint(
                model, optimizer, scaler, epoch, cfg, str(out_dir), suffix="_best"
            )
        ap_small = _ap_allsmall_mean(val_stats)
        if ap_small > best_ap_small:
            best_ap_small = ap_small
            save_checkpoint(
                model,
                optimizer,
                scaler,
                epoch,
                cfg,
                str(out_dir),
                suffix="_best_small",
            )

        save_checkpoint(model, optimizer, scaler, epoch, cfg, str(out_dir))
        metrics["val"].append({"epoch": epoch, **train_stats, **val_stats})
        _write_json(out_dir / "metrics.json", metrics)

    save_final_checkpoint(model, optimizer, scaler, t.epochs - 1, cfg, str(out_dir))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
