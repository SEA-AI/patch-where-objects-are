"""Config dataclasses of the detectors (the YAML files in ``configs/``)."""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field

import yaml

from patch_where_objects_are.config.loader import (
    apply_yaml_overrides,
    dict_to_dataclass,
)


@dataclass
class BackboneConfig:
    """``backbone`` section: ViT backbone and patch selector."""

    type: str = "dinov3_adaptive"
    patch_size: int = 16
    img_size: int = 448
    dinov3_variant: str = "vitb16"
    dinov3_weights_path: str = ""
    # Patch selector: uniform | uniform_mixed | objectness |
    # feature_distortion | entropy | random. objectness, feature_distortion,
    # entropy and random need token_budget, uniform and uniform_mixed need
    # token_budget: null.
    patch_selector: str = "uniform"
    token_budget: int | None = None
    # Propagation of the finest-level scores to coarser nodes: "max"
    # or "mean". objectness, feature_distortion and entropy only.
    selector_pool: str = "max"
    # "finest": score the finest-level cells and propagate by selector_pool.
    # "per_scale" (as Quadformer and APT): score every node at its
    # own level. entropy also applies APT's split thresholds (2.0 bits at
    # 32 px and 64 px), so the token count varies per image up to
    # token_budget. feature_distortion and entropy only.
    selector_scoring: str = "finest"
    # Objectness scorer checkpoint (scripts/train_objectness.py).
    scorer_checkpoint: str = ""


@dataclass
class FpnConfig:
    """``fpn`` section: width of the RT-DETR head input projection."""

    # Width of the RT-DETR head (target of its input projection). The neck has
    # no parameters; the section name is kept for config compatibility.
    hidden_dim: int = 256


@dataclass
class HeadConfig:
    """``head`` section: RT-DETR detection head."""

    hidden_dim: int = 256
    num_queries: int = 300
    num_classes: int = 7
    feat_strides: list[int] = field(default_factory=lambda: [16])
    use_encoder_idx: list[int] = field(default_factory=lambda: [0])
    num_decoder_layers: int = 6
    num_denoising: int = 100
    aux_loss: bool = True
    # Sparse RT-DETR head on the tokens (mixed-resolution detectors only);
    # False = the dense head on the base grid. configs/adaptive
    # set True.
    full_sparse: bool = False


@dataclass
class DataConfig:
    """``data`` section: dataset name, image root and COCO annotation files."""

    # sea | lars | modd2; part of the run name.
    dataset: str = ""
    image_root: str = ""  # directory of the split folders with the images
    train_json: str = ""
    val_json: str = ""
    # Keep frames without annotations during evaluation. Training always drops them.
    eval_include_empty: bool = False


@dataclass
class TrainConfig:
    """``train`` section: schedule, optimizer, evaluation, checkpoints, output, W&B."""

    epochs: int = 50
    batch_size: int = 8
    gpu: int = 0
    num_workers: int = 4
    base_lr: float = 1.0e-4
    min_lr: float = 1.0e-6
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    clip_grad: float = 0.1
    freeze_backbone_layers: int = 6
    unfreeze_epoch: int = 10
    layer_lr_decay: float = 0.75
    # Runs write to <output_root>/<run name>/s<seed>/ (see run_dir).
    output_root: str = "outputs"
    iou_min: float = 0.3
    confidence_threshold: float = 0.001
    nms_iou_threshold: float = 0.5
    use_amp: bool = False
    seed: int = 42
    # Checkpoint that --eval_only evaluates; empty = the run's best checkpoint.
    load_for_eval: str = ""
    # Initialize the weights from a checkpoint without its optimizer and epoch
    # state (warm-start).
    init_from: str = ""
    wandb: bool = False
    wandb_project: str = "pwoa"
    # Run name; empty = derived from the config (see run_name).
    experiment_name: str = ""


@dataclass
class DetectionConfig:
    """Full detector config: one field per YAML section."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fpn: FpnConfig = field(default_factory=FpnConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_detection_config(
    yaml_path: str, overrides: dict | None = None
) -> DetectionConfig:
    """Load a YAML config, apply CLI overrides and resolve its paths.

    String values may use ``${VAR}`` (environment, e.g. ``${DATA_ROOT}``) and
    ``{dataset}`` / ``{seed}`` (from ``data.dataset`` / ``train.seed``).
    """
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        apply_yaml_overrides(raw, overrides)
    cfg = detection_config_from_dict(raw)
    _resolve_strings(cfg)
    return cfg


def detection_config_from_dict(d: dict) -> DetectionConfig:
    """Config from a plain dict (YAML or a checkpoint's stored config)."""
    return dict_to_dataclass(DetectionConfig, d)


def _resolve_strings(cfg: DetectionConfig) -> None:
    fills = {"{dataset}": cfg.data.dataset, "{seed}": str(cfg.train.seed)}
    for section in dataclasses.fields(cfg):
        sub = getattr(cfg, section.name)
        for fld in dataclasses.fields(sub):
            value = getattr(sub, fld.name)
            if not isinstance(value, str):
                continue
            value = os.path.expandvars(value)
            unset = re.findall(r"\$\{?(\w+)", value)
            if unset:
                raise ValueError(
                    f"{section.name}.{fld.name}: environment variable(s) "
                    f"{', '.join(unset)} not set"
                )
            for key, fill in fills.items():
                value = value.replace(key, fill)
            setattr(sub, fld.name, value)


def run_name(cfg: DetectionConfig) -> str:
    """Name of a run: ``train.experiment_name`` if set, else derived from the config."""
    if cfg.train.experiment_name:
        return cfg.train.experiment_name
    ds = cfg.data.dataset
    if not ds:
        raise ValueError("data.dataset must be set (sea | lars | modd2)")
    bb = cfg.backbone
    ft = f"ft{cfg.train.epochs}"
    n = bb.token_budget
    sel = bb.patch_selector
    if sel == "uniform":
        return f"uniform_{ds}_{bb.img_size}"
    if sel == "uniform_mixed":
        return f"adaptive_uniform_mixed_{ds}_{ft}"
    variant = ("_meanpool" if bb.selector_pool == "mean" else "") + (
        "_perscale" if bb.selector_scoring == "per_scale" else ""
    )
    return f"adaptive_{sel}{variant}_{ds}_{ft}_n{n}"


def run_dir(cfg: DetectionConfig) -> str:
    """Output directory of a run: ``<output_root>/<run name>/s<seed>``."""
    return os.path.join(cfg.train.output_root, run_name(cfg), f"s{cfg.train.seed}")
