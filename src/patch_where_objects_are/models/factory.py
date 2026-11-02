"""Build detectors from a config: patch selector, backbone, neck, head."""

from __future__ import annotations

from typing import TYPE_CHECKING

from patch_where_objects_are.models.backbones.dinov3 import DinoV3Backbone, DinoV3Config
from patch_where_objects_are.models.criterion import build_criterion
from patch_where_objects_are.models.detection_head import RTDETRHead
from patch_where_objects_are.models.detector import Detector
from patch_where_objects_are.models.detector_base import DetectionModel
from patch_where_objects_are.models.fpn import SingleScaleNeck

if TYPE_CHECKING:
    from patch_where_objects_are.config.detection import DetectionConfig
    from patch_where_objects_are.models.adaptive_detector import PatchSelector


_BUDGETED_SELECTORS = {
    "objectness",
    "feature_distortion",
    "entropy",
    "random",
}
_ADAPTIVE_SELECTORS = {"uniform", "uniform_mixed"} | _BUDGETED_SELECTORS
_POOL_SELECTORS = {"objectness", "feature_distortion", "entropy"}
_SCORING_SELECTORS = {"feature_distortion", "entropy"}


def _build_adaptive_patch_selector(
    selector: str,
    *,
    patch_size: int,
    token_budget: int | None,
    seed: int,
    scorer_checkpoint: str,
    pool: str = "max",
    scoring: str = "finest",
) -> PatchSelector | None:
    """Validate and build the patch selector of ``dinov3_adaptive``. Returns
    ``None`` for ``uniform`` and ``uniform_mixed``.
    """
    if selector not in _ADAPTIVE_SELECTORS:
        raise NotImplementedError(
            f"Unknown patch_selector {selector!r}; expected one of "
            f"{sorted(_ADAPTIVE_SELECTORS)}"
        )
    needs_budget = selector in _BUDGETED_SELECTORS
    if (token_budget is not None) != needs_budget:
        raise NotImplementedError(
            f"token_budget is required for {sorted(_BUDGETED_SELECTORS)} and "
            f"must be None otherwise; got patch_selector={selector!r}, "
            f"token_budget={token_budget!r}"
        )
    if pool != "max" and selector not in _POOL_SELECTORS:
        raise ValueError(
            f"selector_pool={pool!r} applies to {sorted(_POOL_SELECTORS)} only"
        )
    if scoring != "finest" and selector not in _SCORING_SELECTORS:
        raise ValueError(
            f"selector_scoring={scoring!r} applies to {sorted(_SCORING_SELECTORS)} only"
        )
    if selector == "feature_distortion":
        from patch_where_objects_are.adaptive_patch.feature_distortion import (
            FeatureDistortionSelector,
        )

        return FeatureDistortionSelector(
            patch_size=patch_size,
            token_budget=token_budget,
            scoring=scoring,
            pool=pool,
        )
    if selector in {"entropy", "random"}:
        from patch_where_objects_are.adaptive_patch.selectors import (
            CriterionMapSelector,
        )

        return CriterionMapSelector(
            mode=selector,
            patch_size=patch_size,
            token_budget=token_budget,
            seed=seed,
            pool=pool,
            scoring=scoring,
        )
    if selector == "objectness":
        from patch_where_objects_are.adaptive_patch.objectness import ObjectnessSelector

        if not scorer_checkpoint:
            raise ValueError(
                "patch_selector='objectness' requires backbone.scorer_checkpoint "
                "(trained by scripts/train_objectness.py)"
            )
        return ObjectnessSelector(scorer_checkpoint, token_budget, pool=pool)
    return None  # uniform | uniform_mixed


def build_detector(  # noqa: PLR0913, PLR0917
    img_size: int = 448,
    patch_size: int = 16,
    fpn_hidden_dim: int = 256,
    num_classes: int = 7,
    num_queries: int = 300,
    num_decoder_layers: int = 6,
    num_denoising: int = 100,
    aux_loss: bool = True,
    confidence_threshold: float = 0.001,
    nms_iou_threshold: float = 0.5,
    backbone_type: str = "dinov3_adaptive",
    dinov3_variant: str = "vitb16",
    dinov3_weights_path: str = "",
    adaptive_patch_selector: str = "uniform",
    adaptive_token_budget: int | None = None,
    adaptive_selector_seed: int = 0,
    adaptive_scorer_checkpoint: str = "",
    adaptive_selector_pool: str = "max",
    adaptive_selector_scoring: str = "finest",
    head_feat_strides: list[int] | None = None,
    head_use_encoder_idx: list[int] | None = None,
    head_full_sparse: bool = False,
) -> DetectionModel:
    """Build a detector: backbone, neck and single-scale RT-DETR head."""
    backbone_type = backbone_type.lower()
    adaptive_detector = False
    adaptive_selector = None
    if backbone_type == "dinov3_adaptive":
        # DINOv3 with the single-scale RT-DETR head on the last block's output.
        #   'uniform': DinoV3Backbone + SingleScaleNeck.
        #   'uniform_mixed': AdaptiveDinoV3Backbone + GridBroadcastNeck with
        #     uniform patches; equal to 'uniform'.
        #   all other selectors: AdaptiveDinoV3Backbone + GridBroadcastNeck.
        selector = adaptive_patch_selector.lower()
        adaptive_selector = _build_adaptive_patch_selector(
            selector,
            patch_size=patch_size,
            token_budget=adaptive_token_budget,
            seed=adaptive_selector_seed,
            scorer_checkpoint=adaptive_scorer_checkpoint,
            pool=adaptive_selector_pool,
            scoring=adaptive_selector_scoring,
        )
        dcfg = DinoV3Config(
            variant=dinov3_variant,
            patch_size=patch_size,
            img_size=img_size,
            weights_path=dinov3_weights_path,
        )
        if selector == "uniform":
            backbone = DinoV3Backbone(dcfg)
            fpn = SingleScaleNeck(intermediate_layers=[backbone.depth - 1])
        else:
            from patch_where_objects_are.models.adaptive_detector import (
                GridBroadcastNeck,
            )
            from patch_where_objects_are.models.backbones.dinov3_adaptive import (
                AdaptiveDinoV3Backbone,
            )

            backbone = AdaptiveDinoV3Backbone(dcfg)
            fpn = GridBroadcastNeck(intermediate_layers=[backbone.depth - 1])
            adaptive_detector = True
        embed_dim = backbone.embed_dim
        feat_channels = [embed_dim]
        feat_strides = head_feat_strides or [16]
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")
    if len(feat_channels) != len(feat_strides):
        raise ValueError(
            f"feat_channels and feat_strides must have the same length, got "
            f"{len(feat_channels)} and {len(feat_strides)}"
        )
    if head_use_encoder_idx is None:
        head_use_encoder_idx = [0]
    if any(i < 0 or i >= len(feat_channels) for i in head_use_encoder_idx):
        raise ValueError(
            f"head_use_encoder_idx {head_use_encoder_idx} out of range for "
            f"{len(feat_channels)} feature level(s); single-scale configs need "
            "head.use_encoder_idx: [0]"
        )
    if head_full_sparse and not adaptive_detector:
        raise ValueError(
            "head.full_sparse requires an adaptive mixed-patch detector "
            "(for a uniform control, use patch_selector=uniform_mixed)"
        )
    head = RTDETRHead(
        num_classes=num_classes,
        hidden_dim=fpn_hidden_dim,
        num_queries=num_queries,
        feat_channels=feat_channels,
        feat_strides=feat_strides,
        num_levels=len(feat_channels),
        nhead=8,
        num_encoder_layers=1,
        use_encoder_idx=head_use_encoder_idx,
        num_decoder_layers=num_decoder_layers,
        num_denoising=num_denoising,
        aux_loss=aux_loss,
        full_sparse=head_full_sparse,
    )
    criterion = build_criterion(
        num_classes,
        num_decoder_layers=num_decoder_layers,
        num_denoising=num_denoising,
        aux_loss=aux_loss,
    )
    if adaptive_detector:
        from patch_where_objects_are.models.adaptive_detector import (
            AdaptiveDINOv3Detector,
        )

        return AdaptiveDINOv3Detector(
            backbone=backbone,
            neck=fpn,
            head=head,
            criterion=criterion,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            selector=adaptive_selector,
        )
    return Detector(
        backbone=backbone,
        fpn=fpn,
        head=head,
        criterion=criterion,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )


def build_detector_from_config(cfg: DetectionConfig) -> DetectionModel:
    """Build the detector of a :class:`DetectionConfig`; used by every entry
    point (training, evaluation, export).
    """
    bb, head, t = cfg.backbone, cfg.head, cfg.train
    return build_detector(
        img_size=bb.img_size,
        patch_size=bb.patch_size,
        fpn_hidden_dim=cfg.fpn.hidden_dim,
        num_classes=head.num_classes,
        num_queries=head.num_queries,
        num_decoder_layers=head.num_decoder_layers,
        num_denoising=head.num_denoising,
        aux_loss=head.aux_loss,
        head_feat_strides=head.feat_strides,
        head_use_encoder_idx=head.use_encoder_idx,
        head_full_sparse=head.full_sparse,
        confidence_threshold=t.confidence_threshold,
        nms_iou_threshold=t.nms_iou_threshold,
        backbone_type=bb.type,
        dinov3_variant=bb.dinov3_variant,
        dinov3_weights_path=bb.dinov3_weights_path,
        adaptive_patch_selector=bb.patch_selector,
        adaptive_token_budget=bb.token_budget,
        adaptive_selector_seed=t.seed,
        adaptive_scorer_checkpoint=bb.scorer_checkpoint,
        adaptive_selector_pool=bb.selector_pool,
        adaptive_selector_scoring=bb.selector_scoring,
    )


def checkpoint_allow_missing_prefixes(cfg: DetectionConfig) -> tuple[str, ...]:
    """State-dict prefixes that may be missing when loading a checkpoint into
    ``cfg``'s model. Mixed-resolution models warm-start from uniform
    checkpoints, so the zero-initialized fusion modules, the frozen selector
    (loaded from ``backbone.scorer_checkpoint``) and the zero-initialized
    scale embedding of the sparse head may be absent.
    """
    bb = cfg.backbone
    adaptive = (
        bb.type.lower() == "dinov3_adaptive" and bb.patch_selector.lower() != "uniform"
    )
    prefixes: tuple[str, ...] = (
        ("backbone.mixed_patch_embed.", "selector.") if adaptive else ()
    )
    if cfg.head.full_sparse:
        prefixes += ("head.sparse_scale_embed.",)
    return prefixes
