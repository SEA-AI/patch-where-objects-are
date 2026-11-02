# Adapted from DETR (https://github.com/facebookresearch/detr),
# datasets/coco_eval.py, Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License 2.0.
# That file is based on torchvision's references/detection/coco_eval.py
# (Copyright (c) Soumith Chintala 2016, BSD 3-Clause License, see NOTICE).
#
# _evaluate is a trimmed copy of pycocotools' COCOeval.evaluate
# (https://github.com/cocodataset/cocoapi, Copyright (c) 2014, Piotr Dollar and
# Tsung-Yi Lin, Simplified BSD License, see NOTICE).
#
# Modifications: bbox only, absolute xyxy inputs, configurable start of the IoU
# sweep (iou_min), named stats, TP/FP/FN counts.

"""COCO evaluator with the IoU thresholds of maritime detection.

``iou_min`` shifts the start of the IoU sweep from the standard 0.5 to 0.3,
giving AP@0.3 and AP@[0.3:0.95] next to AP@0.5.

Usage:
    evaluator = CocoEvaluator(coco_gt, iou_types=["bbox"], iou_min=0.3)
    evaluator.update(predictions)   # {img_id: {"boxes" (xyxy abs), "scores", "labels"}}
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    stats = evaluator.summarize()
"""

from __future__ import annotations

import contextlib
import copy
import os

import numpy as np
import torch
from pycocotools.coco import COCO

from patch_where_objects_are.evaluators.adapted_pycocotools.cocoeval import COCOeval
from patch_where_objects_are.utils.misc import all_gather

# Metric keys, in the order of ``COCOeval.stats``.
STAT_KEYS = [
    "AP_0.3:0.95_all",
    "AP_0.3_all",
    "AP_0.5_all",
    "AP_0.3_allsmall",
    "AP_0.3_medium",
    "AP_0.3_large",
    "AP_0.5_allsmall",
    "AP_0.5_medium",
    "AP_0.5_large",
    "AR_0.3:0.95_all_maxDets1",
    "AR_0.3:0.95_all_maxDets10",
    "AR_0.3:0.95_all_maxDets100",
    "AR_0.3:0.95_allsmall_maxDets100",
    "AR_0.3:0.95_medium_maxDets100",
    "AR_0.3:0.95_large_maxDets100",
]


class CocoEvaluator:
    """COCO-style evaluator with configurable minimum IoU threshold."""

    def __init__(
        self,
        coco_gt: COCO,
        iou_types: list[str],
        iou_min: float = 0.3,
    ) -> None:
        assert isinstance(iou_types, (list, tuple))
        self.coco_gt = copy.deepcopy(coco_gt)
        self.iou_min = iou_min
        self.iou_types = iou_types

        self.coco_eval: dict[str, COCOeval] = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(
                self.coco_gt, iouType=iou_type, iou_min=iou_min
            )

        self.img_ids: list[int] = []
        self.eval_imgs: dict[str, list] = {k: [] for k in iou_types}

    def update(self, predictions: dict[int, dict]) -> None:
        """Add predictions for a batch of images.

        Args:
            predictions: ``{img_id: {"boxes": Tensor[N,4] xyxy abs,
                                      "scores": Tensor[N],
                                      "labels": Tensor[N]}}``
        """
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self._prepare(predictions, iou_type)
            with (
                open(os.devnull, "w", encoding="utf-8") as devnull,
                contextlib.redirect_stdout(devnull),
            ):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            _, eval_imgs = _evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self) -> None:
        """Merge evaluation results across processes (no-op for single GPU)."""
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            _create_common_coco_eval(
                self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
            )

    def accumulate(self) -> None:
        """Accumulate per-image evaluation results."""
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self) -> dict[str, float]:
        """Print and return a dict of evaluation statistics."""
        stats: dict[str, float] = {}
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()
            # coco_eval.stats is a fixed-length numpy array from pycocotools.
            # We map available stats to our key names.
            if hasattr(coco_eval, "stats") and coco_eval.stats is not None:
                for i, key in enumerate(STAT_KEYS):
                    if i < len(coco_eval.stats):
                        stats[key] = float(coco_eval.stats[i])
            count_stats = _primary_detection_counts(coco_eval)
            if count_stats:
                stats.update(count_stats)
                iou_thr = _format_metric_threshold(coco_eval.params.iouThrs[0])
                max_det = coco_eval.params.maxDets[-1]
                for area_label in coco_eval.params.areaRngLbl:
                    tp_key = f"TP_{iou_thr}_{area_label}_maxDets{max_det}"
                    fp_key = f"FP_{iou_thr}_{area_label}_maxDets{max_det}"
                    fn_key = f"FN_{iou_thr}_{area_label}_maxDets{max_det}"
                    print(
                        "Detection counts "
                        f"@[ IoU={float(coco_eval.params.iouThrs[0]):.2f} "
                        f"| area={area_label:>8s} | maxDets={max_det:>3d} ] = "
                        f"TP={count_stats[tp_key]:.0f} "
                        f"FP={count_stats[fp_key]:.0f} "
                        f"FN={count_stats[fn_key]:.0f}"
                    )
        return stats

    def _prepare(self, predictions: dict, iou_type: str) -> list[dict]:
        if iou_type == "bbox":
            return self._prepare_detection(predictions)
        raise ValueError(f"Unknown iou_type: {iou_type}")

    @staticmethod
    def _prepare_detection(predictions: dict) -> list[dict]:
        results = []
        for img_id, pred in predictions.items():
            if len(pred.get("boxes", [])) == 0:
                continue
            boxes = pred["boxes"]
            scores = pred["scores"].tolist()
            labels = pred["labels"].tolist()
            # Convert xyxy → xywh for COCO format.
            xywh = _xyxy_to_xywh(boxes).tolist()
            for k, box in enumerate(xywh):
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": int(labels[k]),
                        "bbox": box,
                        "score": float(scores[k]),
                    }
                )
        return results


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=1)


def _primary_detection_counts(coco_eval: COCOeval) -> dict[str, float]:
    """Return TP/FP/FN totals for the primary AP operating point."""
    if not getattr(coco_eval, "eval", None):
        return {}

    iou_idx = 0
    max_det_idx = len(coco_eval.params.maxDets) - 1
    iou_thr = _format_metric_threshold(coco_eval.params.iouThrs[iou_idx])
    max_det = coco_eval.params.maxDets[max_det_idx]

    count_stats = {}
    for eval_key, metric_name in (("tp", "TP"), ("fp", "FP"), ("fn", "FN")):
        values = coco_eval.eval.get(eval_key)
        if values is None:
            return {}
        for area_idx, area_label in enumerate(coco_eval.params.areaRngLbl):
            key = f"{metric_name}_{iou_thr}_{area_label}_maxDets{max_det}"
            count_stats[key] = float(np.sum(values[iou_idx, :, area_idx, max_det_idx]))
    return count_stats


def _format_metric_threshold(value: float) -> str:
    """Format IoU thresholds like existing metric names: 0.30 -> 0.3."""
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _evaluate(coco_eval: COCOeval) -> tuple[list[int], np.ndarray]:
    """Run per-image evaluation (trimmed from pycocotools)."""
    p = coco_eval.params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    coco_eval.params = p
    coco_eval._prepare()

    catIds = p.catIds if p.useCats else [-1]
    computeIoU = coco_eval.computeIoU
    coco_eval.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds
    }

    maxDet = p.maxDets[-1]
    evalImgs = [
        coco_eval.evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
    return p.imgIds, evalImgs


def _merge(img_ids: list[int], eval_imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = list(all_eval_imgs)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]
    return merged_img_ids, merged_eval_imgs


def _create_common_coco_eval(
    coco_eval: COCOeval,
    img_ids: list[int],
    eval_imgs: np.ndarray,
) -> None:
    img_ids, eval_imgs = _merge(img_ids, eval_imgs)
    coco_eval.evalImgs = list(eval_imgs.flatten())
    coco_eval.params.imgIds = list(img_ids)
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
