"""Image transforms and the COCO evaluation."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import nn

from patch_where_objects_are.data.detect_dataset import DetectionDataset, json_split
from patch_where_objects_are.data.transforms import DetectionTransformPipeline
from patch_where_objects_are.evaluators.coco_evaluator import CocoEvaluator
from patch_where_objects_are.training.detect_engine import evaluate


def test_pipeline_resizes_to_a_square_input():
    img = Image.fromarray(np.zeros((128, 160, 3), dtype=np.uint8))
    for augment in (False, True):
        tensor, flipped = DetectionTransformPipeline(img_size=96, augment=augment)(img)
        assert tensor.shape == (3, 96, 96)
        assert isinstance(flipped, bool)
        assert augment or not flipped


def _write_one_frame_split(root, json_name, split, category_id=1):
    """160 x 80 frame in ``root/split`` with a white object at x 20-60, y 10-30."""
    pixels = np.zeros((80, 160, 3), dtype=np.uint8)
    pixels[10:30, 20:60] = 255
    (root / split).mkdir(parents=True)
    Image.fromarray(pixels).save(root / split / "frame.png")
    ann = {"id": 1, "image_id": 1, "category_id": category_id, "bbox": [20, 10, 40, 20]}
    json_path = root / json_name
    json_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "frame.png", "width": 160, "height": 80}
                ],
                "annotations": [ann],
                "categories": [{"id": 1, "name": "boat"}],
            }
        )
    )
    return json_path


@pytest.mark.parametrize(("flip", "cx"), [(False, 0.25), (True, 0.75)])
def test_dataset_box_follows_the_horizontal_flip(tmp_path, flip, cx):
    json_path = _write_one_frame_split(tmp_path, "train.json", "train")
    transform = DetectionTransformPipeline(img_size=64)
    transform.augment.hflip.p = float(flip)
    transform.augment.jitter.p = 0.0
    transform.augment.motion_blur.p = 0.0
    transform.augment.weather.p = 0.0

    image, target = DetectionDataset(str(json_path), str(tmp_path), transform)[0]

    torch.testing.assert_close(target["boxes"], torch.tensor([[cx, 0.25, 0.25, 0.25]]))
    assert target["orig_size"].tolist() == [80, 160]
    # The box covers the object in the returned image.
    ys, xs = torch.nonzero(image.mean(0) > 0, as_tuple=True)
    center = torch.stack([xs.float().mean(), ys.float().mean()]) + 0.5
    torch.testing.assert_close(center / 64, torch.tensor([cx, 0.25]))


def test_dataset_split_comes_from_the_json_file_name(tmp_path):
    # "train" in the data root must not send the val JSON to the train images.
    root = tmp_path / "train_runs"
    json_path = _write_one_frame_split(root, "lars_val_coco.json", "val")

    dataset = DetectionDataset(str(json_path), str(root), img_size=64)

    assert dataset.data_split == "val"
    assert dataset[0][0].shape == (3, 64, 64)


@pytest.mark.parametrize(
    ("json_path", "split"),
    [
        ("/data/train/seaclips/sea-cocovid-val.json", "val"),
        ("/data/val/lars/lars_train_coco.json", "train"),
        ("/data/train/mods/mods_obstacle_test_coco.json", "test"),
        ("/data/coco/instances_val2017.json", "val"),
        ("/data/train/coco.json", None),
        ("/data/lars/lars_trainval_coco.json", None),
    ],
)
def test_json_split_reads_one_split_word_of_the_file_name(json_path, split):
    if split is None:
        with pytest.raises(ValueError, match="exactly one split"):
            json_split(json_path)
    else:
        assert json_split(json_path) == split


def test_dataset_rejects_unknown_category_ids(tmp_path):
    json_path = _write_one_frame_split(tmp_path, "val.json", "val", category_id=7)

    with pytest.raises(ValueError, match="category_id 7"):
        DetectionDataset(str(json_path), str(tmp_path), img_size=64)


class _FixedDetector(nn.Module):
    """One detection at x 16-32, y 16-48 (xyxy pixels of the 64 x 64 input)."""

    def forward(self, images):
        return [
            {
                "boxes": torch.tensor([[16.0, 16.0, 32.0, 48.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in images
        ]


def test_evaluate_rescales_boxes_to_the_original_image_size():
    # 300 x 150 frame: the detection maps to x 75-150, y 37.5-112.5.
    gt = COCO()
    gt.dataset = {
        "info": {},
        "images": [{"id": 1, "width": 300, "height": 150}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "bbox": [75, 37.5, 75, 75],
                "area": 5625,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 3, "name": "boat"}],
    }
    gt.createIndex()
    target = {"image_id": torch.tensor(1), "orig_size": torch.tensor([150, 300])}
    batches = [(torch.zeros(1, 3, 64, 64), [target])]

    stats = evaluate(
        _FixedDetector(),
        batches,
        CocoEvaluator(gt, iou_types=["bbox"], iou_min=0.3),
        torch.device("cpu"),
        coco_category_ids=[3],
    )

    assert stats["AP_0.3:0.95_all"] == pytest.approx(1.0)


def test_evaluator_counts_tp_fp_fn_per_object_size():
    gt = COCO()
    gt.dataset = {
        "info": {},
        "images": [{"id": 1, "width": 300, "height": 260}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [70, 10, 50, 50],
                "area": 2500,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "boat"}],
    }
    gt.createIndex()
    evaluator = CocoEvaluator(gt, iou_types=["bbox"], iou_min=0.3)
    evaluator.update(
        {
            1: {
                "boxes": torch.tensor(
                    [[10.0, 10.0, 30.0, 30.0], [180.0, 10.0, 230.0, 60.0]]
                ),  # xyxy: the small object found, one medium false positive
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([1, 1]),
            }
        }
    )
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    stats = evaluator.summarize()

    assert (stats["TP_0.3_all_maxDets100"], stats["FP_0.3_all_maxDets100"]) == (1, 1)
    assert stats["FN_0.3_all_maxDets100"] == 1
    assert stats["TP_0.3_allsmall_maxDets100"] == 1
    assert stats["FN_0.3_medium_maxDets100"] == 1
