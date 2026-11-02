"""DetectionDataset — images and COCO-format targets for detection.

Loads a CocoVID JSON (frames grouped into videos) and returns one normalized
``(C, H, W)`` image per annotated frame.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from patch_where_objects_are.data.transforms import DetectionTransformPipeline


def sorted_coco_category_ids(json_path: str) -> list[int]:
    """Sorted COCO ``category_id`` values, matching ``DetectionDataset`` label indices.

    Training targets use contiguous labels ``i``; originals in JSON use
    ``sorted_coco_category_ids[i]``.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    cats = data.get("categories", [])
    return sorted(c["id"] for c in cats)


def json_split(json_path: str) -> str:
    """Split (``train`` / ``val`` / ``test``) named in the JSON file name.

    Only the file name counts, so a directory such as ``/data/train_runs`` does
    not pick the split. The name must hold exactly one of the three as a word
    (``lars_val_coco.json``, ``sea-cocovid-train.json``, ``train2017.json``).
    """
    words = set(re.split(r"[^a-z]+", Path(json_path).name.lower()))
    splits = [s for s in ("train", "val", "test") if s in words]
    if len(splits) != 1:
        raise ValueError(
            f"{json_path}: file name must name exactly one split "
            f"(train / val / test), found {splits or 'none'}"
        )
    return splits[0]


class DetectionDataset(Dataset):
    """Detection dataset over a CocoVID JSON (SeaClips, LaRS, MODD2, MODS).

    Samples are ordered by video, then by ``frame_id``.

    Args:
        json_path:      Path to the CocoVID JSON; its file name names the split
                        (``json_split``).
        data_root:      Root directory for frame images (``<data_root>/<split>/``).
        transform:      ``DetectionTransformPipeline`` (None:
                        ``DetectionTransformPipeline(img_size)``).
        img_size:       Target image resolution (used when transform is None).
        include_empty:  Keep frames without annotations (eval only: their
                        detections score as false positives; training on them
                        would add loss terms with no positive targets).
    """

    def __init__(
        self,
        json_path: str,
        data_root: str,
        transform: DetectionTransformPipeline | None = None,
        img_size: int = 448,
        include_empty: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.include_empty = include_empty
        self.transform = transform or DetectionTransformPipeline(img_size=img_size)
        self.data_split = json_split(json_path)

        with open(json_path) as f:
            data = json.load(f)

        # Build lookup structures.
        self._img_info: dict[int, dict] = {img["id"]: img for img in data["images"]}
        self._fid_to_bboxes: dict[int, list[dict]] = {
            img["id"]: [] for img in data["images"]
        }
        for ann in data.get("annotations", []):
            self._fid_to_bboxes[ann["image_id"]].append(ann)

        # Group frames by video; sort by frame_id.
        vid_to_frames: dict[Any, list[int]] = {}
        for img in data["images"]:
            vid = img.get("video_id", 0)
            vid_to_frames.setdefault(vid, []).append(img["id"])
        for fids in vid_to_frames.values():
            fids.sort(key=lambda fid: self._img_info[fid].get("frame_id", fid))

        # Only frames with at least one annotation, unless include_empty is set.
        self._samples: list[int] = [
            fid
            for fids in vid_to_frames.values()
            for fid in fids
            if self.include_empty or self._fid_to_bboxes.get(fid)
        ]

        # Category mapping: id -> contiguous 0-based label.
        cats = data.get("categories", [])
        sorted_cat_ids = sorted(c["id"] for c in cats)
        self._cat_id_map = {cid: i for i, cid in enumerate(sorted_cat_ids)}
        for ann in data.get("annotations", []):
            if ann.get("category_id") not in self._cat_id_map:
                raise ValueError(
                    f"{json_path}: annotation {ann.get('id')} has category_id "
                    f"{ann.get('category_id')}, not in the categories {sorted_cat_ids}"
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        ref_fid = self._samples[idx]
        ref_img_info = self._img_info[ref_fid]
        img = Image.open(
            self.data_root / self.data_split / ref_img_info["file_name"]
        ).convert("RGB")

        image, flipped = self.transform(img)  # (C, H, W), flip flag for the boxes

        H_out, W_out = image.shape[1], image.shape[2]
        img_w = ref_img_info.get("width", W_out)
        img_h = ref_img_info.get("height", H_out)

        boxes = []
        labels = []
        for ann in self._fid_to_bboxes.get(ref_fid, []):
            x, y, bw, bh = ann["bbox"]
            # Normalize to [0, 1] in original image coords, then convert to cxcywh.
            cx = (x + bw / 2) / img_w
            cy = (y + bh / 2) / img_h
            w_n = bw / img_w
            h_n = bh / img_h

            # Apply horizontal flip to bbox.
            if flipped:
                cx = 1.0 - cx

            boxes.append([cx, cy, w_n, h_n])
            labels.append(self._cat_id_map[ann["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
            if labels
            else torch.zeros(0, dtype=torch.long),
            "image_id": torch.tensor(ref_fid),
            "orig_size": torch.tensor([img_h, img_w], dtype=torch.long),
        }
        return image, target
