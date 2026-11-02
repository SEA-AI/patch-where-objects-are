r"""Per-cell targets of a dataset split, cached for the objectness scorer.

For every annotated image, with its boxes scaled to the square input
resolution (default 832, a 52 x 52 base grid at ``p = 16``), the cache holds:

- the ground-truth objectness map ``o*`` (``labels.centernet_heatmap``), the
  training target,
- per-object records (covered cell rectangle, size class), for the recall
  that selects the scorer checkpoint.

Only the annotations are read. Everything is written to one compressed
``.npz``, read by ``scripts/train_objectness.py``.

Run::

    python scripts/compute_cell_scores.py \
        --json $DATA_ROOT/lars/lars_val_coco.json \
        --out $OUTPUT_ROOT/cell_scores/cache_lars_val_832.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from patch_where_objects_are.adaptive_patch.labels import (
    centernet_heatmap,
    object_cell_ranges,
)
from patch_where_objects_are.data.detect_dataset import json_split
from patch_where_objects_are.utils.checkpoint import refuse_overwrite

COCO_SMALL_AREA = 32.0**2
COCO_MEDIUM_AREA = 96.0**2


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        required=True,
        help="CocoVID annotation JSON (split inferred from filename).",
    )
    p.add_argument("--img-size", type=int, default=832)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="Debug: first N images.")
    p.add_argument(
        "--out",
        required=True,
        help="Output .npz (cache_<dataset>_<split>_<img_size>.npz).",
    )
    return p.parse_args()


def _load_split(json_path: str) -> tuple[list[dict], dict[int, list[dict]], str]:
    """Annotated image records, annotations per image id, and split name."""
    split = json_split(json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    anns_by_img: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    images = [img for img in data["images"] if anns_by_img.get(img["id"])]
    return images, anns_by_img, split


def main() -> None:
    """Compute and cache the per-cell targets of the split."""
    args = parse_args()
    refuse_overwrite(args.out)  # never replace existing outputs
    p, size = args.patch_size, args.img_size
    grid_hw = (size // p, size // p)

    images, anns_by_img, split = _load_split(args.json)
    if args.limit:
        images = images[: args.limit]
    print(f"{split}: {len(images)} annotated images, grid {grid_hw} @ {size}")

    ostar = np.empty((len(images), *grid_hw), dtype=np.float16)
    obj_img_idx: list[int] = []
    obj_ranges: list[np.ndarray] = []
    obj_sizecls: list[int] = []
    for i, info in enumerate(images):
        sx, sy = size / info["width"], size / info["height"]
        anns = anns_by_img[info["id"]]
        boxes = torch.tensor(
            [
                [
                    a["bbox"][0] * sx,
                    a["bbox"][1] * sy,
                    (a["bbox"][0] + a["bbox"][2]) * sx,
                    (a["bbox"][1] + a["bbox"][3]) * sy,
                ]
                for a in anns
            ],
            dtype=torch.float32,
        )
        areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).numpy()
        medium_or_large = np.where(areas < COCO_MEDIUM_AREA, 1, 2)
        sizecls = np.where(areas < COCO_SMALL_AREA, 0, medium_or_large)

        ostar[i] = centernet_heatmap(boxes, p, grid_hw).numpy().astype(np.float16)
        obj_img_idx.extend([i] * len(anns))
        obj_ranges.append(object_cell_ranges(boxes, p, grid_hw).numpy())
        obj_sizecls.extend(sizecls.tolist())

    meta = {
        "json": args.json,
        "split": split,
        "img_size": size,
        "patch_size": p,
        "small_area_def": f"operating-resolution area < {COCO_SMALL_AREA}",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        img_ids=np.array([img["id"] for img in images], dtype=np.int64),
        ostar=ostar,
        obj_img_idx=np.array(obj_img_idx, dtype=np.int64),
        obj_ranges=np.concatenate(obj_ranges).astype(np.int16),
        obj_sizecls=np.array(obj_sizecls, dtype=np.uint8),
        meta=np.array(json.dumps(meta)),
    )
    print(f"wrote {args.out} ({len(obj_img_idx)} objects, {len(images)} images)")


if __name__ == "__main__":
    main()
