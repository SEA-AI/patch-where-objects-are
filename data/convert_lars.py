"""Convert the LaRS panoptic annotations to COCO detection JSONs.

LaRS (lars_v1.0.0) provides panoptic annotations whose thing segments carry a
``bbox`` (``[x, y, w, h]`` in pixels). This script keeps the thing categories
(dynamic obstacles: boats, buoys, swimmers, ...) and writes COCO detection
files.

- ``file_name`` gets the ``images/`` prefix, so paths resolve as
  ``<data_root>/<split>/images/<file>.jpg`` with
  ``data_root=$DATA_ROOT/lars/lars_v1.0.0_images``,
- ``area`` is the box area ``w * h``.

Run::

    python data/convert_lars.py [--root $DATA_ROOT/lars] [--splits train val]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=None, help="default: $DATA_ROOT/lars")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <root>).",
    )
    args = p.parse_args()
    if args.root is None:
        if not os.environ.get("DATA_ROOT"):
            p.error("set DATA_ROOT or pass --root")
        args.root = os.path.join(os.environ["DATA_ROOT"], "lars")
    return args


def convert_split(root: Path, split: str, out_dir: Path) -> None:
    src = root / "lars_v1.0.0_annotations" / split / "panoptic_annotations.json"
    with open(src, encoding="utf-8") as f:
        pan = json.load(f)

    thing_cats = [c for c in pan["categories"] if c.get("isthing")]
    thing_ids = {c["id"] for c in thing_cats}

    images = [
        {
            "id": img["id"],
            "file_name": f"images/{img['file_name']}",
            "width": img["width"],
            "height": img["height"],
        }
        for img in pan["images"]
    ]

    annotations = []
    next_id = 1
    for rec in pan["annotations"]:
        for seg in rec["segments_info"]:
            if seg["category_id"] not in thing_ids:
                continue
            x, y, w, h = seg["bbox"]
            annotations.append(
                {
                    "id": next_id,
                    "image_id": rec["image_id"],
                    "category_id": seg["category_id"],
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w) * float(h),
                    "iscrowd": int(seg.get("iscrowd", 0)),
                }
            )
            next_id += 1

    out = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": c["id"], "name": c["name"]} for c in thing_cats],
    }
    out_path = out_dir / f"lars_{split}_coco.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    annotated = {a["image_id"] for a in annotations}
    small = sum(1 for a in annotations if a["area"] < 32 * 32)
    print(
        f"{split}: {len(images)} images ({len(annotated)} annotated), "
        f"{len(annotations)} thing boxes ({small / max(len(annotations), 1):.1%} "
        f"COCO-small) -> {out_path}"
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    for split in args.splits:
        convert_split(root, split, out_dir)


if __name__ == "__main__":
    main()
