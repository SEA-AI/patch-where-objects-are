"""Convert the MODS obstacle annotations to a COCO detection JSON.

MODS (``mods.json``) provides the obstacle boxes (``[x, y, w, h]`` in
pixels) with types ``ship``, ``person``, ``other`` and ``negative``. This
script drops ``negative`` boxes, merges the other types into one category
``obstacle`` (id 1, as in MODD2) and writes ``mods_obstacle_test_coco.json``.
Paths resolve as ``<data_root>/test/<sequence>/frames/<frame>.jpg`` with
``data_root=$DATA_ROOT/mods``. The script links ``test`` to ``sequences``.

Run::

    python data/convert_mods.py [--root $DATA_ROOT/mods]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# All MODS frames have this size.
WIDTH, HEIGHT = 1278, 958


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=None, help="default: $DATA_ROOT/mods")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <root>).",
    )
    args = p.parse_args()
    if args.root is None:
        if not os.environ.get("DATA_ROOT"):
            p.error("set DATA_ROOT or pass --root")
        args.root = os.path.join(os.environ["DATA_ROOT"], "mods")
    return args


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    with open(root / "mods.json", encoding="utf-8") as f:
        mods = json.load(f)

    images, annotations = [], []
    for seq in mods["dataset"]["sequences"]:
        seq_dir = seq["path"].lstrip("/")
        for frame in seq["frames"]:
            image_id = len(images) + 1
            images.append(
                {
                    "id": image_id,
                    "file_name": os.path.join(seq_dir, frame["image_file_name"]),
                    "width": WIDTH,
                    "height": HEIGHT,
                }
            )
            for obj in frame["obstacles"]:
                if obj["type"] == "negative":
                    continue
                x, y, w, h = obj["bbox"]
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )

    out = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "obstacle"}],
    }
    out_path = out_dir / "mods_obstacle_test_coco.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    test_dir = out_dir / "test"
    if not test_dir.exists():
        test_dir.symlink_to((root / "sequences").resolve())
    annotated = {a["image_id"] for a in annotations}
    print(
        f"test: {len(images)} images ({len(annotated)} annotated), "
        f"{len(annotations)} boxes -> {out_path}"
    )


if __name__ == "__main__":
    main()
