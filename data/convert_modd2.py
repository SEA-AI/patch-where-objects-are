"""Convert the MODD2 annotations to COCO detection JSONs.

MODD2 provides one ``.mat`` file per annotated left frame with obstacle boxes
(``[x, y, w, h]`` in pixels, single class). This script writes
``modd2_{train,val}_coco.json`` (category ``obstacle``, id 1) and copies the
annotated frames so that paths resolve as
``<data_root>/<split>/<sequence>/framesRectified/<frame>.jpg`` with
``data_root=$DATA_ROOT/modd2``.

Run::

    python data/convert_modd2.py [--root $DATA_ROOT/modd2]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image
from scipy.io import loadmat

# Sequence order fixes the image ids and with it the sample order in training.
SEQUENCES = {
    "train": [
        "kope67-00-00067272-00067942",
        "kope71-01-00011800-00012000",
        "kope81-00-00019370-00019710",
        "kope71-01-00011210-00011320",
        "kope67-00-00074432-00074612",
        "kope81-00-00015980-00016270",
        "kope71-01-00017650-00017825",
        "kope67-00-00060561-00061461",
        "kope67-00-00061851-00062671",
        "kope81-00-00006800-00007095",
        "kope67-00-00025200-00025670",
        "kope71-01-00011520-00011800",
        "kope81-00-00010940-00011100",
        "kope81-00-00021520-00022080",
        "kope81-00-00022350-00022520",
        "kope67-00-00062671-00063461",
        "kope81-00-00000560-00001080",
        "kope71-01-00014337-00014547",
        "kope67-00-00004500-00005050",
        "kope67-00-00027400-00027650",
        "kope67-00-00040950-00041190",
        "kope81-00-00004330-00004850",
    ],
    "val": [
        "kope75-00-00062200-00062500",
        "kope75-00-00021500-00022160",
        "kope82-00-00012030-00012700",
        "kope75-00-00037550-00037860",
        "kope82-00-00011177-00011797",
        "kope75-00-00013780-00014195",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=None, help="default: $DATA_ROOT/modd2")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for JSONs and frames (default: <root>).",
    )
    args = p.parse_args()
    if args.root is None:
        if not os.environ.get("DATA_ROOT"):
            p.error("set DATA_ROOT or pass --root")
        args.root = os.path.join(os.environ["DATA_ROOT"], "modd2")
    return args


def convert_split(root: Path, split: str, out_dir: Path) -> None:
    ann_root = root / "MODD2_annotations_v2_rectified" / "annotationsV2_rectified"
    img_root = root / "MODD2_video_data_rectified" / "video_data"

    images, annotations = [], []
    missing = 0
    for seq in SEQUENCES[split]:
        rel_dir = Path(seq) / "framesRectified"
        (out_dir / split / rel_dir).mkdir(parents=True, exist_ok=True)
        for mat_file in sorted((ann_root / seq / "ground_truth").glob("*L.mat")):
            src = img_root / rel_dir / f"{mat_file.stem}.jpg"
            # The first annotated frame of each sequence has no image.
            if not src.exists():
                missing += 1
                continue
            shutil.copy2(src, out_dir / split / rel_dir / src.name)
            width, height = Image.open(src).size

            image_id = len(images) + 1
            images.append(
                {
                    "id": image_id,
                    "file_name": str(rel_dir / src.name),
                    "width": width,
                    "height": height,
                }
            )
            for x, y, w, h in loadmat(mat_file)["annotations"][0][0][1]:
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                )

    out = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "obstacle", "supercategory": "object"}],
    }
    out_path = out_dir / f"modd2_{split}_coco.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    annotated = {a["image_id"] for a in annotations}
    print(
        f"{split}: {len(images)} images ({len(annotated)} annotated), "
        f"{len(annotations)} boxes ({missing} without frame) -> {out_path}"
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    for split in args.splits:
        convert_split(root, split, out_dir)


if __name__ == "__main__":
    main()
