#!/usr/bin/env python3
r"""Export LaRS test-set predictions for the MaCVi USV ODCE challenge.

Writes the results JSON expected by
https://github.com/JonNatanael/macvi-usv-odce-toolkit-2024 ::

    {
      "info": {},
      "images": [{"id", "width", "height", "file_name.jpg"}, ...],
      "annotations": [{
        "image_id", "file_name.png",
        "detections": [{"id", "bbox": [x,y,w,h]}, ...]
      }, ...]
    }

Class labels and scores are ignored by the evaluator (class-agnostic F1 @ IoU 0.3);
we still filter by ``--min-score`` before writing. Detection ``id`` is unique
within each image. Empty ``detections`` are kept for frames with no boxes.

Image ids / file names / sizes come from the toolkit's ``lars_test_empty.json``
(under ``$DATA_ROOT/lars/``, see ``data/README.md``) — **not** from
``image_annotations.json``, whose ids differ.

Run with the same config and overrides as the training run; the checkpoint
defaults to that run's best checkpoint::

    python scripts/export_lars_test.py --config configs/adaptive/lars.yaml \
        --backbone.patch_selector objectness --backbone.token_budget 576 \
        --train.seed 0 \
        --min-score 0.30 --gpu 4
    python scripts/export_lars_test.py --config configs/uniform/lars_640.yaml \
        --checkpoint <selected epoch> --min-score 0.30 --gpu 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
parser.add_argument(
    "--checkpoint",
    default=None,
    help="default: the run's best checkpoint (from the config)",
)
parser.add_argument(
    "--out",
    default=None,
    help=(
        "output json (default: lars_test_<checkpoint stem>_min<score>.json next to it)"
    ),
)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--lars-root", default=None, help="default: $DATA_ROOT/lars")
parser.add_argument(
    "--template",
    default=None,
    help="MaCVi empty results skeleton (default: <lars-root>/lars_test_empty.json)",
)
parser.add_argument(
    "--min-score",
    type=float,
    default=0.0,
    help=(
        "drop predictions below this score (model confidence_threshold applies anyway)"
    ),
)
parser.add_argument(
    "--max-dets",
    type=int,
    default=100,
    help="keep at most this many top-scoring predictions per image",
)
args, unknown = parser.parse_known_args()
if args.lars_root is None:
    if not os.environ.get("DATA_ROOT"):
        parser.error("set DATA_ROOT or pass --lars-root")
    args.lars_root = os.path.join(os.environ["DATA_ROOT"], "lars")
if args.template is None:
    args.template = os.path.join(args.lars_root, "lars_test_empty.json")


from patch_where_objects_are.config.loader import parse_overrides  # noqa: E402

overrides = parse_overrides(unknown)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from patch_where_objects_are.config.detection import (  # noqa: E402
    load_detection_config,
    run_dir,
)
from patch_where_objects_are.data.transforms import (  # noqa: E402
    DetectionTransformPipeline,
)
from patch_where_objects_are.models.factory import (  # noqa: E402
    build_detector_from_config,
    checkpoint_allow_missing_prefixes,
)
from patch_where_objects_are.utils.checkpoint import (  # noqa: E402
    best_checkpoint,
    load_checkpoint,
)


class LarsTestImages(torch.utils.data.Dataset):
    """Images-only LaRS test loader keyed by the MaCVi template image list."""

    def __init__(self, images: list[dict], images_dir: Path, img_size: int) -> None:
        self.images = images
        self.images_dir = images_dir
        self.transform = DetectionTransformPipeline(img_size=img_size, augment=False)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        entry = self.images[idx]
        img = Image.open(self.images_dir / entry["file_name"]).convert("RGB")
        # Prefer on-disk size; fall back to template metadata.
        orig_w, orig_h = img.size
        image, _ = self.transform(img)  # (C, H, W)
        return image, entry["id"], (orig_h, orig_w)


def collate_fn(batch):
    images, ids, sizes = zip(*batch, strict=True)
    return torch.stack(images), list(ids), list(sizes)


with open(args.template, encoding="utf-8") as f:
    payload = json.load(f)
if "images" not in payload or "annotations" not in payload:
    raise ValueError(
        f"{args.template}: expected MaCVi skeleton with images+annotations"
    )

ann_by_id = {a["image_id"]: a for a in payload["annotations"]}
missing = [img["id"] for img in payload["images"] if img["id"] not in ann_by_id]
if missing:
    raise ValueError(f"template annotations missing for image_ids: {missing[:5]}...")
for ann in payload["annotations"]:
    ann["detections"] = []  # clear any pre-filled dets

cfg = load_detection_config(args.config, overrides)
t, bb, d = cfg.train, cfg.backbone, cfg.data
if args.checkpoint is None:
    args.checkpoint = str(best_checkpoint(run_dir(cfg)))
if args.out is None:
    ckpt = Path(args.checkpoint)
    args.out = str(ckpt.parent / f"lars_test_{ckpt.stem}_min{args.min_score:.2f}.json")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)

model = build_detector_from_config(cfg).to(device)

# Feature distortion at native resolution needs each image's original size.
needs_orig_size = bool(getattr(model, "eval_needs_orig_size", False))

ckpt_allow_missing = checkpoint_allow_missing_prefixes(cfg)
load_checkpoint(
    model, None, None, args.checkpoint, allow_missing_prefixes=ckpt_allow_missing
)
model.eval()

lars_root = Path(args.lars_root)
dataset = LarsTestImages(
    images=payload["images"],
    images_dir=lars_root / "lars_v1.0.0_images" / "test" / "images",
    img_size=bb.img_size,
)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=t.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)

n_dets = 0
with torch.no_grad():
    for step, (images, ids, sizes) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        in_h, in_w = images.shape[2], images.shape[3]
        meta = None
        if needs_orig_size:
            meta = [
                {"orig_size": torch.tensor([h, w], dtype=torch.long)} for h, w in sizes
            ]
        outs = model(images, targets=meta)
        for res, img_id, (orig_h, orig_w) in zip(outs, ids, sizes, strict=True):
            scores = res["scores"].cpu()
            keep = scores >= args.min_score
            scores = scores[keep]
            order = scores.argsort(descending=True)[: args.max_dets]
            scale = torch.tensor(
                [orig_w / in_w, orig_h / in_h, orig_w / in_w, orig_h / in_h],
                dtype=res["boxes"].dtype,
            )
            boxes = res["boxes"].cpu()[keep][order] * scale  # xyxy, orig pixels
            if boxes.numel() == 0:
                continue
            x1, y1, x2, y2 = boxes.unbind(1)
            xywh = torch.stack([x1, y1, x2 - x1, y2 - y1], dim=1).tolist()
            dets = [
                {"id": k + 1, "bbox": [float(v) for v in box]}
                for k, box in enumerate(xywh)
            ]
            ann_by_id[int(img_id)]["detections"] = dets
            n_dets += len(dets)
        if step % 10 == 0:
            print(f"[{step * t.batch_size}/{len(dataset)}] images done", flush=True)

# Preserve template extras (categories/licenses) for toolkit compatibility.
payload.setdefault("info", {})
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(payload, f)
n_empty = sum(1 for a in payload["annotations"] if not a["detections"])
print(
    f"{n_dets} detections across {len(payload['images'])} images "
    f"({n_empty} empty)  min_score={args.min_score} max_dets={args.max_dets} "
    f"-> {args.out}"
)
