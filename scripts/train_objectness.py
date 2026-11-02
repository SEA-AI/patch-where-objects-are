r"""Train the objectness scorer.

Trains the head of ``ObjectnessScorer`` on the frozen ShuffleNetV2x0.5
features against the ground-truth objectness map ``o*`` of each image, read
from the per-cell caches of ``scripts/compute_cell_scores.py``, with the
weighted BCE ``heatmap_bce_loss``. Each epoch evaluates
recall@k of the predicted maps on the val cache (logged to W&B with
``--wandb``).
The checkpoint with the best small-object recall at the fraction ``--gate-f``
is ``scorer_best.pth``, the last epoch ``scorer_last.pth``.

Run (paper setting: 60 epochs, seeds 0-2)::

    python scripts/train_objectness.py --seed 0 --device cuda:4 \
        --train-cache $OUTPUT_ROOT/cell_scores/cache_lars_train_832.npz \
        --val-cache $OUTPUT_ROOT/cell_scores/cache_lars_val_832.npz \
        --data-root $DATA_ROOT/lars/lars_v1.0.0_images \
        --out-dir $OUTPUT_ROOT/objectness_scorer_lars/s0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from patch_where_objects_are.adaptive_patch.objectness import (
    ObjectnessScorer,
    heatmap_bce_loss,
)
from patch_where_objects_are.adaptive_patch.selector_metrics import (
    object_min_ranks,
    recall_at_budget,
)
from patch_where_objects_are.utils.checkpoint import check_fresh_run_dir

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-channels", type=int, default=96, help="Scorer head width.")
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--data-root", required=True, help="Image root of the dataset.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=10.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--gate-f",
        type=float,
        default=0.12,
        help="Fine-cell fraction f for checkpoint selection (small-object R@f).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Debug: first N images.")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log metrics to Weights & Biases (default off; needs the wandb extra).",
    )
    p.add_argument("--wandb-project", default="pwoa")
    p.add_argument(
        "--wandb-name", default=None, help="W&B run name (default: out-dir name)."
    )
    return p.parse_args()


class _CacheDataset(Dataset):
    """Images (original root, resized square) + per-cell target grids."""

    def __init__(
        self,
        paths: list[Path],
        targets: np.ndarray,
        img_size: int,
        hflip: bool = False,
    ) -> None:
        self.paths = paths
        self.targets = targets
        self.img_size = img_size
        self.hflip = hflip
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = TF.resize(
            img, [self.img_size, self.img_size], interpolation=Image.BILINEAR
        )
        x = (TF.to_tensor(img) - self.mean) / self.std
        t = torch.from_numpy(self.targets[idx].astype(np.float32))
        if self.hflip and torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[-1])
            t = torch.flip(t, dims=[-1])
        return x, t


def _cache_paths(cache: dict[str, np.ndarray], data_root: str) -> list[Path]:
    """Frame paths for the cache's images, in cache order (id-checked)."""
    meta = json.loads(str(cache["meta"]))
    with open(meta["json"], encoding="utf-8") as f:
        data = json.load(f)
    split = meta["split"]
    annotated = {a["image_id"] for a in data["annotations"]}
    images = [img for img in data["images"] if img["id"] in annotated]
    ids = np.array([img["id"] for img in images], dtype=np.int64)
    if not np.array_equal(ids, cache["img_ids"]):
        raise ValueError(f"{meta['json']}: image order does not match cache")
    return [Path(data_root) / split / img["file_name"] for img in images]


@torch.no_grad()
def _predict_maps(
    model: ObjectnessScorer,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Objectness logit maps of every image of a loader, in order."""
    model.eval()
    out = []
    for x, _ in loader:
        out.append(model(x.to(device)).cpu().numpy())
    model.train()
    return np.concatenate(out).astype(np.float64)


def _ckpt_payload(
    model: torch.nn.Module,
    *,
    epoch: int,
    args: argparse.Namespace,
    patch_size: int,
    img_size: int,
    gflops: float,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "epoch": epoch,
        "head_channels": args.head_channels,
        "patch_size": patch_size,
        "img_size": img_size,
        "gflops_per_image": gflops,
        "seed": args.seed,
        "gate_f": args.gate_f,
    }


def _maybe_init_wandb(args: argparse.Namespace) -> Any:
    if not args.wandb:
        return None
    if not WANDB_AVAILABLE:
        raise ImportError(
            "wandb is not installed; drop --wandb or install the wandb extra"
        )
    name = args.wandb_name or Path(args.out_dir).name
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        name=name,
        config={**vars(args), "wandb_name": name},  # with the run name filled in
        dir=args.out_dir,  # W&B files under <out_dir>/wandb
        settings=wandb.Settings(_disable_stats=True),
        job_type="train",
    )


def main() -> None:  # noqa: PLR0915
    """Train the scorer head, evaluate it on val, save the best and last checkpoint."""
    args = parse_args()
    check_fresh_run_dir(args.out_dir)  # never replace existing outputs
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    run = _maybe_init_wandb(args)

    train_cache = dict(np.load(args.train_cache, allow_pickle=True))
    val_cache = dict(np.load(args.val_cache, allow_pickle=True))
    meta = json.loads(str(train_cache["meta"]))
    img_size, p = meta["img_size"], meta["patch_size"]
    gh = gw = img_size // p
    num_cells = gh * gw

    train_paths = _cache_paths(train_cache, args.data_root)
    val_paths = _cache_paths(val_cache, args.data_root)
    if args.limit:
        train_paths = train_paths[: args.limit]

    train_ds = _CacheDataset(
        train_paths,
        train_cache["ostar"][: len(train_paths)],
        img_size,
        hflip=True,
    )
    val_ds = _CacheDataset(val_paths, val_cache["ostar"], img_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ObjectnessScorer(patch_size=p, head_channels=args.head_channels).to(device)
    trainable = [t for t in model.parameters() if t.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_loader)
    )
    n_params = sum(t.numel() for t in model.parameters())
    n_train = sum(t.numel() for t in trainable)
    gflops = model.gflops_per_image(img_size, img_size)
    print(
        f"{n_params / 1e3:.0f}k params "
        f"({n_train / 1e3:.0f}k trainable), "
        f"{gflops:.3f} GFLOPs @ {img_size}"
    )

    small = val_cache["obj_sizecls"] == 0
    obj_img_idx_small = val_cache["obj_img_idx"][small]
    obj_ranges_small = val_cache["obj_ranges"][small].astype(np.int64)
    obj_img_idx_all = val_cache["obj_img_idx"]
    obj_ranges_all = val_cache["obj_ranges"].astype(np.int64)
    fractions = np.array(
        [1 / 64, 1 / 32, 1 / 16, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.33, 0.50]
    )
    gate_idx = int(np.argmin(np.abs(fractions - args.gate_f)))
    gate_f = float(fractions[gate_idx])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"recall": -1.0, "epoch": -1, "recall_all": 0.0}

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for x, t in train_loader:
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            out = model(x)
            loss = heatmap_bce_loss(out, t, pos_weight=args.pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss) * x.shape[0]
            seen += x.shape[0]

        # Recall@k: an object is recalled if one of its cells is among the top k.
        maps = _predict_maps(model, val_loader, device)
        curve_small = recall_at_budget(
            object_min_ranks(maps, obj_img_idx_small, obj_ranges_small),
            num_cells,
            fractions,
        )
        curve_all = recall_at_budget(
            object_min_ranks(maps, obj_img_idx_all, obj_ranges_all),
            num_cells,
            fractions,
        )
        gate_recall_small = float(curve_small[gate_idx])
        gate_recall_all = float(curve_all[gate_idx])
        train_loss = running / seen

        print(
            f"epoch {epoch:2d}  loss {train_loss:.5f}  "
            f"small-R@f={gate_f:.2f} {gate_recall_small:.4f}  "
            f"all-R@f={gate_f:.2f} {gate_recall_all:.4f}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

        if run is not None:
            payload: dict[str, Any] = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/small_R@f": gate_recall_small,
                "val/all_R@f": gate_recall_all,
                "val/gate_f": gate_f,
            }
            for j, f in enumerate(fractions):
                payload[f"val/small_R@f/{f:.4f}"] = float(curve_small[j])
                payload[f"val/all_R@f/{f:.4f}"] = float(curve_all[j])
            run.log(payload, step=epoch)

        if gate_recall_small > best["recall"]:
            best = {
                "recall": gate_recall_small,
                "recall_all": gate_recall_all,
                "epoch": epoch,
            }
            torch.save(
                _ckpt_payload(
                    model,
                    epoch=epoch,
                    args=args,
                    patch_size=p,
                    img_size=img_size,
                    gflops=gflops,
                ),
                out_dir / "scorer_best.pth",
            )

    torch.save(
        _ckpt_payload(
            model,
            epoch=args.epochs - 1,
            args=args,
            patch_size=p,
            img_size=img_size,
            gflops=gflops,
        ),
        out_dir / "scorer_last.pth",
    )

    print(
        f"best epoch {best['epoch']}: small-R@f={gate_f:.2f} {best['recall']:.4f}, "
        f"all-R@f={gate_f:.2f} {best['recall_all']:.4f}"
    )
    print(f"wrote {out_dir / 'scorer_best.pth'} and {out_dir / 'scorer_last.pth'}")
    if run is not None:
        run.summary["best/epoch"] = best["epoch"]
        run.summary["best/small_R@f"] = best["recall"]
        run.summary["best/all_R@f"] = best["recall_all"]
        run.summary["best/gate_f"] = gate_f
        run.finish()


if __name__ == "__main__":
    main()
