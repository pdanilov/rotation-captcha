"""Score how *orientable* each crop of a slice is, using a trained filter model.

For every crop we render it as a disc rotated by N known angles, predict each, and
take the per-crop MEDIAN circular error. A low median means the model recovers the
applied rotation at every angle -> the crop has a clear canonical "up" (orientable).
A high, angle-stable error means the crop is unorientable (a flat floor, a texture,
an edge) -- the tail that injects wrong angle targets and should be dropped.

This is the model-based orientability "thermometer" as a proper tool. Use a filter
model trained on a DISJOINT slice (no leakage): it scores crops it never trained on.

Writes, next to the slice:
  data/raw/coco_objects/<SLICE>/orientability.csv      (filename, orient_median, orient_mae)
  data/raw/coco_objects/<SLICE>/orientability.json     (provenance: checkpoint, n_angles, ...)

    python scripts/score_orientability.py --crops train_from=10000_size=5000_cfg=6be7bb \
        --checkpoint runs/cls-72b-aug__.../model.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rotcaptcha.augment import build_eval_transform
from rotcaptcha.geometry import DEFAULT_SIZE, make_disc_sample
from rotcaptcha.heads import ClassificationHead
from rotcaptcha.metrics import circular_abs_error
from rotcaptcha.model import build_model

ROOT = Path(__file__).resolve().parents[1]
CROPS_BASE = ROOT / "data" / "raw" / "coco_objects"


class DiscAngleDataset(Dataset):
    """Yields every (crop, angle) pair as a rotated disc tensor + its indices."""

    def __init__(self, crop_paths: list[Path], angles: list[float], transform, disc_size: int):
        self.crop_paths = crop_paths
        self.angles = angles
        self.tf = transform
        self.disc_size = disc_size
        self.items = [(i, a) for i in range(len(crop_paths)) for a in angles]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int) -> dict:
        i, angle = self.items[k]
        img = Image.open(self.crop_paths[i]).convert("RGB")
        disc = make_disc_sample(img, angle, size=self.disc_size)
        return {"pixel_values": self.tf(disc), "crop_idx": i, "angle": angle}


def infer_n_bins(state_dict: dict) -> int:
    for key, val in state_dict.items():
        if key.endswith("classifier.1.weight") and val.ndim == 2:
            return int(val.shape[0])
    raise SystemExit("Could not infer n_bins from checkpoint (expected a classifier head).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, help="crop-slice subdir under data/raw/coco_objects/ to score")
    ap.add_argument("--checkpoint", required=True, type=Path, help="filter-model state_dict (.pt)")
    ap.add_argument("--model-name", default="microsoft/resnet-34")
    ap.add_argument("--n-angles", type=int, default=12, help="rotations per crop (evenly spaced over 360)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--disc-size", type=int, default=DEFAULT_SIZE)
    args = ap.parse_args()

    slice_dir = CROPS_BASE / args.crops
    manifest = slice_dir / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"Missing {manifest}")
    with manifest.open(newline="") as f:
        names = [r["filename"] for r in csv.DictReader(f)]
    crop_paths = [slice_dir / n for n in names]
    crop_paths = [p for p in crop_paths if p.exists()]
    print(f"scoring {len(crop_paths)} crops of {args.crops} at {args.n_angles} angles")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    n_bins = infer_n_bins(state_dict)
    head = ClassificationHead(n_bins=n_bins)
    model = build_model(args.model_name, n_bins)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    print(f"loaded {args.checkpoint} (n_bins={n_bins}) on {device}")

    angles = [i / args.n_angles * 360.0 for i in range(args.n_angles)]
    tf = build_eval_transform(args.img_size)
    ds = DiscAngleDataset(crop_paths, angles, tf, args.disc_size)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    errs: dict[int, list[float]] = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(dl, desc="score"):
            logits = model(batch["pixel_values"].to(device)).logits
            pred = head.decode(logits)  # np array of degrees
            true = batch["angle"].numpy() % 360.0
            e = circular_abs_error(pred, true)
            for idx, err in zip(batch["crop_idx"].tolist(), e.tolist(), strict=True):
                errs[idx].append(err)

    rows = []
    for i, path in enumerate(crop_paths):
        e = np.array(errs[i])
        rows.append((path.name, float(np.median(e)), float(e.mean())))

    out_csv = slice_dir / "orientability.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "orient_median", "orient_mae"])
        w.writerows((n, f"{m:.3f}", f"{a:.3f}") for n, m, a in rows)

    med = np.array([m for _, m, _ in rows])
    (slice_dir / "orientability.json").write_text(
        json.dumps(
            {
                "slice": args.crops,
                "checkpoint": str(args.checkpoint),
                "model_name": args.model_name,
                "n_bins": n_bins,
                "n_angles": args.n_angles,
                "n_crops": len(rows),
                "frac_median_gt_10deg": float((med > 10).mean()),
                "frac_median_gt_30deg": float((med > 30).mean()),
                "frac_median_gt_60deg": float((med > 60).mean()),
            },
            indent=2,
        )
    )

    print(f"\nwrote {out_csv}")
    print(
        f"per-crop orient_median: <=10deg {np.mean(med <= 10):.0%}, "
        f">30deg {np.mean(med > 30):.0%}, >60deg {np.mean(med > 60):.0%}"
    )


if __name__ == "__main__":
    main()
