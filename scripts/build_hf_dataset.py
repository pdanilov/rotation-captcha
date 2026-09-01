"""Export the prepared data into HF `datasets` imagefolder layout.

Each split is a folder of images plus a `metadata.csv` (required `file_name`
column + extra metadata), loadable directly with:

    load_dataset("imagefolder", data_dir="data/hf/<name>")

Produced datasets:
  data/hf/coco_objects/{train,validation}/   metadata: file_name, image_id, category
  data/hf/real_caps_labeled/test/            metadata: file_name, angle_cw
  data/hf/real_caps_unlabeled/train/         metadata: file_name

The COCO train/val split is grouped by source `image_id` (all crops from one photo
go to the same split) to prevent scene/background leakage — which is exactly why
crop_coco_objects.py records image_id in its manifest. Use --crop-level-split to
split crops individually instead.

Images are hard-linked into the split folders when possible (no extra disk), with
a copy fallback across filesystems.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "data" / "raw" / "coco_objects"
CROP_MANIFEST = CROPS / "manifest.csv"
REAL = ROOT / "data" / "raw" / "real_caps_labeled"
REAL_CSV = REAL / "labels.csv"
REAL_UNL = ROOT / "data" / "raw" / "real_caps_unlabeled"
HF = ROOT / "data" / "hf"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # hard link: no extra disk, same filesystem
    except OSError:
        shutil.copy2(src, dst)  # fallback across filesystems


def write_split(dst_dir: Path, rows: list[dict], columns: list[str]) -> None:
    """rows: dicts with a 'src' Path + one entry per metadata column."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)
    with (dst_dir / "metadata.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            link_or_copy(r["src"], dst_dir / r["file_name"])
            w.writerow([r[c] for c in columns])


def build_coco(val_frac: float, seed: int, group_by_image: bool) -> None:
    if not CROP_MANIFEST.exists():
        raise SystemExit(f"Missing {CROP_MANIFEST}; run crop_coco_objects.py first.")
    with CROP_MANIFEST.open(newline="") as f:
        recs = [r for r in csv.DictReader(f) if (CROPS / r["filename"]).exists()]

    rng = random.Random(seed)
    if group_by_image:
        by_img: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_img[r["image_id"]].append(r)
        imgs = list(by_img)
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_frac))
        val_imgs = set(imgs[:n_val])
        val = [r for i in val_imgs for r in by_img[i]]
        train = [r for i in imgs[n_val:] for r in by_img[i]]
    else:
        rng.shuffle(recs)
        n_val = max(1, int(len(recs) * val_frac))
        val, train = recs[:n_val], recs[n_val:]

    def to_rows(rs: list[dict]) -> list[dict]:
        return [
            {
                "src": CROPS / r["filename"],
                "file_name": r["filename"],
                "image_id": r["image_id"],
                "category": r["category"],
            }
            for r in rs
        ]

    cols = ["file_name", "image_id", "category"]
    write_split(HF / "coco_objects" / "train", to_rows(train), cols)
    write_split(HF / "coco_objects" / "validation", to_rows(val), cols)
    mode = "grouped-by-image" if group_by_image else "crop-level"
    print(f"coco_objects: train={len(train)} val={len(val)} ({mode} split)")


def build_real_labeled() -> None:
    if not REAL_CSV.exists():
        raise SystemExit(f"Missing {REAL_CSV}; run fetch_real_caps.py first.")
    with REAL_CSV.open(newline="") as f:
        rows = [
            {"src": REAL / r["filename"], "file_name": r["filename"], "angle_cw": r["angle_cw"]}
            for r in csv.DictReader(f)
        ]
    write_split(HF / "real_caps_labeled" / "test", rows, ["file_name", "angle_cw"])
    print(f"real_caps_labeled: test={len(rows)}")


def build_real_unlabeled() -> None:
    imgs = sorted(p for p in REAL_UNL.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        print("real_caps_unlabeled: none found, skipping")
        return
    rows = [{"src": p, "file_name": p.name} for p in imgs]
    write_split(HF / "real_caps_unlabeled" / "train", rows, ["file_name"])
    print(f"real_caps_unlabeled: train={len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--crop-level-split", action="store_true", help="split COCO crops individually instead of grouping by image"
    )
    args = ap.parse_args()

    build_coco(args.val_frac, args.seed, group_by_image=not args.crop_level_split)
    build_real_labeled()
    build_real_unlabeled()
    print(f"\nHF datasets under {HF}/ . Load e.g.:")
    print('  load_dataset("imagefolder", data_dir="data/hf/coco_objects")')
    print('  load_dataset("imagefolder", data_dir="data/hf/real_caps_labeled")')


if __name__ == "__main__":
    main()
