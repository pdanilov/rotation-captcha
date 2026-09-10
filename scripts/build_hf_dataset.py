"""Export the prepared data into HF `datasets` imagefolder layout.

Each split is a folder of images plus a `metadata.csv` (required `file_name`
column + extra metadata), loadable directly with:

    load_dataset("imagefolder", data_dir="data/hf/<name>")

Produced datasets (hf/ mirrors the raw/ layout, incl. the crop slice name):
  data/hf/crops/<SLICE>/{train,validation}/         metadata: file_name, image_id, category
  data/hf/captcha/baidu/labeled_caps/test/                 metadata: file_name, angle_cw
  data/hf/captcha/baidu/unlabeled_caps/{train,validation}/ metadata: file_name

<SLICE> is the crop-slice subdir produced by crop_coco_objects.py (e.g.
'val_cfg=ab12cd', 'train_from=5000_size=5000_cfg=...'); pass it with --crops
(copy the exact name it printed).

The COCO train/val split is grouped by source `image_id` (all crops from one photo
go to the same split) to prevent scene/background leakage — which is exactly why
crop_coco_objects.py records image_id in its manifest. Use --crop-level-split to
split crops individually instead.

Images are hard-linked into the split folders when possible (no extra disk), with
a copy fallback across filesystems.

    python scripts/build_hf_dataset.py --crops train_from=5000_size=5000
    python scripts/build_hf_dataset.py --crops val --no-captcha
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
COCO_BASE = ROOT / "data" / "raw" / "crops"  # holds per-slice subdirs
REAL = ROOT / "data" / "raw" / "captcha" / "baidu" / "labeled_caps"
REAL_CSV = REAL / "labels.csv"
REAL_UNL = ROOT / "data" / "raw" / "captcha" / "baidu" / "unlabeled_caps"
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


def build_coco(crops_dir: Path, val_frac: float, seed: int, group_by_image: bool) -> None:
    manifest = crops_dir / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"Missing {manifest}; run crop_coco_objects.py --split ... first.")
    with manifest.open(newline="") as f:
        recs = [r for r in csv.DictReader(f) if (crops_dir / r["filename"]).exists()]

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
                "src": crops_dir / r["filename"],
                "file_name": r["filename"],
                "image_id": r["image_id"],
                "category": r["category"],
            }
            for r in rs
        ]

    slice_name = crops_dir.name
    cols = ["file_name", "image_id", "category"]
    write_split(HF / "crops" / slice_name / "train", to_rows(train), cols)
    write_split(HF / "crops" / slice_name / "validation", to_rows(val), cols)
    mode = "grouped-by-image" if group_by_image else "crop-level"
    print(f"crops/{slice_name}: train={len(train)} val={len(val)} ({mode} split)")


def build_real_labeled() -> None:
    if not REAL_CSV.exists():
        raise SystemExit(f"Missing {REAL_CSV}; run fetch_real_caps.py first.")
    with REAL_CSV.open(newline="") as f:
        rows = [
            {"src": REAL / r["filename"], "file_name": r["filename"], "angle_cw": r["angle_cw"]}
            for r in csv.DictReader(f)
        ]
    write_split(HF / "captcha" / "baidu" / "labeled_caps" / "test", rows, ["file_name", "angle_cw"])
    print(f"captcha/baidu/labeled_caps: test={len(rows)}")


def build_real_unlabeled(val_frac: float, seed: int) -> None:
    imgs = sorted(p for p in REAL_UNL.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        print("captcha/baidu/unlabeled_caps: none found, skipping")
        return
    # Held-out unlabeled val for the label-free equivariance metric (never touches
    # the labeled test set). Random split — the pool is already pHash-deduped.
    rng = random.Random(seed)
    rng.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_frac))
    splits = {"validation": imgs[:n_val], "train": imgs[n_val:]}
    for split, ps in splits.items():
        rows = [{"src": p, "file_name": p.name} for p in ps]
        write_split(HF / "captcha" / "baidu" / "unlabeled_caps" / split, rows, ["file_name"])
    print(f"captcha/baidu/unlabeled_caps: train={len(splits['train'])} val={len(splits['validation'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--crops",
        default=None,
        help="crop-slice subdir under data/raw/crops/ to export (e.g. 'val', "
        "'train_from=5000_size=5000'). Omit to build only the captcha datasets.",
    )
    ap.add_argument("--no-captcha", action="store_true", help="skip rebuilding the captcha datasets")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--crop-level-split", action="store_true", help="split COCO crops individually instead of grouping by image"
    )
    args = ap.parse_args()

    if not args.crops and args.no_captcha:
        raise SystemExit("Nothing to build: pass --crops <SLICE> and/or drop --no-captcha.")

    if args.crops:
        crops_dir = COCO_BASE / args.crops
        if not crops_dir.is_dir():
            raise SystemExit(f"No such crop slice: {crops_dir}")
        build_coco(crops_dir, args.val_frac, args.seed, group_by_image=not args.crop_level_split)
    if not args.no_captcha:
        build_real_labeled()
        build_real_unlabeled(args.val_frac, args.seed)

    print(f"\nHF datasets under {HF}/ . Load e.g.:")
    if args.crops:
        print(f'  load_dataset("imagefolder", data_dir="data/hf/crops/{args.crops}")')
    print('  load_dataset("imagefolder", data_dir="data/hf/captcha/baidu/labeled_caps")')


if __name__ == "__main__":
    main()
