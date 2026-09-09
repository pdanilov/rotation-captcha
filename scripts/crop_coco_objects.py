"""Crop object-centered squares from COCO using the HF `detection-datasets/coco`
dataset (images + boxes in one place — no zip/annotation downloads). Its parquet
shards are read directly with pyarrow (HF streaming is ~1000x slower here because
every row embeds the image bytes), so --skip/--sample-size slice by shard/row.

Rationale: real rotation captchas show a single recognizable object, upright.
Arbitrary full scenes make a poor synthetic proxy, so instead of rotating whole
images we cut out each annotated object and rotate that.

Crop rule (the "easy hack"): for a bbox of width w and height h, take the square
of side = max(w, h) centred on the bbox centre, so the whole object fits with no
distortion. The square is clamped to stay inside the image (and capped to the
smaller image dimension), so crops are always valid pixels — no padding/bleed.

Note on the source schema (differs from native COCO):
- bbox is [x_min, y_min, x_max, y_max] (XYXY), NOT [x, y, w, h].
- category is a contiguous 0..79 index; names live in the dataset feature.
- there is no `iscrowd` flag, so crowd regions are not filtered.

Output: square object crops in data/raw/coco_objects/<SPLIT_NAME>/, resized to
--out-size, plus a manifest (filename, image_id, category) that build_hf_dataset.py
reads for the grouped-by-image split and metadata, without re-reading annotations.
<SPLIT_NAME> encodes the slice: 'val', 'train', 'train_size=15000',
'train_from=5000_size=5000', ...

    python scripts/crop_coco_objects.py --split val --min-side 64 --max-per-category 1500
    python scripts/crop_coco_objects.py --split train --sample-size 5000 --exclude-categories none
    python scripts/crop_coco_objects.py --split train --skip 5000 --sample-size 5000  # held-out
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

# Rotationally-ambiguous COCO categories to drop by default (see module docstring).
# Round / rotationally symmetric: no canonical orientation.
SYMMETRIC = {
    "bowl",
    "pizza",
    "cake",
    "orange",
    "donut",
    "clock",
    "apple",
    "frisbee",
    "sports ball",
}
# Amorphous or variably-posed: weak or unreliable "up".
AMORPHOUS = {
    "banana",
    "carrot",
    "broccoli",
    "sandwich",
    "hot dog",
    "teddy bear",
    "kite",
    "handbag",
    "backpack",
    "suitcase",
    "cup",
    "wine glass",
}
# Default (moderate) blocklist.
AMBIGUOUS_CATEGORIES = SYMMETRIC | AMORPHOUS


def parse_exclude(arg: str | None) -> set[str]:
    """Turn a --exclude-categories CLI value into a set of category names.

    - None         -> the default moderate blocklist
    - "" / "none"  -> no exclusions
    - "a,b,c"      -> exactly those names
    """
    if arg is None:
        return set(AMBIGUOUS_CATEGORIES)
    arg = arg.strip()
    if arg == "" or arg.lower() == "none":
        return set()
    return {c.strip() for c in arg.split(",") if c.strip()}


CROPS_BASE = ROOT / "data" / "raw" / "coco_objects"


def split_dir_name(split: str, skip: int, sample_size: int | None) -> str:
    """Output subdir name encoding the slice, e.g. 'train', 'train_size=15000',
    'train_from=10000', 'train_from=1000_size=3000'."""
    name = split
    if skip:
        name += f"_from={skip}"
    if sample_size is not None:
        name += f"_size={sample_size}"
    return name


DATASET = "detection-datasets/coco"
# Pin a revision so the derived crops are reproducible (third-party re-host).
REVISION = "cf0b22332314a937e9dc8a1957b21725430bb41d"


def square_crop_box(x, y, w, h, img_w, img_h):
    """Square of side max(w,h) centred on the bbox, clamped inside the image."""
    side = min(max(w, h), img_w, img_h)
    cx, cy = x + w / 2.0, y + h / 2.0
    left = min(max(cx - side / 2.0, 0.0), img_w - side)
    top = min(max(cy - side / 2.0, 0.0), img_h - side)
    return (round(left), round(top), round(left + side), round(top + side))


def iter_coco_rows(split: str, skip: int, sample_size: int | None) -> Iterator[dict]:
    """Yield COCO rows [skip, skip+sample_size) by reading the cached parquet shards
    directly with pyarrow.

    HF streaming is ~1000x slower here (each row embeds image bytes: ~12 s/img vs
    ~90 img/s for pyarrow), so we bypass it. Whole shards before `skip` are skipped
    without reading. Each row is a dict with image_id/width/height/objects and an
    `image` struct carrying the JPEG `bytes`.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        DATASET, repo_type="dataset", revision=REVISION, allow_patterns=[f"data/{split}-*.parquet"]
    )
    shards = sorted(Path(local).glob(f"data/{split}-*.parquet"))
    cols = ["image_id", "width", "height", "objects", "image"]
    seen = yielded = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        nrows = pf.metadata.num_rows
        if seen + nrows <= skip:  # whole shard is before the window — skip without reading
            seen += nrows
            continue
        for batch in pf.iter_batches(batch_size=64, columns=cols):
            for row in batch.to_pylist():
                if seen < skip:
                    seen += 1
                    continue
                if sample_size is not None and yielded >= sample_size:
                    return
                seen += 1
                yielded += 1
                yield row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument(
        "--min-side",
        type=int,
        default=64,
        help="skip objects whose square side is smaller than this",
    )
    ap.add_argument(
        "--out-size",
        type=int,
        default=256,
        help="resize each square crop to this size (px)",
    )
    ap.add_argument("--max-per-image", type=int, default=None)
    ap.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="cap crops per category to reduce class imbalance",
    )
    ap.add_argument("--limit", type=int, default=None, help="max total crops")
    ap.add_argument(
        "--skip",
        type=int,
        default=0,
        help="skip the first M base images. Independent of --sample-size: with it alone "
        "you crop the tail [M, end); with --sample-size you crop [M, M+N). Whole shards "
        "before M are skipped without reading.",
    )
    ap.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="take N base images (after any --skip), in dataset order, instead of the "
        "whole split — reads only the shards needed. Shard order is category-unbiased "
        "(verified: one shard spans all 80 categories), so no shuffle is needed: [0,N) "
        "and [N,2N) are already disjoint, reproducible, and each representative (e.g. "
        "[0,5000) to train a filter model, [5000,10000) to then crop+filter with it).",
    )
    ap.add_argument(
        "--exclude-categories",
        default=None,
        help="comma-separated category names to skip (rotationally "
        "ambiguous). Default: the moderate blocklist "
        f"({len(AMBIGUOUS_CATEGORIES)} cats). Use 'none' to keep all.",
    )
    args = ap.parse_args()

    from datasets import load_dataset_builder  # heavy import; keep it local

    exclude = parse_exclude(args.exclude_categories)
    if exclude:
        print(f"Excluding {len(exclude)} ambiguous categories: {', '.join(sorted(exclude))}")

    # category index -> name, from the builder metadata (pure metadata, no data read).
    # objects is a dict of List(...) features; category is List(ClassLabel).
    builder = load_dataset_builder(DATASET, revision=REVISION)
    cat_names = builder.info.features["objects"]["category"].feature.names
    exclude_idx = {i for i, n in enumerate(cat_names) if n in exclude}

    hi = args.skip + args.sample_size if args.sample_size is not None else "end"
    print(f"Reading {DATASET} {args.split} images [{args.skip}, {hi}) via pyarrow (rev={REVISION[:7]}) ...")
    ds = iter_coco_rows(args.split, args.skip, args.sample_size)

    out_dir = CROPS_BASE / split_dir_name(args.split, args.skip, args.sample_size)
    manifest = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output -> {out_dir}")
    per_cat: dict[int, int] = defaultdict(int)
    per_img: dict[int, int] = defaultdict(int)
    n_written = n_skip_small = n_skip_ambig = n_skip_broken = 0

    with manifest.open("w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["filename", "image_id", "category"])

        for row in tqdm(ds, desc="crop"):
            if args.limit and n_written >= args.limit:
                break
            image_id = row["image_id"]
            img_w, img_h = row["width"], row["height"]
            objs = row["objects"]
            image = None  # decode lazily, only once we keep an object

            for bbox_id, cat, bbox in zip(objs["bbox_id"], objs["category"], objs["bbox"], strict=True):
                if cat in exclude_idx:
                    n_skip_ambig += 1
                    continue
                if args.max_per_image and per_img[image_id] >= args.max_per_image:
                    continue
                if args.max_per_category and per_cat[cat] >= args.max_per_category:
                    continue
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                if min(max(w, h), img_w, img_h) < args.min_side:
                    n_skip_small += 1
                    continue

                if image is None:
                    try:
                        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                    except Exception:
                        n_skip_broken += 1
                        break
                box = square_crop_box(x1, y1, w, h, img_w, img_h)
                crop = image.crop(box)
                if args.out_size:
                    crop = crop.resize((args.out_size, args.out_size), Image.Resampling.BICUBIC)
                name = f"{image_id:012d}_{bbox_id}.jpg"
                crop.save(out_dir / name, quality=92)
                writer.writerow([name, image_id, cat_names[cat]])
                per_cat[cat] += 1
                per_img[image_id] += 1
                n_written += 1

    print(f"\nWrote {n_written} crops -> {out_dir}")
    print(f"  manifest -> {manifest}")
    print(f"  skipped: {n_skip_small} too-small, {n_skip_ambig} ambiguous-category, {n_skip_broken} broken image")


if __name__ == "__main__":
    main()
