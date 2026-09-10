"""Ingest whole *scene* images from Places365 into the same crop-slice layout the
COCO object pipeline uses, so build_hf_dataset / score_orientability /
filter_orientable all work on them unchanged.

Why scenes: the real Baidu captchas are AI-generated *scenes* (skylines, landscapes),
not single objects, and the 180° polarity flips in the MAE tail come from the model
never learning *scene-level* up cues (horizon, sky, gravity). Places365 images have a
canonical up, so they teach exactly that. See docs/roadmap.md "Where the real-test
MAE comes from".

Unlike crop_coco_objects.py there is nothing to crop: each row is already one scene
image. We just center-square + resize to --out-size (make_disc_sample re-squares
anyway; doing it here keeps files small and matches the COCO crops on disk).

Places365 is sorted by class (5000 imgs/class, contiguous), so a sequential first-N
slice would be a handful of classes. Instead we globally shuffle all 39 shards' index
with --seed and take the window --skip:--skip+--sample-size from that permutation, so
every window spans all 365 classes and [0,N)/[N,2N) stay disjoint and reproducible:
[0,10000) trains a disposable filter model, [10000,15000) is the held-out set to
score+filter with it. Shards (all cached) are read one at a time, decoding only the
selected rows.

Output: data/raw/crops/places_<...>_cfg=<hash>/ with the scene images, a manifest
(filename, image_id, category) and crop_config.json. The base dir is shared with COCO on
purpose (it is the training-crop store); the places_ prefix keeps the slice recognizable.

    python scripts/ingest_places_scenes.py --sample-size 10000
    python scripts/ingest_places_scenes.py --skip 10000 --sample-size 5000  # held-out
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections.abc import Iterator
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from rotcaptcha.geometry import center_square

ROOT = Path(__file__).resolve().parents[1]
CROPS_BASE = ROOT / "data" / "raw" / "crops"  # shared training-crop store

DATASET = "ljnlonoljpiljm/places365-256px"
# Pin a revision so the derived slices are reproducible.
REVISION = "2bcb4953a4cd2081ace0c81c97081891551128a2"


def scene_params(args: argparse.Namespace) -> dict:
    """Canonical, content-defining parameters of an ingest run (no result counts)."""
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "skip": args.skip,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "min_side": args.min_side,
        "out_size": args.out_size,
        "limit": args.limit,
    }


def slice_dir_name(params: dict) -> str:
    """places_[from=M_]size=N_cfg=<hash>, matching the COCO slice convention. The
    seed lives in the hash, so shuffled slices never collide with the old sequential
    ones (different params -> different cfg=)."""
    name = "places"
    if params["skip"]:
        name += f"_from={params['skip']}"
    if params["sample_size"] is not None:
        name += f"_size={params['sample_size']}"
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    return f"{name}_cfg={digest}"


def iter_places_rows(skip: int, sample_size: int | None, seed: int) -> Iterator[dict]:
    """Yield a class-DIVERSE sample by globally shuffling all shards, then taking the
    window [skip, skip+sample_size) of the shuffled order.

    Places365 is sorted by class (exactly 5000 imgs/class, contiguous), so a sequential
    first-N slice would be a handful of classes. We build the global index over all 39
    shards, seed-shuffle it, and select our window from that permutation -> every window
    spans the full 365-class spread, and [0,N)/[N,2N) stay disjoint and reproducible.
    Shards are read one at a time (all cached) and only the selected rows are decoded.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    shard_names = sorted(f for f in list_repo_files(DATASET, repo_type="dataset") if f.endswith(".parquet"))
    paths = [hf_hub_download(DATASET, n, repo_type="dataset", revision=REVISION) for n in shard_names]
    offsets, total = [], 0
    for p in paths:  # cheap: parquet footer only
        offsets.append(total)
        total += pq.ParquetFile(p).metadata.num_rows

    order = np.random.default_rng(seed).permutation(total)
    hi = skip + sample_size if sample_size is not None else total
    picked = set(order[skip:hi].tolist())  # global indices we want, in no particular order

    for path, off in zip(paths, offsets, strict=True):
        nrows = pq.ParquetFile(path).metadata.num_rows
        local = [g - off for g in range(off, off + nrows) if g in picked]
        if not local:
            continue
        table = pq.read_table(path, columns=["image", "label"]).take(local)
        yield from table.to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-side", type=int, default=64, help="skip images smaller than this on any side")
    ap.add_argument("--out-size", type=int, default=256, help="resize each square scene to this size (px)")
    ap.add_argument("--limit", type=int, default=None, help="max total scenes")
    ap.add_argument("--skip", type=int, default=0, help="offset into the shuffled order (for a disjoint held-out window)")
    ap.add_argument(
        "--sample-size", type=int, default=None, help="take N images from the shuffled order after --skip"
    )
    ap.add_argument("--seed", type=int, default=0, help="global-shuffle seed (baked into the slice hash)")
    args = ap.parse_args()

    params = scene_params(args)
    out_dir = CROPS_BASE / slice_dir_name(params)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"

    hi = args.skip + args.sample_size if args.sample_size is not None else "end"
    print(f"Sampling {DATASET} shuffled[{args.skip}, {hi}) seed={args.seed} (rev={REVISION[:7]}) ...")
    print(f"Output -> {out_dir}")

    n_written = n_skip_small = n_skip_broken = 0
    with manifest.open("w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["filename", "image_id", "category"])
        for i, row in enumerate(tqdm(iter_places_rows(args.skip, args.sample_size, args.seed), desc="scene")):
            if args.limit and n_written >= args.limit:
                break
            try:
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            except Exception:
                n_skip_broken += 1
                continue
            if min(img.size) < args.min_side:
                n_skip_small += 1
                continue
            img = center_square(img).resize((args.out_size, args.out_size), Image.Resampling.BICUBIC)
            image_id = args.skip + i  # global position = stable id (one scene per image)
            name = f"places_{image_id:08d}.jpg"
            img.save(out_dir / name, quality=92)
            writer.writerow([name, image_id, f"place_{row['label']}"])
            n_written += 1

    record = {**params, "n_scenes": n_written, "n_skip_small": n_skip_small, "n_skip_broken": n_skip_broken}
    (out_dir / "crop_config.json").write_text(json.dumps(record, indent=2))
    print(f"\nWrote {n_written} scenes -> {out_dir}")
    print(f"  skipped: {n_skip_small} too-small, {n_skip_broken} broken")


if __name__ == "__main__":
    main()
