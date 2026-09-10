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
anyway; doing it here keeps files small and matches the COCO crops on disk). Parquet
shards are read directly with pyarrow (rows embed image bytes, so HF streaming is
~1000x slower), and --skip/--sample-size slice by shard/row exactly like the COCO tool:
[0,10000) to train a disposable filter model, [10000,15000) to then score+filter with it.

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
        "min_side": args.min_side,
        "out_size": args.out_size,
        "limit": args.limit,
    }


def slice_dir_name(params: dict) -> str:
    """places_[from=M_]size=N_cfg=<hash>, matching the COCO slice convention."""
    name = "places"
    if params["skip"]:
        name += f"_from={params['skip']}"
    if params["sample_size"] is not None:
        name += f"_size={params['sample_size']}"
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    return f"{name}_cfg={digest}"


def iter_places_rows(skip: int, sample_size: int | None) -> Iterator[dict]:
    """Yield rows [skip, skip+sample_size) by downloading parquet shards lazily.

    Shards are fetched one at a time with hf_hub_download and dropped once the window
    is reached, so a [0,15000) slice pulls only shard 0 (~46k rows) instead of the whole
    ~tens-of-GB dataset (which snapshot_download's glob would grab up front).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    shard_names = sorted(f for f in list_repo_files(DATASET, repo_type="dataset") if f.endswith(".parquet"))
    cols = ["image", "label"]  # image is a struct {bytes, path}
    seen = yielded = 0
    for name in shard_names:
        shard = hf_hub_download(DATASET, name, repo_type="dataset", revision=REVISION)
        pf = pq.ParquetFile(shard)
        nrows = pf.metadata.num_rows
        if seen + nrows <= skip:  # whole shard is before the window
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
    ap.add_argument("--min-side", type=int, default=64, help="skip images smaller than this on any side")
    ap.add_argument("--out-size", type=int, default=256, help="resize each square scene to this size (px)")
    ap.add_argument("--limit", type=int, default=None, help="max total scenes")
    ap.add_argument("--skip", type=int, default=0, help="skip the first M images (whole shards skipped unread)")
    ap.add_argument(
        "--sample-size", type=int, default=None, help="take N images after --skip (reads only the shards needed)"
    )
    args = ap.parse_args()

    params = scene_params(args)
    out_dir = CROPS_BASE / slice_dir_name(params)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"

    hi = args.skip + args.sample_size if args.sample_size is not None else "end"
    print(f"Reading {DATASET} images [{args.skip}, {hi}) via pyarrow (rev={REVISION[:7]}) ...")
    print(f"Output -> {out_dir}")

    n_written = n_skip_small = n_skip_broken = 0
    with manifest.open("w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["filename", "image_id", "category"])
        for i, row in enumerate(tqdm(iter_places_rows(args.skip, args.sample_size), desc="scene")):
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
