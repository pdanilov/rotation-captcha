"""Filter a crop slice down to its *orientable* crops, producing a NEW raw slice.

Filtering is a separate raw-dataset operation, deliberately NOT folded into
build_hf_dataset.py: the filtered set is its own recognizable slice (own dir,
manifest, provenance) whose name flows into training's config.coco_slice -> the
tracker, and build_hf_dataset stays a dumb "folder -> train/val" splitter with no
dependency on a checkpoint or threshold.

Input: a source slice under data/raw/crops/<SRC>/ that has already been
scored by score_orientability.py (so <SRC>/orientability.csv exists). We keep the
crops whose per-crop orient_median <= --max-median and hard-link them into a new
slice, carrying the manifest metadata (image_id, category) unchanged.

Output slice name mirrors the crop convention: a readable stem + a deterministic
hash of the filter params (source slice, threshold, scoring checkpoint), so
different params never collide and identical params reuse the same dir. E.g.
'train_from=10000_size=5000_orient=10_cfg=ab12cd'. Full provenance is written to
filter_config.json.

    python scripts/filter_orientable.py --crops train_from=10000_size=5000_cfg=6be7bb --max-median 10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROPS_BASE = ROOT / "data" / "raw" / "crops"


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # hard link: no extra disk, same filesystem
    except OSError:
        shutil.copy2(src, dst)  # fallback across filesystems


def strip_cfg(name: str) -> str:
    """Drop a trailing '_cfg=<hex>' token so the new name carries a single hash."""
    return name.rsplit("_cfg=", 1)[0] if "_cfg=" in name else name


def filter_params(src_slice: str, max_median: float, score_provenance: dict) -> dict:
    """Canonical, content-defining params of a filter run (no result counts)."""
    return {
        "source_slice": src_slice,
        "max_median_deg": max_median,
        "score_checkpoint": score_provenance.get("checkpoint"),
        "score_model_name": score_provenance.get("model_name"),
        "score_n_angles": score_provenance.get("n_angles"),
        "score_n_bins": score_provenance.get("n_bins"),
    }


def out_slice_name(params: dict) -> str:
    stem = f"{strip_cfg(params['source_slice'])}_orient={params['max_median_deg']:g}"
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    return f"{stem}_cfg={digest}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, help="source crop-slice subdir under data/raw/crops/")
    ap.add_argument(
        "--max-median",
        type=float,
        default=10.0,
        help="keep crops whose orient_median (deg) is <= this; 10 is the empirical "
        "knee (above it crops are visibly mis-orientable). See docs/images.",
    )
    ap.add_argument(
        "--score-csv",
        type=Path,
        default=None,
        help="orientability.csv to read (default: <slice>/orientability.csv)",
    )
    args = ap.parse_args()

    src_dir = CROPS_BASE / args.crops
    manifest = src_dir / "manifest.csv"
    score_csv = args.score_csv or (src_dir / "orientability.csv")
    score_json = src_dir / "orientability.json"
    for p in (manifest, score_csv):
        if not p.exists():
            raise SystemExit(f"Missing {p} (run crop + score_orientability first).")

    provenance = json.loads(score_json.read_text()) if score_json.exists() else {}
    with score_csv.open(newline="") as f:
        median = {r["filename"]: float(r["orient_median"]) for r in csv.DictReader(f)}
    with manifest.open(newline="") as f:
        recs = list(csv.DictReader(f))

    kept, dropped, unscored = [], 0, 0
    for r in recs:
        m = median.get(r["filename"])
        if m is None:
            unscored += 1
            continue
        if m <= args.max_median:
            kept.append(r)
        else:
            dropped += 1

    params = filter_params(args.crops, args.max_median, provenance)
    out_dir = CROPS_BASE / out_slice_name(params)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = out_dir / "manifest.csv"
    with out_manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "image_id", "category"])
        for r in kept:
            link_or_copy(src_dir / r["filename"], out_dir / r["filename"])
            w.writerow([r["filename"], r["image_id"], r["category"]])

    record = {
        **params,
        "n_in": len(recs),
        "n_kept": len(kept),
        "n_dropped": dropped,
        "n_unscored_skipped": unscored,
        "drop_frac": round(dropped / max(1, len(recs)), 4),
    }
    (out_dir / "filter_config.json").write_text(json.dumps(record, indent=2))

    print(f"source {args.crops}: {len(recs)} crops")
    print(f"  kept {len(kept)} (orient_median <= {args.max_median:g}), dropped {dropped}, unscored {unscored}")
    print(f"  -> {out_dir}")
    print(f"  config -> {out_dir / 'filter_config.json'}")
    print(f"\nNext: python scripts/build_hf_dataset.py --crops {out_dir.name} --no-captcha")


if __name__ == "__main__":
    main()
