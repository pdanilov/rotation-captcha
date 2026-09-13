"""Merge several crop slices into one new raw slice (union of their crops).

Used to train on a mix of sources — e.g. the orientability-filtered COCO object
crops plus the filtered Places365 scenes — while keeping the merged set a single
recognizable named slice whose name flows into config.coco_slice -> the tracker.
Crops are hard-linked (no extra disk); manifests are concatenated. The output name
is a readable stem + a deterministic hash of the (sorted) source slice names.

    python scripts/merge_slices.py --crops A_slice B_slice --name coco+places
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
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", nargs="+", required=True, help="two or more source slice subdirs to merge")
    ap.add_argument("--name", required=True, help="readable stem for the merged slice (e.g. coco+places)")
    args = ap.parse_args()

    sources = sorted(args.crops)
    digest = hashlib.sha1(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:6]
    out_dir = CROPS_BASE / f"mix_{args.name}_cfg={digest}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    counts: dict[str, int] = {}
    with (out_dir / "manifest.csv").open("w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["filename", "image_id", "category"])
        for slice_name in sources:
            src_dir = CROPS_BASE / slice_name
            manifest = src_dir / "manifest.csv"
            if not manifest.exists():
                raise SystemExit(f"Missing {manifest}")
            n = 0
            for r in csv.DictReader(manifest.open(newline="")):
                fn = r["filename"]
                if not (src_dir / fn).exists():
                    continue
                if fn in seen:
                    raise SystemExit(f"Filename collision across slices: {fn}")
                seen.add(fn)
                link_or_copy(src_dir / fn, out_dir / fn)
                w.writerow([fn, r["image_id"], r["category"]])
                n += 1
            counts[slice_name] = n

    (out_dir / "merge_config.json").write_text(
        json.dumps({"sources": sources, "counts": counts, "n_total": sum(counts.values())}, indent=2)
    )
    print(f"merged {len(sources)} slices -> {out_dir}  ({sum(counts.values())} crops)")
    for s, n in counts.items():
        print(f"  {s}: {n}")
    print(f"\nNext: python scripts/build_hf_dataset.py --crops {out_dir.name} --no-captcha")


if __name__ == "__main__":
    main()
