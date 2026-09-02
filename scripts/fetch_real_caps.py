"""Download the Baidu rotation-captcha dataset from Kaggle and build the real-cap
sets, deduplicating the unlabeled pool by perceptual hash.

Source: kaggle `ericwang1011/baidu-captcha-dataset` — labeled_caps/ (200, angle in
filename e.g. `100_rot60_00_100.png`) + unlabeled_caps/ (1800). All 152x152 RGB.
Produces (under data/raw/captcha/baidu/, keeping the source's folder names):
  labeled_caps/    200 imgs + labels.csv   -> domain-gap eval set
  unlabeled_caps/  deduped survivors       -> domain adaptation

Unlabeled filtering (pHash, Hamming distance <= --phash-threshold):
  - drop any image matching a labeled/eval image  (prevents eval leakage)
  - drop near-duplicates within the unlabeled set (keep one representative)
Note: pHash is orientation-sensitive, so the *same base scene at a different angle*
is kept (that variety is useful) — only genuine near-identical images are dropped.

Credentials: needs Kaggle auth (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)
unless --local-dir points at an already-extracted dataset folder.

    python scripts/fetch_real_caps.py
    python scripts/fetch_real_caps.py --local-dir /path/to/extracted --phash-threshold 5
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# Keep the source's original folder names (labeled_caps / unlabeled_caps) under a
# per-provider dir so other captcha providers can slot in as siblings.
BAIDU = ROOT / "data" / "raw" / "captcha" / "baidu"
LABELED_OUT = BAIDU / "labeled_caps"
UNLABELED_OUT = BAIDU / "unlabeled_caps"

DATASET = "ericwang1011/baidu-captcha-dataset"

# The clockwise angle is encoded in the labeled filenames, e.g.
# `100_rot60_00_100.png` -> 60 deg. Convention is specific to this source.
_ROT_RE = re.compile(r"rot(\d+(?:\.\d+)?)")


def parse_angle(name: str) -> float | None:
    m = _ROT_RE.search(name)
    return float(m.group(1)) % 360.0 if m else None


# --- perceptual hash (DCT-based pHash, numpy-only) -------------------------------

_HASH_SIZE = 8  # 8x8 low-freq block -> 64-bit hash
_IMG_SIZE = 32  # downscale before DCT


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    return np.cos(np.pi * (2 * x + 1) * k / (2 * n))  # scale-agnostic (median cmp)


_DCT = _dct_matrix(_IMG_SIZE)


def phash(img: Image.Image) -> int:
    """64-bit DCT perceptual hash as an int."""
    a = np.asarray(img.convert("L").resize((_IMG_SIZE, _IMG_SIZE), Image.LANCZOS), float)
    dct = _DCT @ a @ _DCT.T
    low = dct[:_HASH_SIZE, :_HASH_SIZE]
    bits = (low > np.median(low)).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# --- dataset acquisition ---------------------------------------------------------


def download_dataset(dest: Path) -> Path:
    """Download + unzip the Kaggle dataset into `dest`; return that path."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # local: auth on demand

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {DATASET} from Kaggle ...")
    api.dataset_download_files(DATASET, path=str(dest), unzip=True, quiet=False)
    return dest


def _find_subdir(root: Path, name: str) -> Path:
    hits = [p for p in root.rglob(name) if p.is_dir()]
    if not hits:
        raise SystemExit(f"Could not find '{name}/' under {root}")
    return hits[0]


# --- build the two sets ----------------------------------------------------------


def build_labeled(src: Path) -> list[int]:
    """Copy labeled caps, write labels.csv, return their pHashes (for leak checks)."""
    if LABELED_OUT.exists():
        shutil.rmtree(LABELED_OUT)
    LABELED_OUT.mkdir(parents=True)

    hashes: list[int] = []
    rows: list[tuple[str, float]] = []
    n_bad = 0
    for p in sorted(src.glob("*.png")):
        angle = parse_angle(p.name)
        if angle is None:
            n_bad += 1
            continue
        shutil.copy2(p, LABELED_OUT / p.name)
        rows.append((p.name, angle))
        hashes.append(phash(Image.open(p)))

    with (LABELED_OUT / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "angle_cw"])
        w.writerows(rows)
    print(f"labeled: {len(rows)} imgs -> {LABELED_OUT}" + (f" ({n_bad} unparsed angles skipped)" if n_bad else ""))
    return hashes


def build_unlabeled(src: Path, eval_hashes: list[int], threshold: int) -> None:
    """Copy unlabeled caps, dropping eval-overlaps and near-duplicates (pHash)."""
    if UNLABELED_OUT.exists():
        shutil.rmtree(UNLABELED_OUT)
    UNLABELED_OUT.mkdir(parents=True)

    kept_hashes: list[int] = []
    n_total = n_leak = n_dup = n_kept = 0
    for p in sorted(src.glob("*.png")):
        n_total += 1
        h = phash(Image.open(p))
        if any(hamming(h, e) <= threshold for e in eval_hashes):
            n_leak += 1
            continue
        if any(hamming(h, k) <= threshold for k in kept_hashes):
            n_dup += 1
            continue
        shutil.copy2(p, UNLABELED_OUT / p.name)
        kept_hashes.append(h)
        n_kept += 1

    print(
        f"unlabeled: {n_total} scanned -> {n_kept} kept "
        f"(dropped {n_leak} eval-overlap, {n_dup} near-dup) -> {UNLABELED_OUT}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--local-dir", type=Path, default=None, help="use an already-extracted dataset dir instead of Kaggle"
    )
    ap.add_argument(
        "--phash-threshold", type=int, default=5, help="max Hamming distance (of 64) to treat images as duplicates"
    )
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = args.local_dir or download_dataset(Path(tmp))
        labeled_src = _find_subdir(root, "labeled_caps")
        unlabeled_src = _find_subdir(root, "unlabeled_caps")
        eval_hashes = build_labeled(labeled_src)
        build_unlabeled(unlabeled_src, eval_hashes, args.phash_threshold)


if __name__ == "__main__":
    main()
