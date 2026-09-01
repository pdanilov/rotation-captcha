"""Download the real captcha sets from public GitHub repos into data/raw/.

Two sources, both Baidu-style rotation captchas:

- labeled   -> data/raw/real_caps_labeled/   (200 imgs, 152x152) from
  yixiaowang2001/rotate-captcha-solver. The clockwise angle is encoded in the
  filename (e.g. `100_rot60_00_100.png` -> 60 deg); we also write a labels.csv.
  This is the domain-gap evaluation set.
- unlabeled -> data/raw/real_caps_unlabeled/ (30 imgs, 350x350) from
  chencchen/RotateCaptchaBreak. Timestamp filenames, no angle -> for domain
  adaptation / pseudo-labeling / qualitative inspection.

    python scripts/fetch_real_caps.py               # both
    python scripts/fetch_real_caps.py --only labeled
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests

RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"
API_TREE = "https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
EXTS = (".jpg", ".jpeg", ".png")

ROOT = Path(__file__).resolve().parents[1]


# The clockwise angle is encoded in the filename of the yixiaowang2001 captchas,
# e.g. `100_rot60_00_100.png` -> 60 deg. This convention is specific to that
# source, so the parser lives with the source rather than in the shared library.
_ROT_RE = re.compile(r"rot(\d+(?:\.\d+)?)")


def parse_yixiaowang_angle(name: str) -> float | None:
    m = _ROT_RE.search(name)
    return float(m.group(1)) % 360.0 if m else None


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    branch: str
    subdir: str
    out: Path
    # None -> unlabeled; otherwise maps a filename to its clockwise angle.
    label_fn: Callable[[str], float | None] | None = None


SOURCES = {
    "labeled": Source(
        name="labeled",
        repo="yixiaowang2001/rotate-captcha-solver",
        branch="main",
        subdir="caps/raw_labeled_caps",
        out=ROOT / "data" / "raw" / "real_caps_labeled",
        label_fn=parse_yixiaowang_angle,
    ),
    "unlabeled": Source(
        name="unlabeled",
        repo="chencchen/RotateCaptchaBreak",
        branch="master",
        subdir="data/baiduCaptcha",
        out=ROOT / "data" / "raw" / "real_caps_unlabeled",
    ),
}


def list_files(src: Source) -> list[str]:
    r = requests.get(API_TREE.format(repo=src.repo, branch=src.branch), timeout=30)
    r.raise_for_status()
    return sorted(
        n["path"] for n in r.json()["tree"] if n["path"].startswith(src.subdir) and n["path"].lower().endswith(EXTS)
    )


def download(src: Source, path: str) -> str:
    dest = src.out / Path(path).name
    if not dest.exists():
        url = RAW.format(repo=src.repo, branch=src.branch, path=path)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest.name


def fetch(src: Source) -> None:
    src.out.mkdir(parents=True, exist_ok=True)
    paths = list_files(src)
    print(f"[{src.name}] found {len(paths)} captchas; downloading to {src.out} ...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        names = list(ex.map(lambda p: download(src, p), paths))

    if src.label_fn is None:
        print(f"[{src.name}] done: {len(names)} files")
        return

    manifest = src.out / "labels.csv"
    n_bad = 0
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "angle_cw"])
        for name in sorted(names):
            angle = src.label_fn(name)
            if angle is None:
                n_bad += 1
                continue
            w.writerow([name, angle])
    print(
        f"[{src.name}] done: {len(names)} files, labels -> {manifest}"
        + (f" ({n_bad} without parseable angle)" if n_bad else "")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(SOURCES), default=None, help="fetch just one source (default: both)")
    args = ap.parse_args()
    for name in [args.only] if args.only else list(SOURCES):
        fetch(SOURCES[name])


if __name__ == "__main__":
    main()
