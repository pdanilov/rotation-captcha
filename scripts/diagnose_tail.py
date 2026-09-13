"""Diagnose where a model's real-test error comes from (the MAE tail).

Three views on the 200 labeled real caps:
  A. error histogram + flip mass  — is the tail 180-deg flips or broad noise?
  B. confidence x correctness      — split err>45 by R (median): honest-unsure vs
                                     confident-and-wrong (the systematic, fixable part).
  C. self-orientability classes    — a baseline-invariant thermometer: rotate each cap
                                     by known angles, recover the base angle (pred-a),
                                     measure its circular consistency R_c and the
                                     double-angle R_2. orientable (R_c high) /
                                     polarity-ambiguous (axis clear, 2 modes 180 apart) /
                                     scattered (no up).

    python scripts/diagnose_tail.py --checkpoint runs/<run>/model.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rotcaptcha.augment import build_eval_transform
from rotcaptcha.geometry import DEFAULT_SIZE, make_disc_sample
from rotcaptcha.heads import ClassificationHead
from rotcaptcha.metrics import circular_abs_error
from rotcaptcha.model import build_model

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "raw" / "captcha" / "baidu" / "labeled_caps"


def infer_n_bins(state_dict: dict) -> int:
    for key, val in state_dict.items():
        if key.endswith("classifier.1.weight") and val.ndim == 2:
            return int(val.shape[0])
    raise SystemExit("Could not infer n_bins from checkpoint.")


def rank(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r


def rlen(deg: np.ndarray) -> float:
    r = np.deg2rad(deg)
    return float(np.hypot(np.cos(r).mean(), np.sin(r).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--model-name", default="microsoft/resnet-34")
    ap.add_argument("--n-angles", type=int, default=12)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu")
    nb = infer_n_bins(sd)
    head = ClassificationHead(n_bins=nb)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model_name, nb)
    model.load_state_dict(sd)
    model.eval().to(dev)
    tf = build_eval_transform(args.img_size)

    rows = list(csv.DictReader((REAL / "labels.csv").open(newline="")))
    angles = [i / args.n_angles * 360.0 for i in range(args.n_angles)]
    pred, conf, true, rc, r2 = [], [], [], [], []
    with torch.no_grad():
        for r in rows:
            img = Image.open(REAL / r["filename"]).convert("RGB")
            disc = tf(make_disc_sample(img, 0.0, size=DEFAULT_SIZE)).unsqueeze(0).to(dev)
            lg = model(disc).logits
            pred.append(float(head.decode(lg)[0]))
            conf.append(float(head.orientation_vec(lg.cpu()).norm(dim=-1)[0]))
            true.append(float(r["angle_cw"]) % 360.0)
            rec = []
            for a in angles:
                d = tf(make_disc_sample(img, a, size=DEFAULT_SIZE)).unsqueeze(0).to(dev)
                rec.append((float(head.decode(model(d).logits)[0]) - a) % 360.0)
            rec = np.array(rec)
            rc.append(rlen(rec))
            r2.append(rlen(2 * rec))
    pred, conf, true = np.array(pred), np.array(conf), np.array(true)
    rc, r2 = np.array(rc), np.array(r2)
    err = circular_abs_error(pred, true)

    print(f"model={args.checkpoint.parent.name}  n={len(err)}  MAE={err.mean():.1f}  median={np.median(err):.1f}")

    print("\n[A] error histogram")
    edges = [0, 5, 10, 20, 30, 45, 60, 90, 120, 150, 170, 180.01]
    h, _ = np.histogram(err, bins=edges)
    hm = max(h)
    for i in range(len(h)):
        print(f"  [{edges[i]:>5.0f},{edges[i + 1]:>5.0f}) {h[i]:>4}  {'#' * round(h[i] / hm * 40)}")
    print(
        f"  masses: <=10 {np.mean(err <= 10) * 100:.0f}%  45-135 {np.mean((err >= 45) & (err < 135)) * 100:.0f}%  "
        f"FLIP(>=150) {np.mean(err >= 150) * 100:.0f}%   err>45 = {np.sum(err[err > 45]) / err.sum() * 100:.0f}% of MAE"
    )

    print("\n[B] confidence x correctness (R median split)")
    med_conf = np.median(conf)
    hi = conf >= med_conf
    wrong = err > 45
    flip = err >= 150
    print(f"  high-R & right {np.sum(hi & ~wrong):>3}   high-R & wrong {np.sum(hi & wrong):>3}")
    print(f"  low-R  & right {np.sum(~hi & ~wrong):>3}   low-R  & wrong {np.sum(~hi & wrong):>3}")
    cw = hi & wrong
    print(f"  confident-and-wrong: {cw.sum()} (of them flips: {(cw & flip).sum()})")

    print("\n[C] self-orientability classes (baseline-invariant thermometer)")
    orientable = rc >= 0.8
    polarity = (~orientable) & (r2 >= 0.8)
    scattered = (~orientable) & (r2 < 0.8)
    for lab, m in [("orientable", orientable), ("polarity-ambiguous", polarity), ("scattered", scattered)]:
        if m.sum():
            print(
                f"  {lab:<20} n={m.sum():>3} ({m.mean() * 100:4.0f}%)  MAE={err[m].mean():5.1f}  "
                f"median={np.median(err[m]):5.1f}  flips={np.mean(err[m] >= 150) * 100:3.0f}%"
            )


if __name__ == "__main__":
    main()
