"""Circular-angle evaluation metrics (vectorized, numpy).

Angles are clockwise degrees in [0, 360). See `labels.circular_diff` for the
scalar version of the core wrap-around distance.
"""

from __future__ import annotations

import numpy as np


def circular_abs_error(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    """Smallest absolute angular difference (deg), elementwise, in [0, 180]."""
    d = np.abs((pred_deg - true_deg) % 360.0)
    return np.minimum(d, 360.0 - d)


def angle_metrics(
    pred_deg: np.ndarray,
    true_deg: np.ndarray,
    tolerances: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0),
    boundary_band_deg: float = 10.0,
) -> dict[str, float]:
    """Roadmap metric set: circular MAE, median, solved-at-tolerance curve, and
    boundary-band MAE (error restricted to targets near the 0/360 seam)."""
    pred_deg = np.asarray(pred_deg, dtype=float)
    true_deg = np.asarray(true_deg, dtype=float)
    err = circular_abs_error(pred_deg, true_deg)

    out: dict[str, float] = {
        "mae": float(err.mean()),
        "median": float(np.median(err)),
        "n": int(err.size),
    }
    for t in tolerances:
        out[f"acc@{t:g}"] = float((err <= t).mean())

    dist_to_seam = np.minimum(true_deg % 360.0, 360.0 - (true_deg % 360.0))
    near_seam = dist_to_seam <= boundary_band_deg
    out["boundary_mae"] = float(err[near_seam].mean()) if near_seam.any() else float("nan")
    return out


def format_metrics(m: dict[str, float]) -> str:
    acc = " ".join(f"{k}={v:.2f}" for k, v in m.items() if k.startswith("acc@"))
    return f"MAE={m['mae']:.2f} median={m['median']:.2f} boundary_MAE={m['boundary_mae']:.2f} [{acc}] (n={m['n']})"
