"""Angle label helpers with circular awareness.

Angles are clockwise degrees in [0, 360). We expose the common representations so
downstream training code can pick a formulation without re-deriving the math:

- raw degrees
- bin index for classification (with configurable bin count)
- Circular Smooth Label (CSL): a wrapped-Gaussian soft target over the bins, the
  classification baseline's label smoothing
- (sin, cos) unit vector  -> regression target (kept for later experiments)
"""

from __future__ import annotations

import math

import numpy as np


def angle_to_vec(angle_deg: float) -> tuple[float, float]:
    r = math.radians(angle_deg)
    return math.sin(r), math.cos(r)


def vec_to_angle(sin_v: float, cos_v: float) -> float:
    return math.degrees(math.atan2(sin_v, cos_v)) % 360.0


def angle_to_bin(angle_deg: float, n_bins: int = 360) -> int:
    return round(angle_deg / 360.0 * n_bins) % n_bins


def bin_to_angle(idx: int, n_bins: int = 360) -> float:
    return (idx % n_bins) / n_bins * 360.0


def circular_smooth_label(angle_deg: float, n_bins: int = 360, sigma_bins: float = 2.0) -> np.ndarray:
    """Wrapped-Gaussian soft label over `n_bins` bins (Circular Smooth Label).

    Puts a Gaussian bump of width `sigma_bins` around the true angle's bin, wrapping
    across the 0/360 seam, then normalizes to a probability distribution. sigma_bins
    is the label-smoothing width (in bins ~= degrees when n_bins=360).
    """
    centers = np.arange(n_bins, dtype=np.float64)
    true_bin = (angle_deg / 360.0 * n_bins) % n_bins
    d = np.abs(centers - true_bin)
    d = np.minimum(d, n_bins - d)  # circular distance in bins
    w = np.exp(-0.5 * (d / sigma_bins) ** 2)
    return (w / w.sum()).astype(np.float32)


def circular_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees, in [0, 180]."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)
