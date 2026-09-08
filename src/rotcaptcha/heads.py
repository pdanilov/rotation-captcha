"""Angle-task heads: the one piece that varies between formulations.

A head bundles the four task-specific concerns — output width, target
construction, loss, and decode-to-degrees — behind a common interface, so the
backbone, data pipeline, training loop, and metrics stay task-agnostic.

Implemented:
  - ClassificationHead: N angular bins + Circular Smooth Label (the baseline).
  - RegressionHead: (sin, cos) unit-vector regression (needs a warm start; see
    docs/roadmap.md).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from .labels import angle_to_vec, circular_smooth_label


def rotate_vec(u: torch.Tensor, delta_rad: torch.Tensor) -> torch.Tensor:
    """Rotate orientation vectors u=[B,2] (sin, cos) by angle delta (radians).

    Maps the orientation of angle a to that of a+delta:
    (sin a, cos a) -> (sin(a+delta), cos(a+delta)).
    """
    s, c = u[:, 0], u[:, 1]
    cd, sd = torch.cos(delta_rad), torch.sin(delta_rad)
    return torch.stack([s * cd + c * sd, -s * sd + c * cd], dim=-1)


class AngleHead(ABC):
    #: Number of raw outputs the backbone's final linear layer must produce.
    output_dim: int

    @abstractmethod
    def make_target(self, angle_deg: float) -> torch.Tensor:
        """Per-sample training target from the clockwise angle (degrees)."""

    @abstractmethod
    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scalar loss for a batch of raw model outputs [B, output_dim]."""

    @abstractmethod
    def decode(self, output: torch.Tensor | np.ndarray) -> np.ndarray:
        """Raw outputs [B, output_dim] -> clockwise degrees in [0, 360)."""

    @abstractmethod
    def orientation_vec(self, output: torch.Tensor) -> torch.Tensor:
        """Differentiable orientation vector [B, 2] (sin, cos) for equivariance.

        Same direction as `decode`, kept as a tensor so it can flow gradients.
        """


class ClassificationHead(AngleHead):
    """Angular-bin classification with a Circular Smooth Label soft target.

    Decodes with a circular soft-argmax (probability-weighted mean direction),
    which is seam-safe and gives sub-bin resolution.
    """

    def __init__(self, n_bins: int = 360, csl_sigma_bins: float = 2.0):
        self.n_bins = n_bins
        self.csl_sigma_bins = csl_sigma_bins
        self.output_dim = n_bins

    def make_target(self, angle_deg: float) -> torch.Tensor:
        return torch.from_numpy(circular_smooth_label(angle_deg, self.n_bins, self.csl_sigma_bins))

    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -(target * F.log_softmax(output, dim=-1)).sum(dim=-1).mean()

    def decode(self, output: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(output, torch.Tensor):
            output = output.detach().float().cpu().numpy()
        z = output - output.max(axis=1, keepdims=True)
        probs = np.exp(z)
        probs /= probs.sum(axis=1, keepdims=True)
        bin_angles = np.arange(self.n_bins) / self.n_bins * 2 * np.pi
        sin_v = probs @ np.sin(bin_angles)
        cos_v = probs @ np.cos(bin_angles)
        return np.degrees(np.arctan2(sin_v, cos_v)) % 360.0

    def orientation_vec(self, output: torch.Tensor) -> torch.Tensor:
        probs = output.softmax(dim=-1)
        angles = torch.arange(self.n_bins, device=output.device) / self.n_bins * 2 * math.pi
        sin_v = (probs * angles.sin()).sum(dim=-1)
        cos_v = (probs * angles.cos()).sum(dim=-1)
        return torch.stack([sin_v, cos_v], dim=-1)


class RegressionHead(AngleHead):
    """Unit-vector (sin, cos) regression; decode with atan2 (scale-invariant)."""

    output_dim = 2

    def make_target(self, angle_deg: float) -> torch.Tensor:
        return torch.tensor(angle_to_vec(angle_deg), dtype=torch.float32)

    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(F.normalize(output, dim=-1), target)

    def decode(self, output: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(output, torch.Tensor):
            output = output.detach().float().cpu().numpy()
        return np.degrees(np.arctan2(output[:, 0], output[:, 1])) % 360.0

    def orientation_vec(self, output: torch.Tensor) -> torch.Tensor:
        return F.normalize(output, dim=-1)
