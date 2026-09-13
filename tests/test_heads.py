"""Round-trip sanity: make_target(angle) should decode back to ~angle."""

import math

import numpy as np
import pytest
import torch

from rotcaptcha.heads import ClassificationHead, RegressionHead, rotate_vec
from rotcaptcha.labels import circular_diff

ANGLES = [0.0, 1.0, 37.0, 179.5, 180.0, 270.0, 359.0]
HEADS = [RegressionHead(), ClassificationHead(n_bins=360, csl_sigma_bins=2.0)]


def _logits_for(head, angle):
    # A peaked "logits" tensor whose decode is ~angle (CSL for classification,
    # the raw vector for regression), shaped [1, output_dim].
    if isinstance(head, RegressionHead):
        return head.make_target(angle).unsqueeze(0)
    return torch.log(head.make_target(angle).clamp_min(1e-12)).unsqueeze(0)


@pytest.mark.parametrize("angle", ANGLES)
def test_regression_roundtrip(angle):
    head = RegressionHead()
    target = head.make_target(angle).unsqueeze(0)
    (decoded,) = head.decode(target)
    assert circular_diff(decoded, angle) < 1e-3


@pytest.mark.parametrize("angle", ANGLES)
def test_classification_roundtrip(angle):
    # Feeding the CSL soft target as logits should soft-argmax back near the angle
    # (within a bin), including across the 0/360 seam.
    head = ClassificationHead(n_bins=360, csl_sigma_bins=2.0)
    logits = torch.log(head.make_target(angle).clamp_min(1e-12)).unsqueeze(0)
    (decoded,) = head.decode(logits)
    assert circular_diff(decoded, angle) < 1.0


def test_output_dims():
    assert ClassificationHead(n_bins=720).output_dim == 720
    assert RegressionHead().output_dim == 2


def test_decode_accepts_numpy_and_torch():
    head = RegressionHead()
    vec = np.array([[0.0, 1.0]], dtype=np.float32)  # (sin, cos) of 0 deg
    assert circular_diff(head.decode(vec)[0], 0.0) < 1e-3
    assert circular_diff(head.decode(torch.from_numpy(vec))[0], 0.0) < 1e-3


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("delta", [0.0, 10.0, 90.0, 200.0, 359.0])
def test_equivariance_rotate_vec(head, angle, delta):
    # Rotating the orientation vector of `angle` by `delta` should decode to
    # angle + delta (mod 360) — the identity L_eq enforces, incl. across the seam.
    u = head.orientation_vec(_logits_for(head, angle))
    u_rot = rotate_vec(u, torch.tensor([math.radians(delta)]))[0]
    # orientation_vec is always (sin, cos), so decode it with atan2 directly
    decoded = math.degrees(math.atan2(float(u_rot[0]), float(u_rot[1]))) % 360.0
    assert circular_diff(decoded, angle + delta) < 1.0
