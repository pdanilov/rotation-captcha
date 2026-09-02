"""Round-trip sanity: make_target(angle) should decode back to ~angle."""

import numpy as np
import pytest
import torch

from rotcaptcha.heads import ClassificationHead, RegressionHead
from rotcaptcha.labels import circular_diff

ANGLES = [0.0, 1.0, 37.0, 179.5, 180.0, 270.0, 359.0]


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
