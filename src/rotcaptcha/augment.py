"""Image preprocessing and (training-only) domain augmentation.

Two pipelines, both PIL-in / normalized-tensor-out:
  - build_eval_transform: deterministic resize + normalize (val and real-cap eval).
  - build_train_transform: eval pipeline + angle-preserving augmentation to close
    the synthetic->real appearance gap and prevent overfitting to COCO-specific
    low-level cues (see docs/roadmap.md).

Every augmentation here is orientation-preserving (photometric, degradation, or
centered scale/translation). No flips, transposes, or extra rotations — those would
corrupt the angle label.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(img_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_train_transform(img_size: int, strength: float = 1.0) -> v2.Compose:
    """Augmented pipeline. `strength` in [0, 1] scales probabilities/magnitudes."""
    s = strength
    return v2.Compose(
        [
            # scale/framing jitter (centered crop+resize; translation+zoom, no rotation)
            v2.RandomResizedCrop(img_size, scale=(1.0 - 0.2 * s, 1.0), ratio=(1.0, 1.0), antialias=True),
            # photometric
            v2.ColorJitter(brightness=0.3 * s, contrast=0.3 * s, saturation=0.3 * s, hue=0.05 * s),
            v2.RandomGrayscale(p=0.1 * s),
            v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3 * s),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3 * s),
            # compression artifacts (Baidu content was JPEG upstream; our renders are clean)
            v2.RandomApply([v2.JPEG(quality=(40, 90))], p=0.5 * s),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomApply([v2.GaussianNoise(sigma=0.03 * s)], p=0.3 * s),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            v2.RandomErasing(p=0.2 * s, scale=(0.02, 0.1)),
        ]
    )
