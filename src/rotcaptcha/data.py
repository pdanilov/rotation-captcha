"""Datasets and preprocessing (task-agnostic — the head builds the target).

Two dataset roles:
  - SyntheticRotationDataset: COCO object crops -> random clockwise rotation +
    disc mask, on the fly.
  - RealCaptchaDataset: pre-rendered Baidu discs with a known angle_cw label,
    used only for evaluation (never trained on).

Both yield `{"pixel_values": FloatTensor[C,H,W], "target": Tensor, "angle": float}`,
where `target = head.make_target(angle)` — its shape/meaning is the head's concern.
The image `transform` (PIL -> normalized tensor) is supplied by the caller; see
augment.py for the eval and augmented-train pipelines.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from .geometry import DEFAULT_SIZE, make_disc_sample
from .heads import AngleHead

ROOT = Path(__file__).resolve().parents[2]
HF = ROOT / "data" / "hf"

Transform = Callable[[Image.Image], torch.Tensor]


class SyntheticRotationDataset(Dataset):
    """COCO crops rotated by a random clockwise angle and masked to a disc.

    A fresh random angle is drawn per __getitem__, so each epoch sees every base
    image at a new orientation. Set deterministic=True to fix one angle per image
    (reproducible validation).
    """

    def __init__(
        self,
        split: str,
        head: AngleHead,
        transform: Transform,
        coco_slice: str = "val",
        disc_size: int = DEFAULT_SIZE,
        seed: int = 0,
        deterministic: bool = False,
    ):
        data_dir = str(HF / "crops" / coco_slice)
        self.ds = load_dataset("imagefolder", data_dir=data_dir, split=split)
        self.head = head
        self.disc_size = disc_size
        self.tf = transform
        self._base_seed = seed
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        base: Image.Image = self.ds[idx]["image"].convert("RGB")
        if self.deterministic:
            angle = random.Random((self._base_seed, idx).__hash__()).uniform(0.0, 360.0)
        else:
            angle = random.uniform(0.0, 360.0)
        disc = make_disc_sample(base, angle, size=self.disc_size)
        return {
            "pixel_values": self.tf(disc),
            "target": self.head.make_target(angle),
            "angle": angle,
        }


class RealCaptchaDataset(Dataset):
    """Pre-rendered real Baidu discs with a known clockwise angle (eval only)."""

    def __init__(self, subset: str, split: str, head: AngleHead, transform: Transform):
        self.ds = load_dataset("imagefolder", data_dir=str(HF / "captcha" / "baidu" / subset), split=split)
        self.head = head
        self.tf = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        img: Image.Image = row["image"].convert("RGB")
        angle = float(row["angle_cw"]) % 360.0
        return {
            "pixel_values": self.tf(img),
            "target": self.head.make_target(angle),
            "angle": angle,
        }


class UnlabeledPairDataset(Dataset):
    """Two independently-rotated views of an unlabeled real cap, for the label-free
    equivariance loss. Yields `{"v1", "v2", "delta"}` where delta = (r2 - r1) is the
    known clockwise angle *between* the views (the only supervision).
    """

    def __init__(
        self,
        split: str,
        transform: Transform,
        disc_size: int = DEFAULT_SIZE,
        seed: int = 0,
        deterministic: bool = False,
    ):
        data_dir = str(HF / "captcha" / "baidu" / "unlabeled_caps")
        self.ds = load_dataset("imagefolder", data_dir=data_dir, split=split)
        self.tf = transform
        self.disc_size = disc_size
        self._base_seed = seed
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        img: Image.Image = self.ds[idx]["image"].convert("RGB")
        rng = random.Random((self._base_seed, idx).__hash__()) if self.deterministic else random
        r1 = rng.uniform(0.0, 360.0)
        r2 = rng.uniform(0.0, 360.0)
        v1 = make_disc_sample(img, r1, size=self.disc_size)
        v2 = make_disc_sample(img, r2, size=self.disc_size)
        return {
            "v1": self.tf(v1),
            "v2": self.tf(v2),
            "delta": (r2 - r1) % 360.0,
        }


class UnlabeledRealDataset(Dataset):
    """Unlabeled real caps as-is; yields the image + its index (for the pseudo-label
    inference pass — the index lets us map predictions back to specific caps)."""

    def __init__(self, split: str, transform: Transform):
        data_dir = str(HF / "captcha" / "baidu" / "unlabeled_caps")
        self.ds = load_dataset("imagefolder", data_dir=data_dir, split=split)
        self.tf = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        img: Image.Image = self.ds[idx]["image"].convert("RGB")
        return {"pixel_values": self.tf(img), "idx": idx}


class PseudoLabeledDataset(Dataset):
    """Confident real caps with model-assigned pseudo angles (a self-training subset).

    Holds a mutable list of (dataset_index, pseudo_angle) set by `set_items` after
    each relabeling pass; trains on them with L_abs exactly like real labels.
    """

    def __init__(self, split: str, head: AngleHead, transform: Transform):
        data_dir = str(HF / "captcha" / "baidu" / "unlabeled_caps")
        self.ds = load_dataset("imagefolder", data_dir=data_dir, split=split)
        self.head = head
        self.tf = transform
        self.items: list[tuple[int, float]] = []

    def set_items(self, items: list[tuple[int, float]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        idx, angle = self.items[i]
        img: Image.Image = self.ds[idx]["image"].convert("RGB")
        return {"pixel_values": self.tf(img), "target": self.head.make_target(angle)}
