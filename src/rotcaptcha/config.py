"""Training configuration and named run presets (driven by tyro)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .heads import AngleHead, ClassificationHead, RegressionHead

ROOT = Path(__file__).resolve().parents[2]

Task = Literal["classification", "regression"]


@dataclass
class TrainConfig:
    # model
    model_name: str = "microsoft/resnet-34"
    """Pretrained backbone from the transformers hub (pretrained is mandatory)."""
    img_size: int = 224

    # task head
    task: Task = "classification"
    n_bins: int = 360
    """Classification only: number of angular bins over [0, 360)."""
    csl_sigma_bins: float = 2.0
    """Classification only: CSL label-smoothing width, in bins (~degrees at n_bins=360)."""

    # domain augmentation (training synthetic set only; eval is never augmented)
    augment: bool = True
    augment_strength: float = 1.0
    """Scales augmentation probabilities/magnitudes in [0, 1]."""

    # equivariance domain adaptation on unlabeled real caps (0 = off)
    lambda_eq: float = 0.0
    """Weight of the label-free equivariance loss L_eq added to L_abs."""
    eq_warmup_epochs: int = 0
    """Epochs of synthetic-only (L_abs) before enabling L_eq, to build a solid
    absolute anchor before equivariance can drift the real-domain frame."""

    # optimization
    epochs: int = 40
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 0

    # bookkeeping
    max_train_batches: int = 0
    """Cap batches per epoch for a smoke test (0 = full epoch)."""
    out_dir: Path = ROOT / "runs" / "phase0"

    def build_head(self) -> AngleHead:
        if self.task == "classification":
            return ClassificationHead(self.n_bins, self.csl_sigma_bins)
        return RegressionHead()


CONFIGS: dict[str, tuple[str, TrainConfig]] = {
    "default": ("ResNet-34, 360-bin CSL classification + augmentation, 40 epochs", TrainConfig()),
    "noaug": ("default without domain augmentation (Phase-0 A/B baseline)", TrainConfig(augment=False)),
    "eqv": (
        "default + warm-started, small-lambda equivariance on unlabeled real caps",
        TrainConfig(lambda_eq=0.05, eq_warmup_epochs=20),
    ),
    "regression": ("ResNet-34, (sin, cos) regression, 40 epochs", TrainConfig(task="regression")),
    "smoke": (
        "tiny sanity run (1 epoch, 3 batches)",
        TrainConfig(epochs=1, max_train_batches=3, batch_size=16, num_workers=2),
    ),
    "long": ("80-epoch classification run", TrainConfig(epochs=80)),
}
