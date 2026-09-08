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
    n_bins: int = 72
    """Classification only: number of angular bins over [0, 360). 72 (5 deg/bin) beat
    both 360 (too fine to transfer) and 36 (too coarse) on real-test — see roadmap."""
    csl_sigma_bins: float = 1.0
    """Classification only: CSL label-smoothing width, in bins (~5 deg at n_bins=72)."""

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

    # pseudo-labeling / self-training on unlabeled real caps (0 = off)
    lambda_pseudo: float = 0.0
    """Weight of the pseudo-label L_abs on confident real caps."""
    pseudo_warmup_epochs: int = 20
    """Epochs of synthetic-only before the first pseudo-labeling pass (needs a
    usable anchor first — confidence must track correctness)."""
    pseudo_conf_frac: float = 0.25
    """Fraction of unlabeled caps to keep, ranked by resultant-length R (confidence)."""
    pseudo_relabel_every: int = 5
    """Re-run the pseudo-labeling pass every N epochs with the improved model."""

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
    "default": ("ResNet-34, 72-bin CSL classification + augmentation, 40 epochs", TrainConfig()),
    "noaug": ("default without domain augmentation (Phase-0 A/B baseline)", TrainConfig(augment=False)),
    "eqv": (
        "default + warm-started, small-lambda equivariance on unlabeled real caps",
        TrainConfig(lambda_eq=0.05, eq_warmup_epochs=20),
    ),
    "pseudo": (
        "default + R-gated self-training on unlabeled real caps (warm-start)",
        TrainConfig(lambda_pseudo=1.0),
    ),
    "regression": ("ResNet-34, (sin, cos) regression, 40 epochs", TrainConfig(task="regression")),
    "smoke": (
        "tiny sanity run (1 epoch, 3 batches)",
        TrainConfig(epochs=1, max_train_batches=3, batch_size=16, num_workers=2),
    ),
    "long": ("80-epoch classification run", TrainConfig(epochs=80)),
}
