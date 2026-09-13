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

    # data
    coco_slice: str = "val"
    """Which COCO crop slice to train on (subdir under data/hf/crops/, produced
    by crop_coco_objects.py + build_hf_dataset.py). Logged, so each run records its data."""

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
    pseudo_conf_frac: float = 0.5
    """Fraction of unlabeled caps to keep, ranked by resultant-length R (confidence).
    Swept: 0.5 beat 0.25 and 0.75 on real-test (acc@10 0.445 vs 0.404/0.421)."""
    pseudo_relabel_every: int = 5
    """Re-run the pseudo-labeling pass every N epochs with the improved model."""

    # optimization
    epochs: int = 40
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 0

    # weight EMA (0 = off): eval/checkpoint use the averaged weights instead of the
    # raw ones. Tames the epoch-to-epoch real-test swing so the saved model sits near
    # the mean, not on a random syn-selected epoch. 0.999 ~= a few-epoch half-life here.
    ema_decay: float = 0.999

    # early stopping (0 = off; run all `epochs` and save the last checkpoint)
    patience: int = 0
    """Stop if synthetic-val median hasn't improved for this many epochs; when on,
    the *best* (lowest syn-val median) checkpoint is kept instead of the last."""
    min_delta: float = 0.0
    """Minimum syn-val median improvement (deg) that counts as progress."""

    # run metadata (not hyperparameters; logged to the tracker, ignored by training)
    note: str = ""
    """Free-text 'what is this experiment about', logged to the tracker."""
    tags: tuple[str, ...] = ()
    """Short labels for filtering/grouping in the tracker (first tag -> trackio group)."""

    # bookkeeping
    trackio: bool = True
    """Log this run to trackio (--no-trackio to disable). Off for throwaway runs
    (smoke tests) so they don't pollute the dashboard/history."""
    max_train_batches: int = 0
    """Cap batches per epoch for a smoke test (0 = full epoch)."""
    out_dir: Path = ROOT / "runs"

    def build_head(self) -> AngleHead:
        if self.task == "classification":
            return ClassificationHead(self.n_bins, self.csl_sigma_bins)
        return RegressionHead()


def config_slug(cfg: TrainConfig) -> str:
    """Short human-readable stem describing the config (hyperparameters only, no
    metadata). Combined with a random coolname suffix for a collision-free run name."""
    parts = ["cls", f"{cfg.n_bins}b"] if cfg.task == "classification" else ["reg"]
    parts.append("aug" if cfg.augment else "noaug")
    if cfg.lambda_eq > 0:
        parts.append(f"eq{cfg.lambda_eq:g}")
    if cfg.lambda_pseudo > 0:
        parts.append(f"pseudo{cfg.pseudo_conf_frac:g}")
    return "-".join(parts)


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
        "tiny sanity run (1 epoch, 3 batches; not logged to trackio)",
        TrainConfig(epochs=1, max_train_batches=3, batch_size=16, num_workers=2, trackio=False),
    ),
    "long": ("80-epoch classification run", TrainConfig(epochs=80)),
}
