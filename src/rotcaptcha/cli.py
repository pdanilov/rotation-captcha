"""Training CLI for rotation-angle estimation.

Trains a task head (classification or regression, see heads.py) on synthetic COCO
discs with L_abs, optionally adding a label-free equivariance loss L_eq on unlabeled
real captchas for domain adaptation (see docs/roadmap.md). Evaluates on the
synthetic validation split, the real Baidu labeled captchas, and — when equivariance
is on — a label-free equivariance metric on the held-out unlabeled real val split.

    rotcaptcha default                       # 360-bin CSL classification + augmentation
    rotcaptcha eqv                           # + equivariance on unlabeled real caps
    rotcaptcha regression                    # (sin, cos) regression
    rotcaptcha smoke                         # tiny sanity run
    rotcaptcha default --epochs 80 --lr 1e-4 # override any field
"""

from __future__ import annotations

import random
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .augment import build_eval_transform, build_train_transform
from .config import CONFIGS, TrainConfig
from .data import RealCaptchaDataset, SyntheticRotationDataset, UnlabeledPairDataset
from .heads import AngleHead, rotate_vec
from .metrics import angle_metrics, circular_abs_error, format_metrics
from .model import build_model


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def equivariance_loss(model, head: AngleHead, pair: dict) -> torch.Tensor:
    """L_eq: the two views' orientations must differ by exactly delta (label-free).

    Vectors are unit-normalized so the loss can't be trivially minimized by
    collapsing the orientation magnitude (a flat distribution) instead of matching
    the angle.
    """
    u1 = F.normalize(head.orientation_vec(model(pair["v1"]).logits), dim=-1)
    u2 = F.normalize(head.orientation_vec(model(pair["v2"]).logits), dim=-1)
    delta_rad = torch.deg2rad(pair["delta"].float())
    return F.mse_loss(u2, rotate_vec(u1, delta_rad))


@torch.no_grad()
def evaluate(model, loader, head: AngleHead, accelerator: Accelerator) -> dict[str, float]:
    model.eval()
    all_outputs, all_angles = [], []
    for batch in loader:
        outputs = model(batch["pixel_values"]).logits
        outputs, angles = accelerator.gather_for_metrics((outputs, batch["angle"]))
        all_outputs.append(outputs.float().cpu())
        all_angles.append(angles.float().cpu())
    model.train()
    pred_deg = head.decode(torch.cat(all_outputs))
    true_deg = torch.cat(all_angles).numpy() % 360.0
    return angle_metrics(pred_deg, true_deg)


@torch.no_grad()
def evaluate_equivariance(model, loader, head: AngleHead, accelerator: Accelerator) -> dict[str, float]:
    """Label-free: does the decoded angle *between* two views match the known delta?"""
    model.eval()
    out1, out2, deltas = [], [], []
    for batch in loader:
        o1 = model(batch["v1"]).logits
        o2 = model(batch["v2"]).logits
        o1, o2, d = accelerator.gather_for_metrics((o1, o2, batch["delta"]))
        out1.append(o1.float().cpu())
        out2.append(o2.float().cpu())
        deltas.append(d.float().cpu())
    model.train()
    a1 = head.decode(torch.cat(out1))
    a2 = head.decode(torch.cat(out2))
    pred_delta = (a2 - a1) % 360.0
    true_delta = torch.cat(deltas).numpy() % 360.0
    err = circular_abs_error(pred_delta, true_delta)
    return {"eq_mae": float(err.mean()), "eq_median": float(np.median(err)), "n": int(err.size)}


def train(cfg: TrainConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    accelerator = Accelerator(mixed_precision="bf16")
    use_eq = cfg.lambda_eq > 0

    head = cfg.build_head()
    eval_tf = build_eval_transform(cfg.img_size)
    train_tf = build_train_transform(cfg.img_size, cfg.augment_strength) if cfg.augment else eval_tf
    train_ds = SyntheticRotationDataset("train", head, train_tf, seed=cfg.seed)
    syn_val_ds = SyntheticRotationDataset("validation", head, eval_tf, seed=cfg.seed, deterministic=True)
    real_ds = RealCaptchaDataset("labeled_caps", "test", head, eval_tf)

    dl_kw = {"num_workers": cfg.num_workers, "pin_memory": True, "worker_init_fn": seed_worker}
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **dl_kw)
    syn_val_loader = DataLoader(syn_val_ds, batch_size=cfg.batch_size, **dl_kw)
    real_loader = DataLoader(real_ds, batch_size=cfg.batch_size, **dl_kw)

    model = build_model(cfg.model_name, head.output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    if cfg.max_train_batches:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_batches)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * steps_per_epoch)

    model, optimizer, scheduler, train_loader, syn_val_loader, real_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, syn_val_loader, real_loader
    )

    eq_iter = eq_val_loader = None
    if use_eq:
        eq_train_ds = UnlabeledPairDataset("train", eval_tf, seed=cfg.seed)
        eq_val_ds = UnlabeledPairDataset("validation", eval_tf, seed=cfg.seed, deterministic=True)
        eq_train_loader = DataLoader(eq_train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **dl_kw)
        eq_val_loader = DataLoader(eq_val_ds, batch_size=cfg.batch_size, **dl_kw)
        eq_train_loader, eq_val_loader = accelerator.prepare(eq_train_loader, eq_val_loader)
        eq_iter = cycle(eq_train_loader)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, running_eq, seen = 0.0, 0.0, 0
        for step, batch in enumerate(train_loader):
            if cfg.max_train_batches and step >= cfg.max_train_batches:
                break
            outputs = model(batch["pixel_values"]).logits
            loss = head.loss(outputs, batch["target"])
            l_eq = torch.zeros((), device=loss.device)
            if use_eq:
                l_eq = equivariance_loss(model, head, next(eq_iter))
                loss = loss + cfg.lambda_eq * l_eq
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            running_eq += float(l_eq)
            seen += 1

        syn = evaluate(model, syn_val_loader, head, accelerator)
        real = evaluate(model, real_loader, head, accelerator)
        if accelerator.is_main_process:
            eq_note = f" eq_loss={running_eq / max(1, seen):.4f}" if use_eq else ""
            print(f"[epoch {epoch:02d}] loss={running / max(1, seen):.4f}{eq_note}")
            print(f"           synthetic-val | {format_metrics(syn)}")
            print(f"           real-test     | {format_metrics(real)}")
        if use_eq:
            eq = evaluate_equivariance(model, eq_val_loader, head, accelerator)
            if accelerator.is_main_process:
                print(f"           unlabeled-eq  | eq_MAE={eq['eq_mae']:.2f} eq_median={eq['eq_median']:.2f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        aug = "aug" if cfg.augment else "noaug"
        tag = f"{cfg.task}_{aug}" + (f"_eq{cfg.lambda_eq:g}" if use_eq else "")
        ckpt = cfg.out_dir / f"{tag}_{cfg.model_name.replace('/', '_')}.pt"
        torch.save(accelerator.unwrap_model(model).state_dict(), ckpt)
        print(f"\nsaved checkpoint -> {ckpt}")


def main() -> None:
    cfg = tyro.extras.overridable_config_cli(CONFIGS)
    train(cfg)


if __name__ == "__main__":
    main()
