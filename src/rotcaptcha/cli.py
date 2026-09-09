"""Training CLI for rotation-angle estimation.

Trains a task head (classification or regression, see heads.py) on synthetic COCO
discs with L_abs, optionally adding a label-free equivariance loss L_eq on unlabeled
real captchas for domain adaptation (see docs/roadmap.md). Evaluates on the
synthetic validation split, the real Baidu labeled captchas, and — when equivariance
is on — a label-free equivariance metric on the held-out unlabeled real val split.

    rotcaptcha default                        # 72-bin CSL classification + augmentation
    rotcaptcha pseudo                         # + R-gated self-training on unlabeled real
    rotcaptcha regression                     # (sin, cos) regression
    rotcaptcha smoke                          # tiny sanity run
    rotcaptcha default --epochs 80 --lr 1e-4  # override any field
    rotcaptcha pseudo --note "..." --tags a b # tracker metadata

Each run logs to trackio (project "rotation-captcha") and saves its checkpoint under
runs/<config-slug>__<coolname>/model.pt. View runs with `trackio show`.
"""

from __future__ import annotations

import dataclasses
import random
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
import trackio
import tyro
from accelerate import Accelerator
from coolname import generate_slug
from torch.utils.data import DataLoader

from .augment import build_eval_transform, build_train_transform
from .config import CONFIGS, TrainConfig, config_slug
from .data import (
    PseudoLabeledDataset,
    RealCaptchaDataset,
    SyntheticRotationDataset,
    UnlabeledPairDataset,
    UnlabeledRealDataset,
)
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
def compute_pseudo_labels(model, loader, head: AngleHead, accelerator: Accelerator, frac: float):
    """Predict angles for every unlabeled cap, keep the top `frac` by resultant-length
    R (confidence). Returns [(dataset_index, pseudo_angle)] and the kept subset's mean R."""
    model.eval()
    all_logits, all_idx = [], []
    for batch in loader:
        logits = model(batch["pixel_values"]).logits
        logits, idx = accelerator.gather_for_metrics((logits, batch["idx"]))
        all_logits.append(logits.float().cpu())
        all_idx.append(idx.cpu())
    model.train()
    logits = torch.cat(all_logits)
    idx = torch.cat(all_idx).numpy()
    r = head.orientation_vec(logits).norm(dim=-1).numpy()  # circular concentration = confidence
    ang = head.decode(logits)
    order = np.argsort(-r)
    k = max(1, int(len(order) * frac))
    sel = order[:k]
    items = [(int(idx[i]), float(ang[i])) for i in sel]
    return items, float(r[sel].mean())


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

    # unique, human-readable run name: config slug + random coolname suffix
    run_name = f"{config_slug(cfg)}__{generate_slug(2)}"
    run_dir = cfg.out_dir / run_name
    ckpt_path = run_dir / "model.pt"
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        tracker_cfg = dataclasses.asdict(cfg)
        tracker_cfg["out_dir"] = str(cfg.out_dir)
        tracker_cfg["tags"] = list(cfg.tags)
        tracker_cfg |= {"run_name": run_name, "checkpoint": str(ckpt_path)}
        trackio.init(
            project="rotation-captcha",
            name=run_name,
            group=cfg.tags[0] if cfg.tags else None,
            config=tracker_cfg,
        )

    head = cfg.build_head()
    eval_tf = build_eval_transform(cfg.img_size)
    train_tf = build_train_transform(cfg.img_size, cfg.augment_strength) if cfg.augment else eval_tf
    train_ds = SyntheticRotationDataset("train", head, train_tf, coco_slice=cfg.coco_slice, seed=cfg.seed)
    syn_val_ds = SyntheticRotationDataset(
        "validation", head, eval_tf, coco_slice=cfg.coco_slice, seed=cfg.seed, deterministic=True
    )
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

    use_pseudo = cfg.lambda_pseudo > 0
    label_loader = pseudo_ds = pseudo_iter = None
    if use_pseudo:
        label_ds = UnlabeledRealDataset("train", eval_tf)  # clean pass for confident predictions
        label_loader = accelerator.prepare(DataLoader(label_ds, batch_size=cfg.batch_size, **dl_kw))
        pseudo_ds = PseudoLabeledDataset("train", head, train_tf)  # train on pseudo-labels w/ augmentation

    def save_ckpt() -> None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)

    best_metric, best_epoch, since_improve = float("inf"), 0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        eq_active = use_eq and epoch > cfg.eq_warmup_epochs

        pseudo_active = use_pseudo and epoch > cfg.pseudo_warmup_epochs
        if pseudo_active and (epoch - cfg.pseudo_warmup_epochs - 1) % cfg.pseudo_relabel_every == 0:
            items, r_mean = compute_pseudo_labels(model, label_loader, head, accelerator, cfg.pseudo_conf_frac)
            pseudo_ds.set_items(items)
            pl = DataLoader(pseudo_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, **dl_kw)
            pseudo_iter = cycle(accelerator.prepare(pl))
            if accelerator.is_main_process:
                print(f"[epoch {epoch:02d}] relabeled {len(items)} confident caps (mean R={r_mean:.2f})")

        running, running_eq, running_pseudo, seen = 0.0, 0.0, 0.0, 0
        for step, batch in enumerate(train_loader):
            if cfg.max_train_batches and step >= cfg.max_train_batches:
                break
            outputs = model(batch["pixel_values"]).logits
            loss = head.loss(outputs, batch["target"])
            l_eq = torch.zeros((), device=loss.device)
            if eq_active:
                l_eq = equivariance_loss(model, head, next(eq_iter))
                loss = loss + cfg.lambda_eq * l_eq
            l_pseudo = torch.zeros((), device=loss.device)
            if pseudo_active:
                pb = next(pseudo_iter)
                l_pseudo = head.loss(model(pb["pixel_values"]).logits, pb["target"])
                loss = loss + cfg.lambda_pseudo * l_pseudo
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            running_eq += float(l_eq)
            running_pseudo += float(l_pseudo)
            seen += 1

        syn = evaluate(model, syn_val_loader, head, accelerator)
        real = evaluate(model, real_loader, head, accelerator)
        eq = evaluate_equivariance(model, eq_val_loader, head, accelerator) if use_eq else None
        if accelerator.is_main_process:
            notes = ""
            if use_eq:
                notes += " eq_loss=warmup" if not eq_active else f" eq_loss={running_eq / max(1, seen):.4f}"
            if use_pseudo:
                notes += (
                    " pseudo_loss=warmup" if not pseudo_active else f" pseudo_loss={running_pseudo / max(1, seen):.4f}"
                )
            print(f"[epoch {epoch:02d}] loss={running / max(1, seen):.4f}{notes}")
            print(f"           synthetic-val | {format_metrics(syn)}")
            print(f"           real-test     | {format_metrics(real)}")
            if eq is not None:
                print(f"           unlabeled-eq  | eq_MAE={eq['eq_mae']:.2f} eq_median={eq['eq_median']:.2f}")

            logm = {"train/loss": running / max(1, seen)}
            if eq_active:
                logm["train/eq_loss"] = running_eq / max(1, seen)
            if pseudo_active:
                logm["train/pseudo_loss"] = running_pseudo / max(1, seen)
            logm |= {f"syn/{k}": v for k, v in syn.items() if k != "n"}
            logm |= {f"real/{k}": v for k, v in real.items() if k != "n"}
            if eq is not None:
                logm |= {"eq/mae": eq["eq_mae"], "eq/median": eq["eq_median"]}
            trackio.log(logm, step=epoch)

        # early stopping on synthetic-val median (identical across processes after
        # gather, so the stop decision is consistent). When on, keep the BEST ckpt.
        improved = syn["median"] < best_metric - cfg.min_delta
        if improved:
            best_metric, best_epoch, since_improve = syn["median"], epoch, 0
            if cfg.patience:
                save_ckpt()
        else:
            since_improve += 1
        if cfg.patience and since_improve >= cfg.patience:
            if accelerator.is_main_process:
                print(f"early stop @ epoch {epoch}: no syn-median improvement for {cfg.patience} epochs")
            break

    if not cfg.patience:  # no early stopping -> save the final model
        save_ckpt()
    if accelerator.is_main_process:
        best = f" (best syn-median {best_metric:.2f} @ epoch {best_epoch})" if cfg.patience else ""
        print(f"\nsaved checkpoint -> {ckpt_path}{best}")
        trackio.finish()


def main() -> None:
    cfg = tyro.extras.overridable_config_cli(CONFIGS)
    train(cfg)


if __name__ == "__main__":
    main()
