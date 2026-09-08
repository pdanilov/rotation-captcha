"""Phase-0 baseline training CLI: synthetic-only rotation-angle estimation.

Train a task head (classification or regression, see heads.py) on synthetic COCO
discs; evaluate on the synthetic validation split and the real Baidu labeled
captchas. No domain adaptation — this measures the raw synthetic->real gap
(see docs/roadmap.md).

    rotcaptcha default                       # 360-bin CSL classification
    rotcaptcha regression                    # (sin, cos) regression
    rotcaptcha smoke                         # tiny sanity run
    rotcaptcha default --epochs 80 --lr 1e-4 # override any field
"""

from __future__ import annotations

import random

import numpy as np
import torch
import tyro
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .augment import build_eval_transform, build_train_transform
from .config import CONFIGS, TrainConfig
from .data import RealCaptchaDataset, SyntheticRotationDataset
from .heads import AngleHead
from .metrics import angle_metrics, format_metrics
from .model import build_model


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


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


def train(cfg: TrainConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    accelerator = Accelerator(mixed_precision="bf16")

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

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader):
            if cfg.max_train_batches and step >= cfg.max_train_batches:
                break
            outputs = model(batch["pixel_values"]).logits
            loss = head.loss(outputs, batch["target"])
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            seen += 1

        syn = evaluate(model, syn_val_loader, head, accelerator)
        real = evaluate(model, real_loader, head, accelerator)
        if accelerator.is_main_process:
            print(f"[epoch {epoch:02d}] loss={running / max(1, seen):.4f}")
            print(f"           synthetic-val | {format_metrics(syn)}")
            print(f"           real-test     | {format_metrics(real)}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        aug = "aug" if cfg.augment else "noaug"
        ckpt = cfg.out_dir / f"{cfg.task}_{aug}_{cfg.model_name.replace('/', '_')}.pt"
        torch.save(accelerator.unwrap_model(model).state_dict(), ckpt)
        print(f"\nsaved checkpoint -> {ckpt}")


def main() -> None:
    cfg = tyro.extras.overridable_config_cli(CONFIGS)
    train(cfg)


if __name__ == "__main__":
    main()
