"""
Обучение AnimeIFNet (бывший trainAnimeIFNet_v8_2.py).

Curriculum по подсказкам художника — без него обучение нестабильно:
    A (эп. 1–10)   sketch = 0, ROI = весь кадр — удерживаем качество V8
    B (эп. 11–20)  скетч с dropout 70% → 50%, ROI = весь кадр
    C (эп. 21–30)  скетч с dropout 50%, ROI из датасета
    D (эп. 31+)    то же, шлифовка всех четырёх режимов

Валидация всегда без подсказок (sketch = 0, ROI = весь кадр), поэтому
val PSNR напрямую сравним с базовой V8.

Запуск: см. scripts/train.py.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from animeifnet.data import MODEL_INPUT_KEYS, AnimeTripletDataset, split_indices
from animeifnet.losses import AnimeIFNetLoss, LossConfig, psnr
from animeifnet.models import AnimeIFNet


# ═══════════════════════════════════════════════════════════════
# Конфигурация
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    data: str = ""
    save_dir: str = "checkpoints/v8_2"
    init_v8: Optional[str] = None       # инициализация из чекпоинта V8 (рекомендуется)
    resume: Optional[str] = None

    epochs: int = 40
    batch: int = 4
    lr: float = 5e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    grad_clip: float = 0.5
    crop: int = 448
    val_fraction: float = 0.1
    workers: int = 6
    seed: int = 42
    amp: bool = True                    # bf16 autocast на CUDA

    residual_scale: float = 0.7
    corr_levels: int = 4
    corr_radius: int = 5

    # Curriculum: последние эпохи фаз A, B, C
    phase_a_end: int = 10
    phase_b_end: int = 20
    phase_c_end: int = 30

    save_every: int = 5
    limit_train_batches: int = 0        # >0 — только для быстрой проверки
    limit_val_batches: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════
# Curriculum
# ═══════════════════════════════════════════════════════════════

class HintCurriculum:
    """Сколько подсказок модель видит на каждой эпохе."""

    def __init__(self, phase_a_end: int, phase_b_end: int, phase_c_end: int):
        self.a, self.b, self.c = phase_a_end, phase_b_end, phase_c_end

    def phase(self, epoch: int) -> str:
        if epoch <= self.a:
            return "A"
        if epoch <= self.b:
            return "B"
        return "C" if epoch <= self.c else "D"

    def sketch_dropout(self, epoch: int) -> float:
        phase = self.phase(epoch)
        if phase == "A":
            return 1.0
        if phase == "B":  # линейно 70% → 50%
            progress = (epoch - self.a) / max(self.b - self.a, 1)
            return 0.7 - 0.2 * progress
        return 0.5

    def apply(self, sketch: torch.Tensor, roi: torch.Tensor, epoch: int):
        """Обнуляет скетч у части батча и до фазы C заменяет ROI на весь кадр."""
        p = self.sketch_dropout(epoch)
        keep = (torch.rand(sketch.size(0), 1, 1, 1, device=sketch.device) >= p).to(sketch.dtype)
        roi = roi if self.phase(epoch) in ("C", "D") else torch.ones_like(roi)
        return sketch * keep, roi


# ═══════════════════════════════════════════════════════════════
# Помощники
# ═══════════════════════════════════════════════════════════════

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init(worker_id: int):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def build_loaders(cfg: TrainConfig):
    probe = AnimeTripletDataset(cfg.data, crop_size=cfg.crop, augment=False, verbose=False)
    train_idx, val_idx = split_indices(len(probe), cfg.val_fraction, cfg.seed)
    # sketch_dropout=0: скетчем управляет curriculum, иначе dropout был бы двойным
    train_ds = AnimeTripletDataset(cfg.data, cfg.crop, augment=True, indices=train_idx)
    val_ds = AnimeTripletDataset(cfg.data, cfg.crop, augment=False, indices=val_idx)
    common = dict(batch_size=cfg.batch, num_workers=cfg.workers,
                  pin_memory=cfg.device.startswith("cuda"), worker_init_fn=_worker_init,
                  persistent_workers=cfg.workers > 0)
    return (DataLoader(train_ds, shuffle=True, drop_last=True, **common),
            DataLoader(val_ds, shuffle=False, **common))


def build_model(cfg: TrainConfig, device) -> AnimeIFNet:
    kwargs = dict(residual_scale=cfg.residual_scale, corr_levels=cfg.corr_levels,
                  corr_radius=cfg.corr_radius, freeze_backbone=True, splat_feat_ch=16)
    if cfg.init_v8 and not cfg.resume:
        print(f"[init] from V8 checkpoint: {cfg.init_v8}")
        return AnimeIFNet.from_v8_checkpoint(cfg.init_v8, device=device, **kwargs).to(device)
    return AnimeIFNet(**kwargs).to(device)


def lr_lambda(cfg: TrainConfig):
    """Линейный warmup, затем косинус до 1e-6."""
    def fn(e):
        if e < cfg.warmup_epochs:
            return (e + 1) / cfg.warmup_epochs
        prog = (e - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
        return max(0.5 * (1 + math.cos(math.pi * prog)), 1e-6 / cfg.lr)
    return fn


def _to_device(batch: Dict[str, torch.Tensor], device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _autocast(cfg, device):
    return torch.autocast(device.type, dtype=torch.bfloat16,
                          enabled=cfg.amp and device.type == "cuda")


@torch.no_grad()
def validate(model, loader, cfg: TrainConfig, device) -> float:
    """Средний PSNR без подсказок (sketch = 0, ROI = весь кадр)."""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if cfg.limit_val_batches and i >= cfg.limit_val_batches:
            break
        b = _to_device(batch, device)
        with _autocast(cfg, device):
            out = model(*(b[k] for k in MODEL_INPUT_KEYS), t=0.5)
        for j in range(out["pred"].size(0)):
            total += psnr(out["pred"][j], b["it"][j])
            n += 1
    return total / max(n, 1)


# ═══════════════════════════════════════════════════════════════
# Цикл обучения
# ═══════════════════════════════════════════════════════════════

def train(cfg: TrainConfig, loss_cfg: Optional[LossConfig] = None):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(cfg)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} "
          f"(непересекающиеся сцены)")

    model = build_model(cfg, device)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_all:,} params ({n_train:,} trainable)")

    loss_cfg = loss_cfg or LossConfig()
    criterion = AnimeIFNetLoss(loss_cfg).to(device)
    curriculum = HintCurriculum(cfg.phase_a_end, cfg.phase_b_end, cfg.phase_c_end)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))

    start_epoch, best = 1, -math.inf
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        try:  # чекпоинты старого train-скрипта хранят состояние под "optimizer_g"
            opt.load_state_dict(ck.get("optimizer", ck.get("optimizer_g")))
            sched.load_state_dict(ck["scheduler"])
        except (ValueError, KeyError, TypeError) as e:
            print(f"[WARN] optimizer/scheduler state not restored ({e}); "
                  f"начинаем их с нуля")
            for _ in range(ck["epoch"]):
                sched.step()
        start_epoch = ck["epoch"] + 1
        best = ck.get("best_psnr", ck.get("val_psnr", -math.inf))
        print(f"Resumed from epoch {ck['epoch']}, best {best:.2f} dB")

    print(f"\n{'═' * 62}\n  AnimeIFNet training | batch={cfg.batch} crop={cfg.crop} "
          f"epochs={cfg.epochs}\n  curriculum: A[1-{cfg.phase_a_end}] "
          f"B[{cfg.phase_a_end + 1}-{cfg.phase_b_end}] "
          f"C[{cfg.phase_b_end + 1}-{cfg.phase_c_end}] D[{cfg.phase_c_end + 1}+]\n"
          f"  validation: sketch=0, ROI=full\n{'═' * 62}\n")

    log_path = save_dir / "train_log.jsonl"
    for epoch in range(start_epoch, cfg.epochs + 1):
        phase = curriculum.phase(epoch)
        model.train()
        t0 = time.time()
        running: Dict[str, float] = {}
        n_steps, n_skipped = 0, 0

        for i, batch in enumerate(train_loader):
            if cfg.limit_train_batches and i >= cfg.limit_train_batches:
                break
            b = _to_device(batch, device)
            sketch, roi = curriculum.apply(b["sketch"], b["roi"], epoch)

            opt.zero_grad(set_to_none=True)
            with _autocast(cfg, device):
                out = model(*(b[k] for k in MODEL_INPUT_KEYS), t=0.5,
                            sketch_hint=sketch, roi_mask=roi)
                loss, parts = criterion(out, b["it"], b["flow_fwd"], b["flow_bwd"],
                                        roi, epoch, t=0.5)

            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            n_steps += 1
            running["loss"] = running.get("loss", 0.0) + loss.item()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v.item()

        sched.step()
        val_psnr = validate(model, val_loader, cfg, device)
        avg = {k: v / max(n_steps, 1) for k, v in running.items()}
        lr = opt.param_groups[0]["lr"]

        print(f"ep {epoch:03d}/{cfg.epochs} [{phase}] | loss={avg.get('loss', float('nan')):.4f} | "
              f"val_psnr={val_psnr:.2f}dB | lr={lr:.2e} | {time.time() - t0:.0f}s"
              + (f" | skipped {n_skipped} non-finite" if n_skipped else ""))
        print("  " + " | ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items()) if k != "loss"))

        is_best = val_psnr > best
        best = max(best, val_psnr)
        ckpt = {
            "epoch": epoch, "phase": phase,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "val_psnr": val_psnr, "best_psnr": best,
            "config": asdict(cfg), "loss_config": loss_cfg.to_dict(),
        }
        torch.save(ckpt, save_dir / "last.pth")
        if is_best:
            torch.save(ckpt, save_dir / "best.pth")
            print(f"  ★ best → {val_psnr:.2f} dB")
        if cfg.save_every and epoch % cfg.save_every == 0:
            torch.save(ckpt, save_dir / f"ep{epoch:03d}.pth")

        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "phase": phase, "val_psnr": val_psnr,
                                "lr": lr, "skipped": n_skipped, **avg}) + "\n")

        if epoch == cfg.phase_a_end:
            print("  ── Фаза A завершена: val PSNR должен быть ≈ V8 (просадка > 0.3 dB — "
                  "проверь инициализацию из V8).")
        elif epoch == cfg.phase_b_end:
            print("  ── Фаза B завершена: PSNR без скетча не должен заметно упасть.")

    print(f"\nDone. Best val PSNR (sketch=0, ROI=full): {best:.2f} dB → {save_dir / 'best.pth'}")
    return best
