"""
Датасеты AnimeIFNet (триплеты ATD-12K с предрасчитанными признаками).

Бывшие dataset_anime_v8.py + dataset_anime_v8_2.py, собранные в один модуль.

Структура сцены:
    scene/
    ├── frame1.png, frame2.png (эталон), frame3.png
    ├── raft_fwd.npy, raft_bwd.npy      [2, H, W]  RAFT-поток 1→3 и 3→1
    ├── seg0.npy, seg1.npy              [1, H, W]  маски ISNet
    ├── a2s_edge0.npy, a2s_edge1.npy    [1, H, W]  контуры Anime2Sketch
    ├── a2s_dt0.npy, a2s_dt1.npy        [1, H, W]  distance transform контуров
    └── a2s_gt.npy (опционально)        [1, H, W]  контуры эталона → sketch hint

Признаки готовит scripts/precompute_features.py.

Элемент датасета — dict:
    i0, i1, it                [3, S, S]  RGB в [-1, 1]
    edge0, edge1, dt0, dt1    [1, S, S]  в [0, 1]
    flow_fwd, flow_bwd        [2, S, S]  пиксели
    seg0, seg1                [1, S, S]  в [0, 1]
    sketch                    [1, S, S]  контуры эталона или нули
    roi                       [1, S, S]  маска области уточнения
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .augment import flip_flow_h, flip_flow_v, rotate_flow, rotate_spatial

BASE_FEATURES = ["raft_fwd.npy", "raft_bwd.npy", "seg0.npy", "seg1.npy"]
A2S_FEATURES = ["a2s_edge0.npy", "a2s_edge1.npy", "a2s_dt0.npy", "a2s_dt1.npy"]
A2S_GT = "a2s_gt.npy"

# Ключи в том порядке, в котором их принимает AnimeIFNet.forward
MODEL_INPUT_KEYS = ["i0", "i1", "edge0", "edge1", "dt0", "dt1",
                    "flow_fwd", "flow_bwd", "seg0", "seg1"]

# Каналы, которые меняются местами при обращении времени (0 ↔ 1)
_SWAP_PAIRS = [("i0", "i1"), ("flow_fwd", "flow_bwd"), ("seg0", "seg1"),
               ("edge0", "edge1"), ("dt0", "dt1")]
_FLOW_KEYS = ("flow_fwd", "flow_bwd")
_NEAREST_KEYS = ("edge0", "edge1", "sketch")  # бинарные карты — без размытия
_CLAMP_KEYS = ("seg0", "seg1", "edge0", "edge1", "dt0", "dt1", "sketch")

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),  # → [-1, 1]
])


def _find_frame(files: Dict[str, Path], stem: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg"):
        if f"{stem}{ext}" in files:
            return files[f"{stem}{ext}"]
    return None


def scan_scenes(root: Path, require: List[str]) -> Tuple[List[dict], Dict[str, int]]:
    """Сцены с тремя кадрами и всеми файлами из require."""
    samples, skipped = [], {"frames": 0, "base": 0, "a2s": 0}
    for scene in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        files = {f.name.lower(): f for f in scene.iterdir()}
        frames = [_find_frame(files, s) for s in ("frame1", "frame2", "frame3")]
        if not all(frames):
            skipped["frames"] += 1
            continue
        missing = [n for n in require if n not in files]
        if missing:
            skipped["base" if any(m in BASE_FEATURES for m in missing) else "a2s"] += 1
            continue
        samples.append({"f1": frames[0], "f2": frames[1], "f3": frames[2],
                        "dir": scene, "has_sketch": A2S_GT in files})
    return samples, skipped


def _load_npy(scene_dir: Path, name: str) -> torch.Tensor:
    return torch.from_numpy(np.load(str(scene_dir / name)).astype(np.float32))


def _load_sample(s: dict, load_sketch: bool) -> Dict[str, torch.Tensor]:
    d = s["dir"]
    sample = {
        "i0": _to_tensor(Image.open(s["f1"]).convert("RGB")),
        "it": _to_tensor(Image.open(s["f2"]).convert("RGB")),
        "i1": _to_tensor(Image.open(s["f3"]).convert("RGB")),
        "flow_fwd": _load_npy(d, "raft_fwd.npy"),
        "flow_bwd": _load_npy(d, "raft_bwd.npy"),
        "seg0": _load_npy(d, "seg0.npy"),
        "seg1": _load_npy(d, "seg1.npy"),
        "edge0": _load_npy(d, "a2s_edge0.npy"),
        "edge1": _load_npy(d, "a2s_edge1.npy"),
        "dt0": _load_npy(d, "a2s_dt0.npy"),
        "dt1": _load_npy(d, "a2s_dt1.npy"),
    }
    h, w = sample["i0"].shape[-2:]
    sample["sketch"] = (_load_npy(d, A2S_GT) if load_sketch and s["has_sketch"]
                        else torch.zeros(1, h, w))
    return sample


def _upscale_to_min_side(sample: Dict[str, torch.Tensor], min_side: int):
    """Если кадр меньше кропа — увеличить всё согласованно (поток × scale)."""
    h, w = sample["i0"].shape[-2:]
    if min(h, w) >= min_side:
        return sample
    scale = min_side / min(h, w)
    size = (max(min_side, int(h * scale)), max(min_side, int(w * scale)))
    out = {}
    for k, v in sample.items():
        if k in _NEAREST_KEYS:
            out[k] = F.interpolate(v[None], size=size, mode="nearest")[0]
        elif k in ("i0", "i1", "it"):
            out[k] = F.interpolate(v[None], size=size, mode="bicubic",
                                   align_corners=False)[0].clamp(-1, 1)
        else:
            out[k] = F.interpolate(v[None], size=size, mode="bilinear",
                                   align_corners=False)[0]
        if k in _FLOW_KEYS:
            out[k] = out[k] * torch.tensor([size[1] / w, size[0] / h]).view(2, 1, 1)
    return out


# ═══════════════════════════════════════════════════════════════
# Train / val
# ═══════════════════════════════════════════════════════════════

class AnimeTripletDataset(Dataset):
    """
    Кропы триплетов для обучения и валидации.

    Args:
        root_dir:       папка со сценами (например, ATD-12K/train)
        crop_size:      сторона квадратного кропа
        augment:        True — случайный кроп, отражения, повороты, обращение
                        времени и случайная ROI; False — центральный кроп,
                        ROI = весь кадр (детерминированная валидация)
        sketch_dropout: вероятность обнулить скетч-подсказку. В train.py — 0,
                        скетчем управляет curriculum.
        with_roi:       добавлять ли ключ roi
        indices:        подмножество сцен (для разбиения train/val)
    """

    # Доли режимов случайной ROI при augment=True
    ROI_MODES = (("full", 0.30), ("seg", 0.40), ("motion", 0.20), ("bbox", 0.10))

    def __init__(self, root_dir: str, crop_size: int = 448, augment: bool = True,
                 sketch_dropout: float = 0.0, with_roi: bool = True,
                 indices: Optional[List[int]] = None, verbose: bool = True):
        self.root = Path(root_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.sketch_dropout = sketch_dropout
        self.with_roi = with_roi

        samples, skipped = scan_scenes(self.root, BASE_FEATURES + A2S_FEATURES)
        self.samples = [samples[i] for i in indices] if indices is not None else samples

        if verbose:
            print(f"[AnimeTripletDataset] {len(self.samples)} scenes | crop={crop_size} | "
                  f"augment={augment} | sketch_dropout={sketch_dropout}")
            if skipped["base"]:
                print(f"  [WARN] {skipped['base']} scenes skipped — нет RAFT/ISNet признаков")
            if skipped["a2s"]:
                print(f"  [WARN] {skipped['a2s']} scenes skipped — нет Anime2Sketch признаков")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        load_sketch = random.random() >= self.sketch_dropout
        sample = _upscale_to_min_side(_load_sample(s, load_sketch), self.crop_size)
        sample = self._crop(sample)
        if self.augment:
            sample = self._augment(sample)
        for k in _CLAMP_KEYS:
            sample[k] = sample[k].clamp(0, 1)
        sample = {k: v.contiguous() for k, v in sample.items()}
        if self.with_roi:
            sample["roi"] = self._make_roi(sample)
        return sample

    # ── geometry ─────────────────────────────────────────────

    def _crop(self, sample):
        h, w = sample["i0"].shape[-2:]
        cs = self.crop_size
        if self.augment:
            top, left = random.randint(0, h - cs), random.randint(0, w - cs)
        else:
            top, left = (h - cs) // 2, (w - cs) // 2
        return {k: v[:, top:top + cs, left:left + cs] for k, v in sample.items()}

    @staticmethod
    def _augment(sample):
        if random.random() < 0.5:
            sample = {k: flip_flow_h(v) if k in _FLOW_KEYS else v.flip(-1)
                      for k, v in sample.items()}
        if random.random() < 0.15:
            sample = {k: flip_flow_v(v) if k in _FLOW_KEYS else v.flip(-2)
                      for k, v in sample.items()}
        if random.random() < 0.2:
            angle = random.choice([90, 180, 270])
            sample = {k: rotate_flow(v, angle) if k in _FLOW_KEYS else rotate_spatial(v, angle)
                      for k, v in sample.items()}
        # Обращение времени. Скетч не меняется: он всегда для среднего кадра.
        if random.random() < 0.5:
            for a, b in _SWAP_PAIRS:
                sample[a], sample[b] = sample[b], sample[a]
        return sample

    # ── ROI ──────────────────────────────────────────────────

    def _make_roi(self, sample) -> torch.Tensor:
        """
        ROI [1, S, S] в {0, 1}. Валидация — всегда весь кадр.
        Обучение: full 30% / маска персонажа 40% / область движения 20% /
        случайный прямоугольник 10%. Если маска вышла почти пустой — full.
        """
        h, w = sample["seg0"].shape[-2:]
        full = torch.ones(1, h, w)
        if not self.augment:
            return full

        r, acc, mode = random.random(), 0.0, "full"
        for m, p in self.ROI_MODES:
            acc += p
            if r < acc:
                mode = m
                break

        if mode == "seg":
            mask = (torch.maximum(sample["seg0"][:1], sample["seg1"][:1]) > 0.3).float()
        elif mode == "motion":
            mag = sample["flow_fwd"].float().pow(2).sum(0, keepdim=True).add(1e-8).sqrt()
            mask = (mag > mag.mean() + 0.5 * mag.std()).float()
        elif mode == "bbox":
            mask = torch.zeros(1, h, w)
            bh, bw = random.randint(h // 4, h), random.randint(w // 4, w)
            y, x = random.randint(0, h - bh), random.randint(0, w - bw)
            mask[:, y:y + bh, x:x + bw] = 1.0
        else:
            return full
        return mask if mask.sum() >= 64 else full


def split_indices(n: int, val_fraction: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Непересекающиеся индексы train / val (одна перестановка на оба)."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    nv = max(1, int(n * val_fraction))
    return sorted(perm[nv:]), sorted(perm[:nv])


# ═══════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════

PAD_MULTIPLE = 32


def pad_to_multiple(t: torch.Tensor, multiple: int = PAD_MULTIPLE):
    """Replicate-padding [C, H, W] → (padded, (pad_h, pad_w, orig_h, orig_w))."""
    _, h, w = t.shape
    ph, pw = (multiple - h % multiple) % multiple, (multiple - w % multiple) % multiple
    if ph or pw:
        t = F.pad(t[None], (0, pw, 0, ph), mode="replicate")[0]
    return t, (ph, pw, h, w)


def unpad_tensor(t: torch.Tensor, pad_info) -> torch.Tensor:
    """Убрать padding у [C,H,W] или [B,C,H,W]; pad_info = (ph, pw, h, w)."""
    h, w = int(pad_info[2]), int(pad_info[3])
    return t[..., :h, :w]


class AnimeTripletTestDataset(Dataset):
    """
    Полное разрешение без аугментаций, padding до кратности 32.

    Args:
        root_dir:    папка тестовых сцен
        crop_size:   0 — полный кадр, >0 — центральный кроп (быстрый тест)
        load_sketch: подгружать a2s_gt.npy как sketch (иначе нули)
        scenes:      имена сцен, которые нужно взять (порядок сохраняется);
                     None — все сцены с признаками

    Элемент: dict с тензорами как в AnimeTripletDataset, плюс
    has_sketch (bool), scene_name (str), pad_info (ph, pw, h, w).
    Атрибут skipped — сколько сцен отброшено и почему.
    """

    def __init__(self, root_dir: str, crop_size: int = 0, load_sketch: bool = False,
                 scenes: Optional[List[str]] = None, verbose: bool = True):
        self.root = Path(root_dir)
        self.crop_size = crop_size
        self.load_sketch = load_sketch
        samples, self.skipped = scan_scenes(self.root, BASE_FEATURES + A2S_FEATURES)
        self.available = [s["dir"].name for s in samples]
        if scenes is not None:
            by_name = {s["dir"].name: s for s in samples}
            missing = [n for n in scenes if n not in by_name]
            if missing:
                raise ValueError(f"{len(missing)} scenes from the list are missing or have no "
                                 f"features, e.g. {missing[:3]}")
            samples = [by_name[n] for n in scenes]
        self.samples = samples
        if verbose:
            n_skip = self.skipped["base"] + self.skipped["a2s"]
            print(f"[AnimeTripletTestDataset] {len(self.samples)} scenes "
                  f"(of {len(self.available)} with features) | crop={crop_size or 'full'}")
            if n_skip:
                print(f"  [WARN] {n_skip} scenes skipped — нет признаков "
                      f"(RAFT/ISNet: {self.skipped['base']}, Anime2Sketch: {self.skipped['a2s']})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        sample = _load_sample(s, load_sketch=self.load_sketch)
        if self.crop_size > 0:
            h, w = sample["i0"].shape[-2:]
            cs = self.crop_size
            top, left = max(0, (h - cs) // 2), max(0, (w - cs) // 2)
            sample = {k: v[:, top:top + cs, left:left + cs] for k, v in sample.items()}

        pad_info = None
        for k in list(sample):
            sample[k], pi = pad_to_multiple(sample[k])
            pad_info = pad_info or pi
        for k in _CLAMP_KEYS:
            sample[k] = sample[k].clamp(0, 1)
        sample["has_sketch"] = bool(self.load_sketch and s["has_sketch"])
        sample["scene_name"] = s["dir"].name
        sample["pad_info"] = torch.tensor(pad_info)
        return sample
