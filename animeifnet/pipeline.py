"""
animeifnet/pipeline.py — inference-пайплайн AnimeIFNet
══════════════════════════════════════════════════════

Бывший pipeline_v9_geom_fast.py. Внешние признаки (RAFT, ISNet, Anime2Sketch),
подготовка входов, интерполяция одной пары / нескольких кадров / с картой
времени, абляции, метрики и пакетная обработка папки.

Оптимизации скорости:
  1. AMP (bf16 autocast) для RAFT, ISNet, Anime2Sketch и модели.
  2. torch.inference_mode вместо no_grad.
  3. Признаки остаются на GPU, без лишних CPU↔GPU пересылок.
  4. Padding и нормализация на GPU.
  5. RAFT fwd+bwd и ISNet seg0+seg1 считаются одним батчем (B=2).
  6. TTA одним батчем (оригинал + отражение).
  7. LRU-кэш признаков в FeatureProvider.
  8. Несколько промежуточных кадров через generate_multi_t: энкодеры и
     correlation volume считаются один раз на пару.
  9. Опционально: torch.compile, --fast (RAFT max side 768, 12 итераций,
     GPU-NEDT) — быстрее, но может стоить 0.3–1 dB PSNR.

Запуск из корня репозитория:

    python -m animeifnet.pipeline --input ./scene --output ./out \\
        --checkpoint weights/best_v8_2.pth --a2s-weights weights/anime2sketch.pth
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from animeifnet.models import AnimeIFNet


# ══════════════════════════════════════════════════════════════
# Константы и глобальные опции
# ══════════════════════════════════════════════════════════════

PAD_MULTIPLE = 32
NEDT_MAX_DISTANCE = 32.0
A2S_EDGE_FILE0 = "a2s_edge0.npy"
A2S_EDGE_FILE1 = "a2s_edge1.npy"
A2S_DT_FILE0 = "a2s_dt0.npy"
A2S_DT_FILE1 = "a2s_dt1.npy"
A2S_GT_FILE = "a2s_gt.npy"

# Anime2Sketch — UnetGenerator с num_downs=8: вход должен делиться на 2**8.
A2S_PAD_MULTIPLE = 256

# Управляются из CLI / приложения.
USE_AMP = True
AMP_DTYPE = torch.bfloat16
RAFT_ITERS = 20
USE_FEATURE_CACHE = True
# GPU-NEDT считает chebyshev-расстояние, а модель училась на евклидовом (scipy),
# поэтому по умолчанию выключено.
USE_GPU_NEDT = False


def autocast_ctx(device):
    """autocast только на CUDA и только при USE_AMP."""
    if USE_AMP and torch.cuda.is_available() and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return nullcontext()


inference_mode = torch.inference_mode


# ══════════════════════════════════════════════════════════════
# Модели
# ══════════════════════════════════════════════════════════════

def _build_v9_geom(**kwargs):
    from animeifnet.models.animeifnet_v9_geom import AnimeIFNetV9Geom
    return AnimeIFNetV9Geom(**kwargs)


def _build_v8_2(**kwargs):
    # ImageNet-веса не нужны: замороженные backbone лежат в чекпоинте.
    return AnimeIFNet(pretrained=False, **kwargs)


MODEL_REGISTRY = {
    "v8_2": _build_v8_2,        # основная модель
    "v9_geom": _build_v9_geom,  # экспериментальная ветка
}

MODELS_WITH_SKETCH = {"v8_2", "v9_geom"}
MODELS_WITH_ROI = {"v8_2", "v9_geom"}
MODELS_A2S_EDGE = {"v8_2", "v9_geom"}
MODELS_WITH_MULTI_T = {"v8_2"}


def load_model(checkpoint_path, model_name="v8_2", device="cpu",
               corr_levels=4, corr_radius=5, residual_scale=0.7,
               splat_feat_ch=16, compile_model=False):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(MODEL_REGISTRY)}")

    model = MODEL_REGISTRY[model_name](
        corr_levels=corr_levels, corr_radius=corr_radius,
        residual_scale=residual_scale, splat_feat_ch=splat_feat_ch,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

    if compile_model:
        try:
            print("[pipeline] torch.compile(model)... (первый прогон будет дольше)")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}; продолжаем без compile")

    info = f"[pipeline] {model_name} | epoch={ckpt.get('epoch', '?')}"
    if ckpt.get("val_psnr"):
        info += f" | val_psnr={ckpt['val_psnr']:.2f}dB"
    print(info)
    return model


def _unwrap(model):
    """Исходный модуль под torch.compile (у него есть generate_multi_t)."""
    return getattr(model, "_orig_mod", model)


# ══════════════════════════════════════════════════════════════
# Опциональные модули (пост-обработка, Chamfer distance)
# ══════════════════════════════════════════════════════════════

def get_post_processor_cls():
    """PostProcessor из animeifnet/post_process.py (или старого post_process_v9.py)."""
    for mod_name in ("animeifnet.post_process", "post_process_v9"):
        try:
            mod = __import__(mod_name, fromlist=["PostProcessor"])
            return mod.PostProcessor
        except ImportError:
            continue
    return None


def _get_chamfer_fn():
    for mod_name in ("animeifnet.inference_utils", "inference_utils"):
        try:
            mod = __import__(mod_name, fromlist=["chamfer_distance"])
            return mod.chamfer_distance
        except ImportError:
            continue
    return None


# ══════════════════════════════════════════════════════════════
# RAFT / ISNet / Anime2Sketch
# ══════════════════════════════════════════════════════════════

class RAFTFlowEstimator:
    def __init__(self, device, model_size="large"):
        from torchvision.models.optical_flow import (
            raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights)
        if model_size == "large":
            w = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=w).to(device).eval()
        else:
            w = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=w).to(device).eval()
        self.tf = w.transforms()
        self.device = device

    def _resize_in(self, imgs, max_side):
        oh, ow = imgs[0].shape[2:]
        scale = 1.0
        if max_side > 0 and max(oh, ow) > max_side:
            scale = max_side / max(oh, ow)
            imgs = [F.interpolate(x, scale_factor=scale, mode="bilinear",
                                  align_corners=False) for x in imgs]
        return imgs, scale, (oh, ow)

    @staticmethod
    def _resize_out(flow, scale, size):
        if scale != 1.0:
            flow = F.interpolate(flow, size=size, mode="bilinear",
                                 align_corners=False) / scale
        return flow

    @inference_mode()
    def estimate(self, img0, img1, max_side=0, num_iters=None):
        num_iters = num_iters or RAFT_ITERS
        (img0, img1), scale, size = self._resize_in([img0, img1], max_side)
        i0, i1 = self.tf((img0 * 255).byte(), (img1 * 255).byte())
        i0, hw = _pad(i0, 8)
        i1, _ = _pad(i1, 8)
        with autocast_ctx(self.device):
            flow = self.model(i0, i1, num_flow_updates=num_iters)[-1]
        flow = flow[:, :, :hw[0], :hw[1]].float()
        return self._resize_out(flow, scale, size)

    @inference_mode()
    def estimate_pair(self, img0, img1, max_side=0, num_iters=None):
        """flow_fwd и flow_bwd одним батчем (B=2)."""
        num_iters = num_iters or RAFT_ITERS
        (img0, img1), scale, size = self._resize_in([img0, img1], max_side)
        a = torch.cat([img0, img1], dim=0)
        b = torch.cat([img1, img0], dim=0)
        a, b = self.tf((a * 255).byte(), (b * 255).byte())
        a, hw = _pad(a, 8)
        b, _ = _pad(b, 8)
        with autocast_ctx(self.device):
            flow = self.model(a, b, num_flow_updates=num_iters)[-1]
        flow = flow[:, :, :hw[0], :hw[1]].float()
        flow = self._resize_out(flow, scale, size)
        return flow[0:1], flow[1:2]


class ISNetSegmentor:
    def __init__(self, weights_path, dis_repo_path, device, input_size=1024):
        self.device, self.input_size = device, input_size
        sys.path.insert(0, str(Path(dis_repo_path) / "ISNet"))
        try:
            from models.isnet import ISNetDIS
        except ImportError:
            from DIS.ISNet.models.isnet import ISNetDIS
        self.model = ISNetDIS()
        self.model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=False),
            strict=False,
        )
        self.model = self.model.to(device).eval()

    def _run(self, x):
        """x: [B,3,H,W] в [0,1] → маска [B,1,H,W] в [0,1] (per-sample min-max)."""
        oh, ow = x.shape[2:]
        x = F.interpolate(x, size=(self.input_size,) * 2,
                          mode="bilinear", align_corners=False)
        x = x - 0.5  # Normalize(mean=.5, std=1)
        with autocast_ctx(self.device):
            r = self.model(x)
        m = (r[0][0] if isinstance(r, (tuple, list)) else r).float()
        if m.dim() == 3:
            m = m.unsqueeze(1)
        if m.shape[1] > 1:
            m = m[:, :1]
        if m.min() < 0 or m.max() > 1:
            m = torch.sigmoid(m)
        flat = m.reshape(m.shape[0], -1)
        mi = flat.min(dim=1).values.view(-1, 1, 1, 1)
        ma = flat.max(dim=1).values.view(-1, 1, 1, 1)
        m = torch.where(ma - mi > 1e-6, (m - mi) / (ma - mi + 1e-8), m)
        return F.interpolate(m, size=(oh, ow), mode="bilinear",
                             align_corners=False).clamp(0, 1)

    @inference_mode()
    def segment(self, img):
        return self._run(img)

    @inference_mode()
    def segment_pair(self, img0, img1):
        m = self._run(torch.cat([img0, img1], dim=0))
        return m[0:1], m[1:2]


class Anime2SketchEstimator:
    def __init__(self, weights_path, device, repo_path=None):
        self.device = device
        if repo_path:
            sys.path.insert(0, str(repo_path))
        try:
            from Anime2Sketch_master.model.model import UnetGenerator
        except ImportError:
            from anime2sketch.model import UnetGenerator  # type: ignore

        self.net = UnetGenerator(3, 1, 8, 64, norm_layer=nn.InstanceNorm2d,
                                 use_dropout=False)
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.net.load_state_dict(sd, strict=False)
        self.net = self.net.to(device).eval()

    def _run(self, t):
        t, pi = _pad(t, A2S_PAD_MULTIPLE)
        with autocast_ctx(self.device):
            pred = self.net(t)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = _unpad(pred.float(), pi).clamp(-1, 1)
        return (1.0 - (pred + 1.0) / 2.0).cpu().numpy().astype(np.float32)

    @inference_mode()
    def predict_edge(self, img_pil: Image.Image) -> np.ndarray:
        t = transforms.ToTensor()(img_pil).unsqueeze(0).to(self.device) * 2.0 - 1.0
        return self._run(t)[0, 0]

    @inference_mode()
    def predict_edge_pair(self, img0_pil: Image.Image, img1_pil: Image.Image
                          ) -> Tuple[np.ndarray, np.ndarray]:
        t = torch.stack([transforms.ToTensor()(img0_pil),
                         transforms.ToTensor()(img1_pil)]).to(self.device) * 2.0 - 1.0
        pred = self._run(t)
        return pred[0, 0], pred[1, 0]


# ══════════════════════════════════════════════════════════════
# ExternalFeatures + FeatureProvider (с кэшем)
# ══════════════════════════════════════════════════════════════

@dataclass
class ExternalFeatures:
    flow_fwd: torch.Tensor
    flow_bwd: torch.Tensor
    seg0: torch.Tensor
    seg1: torch.Tensor
    a2s_edge0: Optional[torch.Tensor] = None
    a2s_edge1: Optional[torch.Tensor] = None
    a2s_dt0: Optional[torch.Tensor] = None
    a2s_dt1: Optional[torch.Tensor] = None
    a2s_gt: Optional[torch.Tensor] = None


def _img_hash(img: Image.Image) -> str:
    """Ключ кэша: размер + md5 уменьшенной копии (< 1 мс)."""
    small = img.resize((32, 32), Image.NEAREST)
    h = hashlib.md5(np.array(small.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()[:16]
    return f"{img.size[0]}x{img.size[1]}_{h}"


class FeatureProvider:
    """
    Внешние признаки для пары кадров:
      live        — RAFT (+ ISNet, + Anime2Sketch) считаются на лету;
      precomputed — читаются из .npy в папке сцены.
    В live-режиме есть LRU-кэш на cache_size пар.
    """

    def __init__(self, mode="live", device="cpu",
                 raft_size="large", raft_max_side=0,
                 isnet_weights=None, dis_repo=None,
                 a2s_weights=None, a2s_repo=None,
                 need_a2s=False, cache_size=8):
        self.mode = mode
        self.device = device
        self.raft_max_side = raft_max_side
        self.need_a2s = need_a2s
        self.raft = self.isnet = self.a2s = None
        self._cache: Dict[str, ExternalFeatures] = {}
        self._cache_order: List[str] = []
        self._cache_size = cache_size if USE_FEATURE_CACHE else 0

        if mode == "live":
            self.raft = RAFTFlowEstimator(device, raft_size)
            if isnet_weights and dis_repo:
                self.isnet = ISNetSegmentor(isnet_weights, dis_repo, device)
            if need_a2s and a2s_weights:
                try:
                    self.a2s = Anime2SketchEstimator(a2s_weights, device, a2s_repo)
                    print(f"[pipeline] Anime2Sketch loaded from {a2s_weights}")
                except Exception as e:
                    print(f"[WARN] Failed to load Anime2Sketch ({e}); "
                          f"falling back to DoG for edge/dt.")

    def _cache_key(self, img0, img1, gt) -> str:
        return f"{_img_hash(img0)}|{_img_hash(img1)}|" + \
               (_img_hash(gt) if gt is not None else "none")

    def _put_cache(self, key, val):
        if self._cache_size <= 0:
            return
        if key in self._cache:
            self._cache_order.remove(key)
        self._cache[key] = val
        self._cache_order.append(key)
        while len(self._cache_order) > self._cache_size:
            self._cache.pop(self._cache_order.pop(0), None)

    def clear_cache(self):
        self._cache.clear()
        self._cache_order.clear()

    @inference_mode()
    def get_features(self, img0: Image.Image, img1: Image.Image,
                     scene_dir: Optional[Path] = None,
                     img_gt: Optional[Image.Image] = None) -> ExternalFeatures:
        if self.mode == "precomputed":
            return self._load_precomputed(scene_dir)

        key = self._cache_key(img0, img1, img_gt) if self._cache_size > 0 else None
        if key is not None and key in self._cache:
            return self._cache[key]

        t0 = transforms.ToTensor()(img0).unsqueeze(0).to(self.device, non_blocking=True)
        t1 = transforms.ToTensor()(img1).unsqueeze(0).to(self.device, non_blocking=True)

        ff, fb = self.raft.estimate_pair(t0, t1, self.raft_max_side)
        if self.isnet:
            s0, s1 = self.isnet.segment_pair(t0, t1)
        else:
            s0 = s1 = torch.zeros(1, 1, *t0.shape[2:], device=self.device)

        feats = ExternalFeatures(ff, fb, s0, s1)

        if self.need_a2s and self.a2s is not None:
            e0, e1 = self.a2s.predict_edge_pair(img0, img1)
            feats.a2s_edge0 = _tensor1(e0, self.device)
            feats.a2s_edge1 = _tensor1(e1, self.device)
            if USE_GPU_NEDT:
                feats.a2s_dt0 = _compute_nedt_gpu(feats.a2s_edge0)
                feats.a2s_dt1 = _compute_nedt_gpu(feats.a2s_edge1)
            else:
                feats.a2s_dt0 = _tensor1(_compute_nedt(e0), self.device)
                feats.a2s_dt1 = _tensor1(_compute_nedt(e1), self.device)
            if img_gt is not None:
                feats.a2s_gt = _tensor1(self.a2s.predict_edge(img_gt), self.device)

        if key is not None:
            self._put_cache(key, feats)
        return feats

    @staticmethod
    def _load_precomputed(scene_dir: Optional[Path]) -> ExternalFeatures:
        if scene_dir is None:
            raise ValueError("scene_dir required in precomputed mode")
        scene_dir = Path(scene_dir)

        def ld(name):
            return torch.from_numpy(
                np.load(str(scene_dir / name)).astype(np.float32)).unsqueeze(0)

        feats = ExternalFeatures(
            flow_fwd=ld("raft_fwd.npy"),
            flow_bwd=ld("raft_bwd.npy"),
            seg0=ld("seg0.npy").clamp(0, 1),
            seg1=ld("seg1.npy").clamp(0, 1),
        )
        for name, field_n in [
            (A2S_EDGE_FILE0, "a2s_edge0"), (A2S_EDGE_FILE1, "a2s_edge1"),
            (A2S_DT_FILE0, "a2s_dt0"), (A2S_DT_FILE1, "a2s_dt1"),
            (A2S_GT_FILE, "a2s_gt"),
        ]:
            if (scene_dir / name).exists():
                setattr(feats, field_n, ld(name).clamp(0, 1))
        return feats


def _tensor1(arr: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32))[None, None].to(device, non_blocking=True)


# ══════════════════════════════════════════════════════════════
# Edge / DT / NEDT
# ══════════════════════════════════════════════════════════════

def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    k_size = 2 * radius + 1
    coords = torch.arange(k_size, dtype=x.dtype, device=x.device) - radius
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(x, g.view(1, 1, 1, k_size).expand(c, 1, 1, k_size),
                 padding=(0, radius), groups=c)
    return F.conv2d(x, g.view(1, 1, k_size, 1).expand(c, 1, k_size, 1),
                    padding=(radius, 0), groups=c)


def _dog_edge_torch(img_t: torch.Tensor) -> torch.Tensor:
    """DoG-контуры на GPU. img_t: [1,3,H,W] в [0,1] → [1,1,H,W] {0,1}."""
    gray = img_t.mean(dim=1, keepdim=True)
    g1 = _gaussian_blur(gray, sigma=1.0)
    g2 = _gaussian_blur(gray, sigma=1.6)
    return (0.5 + 2.0 * (g2 - g1) - 0.01 < 0.5).float()


def _compute_nedt_gpu(edge: torch.Tensor, max_dist: float = NEDT_MAX_DISTANCE,
                      threshold: float = 0.3) -> torch.Tensor:
    """
    Приближённый distance transform на GPU (chebyshev, через итеративный
    max-pool). [1,1,H,W] в [0,1] → [1,1,H,W] в [0,1]. ~5 мс на 1080p.
    """
    binary = (edge > threshold).float()
    if binary.sum() < 4:
        return torch.ones_like(edge)

    reached = binary.clone()
    dist = torch.zeros_like(edge)
    cur = binary.clone()
    for step in range(1, int(round(max_dist)) + 1):
        cur = F.max_pool2d(cur, kernel_size=3, stride=1, padding=1)
        new_pixels = (reached < 0.5) & (cur > 0.5)
        dist = torch.where(new_pixels, torch.full_like(dist, step), dist)
        reached = torch.maximum(reached, cur)
        if reached.min() > 0.5:
            break
    dist = torch.where(reached < 0.5, torch.full_like(dist, max_dist), dist)
    return (dist / max_dist).clamp(0, 1)


def _compute_nedt(edge: np.ndarray, max_dist: float = NEDT_MAX_DISTANCE) -> np.ndarray:
    """Евклидов NEDT через scipy — как при обучении."""
    from scipy.ndimage import distance_transform_edt
    binary = (edge > 0.5).astype(np.uint8)
    if binary.sum() < 4:
        return np.ones_like(edge, dtype=np.float32)
    dt = distance_transform_edt(1 - binary).astype(np.float32)
    return np.clip(dt, 0, max_dist) / max_dist


def _compute_dt_old(edge: np.ndarray) -> np.ndarray:
    dt = cv2.distanceTransform((1 - edge).astype(np.uint8), cv2.DIST_L2, 3)
    return np.clip(dt / 50.0, 0, 1)


def _edge_and_dt_dog_gpu(img_pil: Image.Image, device) -> Tuple[torch.Tensor, torch.Tensor]:
    t = transforms.ToTensor()(img_pil).unsqueeze(0).to(device, non_blocking=True)
    edge = _dog_edge_torch(t)
    return edge.squeeze(0), _compute_nedt_gpu(edge).squeeze(0)


def _dog_edge_cpu(img: Image.Image) -> np.ndarray:
    gray = np.array(img.convert("L"), np.float32) / 255
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(gray, (0, 0), 1.6)
    return (0.5 + 2 * (g2 - g1) - 0.01 < 0.5).astype(np.float32)


def _edge_and_dt_dog_cpu(img: Image.Image, use_nedt: bool = True
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    edge = _dog_edge_cpu(img)
    dt = _compute_nedt(edge) if use_nedt else _compute_dt_old(edge)
    return (torch.from_numpy(edge).unsqueeze(0),
            torch.from_numpy(dt.astype(np.float32)).unsqueeze(0))


# ══════════════════════════════════════════════════════════════
# Padding / конвертация
# ══════════════════════════════════════════════════════════════

def _pad(t: torch.Tensor, m: int = PAD_MULTIPLE):
    """
    Replicate-padding до кратности m.
    Для [C,H,W] возвращает info = (ph, pw, h, w), для [B,C,H,W] — (h, w).
    """
    h, w = t.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="replicate")
    return t, ((ph, pw, h, w) if t.dim() == 3 else (h, w))


def _unpad(t: torch.Tensor, info):
    h, w = (info[2], info[3]) if len(info) == 4 else info
    return t[..., :h, :w]


def _t2pil(t: torch.Tensor) -> Image.Image:
    t = (t.clamp(-1, 1) + 1) / 2
    return Image.fromarray((t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.5] * 3, [.5] * 3),
])


# ══════════════════════════════════════════════════════════════
# ROI-маски
# ══════════════════════════════════════════════════════════════

def _soft_mask(binary: np.ndarray, h: int, w: int) -> np.ndarray:
    m = cv2.GaussianBlur(binary.astype(np.float32), (11, 11), 0)
    return np.clip(m, 0, 1) if m.sum() >= 64 else np.ones((h, w), np.float32)


def _flow_magnitude(features: ExternalFeatures) -> np.ndarray:
    return np.maximum(
        features.flow_fwd.squeeze(0).norm(dim=0).cpu().numpy(),
        features.flow_bwd.squeeze(0).norm(dim=0).cpu().numpy(),
    )


def _character_mask_fallback(img0: Image.Image, img1: Image.Image,
                             features: ExternalFeatures) -> np.ndarray:
    """
    Маска персонажа без ISNet: персонаж — то, что движется и имеет контуры.
    Сумма motion-diff и величины потока (+ залитые контуры Anime2Sketch),
    порог по 70-му перцентилю, морфология и размытие.
    """
    h, w = np.array(img0).shape[:2]
    a_g = np.array(img0.convert("L"), np.float32)
    b_g = np.array(img1.convert("L"), np.float32)
    if b_g.shape != a_g.shape:
        b_g = cv2.resize(b_g, (a_g.shape[1], a_g.shape[0]))
    diff = cv2.GaussianBlur(np.abs(a_g - b_g), (9, 9), 0)
    diff_n = diff / (diff.max() + 1e-6)

    if features.flow_fwd is not None and features.flow_bwd is not None:
        mag = _flow_magnitude(features)
        if mag.shape != (h, w):
            mag = cv2.resize(mag, (w, h))
        mag_n = mag / (mag.max() + 1e-6)
    else:
        mag_n = np.zeros_like(diff_n)

    saliency = 0.5 * diff_n + 0.5 * mag_n

    if features.a2s_edge0 is not None and features.a2s_edge1 is not None:
        edge_max = np.maximum(features.a2s_edge0.squeeze().cpu().numpy(),
                              features.a2s_edge1.squeeze().cpu().numpy()).astype(np.float32)
        if edge_max.shape != (h, w):
            edge_max = cv2.resize(edge_max, (w, h))
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        filled = cv2.dilate(edge_max, ek, iterations=2)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, ek)
        filled = cv2.GaussianBlur(filled, (15, 15), 0)
        saliency = np.maximum(saliency, filled * 0.7)

    if saliency.sum() < 1e-6:
        return np.ones((h, w), np.float32)
    thr = max(0.20, float(np.percentile(saliency, 70)))
    b = (saliency >= thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    b = cv2.dilate(b, k, iterations=2)
    return _soft_mask(b, h, w)


def build_roi_mask(img0: Image.Image, img1: Image.Image,
                   roi_mode: str = "full",
                   features: Optional[ExternalFeatures] = None,
                   roi_file: Optional[str] = None,
                   roi_array: Optional[np.ndarray] = None) -> np.ndarray:
    """
    roi_mode: full | file | array | seg | flow_mag | motion.
    array — готовая маска [H,W] в [0,1] (например, нарисованная в приложении).
    """
    h, w = np.array(img0).shape[:2]
    if roi_mode == "full":
        return np.ones((h, w), np.float32)

    if roi_mode == "array":
        if roi_array is None:
            return np.ones((h, w), np.float32)
        m = roi_array.astype(np.float32)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(m, 0, 1)

    if roi_mode == "file":
        if not roi_file or not os.path.isfile(roi_file):
            print(f"[WARN] roi_mode=file but roi_file not found: {roi_file}")
            return np.ones((h, w), np.float32)
        m = Image.open(roi_file).convert("L").resize((w, h), Image.BILINEAR)
        return np.clip(np.array(m, np.float32) / 255.0, 0, 1)

    if roi_mode == "seg" and features is not None:
        s = np.maximum(features.seg0.squeeze().cpu().numpy(),
                       features.seg1.squeeze().cpu().numpy())
        if s.max() < 0.05:
            print("[WARN] ROI 'character': ISNet недоступен — используем "
                  "fallback (motion + flow + edge fill). Для качественной "
                  "сегментации запусти с --isnet-weights и --dis-repo.")
            return _character_mask_fallback(img0, img1, features)
        b = (s > 0.3).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        return _soft_mask(cv2.dilate(b, k, iterations=2), h, w)

    if roi_mode == "flow_mag" and features is not None:
        mag = _flow_magnitude(features)
        b = (mag >= max(1, float(np.percentile(mag, 70)))).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        b = cv2.dilate(cv2.morphologyEx(b, cv2.MORPH_CLOSE, k), k)
        return _soft_mask(b, h, w)

    if roi_mode == "motion":
        a = np.array(img0.convert("L"), np.float32)
        b = np.array(img1.convert("L"), np.float32)
        diff = cv2.GaussianBlur(np.abs(a - b), (9, 9), 0)
        mask = (diff >= max(5, float(np.percentile(diff, 75)))).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), k)
        return _soft_mask(mask, h, w)

    return np.ones((h, w), np.float32)


# ══════════════════════════════════════════════════════════════
# Скетч-подсказка
# ══════════════════════════════════════════════════════════════

def load_sketch_image(path: str, h: int, w: int) -> np.ndarray:
    a = np.array(Image.open(path).convert("L").resize((w, h), Image.BILINEAR),
                 np.float32) / 255.0
    if a.mean() > 0.5:  # чёрные линии на белом → инвертируем
        a = 1.0 - a
    return np.clip(a, 0, 1)


def resolve_sketch(sketch_arg: Optional[str], features: ExternalFeatures,
                   h: int, w: int,
                   a2s_live: Optional[Anime2SketchEstimator] = None,
                   img_gt: Optional[Image.Image] = None,
                   device="cpu") -> Optional[torch.Tensor]:
    """None/'none' — без скетча; 'auto' — скетч эталонного кадра; иначе путь к файлу."""
    if sketch_arg is None or sketch_arg == "none":
        return None

    if sketch_arg == "auto":
        if features.a2s_gt is not None:
            sk = features.a2s_gt.to(device, non_blocking=True)
            if sk.shape[-2:] != (h, w):
                sk = F.interpolate(sk, size=(h, w), mode="bilinear", align_corners=False)
            return sk.clamp(0, 1)
        if a2s_live is not None and img_gt is not None:
            return _tensor1(a2s_live.predict_edge(img_gt), device).clamp(0, 1)
        print("[WARN] --sketch auto: GT sketch недоступен, fallback = None")
        return None

    if os.path.isfile(sketch_arg):
        return _tensor1(load_sketch_image(sketch_arg, h, w), device)
    print(f"[WARN] --sketch: файл не найден: {sketch_arg}; fallback = None")
    return None


def _ease(t: float, mode: str = "linear") -> float:
    t = max(0, min(1, t))
    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        return 1 - (1 - t) ** 2
    if mode == "ease_in_out":
        return 4 * t * t * t if t < .5 else 1 - (-2 * t + 2) ** 3 / 2
    return t


# ══════════════════════════════════════════════════════════════
# Подготовка входов
# ══════════════════════════════════════════════════════════════

INPUT_KEYS = ["i0", "i1", "edge0", "edge1", "dt0", "dt1",
              "flow_fwd", "flow_bwd", "seg0", "seg1"]


def _a2s_dt(edge: torch.Tensor, precomputed: Optional[torch.Tensor], device):
    if precomputed is not None:
        return precomputed.squeeze(0).to(device, non_blocking=True)
    if USE_GPU_NEDT:
        return _compute_nedt_gpu(edge.unsqueeze(0)).squeeze(0)
    return _tensor1(_compute_nedt(edge.squeeze(0).cpu().numpy()), device).squeeze(0)


def prepare_inputs(img0: Image.Image, img1: Image.Image,
                   features: ExternalFeatures,
                   model_name: str = "v8_2",
                   roi_mode: str = "full",
                   roi_file: Optional[str] = None,
                   device: str = "cpu",
                   use_nedt: bool = True,
                   roi_array: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
    """Тензоры для forward модели, уже на device и с padding до кратности 32."""
    if model_name in MODELS_A2S_EDGE and features.a2s_edge0 is not None:
        e0 = features.a2s_edge0.squeeze(0).to(device, non_blocking=True)
        e1 = features.a2s_edge1.squeeze(0).to(device, non_blocking=True)
        d0 = _a2s_dt(e0, features.a2s_dt0, device)
        d1 = _a2s_dt(e1, features.a2s_dt1, device)
    elif USE_GPU_NEDT:
        e0, d0 = _edge_and_dt_dog_gpu(img0, device)
        e1, d1 = _edge_and_dt_dog_gpu(img1, device)
    else:
        e0, d0 = (x.to(device, non_blocking=True) for x in _edge_and_dt_dog_cpu(img0, use_nedt))
        e1, d1 = (x.to(device, non_blocking=True) for x in _edge_and_dt_dog_cpu(img1, use_nedt))

    i0 = (transforms.ToTensor()(img0).to(device, non_blocking=True) - 0.5) / 0.5
    i1 = (transforms.ToTensor()(img1).to(device, non_blocking=True) - 0.5) / 0.5

    tensors = [i0, i1, e0, e1, d0, d1,
               features.flow_fwd.squeeze(0).to(device, non_blocking=True),
               features.flow_bwd.squeeze(0).to(device, non_blocking=True),
               features.seg0.squeeze(0).to(device, non_blocking=True),
               features.seg1.squeeze(0).to(device, non_blocking=True)]

    result, pad_info = {}, None
    for key, t in zip(INPUT_KEYS, tensors):
        tp, pi = _pad(t)
        pad_info = pad_info or pi
        result[key] = tp.unsqueeze(0)
    result["pad_info"] = pad_info

    if model_name in MODELS_WITH_ROI:
        mask_np = build_roi_mask(img0, img1, roi_mode, features, roi_file, roi_array)
        rm = torch.from_numpy(mask_np).unsqueeze(0).float().to(device, non_blocking=True)
        result["roi_mask"] = _pad(rm)[0].unsqueeze(0)

    return result


def _pad_sketch(sketch_hint: Optional[torch.Tensor], device) -> Optional[torch.Tensor]:
    if sketch_hint is None:
        return None
    sk = sketch_hint.reshape(1, *sketch_hint.shape[-2:]).to(device, non_blocking=True)
    return _pad(sk)[0].unsqueeze(0)


def _forward_model(model, inputs: Dict[str, torch.Tensor], model_name: str,
                   t_val=0.5, sketch_hint: Optional[torch.Tensor] = None):
    kwargs = {"t": t_val}
    if model_name in MODELS_WITH_SKETCH:
        kwargs["sketch_hint"] = sketch_hint
    if model_name in MODELS_WITH_ROI:
        kwargs["roi_mask"] = inputs.get("roi_mask")
    return model(*(inputs[k] for k in INPUT_KEYS), **kwargs)


# ══════════════════════════════════════════════════════════════
# Интерполяция пары
# ══════════════════════════════════════════════════════════════

def _resize_to(img: Optional[Image.Image], size) -> Optional[Image.Image]:
    if img is None or size is None or img.size == tuple(size):
        return img
    return img.resize(size, Image.LANCZOS)


def _apply_post_processor(pp, out, inputs=None):
    pp_in = {k: out[k].float() for k in ("pred", "warped0", "warped1", "occ_weights")}
    if inputs is not None:
        return pp(pp_in, seg0=inputs.get("seg0"), seg1=inputs.get("seg1"))
    return pp(pp_in)


@inference_mode()
def interpolate_pair(model, img0, img1, features,
                     model_name="v8_2",
                     roi_mode="full", roi_file=None,
                     t=0.5, device="cpu",
                     use_nedt=True, use_tta=False,
                     sketch_hint: Optional[torch.Tensor] = None,
                     save_debug_dir=None,
                     post_processor=None,
                     output_size: Optional[Tuple[int, int]] = None,
                     roi_array: Optional[np.ndarray] = None):
    """
    Один промежуточный кадр.

    Args:
        post_processor: опциональный PostProcessor, применяется к выходу модели.
        output_size:    (W, H) — финальный размер (Lanczos), если инференс шёл
                        на уменьшенном кадре.
        roi_array:      готовая ROI-маска [H,W] при roi_mode="array".
    """
    inputs = prepare_inputs(img0, img1, features, model_name,
                            roi_mode, roi_file, device, use_nedt, roi_array)
    pi = inputs.pop("pad_info")
    sk = _pad_sketch(sketch_hint, device)

    if use_tta:
        # оригинал + горизонтальное отражение одним батчем (B=2)
        batched = {}
        for k, v in inputs.items():
            fv = v.flip(-1)
            if k in ("flow_fwd", "flow_bwd"):
                fv = fv.clone()
                fv[:, 0] *= -1
            batched[k] = torch.cat([v, fv], dim=0)
        sk_b = torch.cat([sk, sk.flip(-1)], dim=0) if sk is not None else None

        with autocast_ctx(device):
            out_b = _forward_model(model, batched, model_name, t, sk_b)

        pred = 0.5 * (out_b["pred"][0:1].float() + out_b["pred"][1:2].float().flip(-1))
        out = {k: (v[0:1] if torch.is_tensor(v) and v.dim() == 4 and v.shape[0] == 2 else v)
               for k, v in out_b.items()}
        out["pred"] = pred.clamp(-1, 1)
    else:
        with autocast_ctx(device):
            out = _forward_model(model, inputs, model_name, t, sk)
        out["pred"] = out["pred"].float()

    if save_debug_dir:
        save_debug(out, pi, save_debug_dir)

    result_pre_post = _t2pil(_unpad(out["pred"][0].float(), pi))

    pred_after_post_pil = None
    if post_processor is not None and all(k in out for k in ("warped0", "warped1", "occ_weights")):
        try:
            pp_result = _apply_post_processor(post_processor, out, inputs)
            out["pred_postprocessed"] = pp_result["pred"]
            out["defect_masks"] = pp_result.get("defect_masks")
            out["pred"] = pp_result["pred"]
            pred_after_post_pil = _t2pil(_unpad(pp_result["pred"][0].float(), pi))
        except Exception as e:
            print(f"[WARN] PostProcessor failed: {e}; используем raw pred")

    pred_up = _unpad(out["pred"][0].float(), pi)
    coarse_up = _unpad(out["coarse"][0].float(), pi) if "coarse" in out else pred_up

    result = {
        "result": _resize_to(_t2pil(pred_up), output_size),
        "result_raw": _resize_to(result_pre_post, output_size),
        "result_postprocessed": _resize_to(pred_after_post_pil, output_size),
        "coarse": _t2pil(coarse_up),
        "roi_mask": None,
        "raw_out": out,
        "pad_info": pi,
    }
    if "pred_local" in out:
        result["pred_local"] = _t2pil(_unpad(out["pred_local"][0].float(), pi))
    if "roi_mask" in inputs:
        result["roi_mask"] = _unpad(inputs["roi_mask"][0], pi).squeeze(0).cpu().numpy()
    return result


def save_debug(out, pad_info, save_dir, prefix=""):
    os.makedirs(save_dir, exist_ok=True)
    if "flow_t0" in out:
        flow = _unpad(out["flow_t0"][0].float().cpu(), pad_info)
        mag = flow.norm(dim=0).numpy()
        mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(str(Path(save_dir) / f"{prefix}flow_mag.png"),
                    cv2.applyColorMap(mag, cv2.COLORMAP_JET))
    if "occ_weights" in out:
        ow = _unpad(out["occ_weights"][0].float().cpu(), pad_info)
        for i, name in enumerate(["disocclusion", "warped0_weight", "warped1_weight"]):
            cv2.imwrite(str(Path(save_dir) / f"{prefix}occ_{name}.png"),
                        (ow[i].numpy() * 255).astype(np.uint8))


def onion_skin(frame1: Image.Image, frame3: Image.Image,
               pred: Image.Image, alpha: float = 0.35) -> Image.Image:
    """Калька: предыдущий кадр красным, следующий синим поверх результата."""
    w, h = pred.size
    base = np.array(pred.convert("RGB"), np.float32)
    f1 = np.array(frame1.resize((w, h), Image.BILINEAR).convert("RGB"), np.float32)
    f3 = np.array(frame3.resize((w, h), Image.BILINEAR).convert("RGB"), np.float32)
    red = f1.copy(); red[..., 1] *= 0.3; red[..., 2] *= 0.3
    blu = f3.copy(); blu[..., 0] *= 0.3; blu[..., 1] *= 0.3
    out = (1 - 2 * alpha) * base + alpha * red + alpha * blu
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ══════════════════════════════════════════════════════════════
# Несколько промежуточных кадров
# ══════════════════════════════════════════════════════════════

@inference_mode()
def interpolate_multi(model, img0, img1, features,
                      model_name="v8_2",
                      num_frames=1, easing="linear", recursive=False,
                      roi_mode="full", roi_file=None,
                      device="cpu", use_nedt=True, use_tta=False,
                      sketch_hint: Optional[torch.Tensor] = None,
                      roi_array: Optional[np.ndarray] = None):
    def pair(a, b, t):
        return interpolate_pair(model, a, b, features, model_name,
                                roi_mode, roi_file, t, device,
                                use_nedt, use_tta, sketch_hint,
                                roi_array=roi_array)["result"]

    if num_frames == 1:
        return [pair(img0, img1, _ease(0.5, easing))]

    t_values = [_ease((i + 1) / (num_frames + 1), easing) for i in range(num_frames)]

    # Общие энкодеры и correlation volume на все t (без TTA).
    base = _unwrap(model)
    if (not recursive and not use_tta and model_name in MODELS_WITH_MULTI_T
            and hasattr(base, "generate_multi_t")):
        inputs = prepare_inputs(img0, img1, features, model_name,
                                roi_mode, roi_file, device, use_nedt, roi_array)
        pi = inputs.pop("pad_info")
        with autocast_ctx(device):
            results = base.generate_multi_t(
                *(inputs[k] for k in INPUT_KEYS), t_values=t_values,
                sketch_hint=_pad_sketch(sketch_hint, device),
                roi_mask=inputs.get("roi_mask"),
            )
        return [_t2pil(_unpad(r["pred"][0].float(), pi)) for r in results]

    if recursive:
        levels = int(math.log2(num_frames + 1))
        if 2 ** levels - 1 == num_frames:
            # Рекурсивная бисекция. Признаки внешних моделей берутся от исходной
            # пары, поэтому для глубоких уровней это приближение.
            cache = {0.0: img0, 1.0: img1}
            for level in range(levels):
                step = 1.0 / (2 ** (level + 1))
                tc = step
                while tc < 1.0:
                    if tc not in cache:
                        cache[tc] = pair(cache[tc - step], cache[tc + step], 0.5)
                    tc += 2 * step
            return [cache[t] for t in sorted(k for k in cache if 0 < k < 1)]

    return [pair(img0, img1, t) for t in t_values]


@inference_mode()
def interpolate_per_region_t(model, img0, img1, features,
                             model_name="v8_2",
                             t_fg=0.6, t_bg=0.4,
                             roi_mode="full", roi_file=None,
                             device="cpu", use_nedt=True, use_tta=False,
                             sketch_hint: Optional[torch.Tensor] = None) -> Image.Image:
    """Разный t для персонажа (seg) и фона; два прохода, смешивание по маске."""
    kw = dict(roi_mode=roi_mode, roi_file=roi_file, device=device,
              use_nedt=use_nedt, use_tta=use_tta, sketch_hint=sketch_hint)
    a = np.array(interpolate_pair(model, img0, img1, features, model_name,
                                  t=t_fg, **kw)["result"], np.float32)
    b = np.array(interpolate_pair(model, img0, img1, features, model_name,
                                  t=t_bg, **kw)["result"], np.float32)

    seg = np.maximum(features.seg0.squeeze().cpu().numpy(),
                     features.seg1.squeeze().cpu().numpy())
    seg = np.clip(cv2.GaussianBlur(seg.astype(np.float32), (11, 11), 0), 0, 1)
    if seg.shape != a.shape[:2]:
        seg = cv2.resize(seg, (a.shape[1], a.shape[0]))
    seg = seg[..., None]
    return Image.fromarray(np.clip(seg * a + (1 - seg) * b, 0, 255).astype(np.uint8))


@inference_mode()
def interpolate_with_tmap(model, img0: Image.Image, img1: Image.Image,
                          features, tmap: np.ndarray,
                          model_name: str = "v8_2",
                          t_levels: int = 5, t_center: float = 0.5,
                          t_range: float = 0.4,
                          roi_mode: str = "full", roi_file: Optional[str] = None,
                          device: str = "cpu", use_nedt: bool = True,
                          use_tta: bool = False,
                          sketch_hint: Optional[torch.Tensor] = None,
                          ) -> Tuple[Image.Image, Dict]:
    """
    Интерполяция с картой времени T(x, y) ∈ [0, 1].

    Диапазон [t_center − t_range, t_center + t_range] делится на t_levels
    уровней, для каждого делается forward (признаки берутся из кэша),
    затем кадры смешиваются гауссовыми весами по расстоянию T(x,y) до уровня.
    """
    t_levels = int(min(max(t_levels, 2), 9))
    t_lo = max(0.0, t_center - t_range)
    t_hi = min(1.0, t_center + t_range)
    levels = np.linspace(t_lo, t_hi, t_levels).astype(np.float32)
    kw = dict(roi_mode=roi_mode, roi_file=roi_file, device=device,
              use_nedt=use_nedt, use_tta=use_tta, sketch_hint=sketch_hint)

    if tmap.std() < 0.02:  # почти равномерная карта → один кадр
        t_real = t_lo + (t_hi - t_lo) * float(tmap.mean())
        r = interpolate_pair(model, img0, img1, features, model_name, t=t_real, **kw)
        return r["result"], {"fallback": "uniform_tmap", "t_used": t_real,
                             "levels": levels.tolist()}

    rendered = [np.array(interpolate_pair(model, img0, img1, features, model_name,
                                          t=float(t), **kw)["result"], np.float32)
                for t in levels]

    H, W = rendered[0].shape[:2]
    tmap_f = tmap.astype(np.float32)
    if tmap_f.shape != (H, W):
        tmap_f = cv2.resize(tmap_f, (W, H), interpolation=cv2.INTER_LINEAR)
    tmap_f = np.clip(cv2.GaussianBlur(tmap_f, (0, 0), sigmaX=2.0), 0.0, 1.0)

    levels_norm = (levels - t_lo) / max(t_hi - t_lo, 1e-6)
    sigma = max(0.5 / (t_levels - 1), 0.08)
    weights = np.exp(-((tmap_f[None] - levels_norm[:, None, None]) ** 2)
                     / (2 * sigma ** 2)).astype(np.float32)
    weights /= weights.sum(0, keepdims=True) + 1e-8

    out = sum(weights[i, :, :, None] * rendered[i] for i in range(t_levels))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), {
        "levels": levels.tolist(),
        "tmap_min": float(tmap_f.min()), "tmap_max": float(tmap_f.max()),
        "tmap_mean": float(tmap_f.mean()), "tmap_std": float(tmap_f.std()),
        "n_forwards": t_levels,
    }


# ══════════════════════════════════════════════════════════════
# Абляции / сравнение
# ══════════════════════════════════════════════════════════════

@inference_mode()
def ablation_run(model, img0, img1, features,
                 sketch_tensor: Optional[torch.Tensor],
                 roi_mode: str = "seg", roi_file: Optional[str] = None,
                 device: str = "cpu", use_nedt: bool = True,
                 model_name: str = "v8_2", t: float = 0.5,
                 roi_array: Optional[np.ndarray] = None) -> Dict[str, Image.Image]:
    """Четыре режима: без подсказок, +скетч, +ROI, +скетч+ROI."""
    def run(rm, sk):
        return interpolate_pair(model, img0, img1, features, model_name,
                                rm, roi_file, t, device, use_nedt, False, sk,
                                roi_array=roi_array)["result"]
    return {
        "baseline": run("full", None),
        "plus_sketch": run("full", sketch_tensor),
        "plus_roi": run(roi_mode, None),
        "plus_both": run(roi_mode, sketch_tensor),
    }


@inference_mode()
def compare_run_v9(model, img0, img1, features,
                   sketch_tensor: Optional[torch.Tensor] = None,
                   roi_mode: str = "full", roi_file: Optional[str] = None,
                   t: float = 0.5, device: str = "cpu", use_nedt: bool = True,
                   output_size: Optional[Tuple[int, int]] = None,
                   model_name: str = "v8_2") -> Dict[str, Image.Image]:
    """
    Вклад пост-обработки: raw / +detail / +полный пайплайн / +полный ×2.
    Модель запускается один раз, к её выходу применяются разные PostProcessor.
    """
    base = interpolate_pair(model, img0, img1, features, model_name,
                            roi_mode, roi_file, t, device, use_nedt,
                            use_tta=False, sketch_hint=sketch_tensor,
                            output_size=output_size)
    raw_out, pi, raw_pil = base["raw_out"], base["pad_info"], base["result"]

    PP = get_post_processor_cls()
    if PP is None:
        print("[WARN] PostProcessor не найден — compare вернёт raw")
        return {k: raw_pil for k in ("raw", "plus_detail", "plus_full", "plus_full_x2")}

    def apply(pp):
        try:
            pred = _unpad(_apply_post_processor(pp, raw_out)["pred"][0].float(), pi)
            return _resize_to(_t2pil(pred), output_size)
        except Exception as e:
            print(f"[WARN] compare PP failed: {e}")
            return raw_pil

    detail = apply(PP(do_detail_injection=True, do_ghost_fix=False,
                      do_color_fix=False, do_sharpen=False, detail_strength=0.6))
    full = apply(PP(do_detail_injection=True, do_ghost_fix=True,
                    do_color_fix=True, do_sharpen=True,
                    detail_strength=0.6, ghost_strength=0.5,
                    color_strength=0.4, sharpen_strength=0.20))
    return {
        "raw": raw_pil,
        "plus_detail": detail,
        "plus_full": full,
        "plus_full_x2": full.resize((full.size[0] * 2, full.size[1] * 2), Image.LANCZOS),
    }


# ══════════════════════════════════════════════════════════════
# Метрики
# ══════════════════════════════════════════════════════════════

def _ssim_simple(p, t, ws=11):
    c1, c2 = .01 ** 2, .03 ** 2
    ch = p.size(1)
    co = torch.arange(ws, dtype=p.dtype, device=p.device) - ws // 2
    g = torch.exp(-co ** 2 / 4.5)
    g /= g.sum()
    k = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0).repeat(ch, 1, 1, 1)
    pad = ws // 2
    mu1 = F.conv2d(p, k, padding=pad, groups=ch)
    mu2 = F.conv2d(t, k, padding=pad, groups=ch)
    s1 = F.conv2d(p * p, k, padding=pad, groups=ch) - mu1 ** 2
    s2 = F.conv2d(t * t, k, padding=pad, groups=ch) - mu2 ** 2
    s12 = F.conv2d(p * t, k, padding=pad, groups=ch) - mu1 * mu2
    return (((2 * mu1 * mu2 + c1) * (2 * s12 + c2)
             / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))).mean())


def evaluate_pair(pred_pil, gt_pil, lpips_metric=None, compute_cd=False,
                  roi_np: Optional[np.ndarray] = None):
    """PSNR + SSIM (+ LPIPS, + Chamfer distance, + PSNR внутри/вне ROI)."""
    pred_t = _to_tensor(pred_pil).unsqueeze(0)
    gt_t = _to_tensor(gt_pil.resize(pred_pil.size)).unsqueeze(0)

    p = (pred_t.clamp(-1, 1) + 1) / 2
    g = (gt_t.clamp(-1, 1) + 1) / 2
    mse = F.mse_loss(p, g).item()
    result = {"psnr": 10 * math.log10(1 / (mse + 1e-8)),
              "ssim": float(_ssim_simple(p, g))}

    if lpips_metric is not None:
        try:
            dev = next(lpips_metric.parameters()).device
            with torch.no_grad():
                result["lpips"] = float(lpips_metric(pred_t.to(dev), gt_t.to(dev)).mean())
        except Exception as e:
            print(f"[WARN] LPIPS failed: {e}")

    if compute_cd:
        chamfer = _get_chamfer_fn()
        if chamfer is not None:
            try:
                cd = chamfer(pred_t[0], gt_t[0])
                result["cd"] = cd["cd_symmetric"]
                result["line_f1"] = cd["line_f1"]
            except Exception as e:
                print(f"[WARN] Chamfer distance failed: {e}")

    if roi_np is not None:
        roi_t = torch.from_numpy(roi_np.astype(np.float32))[None, None]
        if roi_t.shape[-2:] != p.shape[-2:]:
            roi_t = F.interpolate(roi_t, size=p.shape[-2:], mode="bilinear",
                                  align_corners=False)
        roi_t = (roi_t > 0.5).float()
        diff = (p - g) ** 2
        n_in, n_out = roi_t.sum().item(), (1 - roi_t).sum().item()
        if n_in > 0:
            mse_in = (diff * roi_t).sum().item() / (n_in * diff.shape[1])
            result["psnr_roi"] = 10 * math.log10(1 / (mse_in + 1e-8))
        if n_out > 0:
            mse_out = (diff * (1 - roi_t)).sum().item() / (n_out * diff.shape[1])
            result["psnr_preserve"] = 10 * math.log10(1 / (mse_out + 1e-8))

    return result


def is_scene_change(img0: Image.Image, img1: Image.Image) -> bool:
    """Смена сцены — тот же детектор, что и в scripts/evaluate.py."""
    from animeifnet.metrics import is_scene_change as _detect
    if img1.size != img0.size:
        img1 = img1.resize(img0.size, Image.BILINEAR)
    return _detect(_to_tensor(img0), _to_tensor(img1))


# ══════════════════════════════════════════════════════════════
# Пакетная обработка
# ══════════════════════════════════════════════════════════════

def _find_frame(scene_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg"):
        if (scene_dir / f"{stem}{ext}").exists():
            return scene_dir / f"{stem}{ext}"
    return None


def _interpolate_frames(model, im0, im1, feat, model_name, sk_tensor, opts):
    if opts["per_region_t"]:
        return [interpolate_per_region_t(
            model, im0, im1, feat, model_name,
            t_fg=opts["t_fg"], t_bg=opts["t_bg"],
            roi_mode=opts["roi_mode"], roi_file=opts["roi_file"],
            device=opts["device"], use_nedt=opts["use_nedt"],
            use_tta=opts["use_tta"], sketch_hint=sk_tensor)]
    return interpolate_multi(
        model, im0, im1, feat, model_name, opts["num_frames"],
        opts["easing"], opts["recursive"], opts["roi_mode"], opts["roi_file"],
        opts["device"], opts["use_nedt"], opts["use_tta"], sk_tensor)


@inference_mode()
def process_folder(model, feature_provider, input_dir, output_dir,
                   model_name="v8_2",
                   roi_mode="full", roi_file=None,
                   sketch_arg: Optional[str] = None,
                   num_frames=1, easing="linear", recursive=False,
                   use_tta=False, use_nedt=True,
                   per_region_t=False, t_fg=0.6, t_bg=0.4,
                   onion=False, ablation=False, scene_cut=False,
                   device="cpu",
                   save_debug=False, save_masks=False, evaluate=False,
                   compute_cd=False):
    """
    precomputed-режим: input_dir — папка сцен (frame1/frame2/frame3 + .npy).
    live-режим: input_dir — последовательность кадров, между соседними
    вставляются num_frames промежуточных.
    """
    inp, out = Path(input_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    debug_dir = out / "debug" if save_debug else None
    for flag, sub in ((save_debug, "debug"), (save_masks, "masks"), (ablation, "ablation")):
        if flag:
            (out / sub).mkdir(exist_ok=True)

    lpips_m = None
    if evaluate:
        try:
            import lpips
            lpips_m = lpips.LPIPS(net="alex").eval()
        except ImportError:
            print("[WARN] lpips not installed, skipping LPIPS metric")

    opts = dict(num_frames=num_frames, easing=easing, recursive=recursive,
                roi_mode=roi_mode, roi_file=roi_file, device=device,
                use_nedt=use_nedt, use_tta=use_tta,
                per_region_t=per_region_t, t_fg=t_fg, t_bg=t_bg)
    header = dict(num_frames=num_frames, easing=easing, use_tta=use_tta,
                  use_nedt=use_nedt, sketch_arg=sketch_arg, roi_mode=roi_mode,
                  ablation=ablation, per_region_t=per_region_t, onion=onion,
                  scene_cut=scene_cut)
    metrics_all: List[dict] = []
    idx = 1

    def save_mask(name, im0, im1, feat):
        if save_masks and model_name in MODELS_WITH_ROI:
            m = build_roi_mask(im0, im1, roi_mode, feat, roi_file)
            Image.fromarray((m * 255).astype(np.uint8)).save(out / "masks" / f"{name}.png")

    if feature_provider.mode == "precomputed":
        scenes = sorted(d for d in inp.iterdir()
                        if d.is_dir() and (d / "raft_fwd.npy").exists())
        if not scenes:
            raise ValueError(f"No scenes found in {inp}")
        _print_header(model, model_name, len(scenes), **header)

        for i, sd in enumerate(scenes):
            p0, p1, gt = (_find_frame(sd, s) for s in ("frame1", "frame3", "frame2"))
            if not p0 or not p1:
                continue
            im0 = Image.open(p0).convert("RGB")
            im1 = Image.open(p1).convert("RGB")
            im_gt = Image.open(gt).convert("RGB") if gt else None
            feat = feature_provider.get_features(im0, im1, sd, im_gt)
            scene_tag = "SCENE_CUT" if scene_cut and is_scene_change(im0, im1) else ""

            im0.save(out / f"{idx:06d}.png"); idx += 1
            save_mask(sd.name, im0, im1, feat)

            h, w = im0.size[1], im0.size[0]
            sk_tensor = resolve_sketch(sketch_arg, feat, h, w,
                                       a2s_live=feature_provider.a2s,
                                       img_gt=im_gt, device=device)

            if ablation:
                ab = ablation_run(model, im0, im1, feat, sk_tensor,
                                  roi_mode=roi_mode, roi_file=roi_file,
                                  device=device, use_nedt=use_nedt,
                                  model_name=model_name)
                for key, img in ab.items():
                    (out / "ablation" / key).mkdir(exist_ok=True)
                    img.save(out / "ablation" / key / f"{sd.name}.png")
                    if evaluate and im_gt is not None:
                        m = evaluate_pair(img, im_gt, lpips_m, compute_cd=compute_cd)
                        m.update(scene=sd.name, mode=key)
                        if scene_tag:
                            m["scene_tag"] = scene_tag
                        metrics_all.append(m)
                ab["plus_both"].save(out / f"{idx:06d}.png"); idx += 1
            else:
                frames = _interpolate_frames(model, im0, im1, feat, model_name,
                                             sk_tensor, opts)
                if onion:
                    frames = [onion_skin(im0, im1, f) for f in frames]
                roi_for_eval = None
                if evaluate and im_gt is not None and model_name in MODELS_WITH_ROI \
                        and roi_mode != "full":
                    roi_for_eval = build_roi_mask(im0, im1, roi_mode, feat, roi_file)
                for f in frames:
                    f.save(out / f"{idx:06d}.png"); idx += 1
                    if evaluate and im_gt is not None:
                        m = evaluate_pair(f, im_gt, lpips_m, compute_cd=compute_cd,
                                          roi_np=roi_for_eval)
                        m["scene"] = sd.name
                        if scene_tag:
                            m["scene_tag"] = scene_tag
                        metrics_all.append(m)

            if i == len(scenes) - 1:
                im1.save(out / f"{idx:06d}.png"); idx += 1

            line = f"  [{i+1}/{len(scenes)}] {sd.name}" + (f" [{scene_tag}]" if scene_tag else "")
            if metrics_all and metrics_all[-1].get("scene") == sd.name:
                last = metrics_all[-1]
                line += f" | PSNR={last['psnr']:.2f} SSIM={last['ssim']:.3f}"
            print(line)

    else:
        frames_list = sorted(f for f in inp.iterdir()
                             if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if len(frames_list) < 2:
            raise ValueError(f"Need >= 2 frames, found {len(frames_list)}")
        _print_header(model, model_name, len(frames_list),
                      total_out=len(frames_list) + (len(frames_list) - 1) * num_frames,
                      **header)

        for i, fp in enumerate(frames_list):
            img = Image.open(fp).convert("RGB")
            img.save(out / f"{idx:06d}.png"); idx += 1
            if i == len(frames_list) - 1:
                break

            nxt = Image.open(frames_list[i + 1]).convert("RGB")
            pair_name = f"{fp.name} → {frames_list[i+1].name}"
            if scene_cut and is_scene_change(img, nxt):
                print(f"  [{i+1}/{len(frames_list)-1}] {pair_name}  [SCENE_CUT — skip]")
                continue

            feat = feature_provider.get_features(img, nxt)
            save_mask(f"pair_{i:04d}", img, nxt, feat)
            sk_tensor = resolve_sketch(sketch_arg, feat, img.size[1], img.size[0],
                                       a2s_live=feature_provider.a2s, device=device)
            if debug_dir:
                interpolate_pair(model, img, nxt, feat, model_name, roi_mode,
                                 roi_file, 0.5, device, use_nedt, False, sk_tensor,
                                 save_debug_dir=str(debug_dir / f"pair_{i:04d}"))

            frames = _interpolate_frames(model, img, nxt, feat, model_name,
                                         sk_tensor, opts)
            if onion:
                frames = [onion_skin(img, nxt, f) for f in frames]
            for f in frames:
                f.save(out / f"{idx:06d}.png"); idx += 1
            print(f"  [{i+1}/{len(frames_list)-1}] {pair_name}")

    if metrics_all:
        _report_metrics(metrics_all, out)

    total = idx - 1
    print(f"\nDone! {total} frames → {out}")
    print(f"ffmpeg -r 24 -i {out}/%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4")
    return total


def _report_metrics(metrics_all: List[dict], out: Path):
    meta = ("scene", "mode", "scene_tag")
    by_mode: Dict[str, List[dict]] = {}
    for m in metrics_all:
        by_mode.setdefault(m.get("mode", "default"), []).append(m)

    print(f"\n{'=' * 55}")
    summary = {}
    for mode_key, ms in by_mode.items():
        clean = [m for m in ms if m.get("scene_tag") != "SCENE_CUT"]
        keys = [k for k in ms[0] if k not in meta]
        summary[mode_key] = {k: float(np.mean([m[k] for m in clean if k in m]))
                             for k in keys if any(k in m for m in clean)}
        tag = f" ({mode_key})" if mode_key != "default" else ""
        print(f"  AVG METRICS{tag} over {len(clean)} frames (of {len(ms)} total):")
        for k, v in summary[mode_key].items():
            print(f"    {k}: {v:.4f}")
    print(f"{'=' * 55}")

    with open(out / "metrics.json", "w") as f:
        json.dump({"per_frame": metrics_all, "by_mode_avg": summary}, f, indent=2)
    print(f"  Metrics saved to {out / 'metrics.json'}")


def _print_header(model, model_name, n_items, num_frames, easing, use_tta,
                  use_nedt, sketch_arg, roi_mode, ablation, per_region_t,
                  onion, scene_cut, total_out=None):
    print("=" * 55)
    print(f"  Model: {_unwrap(model).__class__.__name__} ({model_name})")
    print(f"  AMP: {USE_AMP} ({AMP_DTYPE}) | RAFT iters: {RAFT_ITERS} | "
          f"GPU-NEDT: {USE_GPU_NEDT} | feature_cache: {USE_FEATURE_CACHE}")
    print(f"  Scenes/frames: {n_items}" +
          (f" | Output frames: {total_out}" if total_out else ""))
    print(f"  Frames/pair: {num_frames} | Easing: {easing} | "
          f"TTA: {use_tta} | NEDT: {use_nedt}")
    extras = [f"sketch={sketch_arg or 'none'}", f"roi={roi_mode}"]
    extras += [name for flag, name in ((per_region_t, "per_region_t=on"),
                                       (onion, "onion=on"),
                                       (scene_cut, "scene_cut=on"),
                                       (ablation, "ABLATION")) if flag]
    print(f"  Extras: {', '.join(extras)}")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def configure(amp=True, amp_dtype="bf16", raft_iters=20,
              feature_cache=True, gpu_nedt=False):
    """Глобальные опции скорости (используются и CLI, и приложением)."""
    global USE_AMP, AMP_DTYPE, RAFT_ITERS, USE_FEATURE_CACHE, USE_GPU_NEDT
    USE_AMP = amp
    AMP_DTYPE = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    RAFT_ITERS = max(1, int(raft_iters))
    USE_FEATURE_CACHE = feature_cache
    USE_GPU_NEDT = gpu_nedt


def apply_fast_preset(args):
    """--fast: RAFT max side 768, 12 итераций, GPU-NEDT (если не заданы явно)."""
    if args.fast:
        if args.raft_max_side == 0:
            args.raft_max_side = 768
        if args.raft_iters == 20:
            args.raft_iters = 12
        args.gpu_nedt = True
    return args


def _build_argparser():
    import argparse
    pa = argparse.ArgumentParser(
        description="AnimeIFNet inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    pa.add_argument("--input", required=True)
    pa.add_argument("--output", required=True)
    pa.add_argument("--checkpoint", required=True)

    pa.add_argument("--model", default="v8_2", choices=list(MODEL_REGISTRY))
    pa.add_argument("--corr-levels", type=int, default=4)
    pa.add_argument("--corr-radius", type=int, default=5)
    pa.add_argument("--residual-scale", type=float, default=0.7)
    pa.add_argument("--splat-feat-ch", type=int, default=16)

    pa.add_argument("--features", default="live", choices=["live", "precomputed"])
    pa.add_argument("--raft-size", default="large", choices=["large", "small"])
    pa.add_argument("--raft-max-side", type=int, default=0,
                    help="Уменьшить вход RAFT для скорости (0 = полное разрешение).")
    pa.add_argument("--raft-iters", type=int, default=20)
    pa.add_argument("--isnet-weights", default=None)
    pa.add_argument("--dis-repo", default=None)
    pa.add_argument("--a2s-weights", default=None)
    pa.add_argument("--a2s-repo", default=None)

    pa.add_argument("--num-frames", type=int, default=1)
    pa.add_argument("--easing", default="linear",
                    choices=["linear", "ease_in", "ease_out", "ease_in_out"])
    pa.add_argument("--recursive", action="store_true")

    pa.add_argument("--per-region-t", action="store_true")
    pa.add_argument("--t-fg", type=float, default=0.6)
    pa.add_argument("--t-bg", type=float, default=0.4)

    pa.add_argument("--onion", action="store_true")
    pa.add_argument("--tta", action="store_true")
    pa.add_argument("--nedt", action="store_true", default=True)
    pa.add_argument("--no-nedt", dest="nedt", action="store_false")

    pa.add_argument("--sketch", default=None,
                    help="none | auto | путь к картинке со скетчем")
    pa.add_argument("--roi", dest="roi_mode", default="full",
                    choices=["full", "seg", "flow_mag", "motion", "file"])
    pa.add_argument("--roi-file", default=None)

    pa.add_argument("--ablation", action="store_true")
    pa.add_argument("--scene-cut", action="store_true")
    pa.add_argument("--save-debug", action="store_true")
    pa.add_argument("--save-masks", action="store_true")
    pa.add_argument("--evaluate", action="store_true")
    pa.add_argument("--compute-cd", action="store_true",
                    help="Считать Chamfer distance (медленно).")

    pa.add_argument("--amp", dest="amp", action="store_true", default=True)
    pa.add_argument("--no-amp", dest="amp", action="store_false")
    pa.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    pa.add_argument("--compile", dest="compile_model", action="store_true")
    pa.add_argument("--no-feature-cache", dest="feature_cache",
                    action="store_false", default=True)
    pa.add_argument("--gpu-nedt", action="store_true", default=False,
                    help="Быстрый DT на GPU (chebyshev). Модель училась на "
                         "евклидовом DT, поэтому по умолчанию выключен.")
    pa.add_argument("--fast", action="store_true",
                    help="raft-max-side 768, raft-iters 12, gpu-nedt: ~×2 быстрее, "
                         "до −1 dB PSNR.")
    return pa


def main(argv=None):
    args = apply_fast_preset(_build_argparser().parse_args(argv))
    configure(args.amp, args.amp_dtype, args.raft_iters,
              args.feature_cache, args.gpu_nedt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | AMP: {USE_AMP} ({AMP_DTYPE}) | "
          f"RAFT iters: {RAFT_ITERS} | RAFT max_side: {args.raft_max_side} | "
          f"GPU-NEDT: {USE_GPU_NEDT}" + (" | --fast" if args.fast else ""))

    model = load_model(args.checkpoint, args.model, device,
                       args.corr_levels, args.corr_radius,
                       args.residual_scale, args.splat_feat_ch,
                       compile_model=args.compile_model)

    need_a2s = (args.model in MODELS_A2S_EDGE and args.features == "live"
                and args.a2s_weights is not None)
    fp = FeatureProvider(
        mode=args.features, device=device,
        raft_size=args.raft_size, raft_max_side=args.raft_max_side,
        isnet_weights=args.isnet_weights, dis_repo=args.dis_repo,
        a2s_weights=args.a2s_weights, a2s_repo=args.a2s_repo,
        need_a2s=need_a2s,
    )
    if args.features == "live" and fp.a2s is None:
        print("[WARN] Anime2Sketch не загружен — edge/dt считаются через DoG; "
              "модель училась на контурах Anime2Sketch.")

    process_folder(
        model, fp, args.input, args.output,
        model_name=args.model,
        roi_mode=args.roi_mode, roi_file=args.roi_file,
        sketch_arg=args.sketch,
        num_frames=args.num_frames, easing=args.easing,
        recursive=args.recursive, use_tta=args.tta, use_nedt=args.nedt,
        per_region_t=args.per_region_t, t_fg=args.t_fg, t_bg=args.t_bg,
        onion=args.onion, ablation=args.ablation, scene_cut=args.scene_cut,
        device=device,
        save_debug=args.save_debug, save_masks=args.save_masks,
        evaluate=args.evaluate, compute_cd=args.compute_cd,
    )


if __name__ == "__main__":
    main()
