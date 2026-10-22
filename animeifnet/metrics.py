"""
Метрики качества и вспомогательные функции для оценки.

Все функции принимают изображения [C, H, W] в диапазоне [-1, 1].
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def calc_psnr(p: torch.Tensor, t: torch.Tensor) -> float:
    p = (p.float().clamp(-1, 1) + 1) * .5
    t = (t.float().clamp(-1, 1) + 1) * .5
    return 10 * math.log10(1 / ((p - t).pow(2).mean().item() + 1e-8))


@torch.no_grad()
def calc_ssim(p: torch.Tensor, t: torch.Tensor, ws: int = 11) -> float:
    """SSIM с гауссовым окном 11×11, σ = 1.5."""
    p = (p.float().clamp(-1, 1)[None] + 1) * .5
    t = (t.float().clamp(-1, 1)[None] + 1) * .5
    c1, c2 = .01 ** 2, .03 ** 2
    ch = p.size(1)
    co = torch.arange(ws, dtype=torch.float32, device=p.device) - ws // 2
    g = torch.exp(-co ** 2 / (2 * 1.5 ** 2))
    g /= g.sum()
    k = (g[:, None] * g[None, :]).expand(ch, 1, ws, ws).contiguous()
    pad = ws // 2
    mu1 = F.conv2d(p, k, padding=pad, groups=ch)
    mu2 = F.conv2d(t, k, padding=pad, groups=ch)
    s1 = F.conv2d(p * p, k, padding=pad, groups=ch) - mu1 ** 2
    s2 = F.conv2d(t * t, k, padding=pad, groups=ch) - mu2 ** 2
    s12 = F.conv2d(p * t, k, padding=pad, groups=ch) - mu1 * mu2
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)
            / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))).mean().item()


@torch.no_grad()
def masked_psnr(p: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    """PSNR только по пикселям mask > 0.5. None, если маска почти пустая."""
    p = (p.float().clamp(-1, 1) + 1) * .5
    t = (t.float().clamp(-1, 1) + 1) * .5
    m = (mask.float().reshape(1, *p.shape[-2:]) > 0.5).float()
    n_pix = m.sum().item()
    if n_pix < 16:
        return None
    mse = ((p - t).pow(2) * m).sum().item() / (n_pix * p.shape[0])
    return 10 * math.log10(1 / (mse + 1e-8))


def is_scene_change(i0: torch.Tensor, i1: torch.Tensor, mse_thr: float = 0.35,
                    hist_thr: float = 0.5) -> bool:
    """
    Смена сцены между кадрами [3, H, W] или [1, 3, H, W] в [-1, 1]:
    большая MSE или большое расстояние Бхаттачарьи между цветовыми гистограммами.
    """
    p0 = (i0.float().reshape(-1, *i0.shape[-3:])[0].clamp(-1, 1) + 1) / 2
    p1 = (i1.float().reshape(-1, *i1.shape[-3:])[0].clamp(-1, 1) + 1) / 2
    if (p0 - p1).pow(2).mean().item() > mse_thr:
        return True
    a = (p0.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    b = (p1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hd = 0.0
    for ch in range(3):
        ha = cv2.calcHist([a], [ch], None, [64], [0, 256])
        hb = cv2.calcHist([b], [ch], None, [64], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        hd += cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)
    return hd / 3 > hist_thr


def sharpen_lines(pred: torch.Tensor, strength: float = 0.3, dilate: int = 3) -> torch.Tensor:
    """
    Постобработка: unsharp mask только вдоль контуров (Canny + дилатация).
    Это не часть модели — метрики после неё отчитываются отдельно.
    """
    if strength <= 0:
        return pred
    pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    pnp = ((pred.clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy()
    gray = cv2.cvtColor(np.clip(pnp, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    med = np.median(gray)
    edges = cv2.Canny(gray, int(max(0, .5 * med)), int(min(255, 1.5 * med)))
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1,) * 2)
        edges = cv2.dilate(edges, k)
    mask = cv2.GaussianBlur(edges.astype(np.float32) / 255, (0, 0), 1.5)[:, :, None]
    blurred = cv2.GaussianBlur(pnp, (0, 0), 1.0)
    sharp = np.clip(pnp + strength * (pnp - blurred), 0, 255)
    result = pnp * (1 - mask) + sharp * mask
    return torch.from_numpy(result.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)


def seg_roi_mask(seg0: torch.Tensor, seg1: torch.Tensor, thr: float = 0.3,
                 dilate_k: int = 15, dilate_iter: int = 2, blur_k: int = 11) -> torch.Tensor:
    """
    ROI персонажа: union(seg0, seg1) > thr → дилатация → размытие.
    [1, H, W] в [0, 1]; если маска почти пустая — весь кадр.
    """
    union = torch.maximum(seg0.reshape(-1, *seg0.shape[-2:])[0],
                          seg1.reshape(-1, *seg1.shape[-2:])[0]).cpu().numpy()
    b = (union > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    b = cv2.dilate(b, k, iterations=dilate_iter)
    m = np.clip(cv2.GaussianBlur(b.astype(np.float32), (blur_k, blur_k), 0), 0, 1)
    if m.sum() < 64:
        m = np.ones_like(m)
    return torch.from_numpy(m.astype(np.float32))[None]
