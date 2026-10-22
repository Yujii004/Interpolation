"""
Функции потерь AnimeIFNet.

Базовые (на финальном кадре pred):
  charb   Charbonnier (робастный L1)
  edge    L1 между картами Собеля — резкость контуров
  raft    согласованность предсказанного потока с линейной аппроксимацией RAFT
  coarse  Charbonnier на coarse-бленде — держит качество warping-ветки
  ssim    1 − SSIM
  warp    L1 каждого warped-кадра к эталону
  lap     L1 по уровням лапласовой пирамиды
  percep  VGG-19 perceptual loss

Подсказки (на pred_local — выходе RefineNet до ROI-композиции):
  roi       Charbonnier к эталону внутри ROI
  preserve  Charbonnier к coarse вне ROI — модель не должна трогать остальное
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Веса и расписание
# ═══════════════════════════════════════════════════════════════

@dataclass
class LossConfig:
    w_char: float = 1.0
    w_edge: float = 0.15
    w_raft: float = 0.10
    w_coarse: float = 0.3
    w_ssim: float = 0.2
    w_warp: float = 0.08
    w_lap: float = 0.3
    w_percep: float = 0.05
    w_roi: float = 0.3
    w_preserve: float = 0.1

    # С какой эпохи включается каждый лосс (charb/edge/raft — с первой)
    coarse_start: int = 3
    ssim_start: int = 5
    warp_start: int = 10
    lap_start: int = 10
    percep_start: int = 30
    percep_warmup: int = 10
    roi_start: int = 30

    def to_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# Отдельные лоссы
# ═══════════════════════════════════════════════════════════════

def charbonnier(p: torch.Tensor, t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((p - t) ** 2 + eps ** 2).mean()


def masked_charbonnier(p, t, mask, eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier, усреднённый только по пикселям маски [B,1,H,W]."""
    diff = torch.sqrt((p - t) ** 2 + eps ** 2)
    return (diff * mask).sum() / (mask.sum() * p.size(1) + 1e-8)


def ssim_loss(p, t, ws: int = 11) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        p, t = (p.float() + 1) / 2, (t.float() + 1) / 2
        c1, c2 = .01 ** 2, .03 ** 2
        ch = p.size(1)
        co = torch.arange(ws, dtype=torch.float32, device=p.device) - ws // 2
        g = torch.exp(-co ** 2 / (2 * 1.5 ** 2))
        g /= g.sum()
        k = (g[:, None] * g[None, :]).expand(ch, 1, ws, ws).contiguous()
        pad = ws // 2
        mu1 = F.conv2d(p, k, padding=pad, groups=ch)
        mu2 = F.conv2d(t, k, padding=pad, groups=ch)
        s1 = F.conv2d(p * p, k, padding=pad, groups=ch) - mu1 * mu1
        s2 = F.conv2d(t * t, k, padding=pad, groups=ch) - mu2 * mu2
        s12 = F.conv2d(p * t, k, padding=pad, groups=ch) - mu1 * mu2
        sm = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / (
            (mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
        return 1 - sm.mean()


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    g = x.mean(1, keepdim=True)
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=x.device, dtype=x.dtype)
    ky = kx.transpose(-1, -2)
    return torch.sqrt(F.conv2d(g, kx, padding=1) ** 2 + F.conv2d(g, ky, padding=1) ** 2 + 1e-6)


def edge_loss(p, t) -> torch.Tensor:
    return F.l1_loss(sobel_edges(p), sobel_edges(t))


def warp_loss(w0, w1, t) -> torch.Tensor:
    return 0.5 * (F.l1_loss(w0, t) + F.l1_loss(w1, t))


def raft_consistency(ft0, ft1, flow_fwd, flow_bwd, t: float = 0.5) -> torch.Tensor:
    """Поток t→0 / t→1 должен быть близок к линейной аппроксимации из RAFT."""
    tt0 = 0.5 * (-t * flow_fwd + (1 - t) * flow_bwd)
    tt1 = 0.5 * ((1 - t) * flow_fwd + (-t) * flow_bwd)
    return 0.5 * (F.smooth_l1_loss(ft0, tt0.detach()) + F.smooth_l1_loss(ft1, tt1.detach()))


class LaplacianPyramidLoss(nn.Module):
    def __init__(self, levels: int = 4):
        super().__init__()
        self.levels = levels
        self.weights = [2 ** i for i in range(levels)]
        k = torch.tensor([1., 4., 6., 4., 1.]) / 16.0
        self.register_buffer("gk", (k[:, None] * k[None, :])[None, None])

    def _down(self, x):
        c = x.size(1)
        return F.conv2d(F.pad(x, [2] * 4, mode="reflect"),
                        self.gk.expand(c, 1, 5, 5), groups=c)[:, :, ::2, ::2]

    def forward(self, pred, target):
        loss = pred.new_tensor(0.)
        p, t = pred, target
        for w in self.weights:
            pd, td = self._down(p), self._down(t)
            pu = F.interpolate(pd, size=p.shape[2:], mode="bilinear", align_corners=False)
            tu = F.interpolate(td, size=t.shape[2:], mode="bilinear", align_corners=False)
            loss = loss + w * F.l1_loss(p - pu, t - tu)
            p, t = pd, td
        return loss + self.weights[-1] * F.l1_loss(p, t)


class VGGPerceptualLoss(nn.Module):
    """L1 между признаками VGG-19 (relu1_2 … relu4_4)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        ch = list(vgg.children())
        self.stages = nn.ModuleList([nn.Sequential(*ch[:4]), nn.Sequential(*ch[4:9]),
                                     nn.Sequential(*ch[9:17]), nn.Sequential(*ch[17:26])])
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))
        self.w = [1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6]

    def train(self, mode: bool = True):
        return super().train(False)  # всегда eval

    def _norm(self, x):
        return ((x.float() + 1) / 2 - self.mean) / self.std

    def forward(self, pred, target):
        p = self._norm(pred)
        with torch.no_grad():
            t = self._norm(target)
            t_feats = []
            for s in self.stages:
                t = s(t)
                t_feats.append(t)
        loss = pred.new_tensor(0.)
        for s, w, tf in zip(self.stages, self.w, t_feats):
            p = s(p)
            loss = loss + w * F.l1_loss(p, tf)
        return loss


# ═══════════════════════════════════════════════════════════════
# Суммарный лосс
# ═══════════════════════════════════════════════════════════════

class AnimeIFNetLoss(nn.Module):
    """
    Суммарный лосс с расписанием по эпохам.
    forward(out, target, flow_fwd, flow_bwd, roi, epoch, t) → (total, {name: value})
    """

    def __init__(self, cfg: LossConfig = None, vgg_pretrained: bool = True):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.lap = LaplacianPyramidLoss(levels=4)
        self.vgg = VGGPerceptualLoss(pretrained=vgg_pretrained) if self.cfg.w_percep > 0 else None

    def forward(self, out, target, flow_fwd, flow_bwd, roi, epoch: int,
                t: float = 0.5) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        c = self.cfg
        pred, pred_local, coarse = out["pred"], out["pred_local"], out["coarse"]
        L = {
            "charb": c.w_char * charbonnier(pred, target),
            "edge": c.w_edge * edge_loss(pred, target),
            "raft": c.w_raft * raft_consistency(out["flow_t0"], out["flow_t1"],
                                                flow_fwd, flow_bwd, t),
        }
        if epoch >= c.coarse_start:
            L["coarse"] = c.w_coarse * charbonnier(coarse, target)
        if epoch >= c.ssim_start:
            L["ssim"] = c.w_ssim * ssim_loss(pred, target)
        if epoch >= c.warp_start:
            L["warp"] = c.w_warp * warp_loss(out["warped0"], out["warped1"], target)
        if epoch >= c.lap_start:
            L["lap"] = c.w_lap * self.lap(pred, target)
        if self.vgg is not None and epoch >= c.percep_start:
            warmup = min(1.0, (epoch - c.percep_start) / max(c.percep_warmup, 1))
            L["percep"] = c.w_percep * warmup * self.vgg(pred, target)

        # ROI-лоссы — только когда маска действительно что-то ограничивает
        if epoch >= c.roi_start and 0.02 < roi.mean().item() < 0.98:
            L["roi"] = c.w_roi * masked_charbonnier(pred_local, target, roi)
            L["preserve"] = c.w_preserve * masked_charbonnier(pred_local, coarse, 1.0 - roi)

        return sum(L.values()), L


def psnr(p: torch.Tensor, t: torch.Tensor) -> float:
    """PSNR для изображений в [-1, 1]."""
    p, t = (p.clamp(-1, 1) + 1) / 2, (t.clamp(-1, 1) + 1) / 2
    mse = F.mse_loss(p.float(), t.float()).item()
    return 10 * torch.log10(torch.tensor(1 / (mse + 1e-8))).item()
