"""
Строительные блоки AnimeIFNet.

Здесь собраны все переиспользуемые модули, которые раньше были разбросаны
по animeIFNet_v6 / v7 / v7_3 / v8. Ни один блок не менял ни архитектуру,
ни имена своих подмодулей, поэтому ключи state_dict совпадают со старыми
чекпоинтами.

Содержимое:
  Утилиты          warp, forward_splat, conv_act, conv_act_d
  Энкодеры         SharedEncoder, ContextEncoder, PretrainedContextEncoder,
                   FrozenResNetExtractor
  Оценка потока    CorrBlock, CorrCompressor, IFBlock, FlowRefineBlock,
                   ConvexUpsample
  Синтез           DTMBlock, ForwardSplatBranch,
                   ChannelAttention, SpatialSelfAttention, AttentionBottleneck

Соглашения по диапазонам:
  RGB: [-1, 1];  edge / dt / seg: [0, 1];  flow: пиксели.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "warp", "forward_splat", "conv_act", "conv_act_d",
    "SharedEncoder", "ContextEncoder", "PretrainedContextEncoder",
    "FrozenResNetExtractor",
    "CorrBlock", "CorrCompressor", "IFBlock", "FlowRefineBlock", "ConvexUpsample",
    "DTMBlock", "ForwardSplatBranch",
    "ChannelAttention", "SpatialSelfAttention", "AttentionBottleneck",
]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ═══════════════════════════════════════════════════════════════
# Утилиты
# ═══════════════════════════════════════════════════════════════

def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warping: out(x) = img(x + flow(x)), билинейно, border padding."""
    b, c, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=img.device),
        torch.arange(w, dtype=torch.float32, device=img.device),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
    grid = base + flow.float()
    norm_x = 2.0 * grid[:, 0] / max(w - 1, 1) - 1.0
    norm_y = 2.0 * grid[:, 1] / max(h - 1, 1) - 1.0
    grid_norm = torch.stack([norm_x, norm_y], dim=-1)
    return F.grid_sample(
        img.float(), grid_norm,
        mode="bilinear", padding_mode="border", align_corners=True,
    ).to(dtype=img.dtype)


def forward_splat(src: torch.Tensor, flow: torch.Tensor,
                  metric: Optional[torch.Tensor] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Билинейный forward splatting (pure PyTorch, scatter_add).

    Каждый пиксель источника переносится вперёд по flow и распределяется
    между 4 соседними пикселями цели; конфликты усредняются по весам.

    Args:
        src:    [B, C, H, W] изображение или признаки
        flow:   [B, 2, H, W] смещение src → target
        metric: [B, 1, H, W] опциональная метрика Z для softmax splatting

    Returns:
        output [B, C, H, W], coverage mask [B, 1, H, W]
    """
    B, C, H, W = src.shape
    device, dtype = src.device, src.dtype
    N = H * W

    if metric is not None:
        src_w = src * torch.exp(metric.clamp(-10, 10)).to(dtype)
    else:
        src_w = src

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    target_x = grid_x.unsqueeze(0) + flow[:, 0].to(dtype)
    target_y = grid_y.unsqueeze(0) + flow[:, 1].to(dtype)

    x0 = target_x.floor().long()
    y0 = target_y.floor().long()
    x0f, y0f = x0.to(dtype), y0.to(dtype)

    wa = ((x0f + 1) - target_x) * ((y0f + 1) - target_y)
    wb = (target_x - x0f) * ((y0f + 1) - target_y)
    wc = ((x0f + 1) - target_x) * (target_y - y0f)
    wd = (target_x - x0f) * (target_y - y0f)

    output = src.new_zeros(B, C, H, W)
    weight = src.new_zeros(B, 1, H, W)
    src_flat = src_w.reshape(B, C, N)

    for cx, cy, cw in ((x0, y0, wa), (x0 + 1, y0, wb),
                       (x0, y0 + 1, wc), (x0 + 1, y0 + 1, wd)):
        valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        idx = (cy.clamp(0, H - 1) * W + cx.clamp(0, W - 1)).reshape(B, 1, N)
        w_masked = (cw * valid.to(dtype)).reshape(B, 1, N)
        output.reshape(B, C, N).scatter_add_(
            2, idx.expand(B, C, N), src_flat * w_masked.expand_as(src_flat))
        weight.reshape(B, 1, N).scatter_add_(2, idx, w_masked)

    mask = (weight > 1e-4).to(dtype)
    return output / (weight + 1e-8), mask


def conv_act(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.PReLU(out_ch),
    )


def conv_act_d(in_ch: int, out_ch: int, dilation: int = 1) -> nn.Sequential:
    """Conv 3x3 + PReLU с заданным dilation."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
        nn.PReLU(out_ch),
    )


# ═══════════════════════════════════════════════════════════════
# Энкодеры
# ═══════════════════════════════════════════════════════════════

class SharedEncoder(nn.Module):
    """Сиамский энкодер для matching. Вход: RGB(3)+edge(1)+dt(1)+seg(1)=6.
    Выход: признаки на 1/2 (48ch), 1/4 (96ch), 1/8 (128ch)."""

    def __init__(self, in_ch: int = 6):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, 48, 3, stride=2, padding=1), nn.PReLU(48),
            nn.Conv2d(48, 48, 3, padding=1), nn.PReLU(48),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.PReLU(96),
            nn.Conv2d(96, 96, 3, padding=1), nn.PReLU(96),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.PReLU(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.PReLU(128),
        )

    def forward(self, x: torch.Tensor):
        f_s2 = self.layer1(x)
        f_s4 = self.layer2(f_s2)
        f_s8 = self.layer3(f_s4)
        return f_s2, f_s4, f_s8


class ContextEncoder(nn.Module):
    """Лёгкий обучаемый энкодер контекста для синтеза (вариант без
    предобученного backbone). Выход: ctx_s2 (32ch, 1/2), ctx_s4 (48ch, 1/4)."""

    def __init__(self, in_ch: int = 6):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.PReLU(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.PReLU(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.PReLU(48),
            nn.Conv2d(48, 48, 3, padding=1), nn.PReLU(48),
        )

    def forward(self, x: torch.Tensor):
        ctx_s2 = self.layer1(x)
        ctx_s4 = self.layer2(ctx_s2)
        return ctx_s2, ctx_s4


class PretrainedContextEncoder(nn.Module):
    """
    Энкодер контекста на базе EfficientNet-B0 (первые 3 стадии).

    RGB → backbone; edge+dt+seg → лёгкая обучаемая ветка;
    конкатенация + 1x1 проекция → ctx_s2 (32ch, H/2), ctx_s4 (48ch, H/4).

    Args:
        freeze_backbone: заморозить EfficientNet (разморозка — unfreeze_backbone()).
        pretrained: грузить ImageNet-веса. Для инференса из чекпоинта можно
            передать False: веса backbone всё равно лежат в state_dict.
    """

    def __init__(self, freeze_backbone: bool = True, pretrained: bool = True):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        eff = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.bb_s2 = nn.Sequential(eff.features[0], eff.features[1])  # 16ch @ H/2
        self.bb_s4 = eff.features[2]                                   # 24ch @ H/4

        if freeze_backbone:
            for p in list(self.bb_s2.parameters()) + list(self.bb_s4.parameters()):
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).reshape(1, 3, 1, 1))

        self.aux_s2 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.PReLU(16))
        self.aux_s4 = nn.Sequential(nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.PReLU(16))
        self.proj_s2 = nn.Sequential(nn.Conv2d(16 + 16, 32, 1), nn.PReLU(32))
        self.proj_s4 = nn.Sequential(nn.Conv2d(24 + 16, 48, 1), nn.PReLU(48))

    def _norm_rgb(self, x):
        return ((x + 1) / 2 - self.mean) / self.std

    def unfreeze_backbone(self):
        for p in list(self.bb_s2.parameters()) + list(self.bb_s4.parameters()):
            p.requires_grad = True

    def forward(self, x: torch.Tensor):
        """x: [B, 6, H, W] = RGB(3) + edge + dt + seg."""
        rgb = self._norm_rgb(x[:, :3])
        aux = x[:, 3:]
        bb_s2 = self.bb_s2(rgb)
        bb_s4 = self.bb_s4(bb_s2)
        aux_s2 = self.aux_s2(aux)
        aux_s4 = self.aux_s4(aux_s2)
        ctx_s2 = self.proj_s2(torch.cat([bb_s2, aux_s2], dim=1))
        ctx_s4 = self.proj_s4(torch.cat([bb_s4, aux_s4], dim=1))
        return ctx_s2, ctx_s4


class FrozenResNetExtractor(nn.Module):
    """
    Замороженный ResNet-50 (conv1 … layer1) → 256ch @ H/4,
    плюс обучаемая 1x1 проекция в out_ch. BatchNorm backbone всегда в eval.
    """

    def __init__(self, out_ch: int = 16, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
        )
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

        self.proj = nn.Sequential(nn.Conv2d(256, out_ch, 1), nn.PReLU(out_ch))
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).reshape(1, 3, 1, 1))

    def _norm(self, x):
        return ((x + 1) / 2 - self.mean) / self.std

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] в [-1, 1] → [B, out_ch, H/4, W/4]."""
        with torch.no_grad():
            feat = self.features(self._norm(x))
        return self.proj(feat)


# ═══════════════════════════════════════════════════════════════
# Оценка потока
# ═══════════════════════════════════════════════════════════════

class CorrBlock:
    """
    Пирамида 4D cost volume (RAFT-style) с segment-guided модуляцией:
        corr *= alpha + (1 - alpha) * seg_compat,
        seg_compat[i, j] = s_src[i] * s_tgt[j] + (1 - s_src[i]) * (1 - s_tgt[j]).
    Пары пикселей из разных слоёв (fg/bg) ослабляются, но не обнуляются.
    Не nn.Module: обучаемых параметров нет.
    """

    def __init__(self, f_src: torch.Tensor, f_tgt: torch.Tensor,
                 seg_src: Optional[torch.Tensor] = None,
                 seg_tgt: Optional[torch.Tensor] = None,
                 num_levels: int = 4, radius: int = 5, seg_alpha: float = 0.3):
        self.num_levels = num_levels
        self.radius = radius

        b, c, h, w = f_src.shape
        self.h, self.w = h, w

        f_src = f_src.detach()
        f_tgt = f_tgt.detach()
        f_src = f_src / (f_src.norm(dim=1, keepdim=True) + 1e-6)
        f_tgt = f_tgt / (f_tgt.norm(dim=1, keepdim=True) + 1e-6)

        f_src_flat = f_src.reshape(b, c, h * w).half()
        f_tgt_flat = f_tgt.reshape(b, c, h * w).half()
        corr = torch.bmm(f_src_flat.permute(0, 2, 1), f_tgt_flat).float()
        del f_src_flat, f_tgt_flat
        corr = corr.reshape(b, h * w, h, w)

        if seg_src is not None and seg_tgt is not None:
            seg_s = seg_src.detach().reshape(b, 1, h * w).half()
            seg_t = seg_tgt.detach().reshape(b, 1, h * w).half()
            fg = torch.bmm(seg_s.permute(0, 2, 1), seg_t)
            bg = torch.bmm((1 - seg_s).permute(0, 2, 1), (1 - seg_t))
            del seg_s, seg_t
            seg_compat = (fg + bg).float().reshape(b, h * w, h, w)
            del fg, bg
            corr = corr * (seg_alpha + (1.0 - seg_alpha) * seg_compat)

        self.corr_pyramid = [corr]
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def lookup(self, flow_s8: torch.Tensor) -> torch.Tensor:
        r = self.radius
        b = flow_s8.shape[0]
        h, w = self.h, self.w
        dev, dt = flow_s8.device, flow_s8.dtype

        yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=dt),
                                torch.arange(w, device=dev, dtype=dt), indexing="ij")
        cx = xx.unsqueeze(0).expand(b, -1, -1) + flow_s8[:, 0]
        cy = yy.unsqueeze(0).expand(b, -1, -1) + flow_s8[:, 1]

        offsets = torch.arange(-r, r + 1, device=dev, dtype=dt)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")

        out_levels = []
        for level in range(self.num_levels):
            corr = self.corr_pyramid[level]
            _, _, hl, wl = corr.shape
            scale = 2 ** level
            corr_flat = corr.reshape(b * h * w, 1, hl, wl)
            cx_flat = (cx / scale).reshape(b * h * w)
            cy_flat = (cy / scale).reshape(b * h * w)

            sx = 2.0 * (cx_flat[:, None, None] + dx[None]) / max(wl - 1, 1) - 1.0
            sy = 2.0 * (cy_flat[:, None, None] + dy[None]) / max(hl - 1, 1) - 1.0
            grid = torch.stack([sx, sy], dim=-1)

            sampled = F.grid_sample(corr_flat, grid, align_corners=True,
                                    mode="bilinear", padding_mode="zeros")
            sampled = sampled.reshape(b, h, w, (2 * r + 1) ** 2).permute(0, 3, 1, 2)
            out_levels.append(sampled)

        return torch.cat(out_levels, dim=1)


class CorrCompressor(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.PReLU(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IFBlock(nn.Module):
    """
    Один шаг итеративной оценки потока на 1/8 (U-Net-подобный).
    Выход: Δflow (4 = t→0 и t→1) и Δocc-логиты (3 канала):
      0 — disocclusion, 1 — вес warped0, 2 — вес warped1.
    """

    def __init__(self, in_ch: int, c_hidden: int = 192):
        super().__init__()
        self.conv0 = conv_act(in_ch, 96, stride=2)
        self.conv1 = conv_act(96, 128, stride=2)
        self.conv2 = conv_act(128, c_hidden, stride=2)
        self.up2 = nn.ConvTranspose2d(c_hidden, 128, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1)
        self.up0 = nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1)
        self.act2 = nn.PReLU(128)
        self.act1 = nn.PReLU(96)
        self.head = nn.Conv2d(48, 7, 3, padding=1)

    @staticmethod
    def _match(up, skip):
        if up.shape[2:] != skip.shape[2:]:
            return up[:, :, :skip.shape[2], :skip.shape[3]]
        return up

    def forward(self, x: torch.Tensor):
        in_h, in_w = x.shape[2], x.shape[3]
        f0 = self.conv0(x)
        f1 = self.conv1(f0)
        f2 = self.conv2(f1)
        d2 = self.act2(self._match(self.up2(f2), f1) + f1)
        d1 = self.act1(self._match(self.up1(d2), f0) + f0)
        out = self.head(self.up0(d1))
        if out.shape[2] != in_h or out.shape[3] != in_w:
            out = out[:, :, :in_h, :in_w]
        return out[:, :4], out[:, 4:7]


class FlowRefineBlock(nn.Module):
    """
    Уточнение потока на 1/4 с dilated-свёртками (RF 3 → 7 → 15).
    Вход: flow(4) + match(192) + ctx(96) + warped(6) + occ(3) + seg(2) = 303.
    """

    IN_CH = 4 + 192 + 96 + 6 + 3 + 2  # 303

    def __init__(self):
        super().__init__()
        self.compress = conv_act(self.IN_CH, 128)
        self.d1 = conv_act_d(128, 128, dilation=1)
        self.d2 = conv_act_d(128, 128, dilation=2)
        self.d4 = conv_act_d(128, 96, dilation=4)
        self.fuse = conv_act(96, 64)
        self.head = nn.Conv2d(64, 7, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.fuse(self.d4(self.d2(self.d1(self.compress(x)))))
        out = self.head(x)
        return out[:, :4], out[:, 4:7]


class ConvexUpsample(nn.Module):
    """Обучаемый convex upsampling потока: 1/4 → 1/2 (convex) → full (bilinear)."""

    def __init__(self, feat_ch: int = 96, flow_ch: int = 4):
        super().__init__()
        self.flow_ch = flow_ch
        self.weight_net = nn.Sequential(
            nn.Conv2d(feat_ch + flow_ch, 128, 3, padding=1),
            nn.PReLU(128),
            nn.Conv2d(128, flow_ch * 9, 1),
        )

    def forward(self, flow_s4, f0_s2, f1_s2):
        b, c, h4, w4 = flow_s4.shape
        h2, w2 = f0_s2.shape[2], f0_s2.shape[3]

        flow_s2 = F.interpolate(flow_s4, size=(h2, w2), mode="bilinear",
                                align_corners=False) * 2.0
        guide = torch.cat([f0_s2, f1_s2, flow_s2], dim=1)
        weights = torch.softmax(self.weight_net(guide).reshape(b, c, 9, h2, w2), dim=2)

        flow_unf = F.unfold(F.pad(flow_s4, [1, 1, 1, 1], mode="replicate"),
                            kernel_size=3, padding=0)
        flow_unf = flow_unf.reshape(b, c * 9, h4, w4)
        flow_unf_s2 = F.interpolate(flow_unf, size=(h2, w2), mode="nearest")
        flow_unf_s2 = flow_unf_s2.reshape(b, c, 9, h2, w2) * 2.0

        flow_s2_refined = (weights * flow_unf_s2).sum(dim=2)
        return F.interpolate(flow_s2_refined, scale_factor=2, mode="bilinear",
                             align_corners=False) * 2.0


# ═══════════════════════════════════════════════════════════════
# Синтез
# ═══════════════════════════════════════════════════════════════

class DTMBlock(nn.Module):
    """
    Distance Transform Module (идея из EISAI): модулирует residual по
    нормированному расстоянию до ближайшей линии. В плоских заливках (dt≈1)
    warp ненадёжен, и DTM усиливает коррекцию. ~11K параметров.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.PReLU(16),
            nn.Conv2d(16, 16, 3, padding=1), nn.PReLU(16),
            nn.Conv2d(16, 3, 1), nn.Sigmoid(),
        )

    def forward(self, residual, dt0, dt1):
        mod = self.net(torch.cat([residual, dt0, dt1], dim=1))
        return residual * (0.5 + 0.5 * mod)


class ForwardSplatBranch(nn.Module):
    """
    Forward splatting RGB и ResNet-признаков обоих кадров к моменту t на H/4,
    взаимное заполнение пустот (occlusion infilling) и upsample до full res.
    Потоки к t линейно масштабируются из RAFT: 0→t = t·fwd, 1→t = (1−t)·bwd.
    """

    def __init__(self, feat_ch: int = 16, pretrained: bool = True):
        super().__init__()
        self.feat_ch = feat_ch
        self.resnet = FrozenResNetExtractor(out_ch=feat_ch, pretrained=pretrained)

    def _splat_one(self, img, feat, flow_to_t):
        h4, w4 = feat.shape[2], feat.shape[3]
        flow_s4 = F.interpolate(flow_to_t, size=(h4, w4), mode="bilinear",
                                align_corners=False) / 4.0
        img_s4 = F.interpolate(img, size=(h4, w4), mode="bilinear", align_corners=False)
        splat_rgb, mask = forward_splat(img_s4, flow_s4)
        splat_feat, _ = forward_splat(feat, flow_s4)
        return splat_rgb, splat_feat, mask

    def forward(self, i0, i1, flow_fwd, flow_bwd, t=0.5):
        _, _, H, W = i0.shape
        feat0 = self.resnet(i0)
        feat1 = self.resnet(i1)

        sr0, sf0, sm0 = self._splat_one(i0, feat0, t * flow_fwd)
        sr1, sf1, sm1 = self._splat_one(i1, feat1, (1 - t) * flow_bwd)

        sf0_filled = sf0 * sm0 + sf1 * (1 - sm0)
        sf1_filled = sf1 * sm1 + sf0 * (1 - sm1)
        sr0_filled = sr0 * sm0 + sr1 * (1 - sm0)
        sr1_filled = sr1 * sm1 + sr0 * (1 - sm1)

        def up(x):
            return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return {
            "splat_rgb0": up(sr0_filled), "splat_rgb1": up(sr1_filled),
            "splat_feat0": up(sf0_filled), "splat_feat1": up(sf1_filled),
            "splat_mask0": up(sm0), "splat_mask1": up(sm1),
        }


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention по пространству (pre-LayerNorm, residual)."""

    def __init__(self, channels, num_heads=4, proj_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Conv2d(channels, proj_dim, 1)
        self.k = nn.Conv2d(channels, proj_dim, 1)
        self.v = nn.Conv2d(channels, proj_dim, 1)
        self.out = nn.Conv2d(proj_dim, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        xn = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        N = H * W
        q = self.q(xn).reshape(B, self.num_heads, self.head_dim, N)
        k = self.k(xn).reshape(B, self.num_heads, self.head_dim, N)
        v = self.v(xn).reshape(B, self.num_heads, self.head_dim, N)
        attn = (q.transpose(-2, -1) @ k * self.scale).softmax(-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        return x + self.out(out)


class AttentionBottleneck(nn.Module):
    """Strided conv-блок → channel attention → spatial self-attention."""

    def __init__(self, in_ch, out_ch, num_heads=4, proj_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch), nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.PReLU(out_ch),
        )
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialSelfAttention(out_ch, num_heads, proj_dim)

    def forward(self, x):
        return self.sa(self.ca(self.conv(x)))
