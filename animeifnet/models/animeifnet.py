"""
AnimeIFNet — интерполяция промежуточных кадров для 2D-анимации.

Финальная модель (бывшая AnimeIFNetV8_2), собранная из цепочки
v6 → v7 → v7.3 → v8 → v8.2 в один файл без наследования между версиями.
Имена подмодулей сохранены, поэтому чекпоинты V8.2 загружаются как есть
(load_state_dict(strict=True)).

Пайплайн:
  1. IFNet — итеративная оценка двунаправленного потока к моменту t:
       • инициализация от RAFT (flow_fwd / flow_bwd);
       • segment-guided correlation volume на 1/8;
       • 5 итераций IFBlock на 1/8 → FlowRefineBlock на 1/4 → convex upsample;
       • 3-канальная окклюзия (disocclusion / warped0 / warped1);
       • t — скаляр или per-pixel карта [B,1,H,W].
  2. Backward warp обоих кадров + взвешенный blend → coarse.
  3. ForwardSplatBranch — forward splatting RGB и признаков ResNet-50.
  4. RefineNet (U-Net + attention bottleneck) → residual.
  5. DTMBlock — модуляция residual по distance transform до контуров.
  6. Опциональные подсказки художника:
       sketch_hint — эскиз целевой формы, roi_mask — где разрешено уточнять;
       pred = roi * (coarse + residual) + (1 − roi) * coarse.

Входы forward():
  i0, i1       [B,3,H,W]  RGB в [-1, 1]
  edge0, edge1 [B,1,H,W]  карты контуров в [0, 1]
  dt0, dt1     [B,1,H,W]  нормированный distance transform до контуров
  flow_fwd     [B,2,H,W]  RAFT-поток 0 → 1 (пиксели)
  flow_bwd     [B,2,H,W]  RAFT-поток 1 → 0
  seg0, seg1   [B,1,H,W]  маска переднего плана
  t            float или [B,1,H,W]
  sketch_hint, roi_mask   [B,1,H,W] в [0, 1], опционально
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    warp, SharedEncoder, PretrainedContextEncoder,
    CorrBlock, CorrCompressor, IFBlock, FlowRefineBlock, ConvexUpsample,
    DTMBlock, ForwardSplatBranch, AttentionBottleneck,
)

TimeArg = Union[float, int, torch.Tensor]

__all__ = ["IFNet", "RefineNet", "AnimeIFNet"]


def _resize(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


# ═══════════════════════════════════════════════════════════════
# 1. IFNet — оценка потока
# ═══════════════════════════════════════════════════════════════

class IFNet(nn.Module):
    """
    Итеративная оценка потока t→0 / t→1.

    Работа разделена на две стадии, чтобы при генерации нескольких кадров
    тяжёлая часть (энкодеры + correlation volume) считалась один раз:
      encode_shared(...)      — всё, что не зависит от t;
      estimate_flow(shared, t) — поток и окклюзия для конкретного t.

    Вход IFBlock на 1/8:
      corr fwd+bwd(128) + f_s8(256) + warped(6) + flow(4) + occ(3) + t(1)
      + edge(2) + dt(2) + seg(2) = 404
    """

    IN_CH = 64 + 64 + 256 + 6 + 4 + 3 + 1 + 2 + 2 + 2  # 404

    def __init__(self, corr_levels: int = 4, corr_radius: int = 5,
                 num_iters: int = 5, freeze_backbone: bool = True,
                 pretrained: bool = True):
        super().__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius

        self.encoder = SharedEncoder(in_ch=6)
        self.ctx_encoder = PretrainedContextEncoder(
            freeze_backbone=freeze_backbone, pretrained=pretrained)

        corr_raw_ch = corr_levels * (2 * corr_radius + 1) ** 2
        self.corr_compress_fwd = CorrCompressor(in_ch=corr_raw_ch, out_ch=64)
        self.corr_compress_bwd = CorrCompressor(in_ch=corr_raw_ch, out_ch=64)

        self.blocks = nn.ModuleList(
            [IFBlock(self.IN_CH, c_hidden=192) for _ in range(num_iters)])
        self.flow_refine = FlowRefineBlock()
        self.convex_up = ConvexUpsample(feat_ch=96, flow_ch=4)

    # ── t helpers ─────────────────────────────────────────────

    @staticmethod
    def _make_t_ch(t: TimeArg, b, h, w, device, dtype) -> torch.Tensor:
        if isinstance(t, (int, float)):
            return torch.full((b, 1, h, w), float(t), device=device, dtype=dtype)
        if t.shape[2] != h or t.shape[3] != w:
            return _resize(t, (h, w))
        return t

    @staticmethod
    def _t_scalar(t: TimeArg) -> float:
        return float(t) if isinstance(t, (int, float)) else float(t.mean().item())

    @staticmethod
    def _init_flow_from_raft(flow_fwd, flow_bwd, h8, w8, t: float):
        """Линейная аппроксимация потоков t→0 и t→1 из RAFT на масштабе 1/8."""
        fwd_s8 = _resize(flow_fwd, (h8, w8)) / 8.0
        bwd_s8 = _resize(flow_bwd, (h8, w8)) / 8.0
        flow_t0 = 0.5 * (-t * fwd_s8 + (1 - t) * bwd_s8)
        flow_t1 = 0.5 * ((1 - t) * fwd_s8 + (-t) * bwd_s8)
        return torch.cat([flow_t0, flow_t1], dim=1)

    # ── t-independent part ───────────────────────────────────

    def encode_shared(self, i0, i1, edge0, edge1, dt0, dt1,
                      seg0, seg1, flow_fwd, flow_bwd) -> Dict:
        b, _, h, w = i0.shape
        enc_in0 = torch.cat([i0, edge0, dt0, seg0], dim=1)
        enc_in1 = torch.cat([i1, edge1, dt1, seg1], dim=1)

        f0_s2, f0_s4, f0_s8 = self.encoder(enc_in0)
        f1_s2, f1_s4, f1_s8 = self.encoder(enc_in1)
        ctx0_s2, ctx0_s4 = self.ctx_encoder(enc_in0)
        ctx1_s2, ctx1_s4 = self.ctx_encoder(enc_in1)

        h8, w8 = f0_s8.shape[2:]
        h4, w4 = f0_s4.shape[2:]

        seg0_s8, seg1_s8 = _resize(seg0, (h8, w8)), _resize(seg1, (h8, w8))
        corr_fwd = CorrBlock(f0_s8, f1_s8, seg_src=seg0_s8, seg_tgt=seg1_s8,
                             num_levels=self.corr_levels, radius=self.corr_radius)
        corr_bwd = CorrBlock(f1_s8, f0_s8, seg_src=seg1_s8, seg_tgt=seg0_s8,
                             num_levels=self.corr_levels, radius=self.corr_radius)

        return {
            "b": b, "h": h, "w": w, "h8": h8, "w8": w8, "h4": h4, "w4": w4,
            "f0_s2": f0_s2, "f1_s2": f1_s2,
            "f0_s4": f0_s4, "f1_s4": f1_s4,
            "f0_s8": f0_s8, "f1_s8": f1_s8,
            "ctx0_s2": ctx0_s2, "ctx1_s2": ctx1_s2,
            "ctx0_s4": ctx0_s4, "ctx1_s4": ctx1_s4,
            "corr_fwd": corr_fwd, "corr_bwd": corr_bwd,
            "i0": i0, "i1": i1,
            "i0_s8": _resize(i0, (h8, w8)), "i1_s8": _resize(i1, (h8, w8)),
            "edge0_s8": _resize(edge0, (h8, w8)), "edge1_s8": _resize(edge1, (h8, w8)),
            "dt0_s8": _resize(dt0, (h8, w8)), "dt1_s8": _resize(dt1, (h8, w8)),
            "seg0_s8": seg0_s8, "seg1_s8": seg1_s8,
            "seg0_s4": _resize(seg0, (h4, w4)), "seg1_s4": _resize(seg1, (h4, w4)),
            "flow_fwd": flow_fwd, "flow_bwd": flow_bwd,
        }

    # ── t-dependent part ─────────────────────────────────────

    def estimate_flow(self, shared: Dict, t: TimeArg = 0.5):
        """
        Returns:
            flow_t0, flow_t1   [B,2,H,W]
            occ_logits         [B,3,H,W]
            occ_weights        [B,3,H,W]  softmax(occ_logits)
            warped0, warped1   [B,3,H,W]
        """
        S = shared
        b, h, w = S["b"], S["h"], S["w"]
        h8, w8, h4, w4 = S["h8"], S["w8"], S["h4"], S["w4"]
        dev, dtype = S["i0"].device, S["i0"].dtype

        t_ch_s8 = self._make_t_ch(t, b, h8, w8, dev, dtype)
        flow = self._init_flow_from_raft(S["flow_fwd"], S["flow_bwd"], h8, w8,
                                         self._t_scalar(t))
        occ_logits = torch.zeros(b, 3, h8, w8, device=dev, dtype=dtype)
        warped0_s8 = warp(S["i0_s8"], flow[:, :2])
        warped1_s8 = warp(S["i1_s8"], flow[:, 2:4])

        # Итерации на 1/8
        for block in self.blocks:
            cf = self.corr_compress_fwd(S["corr_fwd"].lookup(flow[:, 2:4]))
            cb = self.corr_compress_bwd(S["corr_bwd"].lookup(flow[:, :2]))
            x = torch.cat([cf, cb, S["f0_s8"], S["f1_s8"], warped0_s8, warped1_s8,
                           flow, occ_logits, t_ch_s8,
                           S["edge0_s8"], S["edge1_s8"], S["dt0_s8"], S["dt1_s8"],
                           S["seg0_s8"], S["seg1_s8"]], dim=1)
            d_flow, d_occ = block(x)
            flow, occ_logits = flow + d_flow, occ_logits + d_occ
            warped0_s8 = warp(S["i0_s8"], flow[:, :2])
            warped1_s8 = warp(S["i1_s8"], flow[:, 2:4])

        # Уточнение на 1/4
        flow_s4 = _resize(flow, (h4, w4)) * 2
        occ_s4 = _resize(occ_logits, (h4, w4))
        w0_s4 = warp(_resize(S["i0"], (h4, w4)), flow_s4[:, :2])
        w1_s4 = warp(_resize(S["i1"], (h4, w4)), flow_s4[:, 2:4])

        refine_in = torch.cat([flow_s4, S["f0_s4"], S["f1_s4"],
                               S["ctx0_s4"], S["ctx1_s4"],
                               w0_s4, w1_s4, occ_s4, S["seg0_s4"], S["seg1_s4"]], dim=1)
        d_flow_s4, d_occ_s4 = self.flow_refine(refine_in)
        flow_s4, occ_s4 = flow_s4 + d_flow_s4, occ_s4 + d_occ_s4

        # Full resolution
        flow_full = self.convex_up(flow_s4, S["f0_s2"], S["f1_s2"])
        occ_full = _resize(occ_s4, (h, w))
        occ_w = torch.softmax(occ_full, dim=1)
        ft0, ft1 = flow_full[:, :2], flow_full[:, 2:4]
        return ft0, ft1, occ_full, occ_w, warp(S["i0"], ft0), warp(S["i1"], ft1)

    def forward(self, i0, i1, edge0, edge1, dt0, dt1, flow_fwd, flow_bwd,
                seg0, seg1, t: TimeArg = 0.5):
        shared = self.encode_shared(i0, i1, edge0, edge1, dt0, dt1,
                                    seg0, seg1, flow_fwd, flow_bwd)
        return self.estimate_flow(shared, t)


# ═══════════════════════════════════════════════════════════════
# 2. RefineNet — синтез residual
# ═══════════════════════════════════════════════════════════════

class RefineNet(nn.Module):
    """
    U-Net с attention bottleneck.

    Вход (134 канала при splat_feat_ch=16):
      backward-warp пакет (92):
        coarse, warped0, warped1, i0, i1 (15) + flow_t0, flow_t1 (4)
        + occ_weights (3) + edge, dt, seg ×2 (6) + ctx ×2 (64)
      forward-splat пакет (6 + 2·splat_feat_ch + 2):
        splat_rgb ×2, splat_feat ×2, splat_mask ×2
      подсказки (2): sketch_hint, roi_mask
    """

    BACKWARD_CH = 92
    HINT_CH = 2

    def __init__(self, splat_feat_ch: int = 16, attn_heads: int = 4,
                 attn_proj_dim: int = 64):
        super().__init__()
        in_ch = self.BACKWARD_CH + 6 + splat_feat_ch * 2 + 2 + self.HINT_CH
        self.e1 = self._block(in_ch, 64)
        self.e2 = self._block(64, 128, downsample=True)
        self.e3 = self._block(128, 256, downsample=True)
        self.bn = AttentionBottleneck(256, 384, attn_heads, attn_proj_dim)
        self.d3 = self._block(384 + 256, 256)
        self.d2 = self._block(256 + 128, 128)
        self.d1 = self._block(128 + 64, 64)
        self.head = nn.Conv2d(64, 3, 3, padding=1)

    @staticmethod
    def _block(in_ch, out_ch, downsample=False):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2 if downsample else 1, padding=1),
            nn.BatchNorm2d(out_ch), nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.PReLU(out_ch),
        )

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        b = self.bn(s3)
        d = self.d3(torch.cat([_resize(b, s3.shape[2:]), s3], 1))
        d = self.d2(torch.cat([_resize(d, s2.shape[2:]), s2], 1))
        d = self.d1(torch.cat([_resize(d, s1.shape[2:]), s1], 1))
        return self.head(d)


# ═══════════════════════════════════════════════════════════════
# 3. AnimeIFNet — полная модель
# ═══════════════════════════════════════════════════════════════

class AnimeIFNet(nn.Module):
    """
    Полная модель. Без подсказок (sketch_hint=None, roi_mask=None)
    работает в обычном режиме интерполяции.

    Returns dict:
      pred         финальный кадр (после ROI-композиции)
      pred_local   выход до ROI-композиции (для roi/preserve loss)
      coarse, warped0, warped1, flow_t0, flow_t1,
      occ_weights, occ_logits, disocclusion, residual, raw_residual
    """

    def __init__(self, residual_scale: float = 0.7, corr_levels: int = 4,
                 corr_radius: int = 5, freeze_backbone: bool = True,
                 splat_feat_ch: int = 16, pretrained: bool = True):
        super().__init__()
        self.ifnet = IFNet(corr_levels=corr_levels, corr_radius=corr_radius,
                           freeze_backbone=freeze_backbone, pretrained=pretrained)
        self.splat = ForwardSplatBranch(feat_ch=splat_feat_ch, pretrained=pretrained)
        self.refine = RefineNet(splat_feat_ch=splat_feat_ch)
        self.dtm = DTMBlock()
        self.residual_scale = residual_scale

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _prepare_hints(i0, sketch_hint, roi_mask):
        B, _, H, W = i0.shape
        if sketch_hint is None:
            sketch_hint = i0.new_zeros(B, 1, H, W)
        elif sketch_hint.shape[-2:] != (H, W):
            sketch_hint = _resize(sketch_hint, (H, W))
        if roi_mask is None:
            roi_mask = i0.new_ones(B, 1, H, W)
        elif roi_mask.shape[-2:] != (H, W):
            roi_mask = _resize(roi_mask, (H, W))
        return sketch_hint, roi_mask

    @staticmethod
    def _full_res_context(shared, h, w):
        return _resize(shared["ctx0_s2"], (h, w)), _resize(shared["ctx1_s2"], (h, w))

    def _compose(self, ft0, ft1, occ_l, occ_w, w0, w1,
                 i0, i1, e0, e1, d0, d1, s0, s1, c0f, c1f,
                 splat_data, sketch_hint, roi_mask) -> Dict[str, torch.Tensor]:
        # Coarse blend в fp32 (стабильно под AMP). Канал 0 — disocclusion,
        # в blend не участвует: эти области должен синтезировать RefineNet.
        with torch.amp.autocast("cuda", enabled=False):
            wt0 = occ_w[:, 0:1].float()
            wt1 = occ_w[:, 1:2].float()
            wt2 = occ_w[:, 2:3].float()
            ws = wt1 + wt2 + 1e-6
            coarse = ((wt1 / ws) * w0.float() + (wt2 / ws) * w1.float()).to(w0.dtype)

        ri = torch.cat([
            coarse, w0, w1, i0, i1,
            ft0, ft1, occ_w,
            e0, e1, d0, d1, s0, s1,
            c0f, c1f,
            splat_data["splat_rgb0"], splat_data["splat_rgb1"],
            splat_data["splat_feat0"], splat_data["splat_feat1"],
            splat_data["splat_mask0"], splat_data["splat_mask1"],
            sketch_hint, roi_mask,
        ], dim=1)

        raw_res = self.refine(ri) * self.residual_scale
        residual = self.dtm(raw_res, d0, d1)
        pred_local = (coarse + residual).clamp(-1, 1)
        pred = (roi_mask * pred_local + (1.0 - roi_mask) * coarse).clamp(-1, 1)

        return {
            "pred": pred, "pred_local": pred_local, "coarse": coarse,
            "warped0": w0, "warped1": w1, "flow_t0": ft0, "flow_t1": ft1,
            "occ_weights": occ_w, "occ_logits": occ_l, "disocclusion": wt0,
            "residual": residual, "raw_residual": raw_res,
        }

    # ── public API ───────────────────────────────────────────

    def forward(self, i0, i1, edge0, edge1, dt0, dt1, flow_fwd, flow_bwd,
                seg0, seg1, t: TimeArg = 0.5,
                sketch_hint: Optional[torch.Tensor] = None,
                roi_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        sketch_hint, roi_mask = self._prepare_hints(i0, sketch_hint, roi_mask)

        S = self.ifnet.encode_shared(i0, i1, edge0, edge1, dt0, dt1,
                                     seg0, seg1, flow_fwd, flow_bwd)
        ft0, ft1, ol, ow, w0, w1 = self.ifnet.estimate_flow(S, t)
        splat_data = self.splat(i0, i1, flow_fwd, flow_bwd, t=t)
        c0, c1 = self._full_res_context(S, *i0.shape[2:])

        return self._compose(ft0, ft1, ol, ow, w0, w1,
                             i0, i1, edge0, edge1, dt0, dt1, seg0, seg1, c0, c1,
                             splat_data, sketch_hint, roi_mask)

    def generate_multi_t(self, i0, i1, edge0, edge1, dt0, dt1, flow_fwd, flow_bwd,
                         seg0, seg1, t_values: Sequence[TimeArg] = (0.25, 0.5, 0.75),
                         sketch_hint: Optional[torch.Tensor] = None,
                         roi_mask: Optional[torch.Tensor] = None):
        """Несколько промежуточных кадров: энкодеры и correlation volume
        считаются один раз, поток и синтез — для каждого t."""
        sketch_hint, roi_mask = self._prepare_hints(i0, sketch_hint, roi_mask)
        S = self.ifnet.encode_shared(i0, i1, edge0, edge1, dt0, dt1,
                                     seg0, seg1, flow_fwd, flow_bwd)
        c0, c1 = self._full_res_context(S, *i0.shape[2:])

        results = []
        for t in t_values:
            ft0, ft1, ol, ow, w0, w1 = self.ifnet.estimate_flow(S, t)
            splat_data = self.splat(i0, i1, flow_fwd, flow_bwd, t=t)
            out = self._compose(ft0, ft1, ol, ow, w0, w1,
                                i0, i1, edge0, edge1, dt0, dt1, seg0, seg1, c0, c1,
                                splat_data, sketch_hint, roi_mask)
            out["t"] = t
            results.append(out)
        return results

    def unfreeze_backbone(self):
        self.ifnet.ctx_encoder.unfreeze_backbone()

    # ── checkpoints ──────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, path: str, device="cpu", strict: bool = True, **kwargs):
        """Загрузить обученную модель. ImageNet-веса не скачиваются:
        замороженные backbone тоже лежат в чекпоинте."""
        kwargs.setdefault("pretrained", False)
        model = cls(**kwargs).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=strict)
        return model.eval()

    @classmethod
    def from_v8_checkpoint(cls, path: str, device="cpu", **kwargs):
        """
        Инициализация из чекпоинта V8 (без подсказок). У первого conv
        RefineNet вход шире на 2 канала: старые веса копируются,
        новые (sketch, roi) обнуляются, так что при sketch=0, roi=1
        модель стартует с поведением V8.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"V8 checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        src = ckpt.get("model", ckpt)

        model = cls(**kwargs).to(device)
        own = model.state_dict()
        copied, expanded, skipped = 0, 0, []

        for k, v in src.items():
            if k not in own:
                skipped.append(k)
                continue
            tgt = own[k]
            if tgt.shape == v.shape:
                own[k] = v.to(device)
                copied += 1
            elif (tgt.ndim == v.ndim >= 2 and tgt.shape[0] == v.shape[0]
                  and tgt.shape[2:] == v.shape[2:] and tgt.shape[1] > v.shape[1]):
                new_w = torch.zeros_like(tgt)
                new_w[:, :v.shape[1]] = v.to(device)
                own[k] = new_w
                expanded += 1
            else:
                skipped.append(f"{k}: {list(v.shape)} vs {list(tgt.shape)}")

        model.load_state_dict(own, strict=False)
        print(f"[from_v8] copied={copied}, expanded={expanded}, skipped={len(skipped)}")
        if skipped:
            print(f"  first skipped: {skipped[:5]}")
        return model
