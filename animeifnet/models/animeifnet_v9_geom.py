"""
animeifnet_v9_geom.py
─────────────────────
Experimental geometry-first branch on top of V8/V8.2.

Implements first versions of:
  1) sketch guidance inside the flow estimator, not only RefineNet;
  2) auxiliary target line/edge prediction head;
  3) segment-guided coarse / multi-region piece-flow initialization;
  4) ConvGRU recurrent flow refinement instead of independent IFBlock cascade;
  5) H/2 forward splatting for RGB + edge, H/4 for deep ResNet features;
  6) reliability/gating network over backward warp, RAFT warp, piece-flow warp, splat;
  7) TimingNet that predicts a per-pixel t-map for non-linear cartoon timing.

The code intentionally keeps the public forward interface close to V8.2:
    forward(i0, i1, edge0, edge1, dt0, dt1, flow_fwd, flow_bwd,
            seg0, seg1, t=0.5, sketch_hint=None, roi_mask=None)

Expected ranges:
  RGB: [-1, 1]
  edge/dt/seg/sketch/roi: [0, 1]
  flow_fwd: frame0 -> frame1, pixels
  flow_bwd: frame1 -> frame0, pixels
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    warp, CorrBlock, CorrCompressor, ConvexUpsample,
    DTMBlock, forward_splat, FrozenResNetExtractor, AttentionBottleneck,
)
from .animeifnet import IFNet

TensorOrFloat = Union[torch.Tensor, float, int]


# ═══════════════════════════════════════════════════════════════
# Small utilities
# ═══════════════════════════════════════════════════════════════

def _make_t_map(t: TensorOrFloat, b: int, h: int, w: int,
                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return [B,1,H,W] t-map from scalar or tensor input."""
    if isinstance(t, (float, int)):
        return torch.full((b, 1, h, w), float(t), device=device, dtype=dtype)
    if t.dim() == 3:
        t = t.unsqueeze(1)
    if t.shape[-2:] != (h, w):
        t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.to(device=device, dtype=dtype).clamp(0.0, 1.0)


def _scale_flow_to(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize a pixel flow to target spatial size and scale dx/dy accordingly."""
    _, _, h, w = flow.shape
    th, tw = size
    out = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    out = out.clone()
    out[:, 0:1] *= float(tw) / max(float(w), 1.0)
    out[:, 1:2] *= float(th) / max(float(h), 1.0)
    return out


def _masked_avg_flow(flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a spatially constant [B,2,1,1] average flow under soft mask."""
    denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (flow * mask).sum(dim=(2, 3), keepdim=True) / denom


def _local_avg_flow_grid(
    flow: torch.Tensor,
    mask: torch.Tensor,
    grid_size: int = 6,
    min_coverage: float = 0.02,
) -> torch.Tensor:
    """
    Multi-region piece-flow primitive.

    Instead of one spatially-constant average motion for the whole mask, split the
    image into a coarse grid, estimate an average flow for every occupied cell and
    bilinearly upsample that flow field back to the original resolution. Empty
    cells fall back to the global masked average, which keeps the prior stable on
    tiny masks and at the image borders.

    This is intentionally pure PyTorch and checkpoint-compatible: it has no new
    parameters and can be used during both training and inference without CPU
    connected-component calls.
    """
    b, _, h, w = flow.shape
    gh, gw = min(grid_size, h), min(grid_size, w)

    mask = mask.clamp(0, 1)
    global_avg = _masked_avg_flow(flow, mask)  # [B,2,1,1]

    # adaptive_avg_pool2d gives mean values per cell. Dividing pooled flow*mask
    # by pooled mask is equivalent to a masked average inside each cell.
    mask_g = F.adaptive_avg_pool2d(mask, (gh, gw))
    flow_g = F.adaptive_avg_pool2d(flow * mask, (gh, gw)) / mask_g.clamp_min(1e-6)

    valid_g = (mask_g > min_coverage).to(flow.dtype)
    flow_g = valid_g * flow_g + (1.0 - valid_g) * global_avg.expand(-1, -1, gh, gw)

    local = F.interpolate(flow_g, size=(h, w), mode="bilinear", align_corners=False)
    valid = F.interpolate(valid_g, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
    return valid * local + (1.0 - valid) * global_avg


def _piecewise_flow(flow: torch.Tensor, seg: torch.Tensor, grid_size: int = 6) -> torch.Tensor:
    """
    Multi-region piece-flow prior.

    Previous version used only two constants: one foreground flow and one
    background flow. This version keeps the same public call-site but makes both
    foreground and background spatially local, so separate body parts/objects can
    receive different coarse motion even before the learned ConvGRU refinement.
    """
    seg = seg.clamp(0, 1)
    fg_local = _local_avg_flow_grid(flow, seg, grid_size=grid_size)
    bg_local = _local_avg_flow_grid(flow, 1.0 - seg, grid_size=grid_size)
    return seg * fg_local + (1.0 - seg) * bg_local


def _raft_mid_flows(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor,
                    t_map: torch.Tensor) -> torch.Tensor:
    """Linear bidirectional RAFT approximation, returns [B,4,H,W]."""
    flow_t0 = 0.5 * (-t_map * flow_fwd + (1.0 - t_map) * flow_bwd)
    flow_t1 = 0.5 * ((1.0 - t_map) * flow_fwd - t_map * flow_bwd)
    return torch.cat([flow_t0, flow_t1], dim=1)


# ═══════════════════════════════════════════════════════════════
# 1) Sketch into geometry branch
# ═══════════════════════════════════════════════════════════════

class SketchPyramidEncoder(nn.Module):
    """Lightweight multi-scale encoder for target sketch/line hint."""
    def __init__(self, c2: int = 16, c4: int = 24, c8: int = 32):
        super().__init__()
        self.s2 = nn.Sequential(
            nn.Conv2d(1, c2, 3, stride=2, padding=1), nn.PReLU(c2),
            nn.Conv2d(c2, c2, 3, padding=1), nn.PReLU(c2),
        )
        self.s4 = nn.Sequential(
            nn.Conv2d(c2, c4, 3, stride=2, padding=1), nn.PReLU(c4),
            nn.Conv2d(c4, c4, 3, padding=1), nn.PReLU(c4),
        )
        self.s8 = nn.Sequential(
            nn.Conv2d(c4, c8, 3, stride=2, padding=1), nn.PReLU(c8),
            nn.Conv2d(c8, c8, 3, padding=1), nn.PReLU(c8),
        )

    def forward(self, sketch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f2 = self.s2(sketch)
        f4 = self.s4(f2)
        f8 = self.s8(f4)
        return f2, f4, f8


# ═══════════════════════════════════════════════════════════════
# 4) ConvGRU recurrent flow update
# ═══════════════════════════════════════════════════════════════

class ConvGRUCell(nn.Module):
    def __init__(self, x_ch: int, h_ch: int):
        super().__init__()
        self.z = nn.Conv2d(x_ch + h_ch, h_ch, 3, padding=1)
        self.r = nn.Conv2d(x_ch + h_ch, h_ch, 3, padding=1)
        self.q = nn.Conv2d(x_ch + h_ch, h_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.z(hx))
        r = torch.sigmoid(self.r(hx))
        q = torch.tanh(self.q(torch.cat([r * h, x], dim=1)))
        return (1.0 - z) * h + z * q


class FlowGRUUpdateBlock(nn.Module):
    """Compress large RAFT-like input, run ConvGRU, predict Δflow and Δocc."""
    def __init__(self, in_ch: int, hidden_ch: int = 192):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1), nn.PReLU(hidden_ch),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.PReLU(hidden_ch),
        )
        self.gru = ConvGRUCell(hidden_ch, hidden_ch)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_ch, 96, 3, padding=1), nn.PReLU(96),
            nn.Conv2d(96, 7, 3, padding=1),
        )
        # Stable first checkpoint loading / warm start: recurrent path starts as no-op.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        z = self.in_proj(x)
        if h is None:
            h = z.new_zeros(z.size(0), self.hidden_ch, z.size(2), z.size(3))
        h = self.gru(z, h)
        out = self.head(h)
        return out[:, :4], out[:, 4:7], h


class FlowRefineBlockSketch(nn.Module):
    """H/4 refinement with sketch features, t-map and piece-flow prior."""
    # flow 4 + matching 192 + context 96 + warped RGB 6 + occ 3 + seg 2
    # + sketch_s4 24 + piece_flow 4 + t_map 1 = 332
    IN_CH = 4 + 192 + 96 + 6 + 3 + 2 + 24 + 4 + 1

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(self.IN_CH, 160, 3, padding=1), nn.PReLU(160),
            nn.Conv2d(160, 160, 3, padding=2, dilation=2), nn.PReLU(160),
            nn.Conv2d(160, 128, 3, padding=4, dilation=4), nn.PReLU(128),
            nn.Conv2d(128, 96, 3, padding=1), nn.PReLU(96),
            nn.Conv2d(96, 7, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[:, :4], out[:, 4:7]


class AuxLineHead(nn.Module):
    """Predict target line map from the geometry branch at H/4."""
    # flow 4 + occ 3 + warped edges 2 + sketch features 24 + seg 2 = 35
    IN_CH = 4 + 3 + 2 + 24 + 2

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(self.IN_CH, 64, 3, padding=1), nn.PReLU(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.PReLU(64),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IFNetSketchGRU(IFNet):
    """
    IFNet-compatible estimator with sketch-aware ConvGRU recurrent updates.

    Keeps the same encode/estimate split, but estimate_flow() returns an extra aux dict:
      ft0, ft1, occ_logits_full, occ_weights, warped0, warped1, aux
    """

    UPDATE_IN_CH = 64 + 64 + 256 + 6 + 4 + 3 + 1 + 2 + 2 + 2 + 32 + 4  # 440

    def __init__(self, corr_levels=4, corr_radius=5, freeze_backbone=True,
                 recurrent_steps: int = 5):
        super().__init__(corr_levels=corr_levels, corr_radius=corr_radius,
                         freeze_backbone=freeze_backbone)
        # The old independent IFBlock cascade is intentionally not used.
        # Remove it so the experimental branch does not train dead parameters.
        self.blocks = nn.ModuleList()
        self.recurrent_steps = recurrent_steps
        self.sketch_encoder = SketchPyramidEncoder()
        self.update_gru = FlowGRUUpdateBlock(self.UPDATE_IN_CH, hidden_ch=192)
        self.flow_refine = FlowRefineBlockSketch()
        self.line_head = AuxLineHead()
        # Start close to RAFT; sigmoid(-1.5) ≈ 0.18 piece-flow contribution.
        self.piece_blend_logit = nn.Parameter(torch.tensor(-1.5))

    def encode_shared(self, i0, i1, edge0, edge1, dt0, dt1,
                      seg0, seg1, flow_fwd, flow_bwd,
                      sketch_hint: Optional[torch.Tensor] = None):
        b, _, h, w = i0.shape
        if sketch_hint is None:
            sketch_hint = i0.new_zeros(b, 1, h, w)
        if sketch_hint.shape[-2:] != (h, w):
            sketch_hint = F.interpolate(sketch_hint, size=(h, w),
                                        mode="bilinear", align_corners=False)
        sketch_hint = sketch_hint.clamp(0, 1)

        enc_in0 = torch.cat([i0, edge0, dt0, seg0], dim=1)
        enc_in1 = torch.cat([i1, edge1, dt1, seg1], dim=1)

        f0_s2, f0_s4, f0_s8 = self.encoder(enc_in0)
        f1_s2, f1_s4, f1_s8 = self.encoder(enc_in1)
        ctx0_s2, ctx0_s4 = self.ctx_encoder(enc_in0)
        ctx1_s2, ctx1_s4 = self.ctx_encoder(enc_in1)
        sk_s2, sk_s4, sk_s8 = self.sketch_encoder(sketch_hint)

        h8, w8 = f0_s8.shape[2:]
        h4, w4 = f0_s4.shape[2:]

        seg0_s8 = F.interpolate(seg0, size=(h8, w8), mode="bilinear", align_corners=False)
        seg1_s8 = F.interpolate(seg1, size=(h8, w8), mode="bilinear", align_corners=False)
        seg0_s4 = F.interpolate(seg0, size=(h4, w4), mode="bilinear", align_corners=False)
        seg1_s4 = F.interpolate(seg1, size=(h4, w4), mode="bilinear", align_corners=False)

        corr_fwd = CorrBlock(f0_s8, f1_s8, seg_src=seg0_s8, seg_tgt=seg1_s8,
                             num_levels=self.corr_levels, radius=self.corr_radius)
        corr_bwd = CorrBlock(f1_s8, f0_s8, seg_src=seg1_s8, seg_tgt=seg0_s8,
                             num_levels=self.corr_levels, radius=self.corr_radius)

        i0_s8 = F.interpolate(i0, size=(h8, w8), mode="bilinear", align_corners=False)
        i1_s8 = F.interpolate(i1, size=(h8, w8), mode="bilinear", align_corners=False)
        edge0_s8 = F.interpolate(edge0, size=(h8, w8), mode="bilinear", align_corners=False)
        edge1_s8 = F.interpolate(edge1, size=(h8, w8), mode="bilinear", align_corners=False)
        dt0_s8 = F.interpolate(dt0, size=(h8, w8), mode="bilinear", align_corners=False)
        dt1_s8 = F.interpolate(dt1, size=(h8, w8), mode="bilinear", align_corners=False)
        edge0_s4 = F.interpolate(edge0, size=(h4, w4), mode="bilinear", align_corners=False)
        edge1_s4 = F.interpolate(edge1, size=(h4, w4), mode="bilinear", align_corners=False)

        return {
            "b": b, "h": h, "w": w, "h8": h8, "w8": w8, "h4": h4, "w4": w4,
            "f0_s2": f0_s2, "f1_s2": f1_s2,
            "f0_s4": f0_s4, "f1_s4": f1_s4,
            "f0_s8": f0_s8, "f1_s8": f1_s8,
            "ctx0_s2": ctx0_s2, "ctx1_s2": ctx1_s2,
            "ctx0_s4": ctx0_s4, "ctx1_s4": ctx1_s4,
            "sk_s2": sk_s2, "sk_s4": sk_s4, "sk_s8": sk_s8,
            "corr_fwd": corr_fwd, "corr_bwd": corr_bwd,
            "i0": i0, "i1": i1,
            "i0_s8": i0_s8, "i1_s8": i1_s8,
            "edge0_s8": edge0_s8, "edge1_s8": edge1_s8,
            "edge0_s4": edge0_s4, "edge1_s4": edge1_s4,
            "dt0_s8": dt0_s8, "dt1_s8": dt1_s8,
            "seg0_s8": seg0_s8, "seg1_s8": seg1_s8,
            "seg0_s4": seg0_s4, "seg1_s4": seg1_s4,
            "flow_fwd": flow_fwd, "flow_bwd": flow_bwd,
            "sketch_hint": sketch_hint,
        }

    def _flow_inits_s8(self, S: Dict, t: TensorOrFloat):
        b, h8, w8 = S["b"], S["h8"], S["w8"]
        dev, dtype = S["i0"].device, S["i0"].dtype
        t8 = _make_t_map(t, b, h8, w8, dev, dtype)

        fwd_s8 = _scale_flow_to(S["flow_fwd"], (h8, w8))
        bwd_s8 = _scale_flow_to(S["flow_bwd"], (h8, w8))
        raft_init = _raft_mid_flows(fwd_s8, bwd_s8, t8)

        piece_fwd = _piecewise_flow(fwd_s8.detach(), S["seg0_s8"].detach())
        piece_bwd = _piecewise_flow(bwd_s8.detach(), S["seg1_s8"].detach())
        piece_init = _raft_mid_flows(piece_fwd, piece_bwd, t8)
        blend = torch.sigmoid(self.piece_blend_logit)
        init = (1.0 - blend) * raft_init + blend * piece_init
        return init, raft_init, piece_init, t8

    def estimate_flow(self, shared: Dict, t: TensorOrFloat = 0.5):
        S = shared
        b, h, w = S["b"], S["h"], S["w"]
        h4, w4 = S["h4"], S["w4"]
        dev, dtype = S["i0"].device, S["i0"].dtype

        flow, raft_init_s8, piece_init_s8, t_s8 = self._flow_inits_s8(S, t)
        occ_logits = torch.zeros(b, 3, S["h8"], S["w8"], device=dev, dtype=dtype)
        hidden = None

        warped0_s8 = warp(S["i0_s8"], flow[:, :2])
        warped1_s8 = warp(S["i1_s8"], flow[:, 2:4])

        for _ in range(self.recurrent_steps):
            cf = self.corr_compress_fwd(S["corr_fwd"].lookup(flow[:, 2:4]))
            cb = self.corr_compress_bwd(S["corr_bwd"].lookup(flow[:, :2]))
            x = torch.cat([
                cf, cb, S["f0_s8"], S["f1_s8"],
                warped0_s8, warped1_s8,
                flow, occ_logits, t_s8,
                S["edge0_s8"], S["edge1_s8"], S["dt0_s8"], S["dt1_s8"],
                S["seg0_s8"], S["seg1_s8"],
                S["sk_s8"], piece_init_s8,
            ], dim=1)
            d_flow, d_occ, hidden = self.update_gru(x, hidden)
            flow = flow + d_flow
            occ_logits = occ_logits + d_occ
            warped0_s8 = warp(S["i0_s8"], flow[:, :2])
            warped1_s8 = warp(S["i1_s8"], flow[:, 2:4])

        flow_s4 = F.interpolate(flow, size=(h4, w4), mode="bilinear", align_corners=False) * 2.0
        occ_s4 = F.interpolate(occ_logits, size=(h4, w4), mode="bilinear", align_corners=False)
        piece_s4 = F.interpolate(piece_init_s8, size=(h4, w4), mode="bilinear", align_corners=False) * 2.0
        t_s4 = _make_t_map(t, b, h4, w4, dev, dtype)

        i0_s4 = F.interpolate(S["i0"], size=(h4, w4), mode="bilinear", align_corners=False)
        i1_s4 = F.interpolate(S["i1"], size=(h4, w4), mode="bilinear", align_corners=False)
        warped0_s4 = warp(i0_s4, flow_s4[:, :2])
        warped1_s4 = warp(i1_s4, flow_s4[:, 2:4])

        refine_in = torch.cat([
            flow_s4,
            S["f0_s4"], S["f1_s4"],
            S["ctx0_s4"], S["ctx1_s4"],
            warped0_s4, warped1_s4,
            occ_s4,
            S["seg0_s4"], S["seg1_s4"],
            S["sk_s4"], piece_s4, t_s4,
        ], dim=1)
        d_flow_s4, d_occ_s4 = self.flow_refine(refine_in)
        flow_s4 = flow_s4 + d_flow_s4
        occ_s4 = occ_s4 + d_occ_s4

        # Aux line prediction from geometry state.
        edge0_w_s4 = warp(S["edge0_s4"], flow_s4[:, :2])
        edge1_w_s4 = warp(S["edge1_s4"], flow_s4[:, 2:4])
        line_logits_s4 = self.line_head(torch.cat([
            flow_s4, occ_s4, edge0_w_s4, edge1_w_s4,
            S["sk_s4"], S["seg0_s4"], S["seg1_s4"],
        ], dim=1))
        line_logits = F.interpolate(line_logits_s4, size=(h, w), mode="bilinear", align_corners=False)
        line_pred = torch.sigmoid(line_logits)

        flow_full = self.convex_up(flow_s4, S["f0_s2"], S["f1_s2"])
        occ_full = F.interpolate(occ_s4, size=(h, w), mode="bilinear", align_corners=False)
        occ_w = torch.softmax(occ_full, dim=1)
        ft0, ft1 = flow_full[:, :2], flow_full[:, 2:4]

        # Full-res priors for gating/diagnostics.
        t_full = _make_t_map(t, b, h, w, dev, dtype)
        fwd_full = S["flow_fwd"]
        bwd_full = S["flow_bwd"]
        raft_full = _raft_mid_flows(fwd_full, bwd_full, t_full)
        # Recompute full-res piece flow from upsampled soft masks for gating.
        seg0_full = F.interpolate(S["seg0_s8"], size=(h, w), mode="bilinear", align_corners=False)
        seg1_full = F.interpolate(S["seg1_s8"], size=(h, w), mode="bilinear", align_corners=False)
        piece_fwd_full = _piecewise_flow(fwd_full.detach(), seg0_full.detach())
        piece_bwd_full = _piecewise_flow(bwd_full.detach(), seg1_full.detach())
        piece_full = _raft_mid_flows(piece_fwd_full, piece_bwd_full, t_full)

        aux = {
            "line_logits": line_logits,
            "line_pred": line_pred,
            "raft_flow_full": raft_full,
            "piece_flow_full": piece_full,
            "piece_blend": torch.sigmoid(self.piece_blend_logit).detach(),
        }
        return ft0, ft1, occ_full, occ_w, warp(S["i0"], ft0), warp(S["i1"], ft1), aux


# ═══════════════════════════════════════════════════════════════
# 7) TimingNet: per-pixel t-map
# ═══════════════════════════════════════════════════════════════

class TimingNet(nn.Module):
    """Predict a residual t-map; initialized as identity t_map=t."""
    def __init__(self, hidden: int = 32, max_delta: float = 0.25):
        super().__init__()
        self.max_delta = max_delta
        in_ch = 3 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 2 + 2 + 1  # 17
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.PReLU(hidden),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.PReLU(hidden),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, i0, i1, edge0, edge1, dt0, dt1, seg0, seg1,
                flow_fwd, flow_bwd, t: TensorOrFloat = 0.5) -> torch.Tensor:
        b, _, h, w = i0.shape
        t0 = _make_t_map(t, b, h, w, i0.device, i0.dtype)
        flow_mag_f = flow_fwd.norm(dim=1, keepdim=True)
        flow_mag_b = flow_bwd.norm(dim=1, keepdim=True)
        # Normalize roughly by frame diagonal to keep magnitudes stable.
        norm = max(float(h * h + w * w) ** 0.5, 1.0)
        x = torch.cat([
            i0, i1, edge0, edge1, dt0, dt1, seg0, seg1,
            flow_fwd / norm, flow_bwd / norm, t0,
        ], dim=1)
        delta = self.max_delta * torch.tanh(self.net(x))
        return (t0 + delta).clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════
# 5) H/2 splatting for RGB/edge, H/4 for deep features
# ═══════════════════════════════════════════════════════════════

class ForwardSplatBranchH2(nn.Module):
    def __init__(self, feat_ch: int = 16):
        super().__init__()
        self.feat_ch = feat_ch
        self.resnet = FrozenResNetExtractor(out_ch=feat_ch)

    def _splat_to(self, src: torch.Tensor, flow_to_t: torch.Tensor,
                  size: Tuple[int, int], metric: Optional[torch.Tensor] = None):
        src_s = F.interpolate(src, size=size, mode="bilinear", align_corners=False)
        flow_s = _scale_flow_to(flow_to_t, size)
        metric_s = None
        if metric is not None:
            metric_s = F.interpolate(metric, size=size, mode="bilinear", align_corners=False)
        return forward_splat(src_s, flow_s, metric_s)

    def forward(self, i0, i1, edge0, edge1, flow_fwd, flow_bwd, t: TensorOrFloat = 0.5):
        b, _, h, w = i0.shape
        t_full = _make_t_map(t, b, h, w, i0.device, i0.dtype)
        flow_0_to_t = t_full * flow_fwd
        flow_1_to_t = (1.0 - t_full) * flow_bwd

        feat0 = self.resnet(i0)
        feat1 = self.resnet(i1)
        h4, w4 = feat0.shape[-2:]
        h2, w2 = max(h // 2, 1), max(w // 2, 1)

        sr0, sm0 = self._splat_to(i0, flow_0_to_t, (h2, w2))
        sr1, sm1 = self._splat_to(i1, flow_1_to_t, (h2, w2))
        se0, _ = self._splat_to(edge0, flow_0_to_t, (h2, w2))
        se1, _ = self._splat_to(edge1, flow_1_to_t, (h2, w2))

        flow0_s4 = _scale_flow_to(flow_0_to_t, (h4, w4))
        flow1_s4 = _scale_flow_to(flow_1_to_t, (h4, w4))
        sf0, _ = forward_splat(feat0, flow0_s4)
        sf1, _ = forward_splat(feat1, flow1_s4)

        # Mutual infilling: empty splat pixels borrow from the other direction.
        sr0_f = sr0 * sm0 + sr1 * (1.0 - sm0)
        sr1_f = sr1 * sm1 + sr0 * (1.0 - sm1)
        se0_f = se0 * sm0 + se1 * (1.0 - sm0)
        se1_f = se1 * sm1 + se0 * (1.0 - sm1)
        sf0_f = sf0
        sf1_f = sf1

        up = lambda x: F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return {
            "splat_rgb0": up(sr0_f), "splat_rgb1": up(sr1_f),
            "splat_edge0": up(se0_f), "splat_edge1": up(se1_f),
            "splat_feat0": up(sf0_f), "splat_feat1": up(sf1_f),
            "splat_mask0": up(sm0), "splat_mask1": up(sm1),
        }


# ═══════════════════════════════════════════════════════════════
# 6) Reliability/gating network
# ═══════════════════════════════════════════════════════════════

class ReliabilityGateNet(nn.Module):
    """Per-pixel selector over 4 candidates: backward, RAFT, piece, splat."""
    def __init__(self):
        super().__init__()
        # 4 RGB candidates = 12, occ 3, splat masks 2, edge/sketch/roi 4, flow mags 4 = 25
        in_ch = 12 + 3 + 2 + 4 + 4
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 48, 3, padding=1), nn.PReLU(48),
            nn.Conv2d(48, 48, 3, padding=1), nn.PReLU(48),
            nn.Conv2d(48, 4, 1),
        )
        # Small bias toward backward/coarse candidate at initialization.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.0)
        with torch.no_grad():
            self.net[-1].bias[0] = 5.0  # softmax([5,0,0,0]) ≈ [0.96, 0.013, 0.013, 0.013]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)


# ═══════════════════════════════════════════════════════════════
# RefineNet for geometry branch
# ═══════════════════════════════════════════════════════════════

class RefineNetV9Geom(nn.Module):
    """
    Input channels:
      V8 backward package: 92
      H/2/H/4 splat package: RGB6 + edge2 + feat32 + mask2 = 42
      sketch + ROI: 2
      gate weights: 4
      t_map + aux line pred: 2
      Total: 142
    """
    IN_CH = 92 + 42 + 2 + 4 + 2

    def __init__(self, splat_feat_ch=16, attn_heads=4, attn_proj_dim=64):
        super().__init__()
        in_ch = 92 + 6 + 2 + splat_feat_ch * 2 + 2 + 2 + 4 + 2
        self.e1 = self._block(in_ch, 64, downsample=False)
        self.e2 = self._block(64, 128, downsample=True)
        self.e3 = self._block(128, 256, downsample=True)
        self.bn = AttentionBottleneck(256, 384, attn_heads, attn_proj_dim)
        self.d3 = self._block(384 + 256, 256, downsample=False)
        self.d2 = self._block(256 + 128, 128, downsample=False)
        self.d1 = self._block(128 + 64, 64, downsample=False)
        self.head = nn.Conv2d(64, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _block(in_ch, out_ch, downsample=False):
        stride = 2 if downsample else 1
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch), nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.PReLU(out_ch),
        )

    @staticmethod
    def _up(x, target):
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            s1 = self.e1(x)
            s2 = self.e2(s1)
            s3 = self.e3(s2)
            b = self.bn(s3)
            d = self.d3(torch.cat([self._up(b, s3), s3], 1))
            d = self.d2(torch.cat([self._up(d, s2), s2], 1))
            d = self.d1(torch.cat([self._up(d, s1), s1], 1))
            return self.head(d)


# ═══════════════════════════════════════════════════════════════
# Full model
# ═══════════════════════════════════════════════════════════════

class AnimeIFNetV9Geom(nn.Module):
    def __init__(self, residual_scale=0.7, corr_levels=4, corr_radius=5,
                 freeze_backbone=True, splat_feat_ch=16,
                 recurrent_steps: int = 5, use_timing_net: bool = True):
        super().__init__()
        self.ifnet = IFNetSketchGRU(
            corr_levels=corr_levels, corr_radius=corr_radius,
            freeze_backbone=freeze_backbone, recurrent_steps=recurrent_steps,
        )
        self.timing = TimingNet()
        self.use_timing_net = use_timing_net
        self.splat = ForwardSplatBranchH2(feat_ch=splat_feat_ch)
        self.gate = ReliabilityGateNet()
        self.refine = RefineNetV9Geom(splat_feat_ch=splat_feat_ch)
        self.dtm = DTMBlock()
        self.residual_scale = residual_scale

    def _candidate_warps(self, i0, i1, flow4: torch.Tensor, occ_w: torch.Tensor,
                         splat_data: Dict[str, torch.Tensor]):
        ft0, ft1 = flow4[:, :2], flow4[:, 2:4]
        w0 = warp(i0, ft0)
        w1 = warp(i1, ft1)
        wt1, wt2 = occ_w[:, 1:2], occ_w[:, 2:3]
        coarse = (wt1 * w0 + wt2 * w1) / (wt1 + wt2 + 1e-8)
        splat = (
            splat_data["splat_rgb0"] * splat_data["splat_mask0"] +
            splat_data["splat_rgb1"] * splat_data["splat_mask1"]
        ) / (splat_data["splat_mask0"] + splat_data["splat_mask1"] + 1e-8)
        return coarse, w0, w1, splat

    def _compose(self, ft0, ft1, occ_l, occ_w, w0, w1,
                 i0, i1, e0, e1, d0, d1, s0, s1, c0f, c1f,
                 splat_data: Dict[str, torch.Tensor], aux: Dict[str, torch.Tensor],
                 sketch_hint: torch.Tensor, roi_mask: torch.Tensor,
                 t_map: torch.Tensor):
        wt0, wt1, wt2 = occ_w[:, 0:1], occ_w[:, 1:2], occ_w[:, 2:3]
        backward = (wt1 * w0 + wt2 * w1) / (wt1 + wt2 + 1e-8)

        raft4 = aux["raft_flow_full"]
        piece4 = aux["piece_flow_full"]
        raft_c, _, _, _ = self._candidate_warps(i0, i1, raft4, occ_w, splat_data)
        piece_c, _, _, _ = self._candidate_warps(i0, i1, piece4, occ_w, splat_data)
        splat_c = (
            splat_data["splat_rgb0"] * splat_data["splat_mask0"] +
            splat_data["splat_rgb1"] * splat_data["splat_mask1"]
        ) / (splat_data["splat_mask0"] + splat_data["splat_mask1"] + 1e-8)

        norm = max(float(i0.shape[-2] * i0.shape[-2] + i0.shape[-1] * i0.shape[-1]) ** 0.5, 1.0)
        gate_in = torch.cat([
            backward, raft_c, piece_c, splat_c,
            occ_w, splat_data["splat_mask0"], splat_data["splat_mask1"],
            e0, e1, sketch_hint, roi_mask,
            ft0 / norm, ft1 / norm,
        ], dim=1)
        gates = self.gate(gate_in)
        coarse = (gates[:, 0:1] * backward + gates[:, 1:2] * raft_c +
                  gates[:, 2:3] * piece_c + gates[:, 3:4] * splat_c)

        ri = torch.cat([
            # V8-style backward package (92), but with gated coarse.
            coarse, w0, w1, i0, i1,
            ft0, ft1, occ_w,
            e0, e1, d0, d1, s0, s1,
            c0f, c1f,
            # H/2/H/4 splat package (42)
            splat_data["splat_rgb0"], splat_data["splat_rgb1"],
            splat_data["splat_edge0"], splat_data["splat_edge1"],
            splat_data["splat_feat0"], splat_data["splat_feat1"],
            splat_data["splat_mask0"], splat_data["splat_mask1"],
            # Control/reliability/timing/aux
            sketch_hint, roi_mask, gates, t_map, aux["line_pred"],
        ], dim=1)

        raw_res = self.refine(ri) * self.residual_scale
        residual = self.dtm(raw_res, d0, d1)
        pred_local = (coarse + residual).clamp(-1, 1)
        pred = (roi_mask * pred_local + (1.0 - roi_mask) * coarse).clamp(-1, 1)

        return {
            "pred": pred,
            "pred_local": pred_local,
            "coarse": coarse,
            "coarse_backward": backward,
            "coarse_raft": raft_c,
            "coarse_piece": piece_c,
            "coarse_splat": splat_c,
            "warped0": w0, "warped1": w1,
            "flow_t0": ft0, "flow_t1": ft1,
            "occ_weights": occ_w, "occ_logits": occ_l,
            "disocclusion": wt0,
            "gate_weights": gates,
            "t_map": t_map,
            "line_logits": aux["line_logits"],
            "line_pred": aux["line_pred"],
            "residual": residual, "raw_residual": raw_res,
        }

    def forward(self, i0, i1, edge0, edge1, dt0, dt1,
                flow_fwd, flow_bwd, seg0, seg1,
                t: TensorOrFloat = 0.5,
                sketch_hint: Optional[torch.Tensor] = None,
                roi_mask: Optional[torch.Tensor] = None):
        b, _, h, w = i0.shape
        if sketch_hint is None:
            sketch_hint = i0.new_zeros(b, 1, h, w)
        if roi_mask is None:
            roi_mask = i0.new_ones(b, 1, h, w)
        if sketch_hint.shape[-2:] != (h, w):
            sketch_hint = F.interpolate(sketch_hint, size=(h, w), mode="bilinear", align_corners=False)
        if roi_mask.shape[-2:] != (h, w):
            roi_mask = F.interpolate(roi_mask, size=(h, w), mode="bilinear", align_corners=False)
        sketch_hint = sketch_hint.clamp(0, 1)
        roi_mask = roi_mask.clamp(0, 1)

        t_map = self.timing(i0, i1, edge0, edge1, dt0, dt1, seg0, seg1,
                            flow_fwd, flow_bwd, t) if self.use_timing_net else _make_t_map(
            t, b, h, w, i0.device, i0.dtype)

        S = self.ifnet.encode_shared(i0, i1, edge0, edge1, dt0, dt1,
                                     seg0, seg1, flow_fwd, flow_bwd,
                                     sketch_hint=sketch_hint)
        ft0, ft1, ol, ow, w0, w1, aux = self.ifnet.estimate_flow(S, t_map)
        splat_data = self.splat(i0, i1, edge0, edge1, flow_fwd, flow_bwd, t=t_map)

        c0 = F.interpolate(S["ctx0_s2"], size=(h, w), mode="bilinear", align_corners=False)
        c1 = F.interpolate(S["ctx1_s2"], size=(h, w), mode="bilinear", align_corners=False)
        return self._compose(ft0, ft1, ol, ow, w0, w1,
                             i0, i1, edge0, edge1, dt0, dt1, seg0, seg1, c0, c1,
                             splat_data, aux, sketch_hint, roi_mask, t_map)

    def generate_multi_t(self, i0, i1, edge0, edge1, dt0, dt1,
                         flow_fwd, flow_bwd, seg0, seg1,
                         t_values=(0.25, 0.5, 0.75),
                         sketch_hint: Optional[torch.Tensor] = None,
                         roi_mask: Optional[torch.Tensor] = None):
        return [self.forward(i0, i1, edge0, edge1, dt0, dt1,
                             flow_fwd, flow_bwd, seg0, seg1,
                             t=t, sketch_hint=sketch_hint, roi_mask=roi_mask)
                for t in t_values]

    def unfreeze_backbone(self):
        if hasattr(self.ifnet.ctx_encoder, "unfreeze_backbone"):
            self.ifnet.ctx_encoder.unfreeze_backbone()

    @staticmethod
    def from_v8_checkpoint(checkpoint_path, device="cpu", **kwargs):
        """Best-effort warm start from V8/V8.2: copy matching tensors and expand first convs."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        src = ckpt.get("model", ckpt)
        model = AnimeIFNetV9Geom(**kwargs).to(device)
        dst = model.state_dict()
        copied, expanded, skipped = 0, 0, []
        for k, v in src.items():
            if k not in dst:
                skipped.append(k)
                continue
            target = dst[k]
            if target.shape == v.shape:
                dst[k] = v.to(device)
                copied += 1
            elif (target.ndim == v.ndim and target.ndim >= 2 and
                  target.shape[0] == v.shape[0] and target.shape[2:] == v.shape[2:] and
                  target.shape[1] > v.shape[1]):
                new_v = target.clone()
                new_v[:, :v.shape[1]] = v.to(device)
                new_v[:, v.shape[1]:] = 0
                dst[k] = new_v
                expanded += 1
            else:
                skipped.append(f"{k}: {list(v.shape)} -> {list(target.shape)}")
        model.load_state_dict(dst, strict=False)
        print(f"[from_v8_checkpoint] copied={copied}, expanded={expanded}, skipped={len(skipped)}")
        if skipped:
            print(f"  first skipped: {skipped[:8]}")
        return model


if __name__ == "__main__":
    print("AnimeIFNetV9Geom self-test")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = AnimeIFNetV9Geom(freeze_backbone=True, recurrent_steps=2).to(dev).eval()
    B, H, W = 1, 128, 128
    x = torch.randn(B, 3, H, W, device=dev).clamp(-1, 1)
    y = torch.randn(B, 3, H, W, device=dev).clamp(-1, 1)
    one = torch.rand(B, 1, H, W, device=dev)
    flow = torch.zeros(B, 2, H, W, device=dev)
    with torch.no_grad():
        out = m(x, y, one, one, one, one, flow, flow, one, one, sketch_hint=one)
    print({k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v) and k in ["pred", "t_map", "line_pred", "gate_weights"]})