"""
app.py — Gradio-интерфейс AnimeIFNet (таймлайн ключевых кадров)
══════════════════════════════════════════════════════════════

Бывший app_v8_4_light.py.
  • Таймлайн из N ключевых кадров, генерация выбранной пары или всей шкалы.
  • Холст скетча-подсказки (авто-скетч v2 кладётся в редактируемый слой).
  • Холст ROI с пресетами (персонаж / движение / поток) и ручной правкой.
  • Вкладка «Сравнение ×4»: без подсказок / +скетч / +ROI / +оба.
  • Отладочные карты (поток, дизокклюзия, невязка), экспорт PNG / GIF / ZIP.

Запуск из корня репозитория:

    python app.py --checkpoint weights/best_v8_2.pth \\
        [--a2s-weights weights/anime2sketch.pth] [--share]

Для 1920×1080 в интерфейсе стоит выбрать «Макс. сторона при инференсе» 960/1280.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

import animeifnet.pipeline as pl
from animeifnet.pipeline import (
    MODEL_REGISTRY, MODELS_WITH_ROI, MODELS_WITH_SKETCH, MODELS_A2S_EDGE,
    ExternalFeatures, FeatureProvider,
    ablation_run, build_roi_mask, evaluate_pair, interpolate_multi,
    interpolate_pair, is_scene_change, load_model, onion_skin, _tensor1,
)

PostProcessor = pl.get_post_processor_cls()


# ══════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════

class AppState:
    def __init__(self):
        self.model = None
        self.model_name = "v8_2"
        self.fp: Optional[FeatureProvider] = None
        self.device = torch.device("cpu")
        self.lpips_metric = None

    def init(self, checkpoint, model_name, a2s_weights=None, a2s_repo=None,
             isnet_weights=None, dis_repo=None, raft_size="large",
             raft_max_side=0, compile_model=False, cache_size=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[app] device: {self.device}")

        self.model = load_model(checkpoint, model_name, self.device,
                                compile_model=compile_model)
        self.model_name = model_name
        self.fp = FeatureProvider(
            mode="live", device=self.device,
            raft_size=raft_size, raft_max_side=raft_max_side,
            isnet_weights=isnet_weights, dis_repo=dis_repo,
            a2s_weights=a2s_weights, a2s_repo=a2s_repo,
            need_a2s=(model_name in MODELS_A2S_EDGE and a2s_weights is not None),
            cache_size=cache_size,
        )

        try:
            import lpips
            self.lpips_metric = lpips.LPIPS(net="alex").to(self.device).eval()
            print("[app] LPIPS loaded")
        except Exception:
            self.lpips_metric = None
            print("[app] LPIPS unavailable; PSNR/SSIM only")

    def status_html(self) -> str:
        def pill(label: str, ok: bool) -> str:
            colors = ("#fff0eb", "#c43a13", "#ffd8c7") if ok else ("#f4f4ef", "#76767c", "#e0e0db")
            style = (f"padding:3px 9px;border-radius:999px;background:{colors[0]};"
                     f"color:{colors[1]};border:1px solid {colors[2]};"
                     "font:600 10.5px/1 'JetBrains Mono',ui-monospace,monospace;"
                     "margin-right:6px;letter-spacing:.02em;")
            return f"<span style='{style}'>{label} {'✓' if ok else '—'}</span>"

        fp = self.fp
        return (
            "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin-top:6px'>"
            + pill("CUDA", str(self.device).startswith("cuda"))
            + pill(self.model_name, self.model is not None)
            + pill("AMP", pl.USE_AMP)
            + pill("A2S", fp is not None and fp.a2s is not None)
            + pill("ISNet", fp is not None and fp.isnet is not None)
            + pill("LPIPS", self.lpips_metric is not None)
            + "</div>"
        )


STATE = AppState()


# ══════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════

def _np_to_pil(arr) -> Optional[Image.Image]:
    if arr is None:
        return None
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    if isinstance(arr, str):
        return Image.open(arr).convert("RGB")
    if isinstance(arr, np.ndarray):
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return None


def _pil_to_np(img: Optional[Image.Image]) -> Optional[np.ndarray]:
    return None if img is None else np.array(img.convert("RGB"), dtype=np.uint8)


def _open_file_image(f) -> Image.Image:
    path = f if isinstance(f, str) else getattr(f, "name", None)
    if path is None:
        raise ValueError(f"Unsupported file object: {type(f)}")
    return Image.open(path).convert("RGB")


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask, 0, 1)


def _first_layer(editor_dict) -> Optional[np.ndarray]:
    layers = (editor_dict or {}).get("layers") or []
    if not layers or layers[0] is None:
        return None
    layer = layers[0]
    return np.array(layer) if isinstance(layer, Image.Image) else layer


def _extract_roi_from_editor(editor_dict, h, w) -> Optional[np.ndarray]:
    if not isinstance(editor_dict, dict):
        return None
    layer = _first_layer(editor_dict)
    if layer is None:
        return None
    if layer.ndim == 3 and layer.shape[-1] == 4:
        mask = layer[..., 3].astype(np.float32) / 255.0
    elif layer.ndim == 3 and layer.shape[-1] == 3:
        mask = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        mask = layer.astype(np.float32) / 255.0
    if mask.sum() < 10:
        return None
    return _resize_mask(mask, h, w)


def _extract_sketch_from_editor(editor_dict, h, w) -> Optional[np.ndarray]:
    if not isinstance(editor_dict, dict):
        return None
    layer = _first_layer(editor_dict)
    if layer is not None and layer.ndim == 3 and layer.shape[-1] == 4:
        alpha = layer[..., 3].astype(np.float32) / 255.0
        if alpha.sum() >= 10:
            return _resize_mask(alpha, h, w)
    comp = editor_dict.get("composite")
    if comp is not None:
        c = comp if isinstance(comp, np.ndarray) else np.array(comp)
        if c.ndim == 3 and c.shape[-1] in (3, 4):
            gray = cv2.cvtColor(c[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = c.astype(np.float32) / 255.0
        inv = 1.0 - gray
        if inv.sum() >= 20:
            return _resize_mask(inv, h, w)
    return None


# ── Значения ImageEditor ──────────────────────────────────────
# Автоматический результат (авто-скетч / пресет ROI) кладётся в layer[0]
# как RGBA-слой, чтобы его можно было дорисовать кистью или стереть.

def _white_background(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _composite(bg: np.ndarray, layer: np.ndarray) -> np.ndarray:
    a = layer[..., 3:4].astype(np.float32) / 255.0
    return (bg.astype(np.float32) * (1 - a)
            + layer[..., :3].astype(np.float32) * a).astype(np.uint8)


def _editor_value(bg: np.ndarray, layer: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if layer is None:
        layer = np.zeros((*bg.shape[:2], 4), dtype=np.uint8)
    return {"background": bg, "layers": [layer], "composite": _composite(bg, layer)}


def _empty_sketch_editor_value(frame: Optional[Image.Image]) -> Dict[str, Any]:
    if frame is None:
        return _editor_value(_white_background(360, 640))
    w, h = frame.size
    return _editor_value(_white_background(h, w))


def _empty_roi_editor_value(frame: Optional[Image.Image]) -> Dict[str, Any]:
    if frame is None:
        return _editor_value(np.full((360, 640, 3), 240, dtype=np.uint8))
    return _editor_value(np.array(frame.convert("RGB")))


def _sketch_editor_value_from_mask(sk: np.ndarray) -> Dict[str, Any]:
    """Белый фон + чёрный слой с alpha = sk."""
    h, w = sk.shape[:2]
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[..., 3] = np.clip(sk * 255.0, 0, 255).astype(np.uint8)
    return _editor_value(_white_background(h, w), layer)


def _roi_editor_value_from_mask(frame: Image.Image, mask: np.ndarray,
                                alpha_strength: float = 0.55) -> Dict[str, Any]:
    """Кадр как фон + красный полупрозрачный слой с alpha = mask."""
    bg = np.array(frame.convert("RGB"))
    h, w = bg.shape[:2]
    mask = _resize_mask(mask, h, w)
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[..., :3] = (255, 59, 48)
    layer[..., 3] = np.clip(mask * 255.0 * alpha_strength, 0, 255).astype(np.uint8)
    return _editor_value(bg, layer)


# ── Совместимость с разными версиями Gradio ──────────────────

def _safe_brush(colors, default_size):
    try:
        return gr.Brush(colors=colors, default_size=default_size, color_mode="defaults")
    except TypeError:
        return gr.Brush(colors=colors, default_size=default_size)


def _safe_eraser(default_size):
    try:
        return gr.Eraser(default_size=default_size)
    except TypeError:
        return gr.Eraser()


def _safe_image_editor(**kwargs):
    try:
        return gr.ImageEditor(**kwargs)
    except TypeError:
        for k in ("sources", "transforms", "canvas_size", "fixed_canvas"):
            kwargs.pop(k, None)
        return gr.ImageEditor(**kwargs)


# ── Картинки для отладки ─────────────────────────────────────

def _maybe_downscale(img: Image.Image, max_side: int, mult: int = 32
                     ) -> Tuple[Image.Image, Optional[Tuple[int, int]]]:
    if max_side <= 0:
        return img, None
    w, h = img.size
    if max(w, h) <= max_side:
        return img, None
    scale = max_side / max(w, h)
    new_w = ((max(mult, int(round(w * scale))) + mult - 1) // mult) * mult
    new_h = ((max(mult, int(round(h * scale))) + mult - 1) // mult) * mult
    return img.resize((new_w, new_h), Image.BICUBIC), (w, h)


def _flow_to_rgb(flow: torch.Tensor) -> np.ndarray:
    fx, fy = flow[0].float().cpu().numpy(), flow[1].float().cpu().numpy()
    mag = np.sqrt(fx ** 2 + fy ** 2)
    hsv = np.zeros((*mag.shape, 3), np.float32)
    hsv[..., 0] = (np.arctan2(fy, fx) + np.pi) / (2 * np.pi) * 179
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _heatmap(x: torch.Tensor) -> np.ndarray:
    x = (x[0] if x.dim() == 3 else x).float().cpu().numpy()
    x = ((x - x.min()) / (x.max() - x.min() + 1e-8) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(x, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def _crop(t, pad_info):
    h, w = (pad_info[2], pad_info[3]) if len(pad_info) == 4 else pad_info
    return t[:, :h, :w]


def _create_demo_keyframe(pose: float, size=(640, 360)) -> Image.Image:
    """Заглушка-кадр для проверки интерфейса без своих файлов."""
    w, h = size
    img = Image.new("RGB", size, (20, 22, 28))
    d = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    for x in range(0, w, 40):
        d.line((x, 0, x, h), fill=(28, 30, 36))
    for y in range(0, h, 40):
        d.line((0, y, w, y), fill=(28, 30, 36))
    head_y = cy - 80 - int(pose * 15)
    limb = (235, 232, 225)
    d.ellipse((cx - 28, head_y - 28, cx + 28, head_y + 28),
              fill=(238, 207, 170), outline=(120, 90, 55), width=3)
    d.line((cx, head_y + 28, cx, head_y + 115), fill=limb, width=8)
    arm = np.deg2rad(-40 + pose * 120)
    d.line((cx, head_y + 60, cx + int(np.cos(arm) * 95), head_y + 60 + int(np.sin(arm) * 95)),
           fill=limb, width=8)
    d.line((cx, head_y + 60, cx - 80, head_y + 110), fill=limb, width=8)
    d.line((cx, head_y + 115, cx - 35, head_y + 180), fill=limb, width=8)
    d.line((cx, head_y + 115, cx + 35, head_y + 180), fill=limb, width=8)
    d.text((12, 10), f"demo pose={pose:.2f}", fill=(120, 120, 130))
    return img


# ── Таймлайн ─────────────────────────────────────────────────

def _timeline_sequence(keyframes: List[Image.Image],
                       generated: Dict[int, List[Image.Image]]) -> List[Tuple[Image.Image, str]]:
    items: List[Tuple[Image.Image, str]] = []
    for i, kf in enumerate(keyframes):
        items.append((kf, f"K{i+1:02d}"))
        if i < len(keyframes) - 1:
            items += [(fr, f"K{i+1:02d}.{j+1:02d}") for j, fr in enumerate(generated.get(i, []))]
    return items


def _build_timeline_gallery(state: Optional[Dict[str, Any]]) -> List[Tuple[Image.Image, str]]:
    if not state or not state.get("keyframes"):
        return []
    keyframes = state["keyframes"]
    generated = state.get("generated", {})
    selected = int(state.get("selected_pair", 0))
    items: List[Tuple[Image.Image, str]] = []
    for i, kf in enumerate(keyframes):
        items.append((kf, f"K{i+1:02d}" + (" ★" if i in (selected, selected + 1) else "")))
        if i == len(keyframes) - 1:
            break
        frames = generated.get(i, [])
        if frames:
            items += [(fr, f"I{i+1:02d}-{j+1:02d}") for j, fr in enumerate(frames)]
        else:
            ph = Image.new("RGB", (120, 80), (24, 24, 28))
            d = ImageDraw.Draw(ph)
            color = (255, 106, 61) if i == selected else (80, 80, 88)
            d.line((10, 40, 110, 40), fill=color, width=2)
            d.text((40, 30), f"+{i+1}", fill=color)
            items.append((ph, f"gap {i+1}"))
    return items


def _pair_choices(state: Optional[Dict[str, Any]]) -> Tuple[List[str], Optional[str]]:
    if not state or len(state.get("keyframes", [])) < 2:
        return [], None
    n = len(state["keyframes"])
    choices = [f"K{i+1:02d} → K{i+2:02d}" for i in range(n - 1)]
    idx = min(max(int(state.get("selected_pair", 0)), 0), n - 2)
    return choices, choices[idx]


def _pair_index_from_value(pair_value: Any, state: Dict[str, Any]) -> int:
    if not state or len(state.get("keyframes", [])) < 2:
        return 0
    choices, _ = _pair_choices(state)
    if pair_value in choices:
        return choices.index(pair_value)
    try:
        return int(pair_value)
    except (TypeError, ValueError):
        return int(state.get("selected_pair", 0))


def _require_pair(state):
    if not state or len(state.get("keyframes", [])) < 2:
        raise gr.Error("Сначала загрузите минимум два ключевых кадра.")


# ── Авто-скетч ───────────────────────────────────────────────

def _resize_flow(flow: np.ndarray, h: int, w: int) -> np.ndarray:
    if flow.shape[-2:] == (h, w):
        return flow
    sx, sy = w / flow.shape[-1], h / flow.shape[-2]
    return np.stack([cv2.resize(flow[0], (w, h)) * sx, cv2.resize(flow[1], (w, h)) * sy])


def _build_auto_sketch_v2(img0: Image.Image, img1: Image.Image,
                          features: ExternalFeatures, t: float,
                          a2s=None) -> Optional[np.ndarray]:
    """
    Авто-скетч v2: контуры обоих кадров (Anime2Sketch), приближённо
    перенесённые в момент t по RAFT-потоку, затем max-смешивание.
    """
    h, w = np.array(img0).shape[:2]
    if features.a2s_edge0 is not None and features.a2s_edge1 is not None:
        e0 = features.a2s_edge0.squeeze().cpu().numpy().astype(np.float32)
        e1 = features.a2s_edge1.squeeze().cpu().numpy().astype(np.float32)
    elif a2s is not None:
        e0, e1 = a2s.predict_edge_pair(img0, img1)
    else:
        return None
    e0, e1 = _resize_mask(e0, h, w), _resize_mask(e1, h, w)

    ff = _resize_flow(features.flow_fwd.squeeze(0).float().cpu().numpy(), h, w)
    fb = _resize_flow(features.flow_bwd.squeeze(0).float().cpu().numpy(), h, w)

    t = float(t)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    w0 = cv2.remap(e0, xx - t * ff[0], yy - t * ff[1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w1 = cv2.remap(e1, xx - (1 - t) * fb[0], yy - (1 - t) * fb[1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sk = cv2.GaussianBlur(np.maximum(w0 * (1 - t), w1 * t), (0, 0), 0.35)
    return np.clip(sk, 0, 1).astype(np.float32)


def _post_processor_from_params(do_post, detail, ghost, color, sharpen):
    if not do_post or PostProcessor is None:
        return None
    return PostProcessor(
        do_detail_injection=detail > 0, do_ghost_fix=ghost > 0,
        do_color_fix=color > 0, do_sharpen=sharpen > 0,
        detail_strength=detail, ghost_strength=ghost,
        color_strength=color, sharpen_strength=sharpen,
    )


# ══════════════════════════════════════════════════════════════
# Генерация
# ══════════════════════════════════════════════════════════════

def _prepare_pair(f1, f3, gt, t_val, infer_max_side,
                  sketch_editor, use_auto_sketch, roi_editor, roi_preset,
                  allow_manual_controls=True) -> Dict[str, Any]:
    """Уменьшение кадров, внешние признаки, скетч и ROI для пары."""
    if STATE.model is None or STATE.fp is None:
        raise gr.Error("Модель не загружена. Запустите app с --checkpoint ...")

    if f3.size != f1.size:
        f3 = f3.resize(f1.size, Image.BILINEAR)
    if gt is not None and gt.size != f1.size:
        gt = gt.resize(f1.size, Image.BILINEAR)

    orig_w, orig_h = f1.size
    f1_inf, orig_size = _maybe_downscale(f1, int(infer_max_side))
    f3_inf, _ = _maybe_downscale(f3, int(infer_max_side))
    gt_inf = _maybe_downscale(gt, int(infer_max_side))[0] if gt is not None else None
    w, h = f1_inf.size

    feats = STATE.fp.get_features(f1_inf, f3_inf, img_gt=gt_inf)

    sketch_tensor = None
    if STATE.model_name in MODELS_WITH_SKETCH:
        sketch_np = None
        if allow_manual_controls:
            sketch_np = _extract_sketch_from_editor(sketch_editor, h, w)
        if sketch_np is None and use_auto_sketch:
            sketch_np = _build_auto_sketch_v2(f1_inf, f3_inf, feats, float(t_val), STATE.fp.a2s)
        if sketch_np is not None:
            sketch_tensor = _tensor1(sketch_np, STATE.device)

    roi_mode, roi_array = "full", None
    if STATE.model_name in MODELS_WITH_ROI:
        if allow_manual_controls:
            roi_array = _extract_roi_from_editor(roi_editor, h, w)
        if roi_array is not None:
            roi_mode = "array"
        elif roi_preset not in ("none", "full"):
            roi_mode = {"character": "seg", "flow": "flow_mag"}.get(roi_preset, roi_preset)

    return {
        "f1": f1, "f3": f3, "gt": gt, "f1_inf": f1_inf, "f3_inf": f3_inf,
        "feats": feats, "sketch": sketch_tensor,
        "roi_mode": roi_mode, "roi_array": roi_array,
        "output_size": (orig_w, orig_h) if orig_size is not None else None,
        "perf_info": (f"_⚡ Inference: {w}×{h} → upscale {orig_w}×{orig_h}_\n\n"
                      if orig_size is not None else ""),
    }


def _generate_pair_core(f1, f3, gt, t_val, easing, num_frames,
                        sketch_editor, use_auto_sketch, roi_editor, roi_preset,
                        use_recursive, use_onion, use_tta, scene_cut,
                        infer_max_side, do_post, detail_str, ghost_str,
                        color_str, sharpen_str, allow_manual_controls=True):
    P = _prepare_pair(f1, f3, gt, t_val, infer_max_side, sketch_editor,
                      use_auto_sketch, roi_editor, roi_preset, allow_manual_controls)
    f1, f3, gt, output_size = P["f1"], P["f3"], P["gt"], P["output_size"]

    pp = _post_processor_from_params(do_post, detail_str, ghost_str, color_str, sharpen_str)
    if do_post and pp is None:
        gr.Warning("PostProcessor недоступен — положите post_process.py в animeifnet/")

    common = dict(roi_mode=P["roi_mode"], roi_file=None, device=STATE.device,
                  use_nedt=True, sketch_hint=P["sketch"], roi_array=P["roi_array"])

    # Отладочный forward: при одном кадре — он же результат.
    r = interpolate_pair(STATE.model, P["f1_inf"], P["f3_inf"], P["feats"], STATE.model_name,
                         t=float(t_val) if int(num_frames) == 1 else 0.5,
                         use_tta=use_tta if int(num_frames) == 1 else False,
                         post_processor=pp, output_size=output_size, **common)
    if int(num_frames) == 1:
        frames = [r["result"]]
    else:
        frames = interpolate_multi(STATE.model, P["f1_inf"], P["f3_inf"], P["feats"],
                                   STATE.model_name, num_frames=int(num_frames),
                                   easing=easing, recursive=use_recursive,
                                   use_tta=use_tta, **common)
        if output_size is not None:
            frames = [fr.resize(output_size, Image.LANCZOS) for fr in frames]

    if use_onion:
        frames = [onion_skin(f1, f3, fr) for fr in frames]

    raw_out, pad_info = r["raw_out"], r["pad_info"]
    flow_dbg = _flow_to_rgb(_crop(raw_out["flow_t0"][0], pad_info)) if "flow_t0" in raw_out else None
    disocc_dbg = (_heatmap(_crop(raw_out["occ_weights"][0], pad_info)[0:1])
                  if "occ_weights" in raw_out else None)
    residual_dbg = (_heatmap(_crop(raw_out["residual"][0], pad_info).abs().mean(0, keepdim=True))
                    if "residual" in raw_out else None)

    main_frame = frames[len(frames) // 2]

    md = P["perf_info"]
    if scene_cut and is_scene_change(f1, f3):
        md = "⚠️ Возможна смена сцены — результат может быть некорректным.\n\n" + md
    if gt is not None:
        m = evaluate_pair(main_frame, gt, STATE.lpips_metric)
        md += (f"### vs GT\n\n- **PSNR**: `{m['psnr']:.2f} dB`\n"
               f"- **SSIM**: `{m['ssim']:.4f}`\n")
        if "lpips" in m:
            md += f"- **LPIPS**: `{m['lpips']:.4f}`\n"
    else:
        md += "_Загрузите GT для метрик._"
    if pp is not None:
        md += (f"\n\n_🔧 Post: detail={detail_str:.2f}, ghost={ghost_str:.2f}, "
               f"color={color_str:.2f}, sharpen={sharpen_str:.2f}_")
    if P["sketch"] is not None:
        md += "\n\n_✏ Sketch hint active_"
    if P["roi_mode"] != "full":
        md += f"\n\n_◉ ROI: {'нарисованная' if P['roi_mode'] == 'array' else P['roi_mode']}_"

    return {"frames": frames, "main_frame": main_frame, "metrics_md": md,
            "flow_dbg": flow_dbg, "disocc_dbg": disocc_dbg,
            "residual_dbg": residual_dbg, "roi_mode": P["roi_mode"]}


def _export_sequence(seq: List[Image.Image], prefix="timeline"
                     ) -> Tuple[str, Optional[str], str]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="animeifnet_"))
    png_path = str(tmp_dir / f"{prefix}_pred.png")
    (seq[len(seq) // 2] if seq else Image.new("RGB", (64, 64))).save(png_path)

    gif_path = None
    if len(seq) > 1:
        gif_path = str(tmp_dir / f"{prefix}.gif")
        seq[0].save(gif_path, save_all=True, append_images=seq[1:], duration=120, loop=0)

    zip_path = str(tmp_dir / f"{prefix}_frames.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i, fr in enumerate(seq):
            p = tmp_dir / f"{i+1:05d}.png"
            fr.save(p)
            zf.write(p, p.name)
    return png_path, gif_path, zip_path


# ══════════════════════════════════════════════════════════════
# Действия интерфейса
# ══════════════════════════════════════════════════════════════

def _loaded_outputs(keyframes, label):
    state = {"keyframes": keyframes, "generated": {}, "selected_pair": 0}
    choices, value = _pair_choices(state)
    return (state, _build_timeline_gallery(state),
            gr.update(choices=choices, value=value),
            _pil_to_np(keyframes[0]), _pil_to_np(keyframes[1]),
            _empty_sketch_editor_value(keyframes[0]),
            _empty_roi_editor_value(keyframes[0]),
            f"### Timeline\n\n{label}: **{len(keyframes)}** keyframes. Выбрана пара `{value}`.")


def action_load_keyframes(files):
    if not files or len(files) < 2:
        raise gr.Error("Выберите минимум два изображения.")
    keyframes = [_open_file_image(f) for f in files]
    base = keyframes[0].size
    keyframes = [im if im.size == base else im.resize(base, Image.BILINEAR) for im in keyframes]
    return _loaded_outputs(keyframes, "Загружено")


def action_load_demo():
    return _loaded_outputs([_create_demo_keyframe(p) for p in (0.0, 0.35, 1.0)], "Demo")


def action_select_pair(pair_value, state):
    if not state or len(state.get("keyframes", [])) < 2:
        return state, None, None, None, None, []
    idx = _pair_index_from_value(pair_value, state)
    state = dict(state, selected_pair=idx)
    kf = state["keyframes"]
    return (state, _pil_to_np(kf[idx]), _pil_to_np(kf[idx + 1]),
            _empty_sketch_editor_value(kf[idx]), _empty_roi_editor_value(kf[idx]),
            _build_timeline_gallery(state))


def action_generate_selected(state, pair_value, gt_img, t_val, easing, num_frames,
                             sketch_editor, use_auto_sketch, roi_editor, roi_preset,
                             use_recursive, use_onion, use_tta, scene_cut,
                             infer_max_side, do_post, detail_str, ghost_str,
                             color_str, sharpen_str):
    _require_pair(state)
    idx = _pair_index_from_value(pair_value, state)
    kf = state["keyframes"]
    out = _generate_pair_core(
        kf[idx], kf[idx + 1], _np_to_pil(gt_img), t_val, easing, num_frames,
        sketch_editor, use_auto_sketch, roi_editor, roi_preset,
        use_recursive, use_onion, use_tta, scene_cut,
        infer_max_side, do_post, detail_str, ghost_str, color_str, sharpen_str,
    )
    state = dict(state, generated={**state.get("generated", {}), idx: out["frames"]},
                 selected_pair=idx)
    png, gif, zp = _export_sequence([kf[idx]] + out["frames"] + [kf[idx + 1]],
                                    prefix=f"pair_{idx+1:02d}")
    seq_gallery = [(fr, f"I{idx+1:02d}-{i+1:02d}") for i, fr in enumerate(out["frames"])]
    return (state, out["main_frame"], out["main_frame"], out["metrics_md"],
            out["flow_dbg"], out["disocc_dbg"], out["residual_dbg"],
            gr.update(value=seq_gallery, visible=True), _build_timeline_gallery(state),
            png, gif, zp, gr.Tabs(selected="output"), list(out["frames"]))


def action_generate_all(state, pair_value, gt_img, t_val, easing, num_frames,
                        sketch_editor, use_auto_sketch, roi_editor, roi_preset,
                        use_recursive, use_onion, use_tta, scene_cut,
                        infer_max_side, do_post, detail_str, ghost_str,
                        color_str, sharpen_str):
    _require_pair(state)
    keyframes = state["keyframes"]
    selected = _pair_index_from_value(pair_value, state)
    gt = _np_to_pil(gt_img)
    generated: Dict[int, List[Image.Image]] = {}
    lines = ["### Timeline generation", ""]
    debug = (None, None, None)

    for idx in range(len(keyframes) - 1):
        # Ручные скетч, ROI и эталон относятся только к выбранной паре.
        is_sel = idx == selected
        out = _generate_pair_core(
            keyframes[idx], keyframes[idx + 1], gt if is_sel else None,
            t_val, easing, num_frames, sketch_editor, use_auto_sketch,
            roi_editor, roi_preset, False, use_onion, use_tta, scene_cut,
            infer_max_side, do_post, detail_str, ghost_str, color_str, sharpen_str,
            allow_manual_controls=is_sel,
        )
        generated[idx] = out["frames"]
        lines.append(f"- `K{idx+1:02d} → K{idx+2:02d}`: {len(out['frames'])} inbetween(s), "
                     f"ROI `{out['roi_mode']}`")
        if is_sel:
            debug = (out["flow_dbg"], out["disocc_dbg"], out["residual_dbg"])

    state = dict(state, generated=generated, selected_pair=selected)
    seq = _timeline_sequence(keyframes, generated)
    full_seq = [img for img, _ in seq]
    png, gif, zp = _export_sequence(full_seq, prefix="full_timeline")
    main_frame = generated[selected][len(generated[selected]) // 2]
    return (state, main_frame, main_frame, "\n".join(lines), *debug,
            gr.update(value=seq, visible=True), _build_timeline_gallery(state),
            png, gif, zp, gr.Tabs(selected="output"), full_seq)


def action_compare(state, pair_value, t_val, infer_max_side,
                   sketch_editor, use_auto_sketch, roi_editor, roi_preset):
    """Сравнение ×4 для выбранной пары: без подсказок / +скетч / +ROI / +оба."""
    _require_pair(state)
    idx = _pair_index_from_value(pair_value, state)
    kf = state["keyframes"]
    P = _prepare_pair(kf[idx], kf[idx + 1], None, t_val, infer_max_side,
                      sketch_editor, use_auto_sketch, roi_editor, roi_preset)
    if P["sketch"] is None:
        gr.Warning("Скетч пуст — «+ sketch» совпадёт с базовым. Нарисуйте скетч "
                   "или включите авто-скетч.")
    roi_mode = P["roi_mode"]
    if roi_mode == "full":
        roi_mode = "seg"  # без маски сравниваем с маской персонажа
    ab = ablation_run(STATE.model, P["f1_inf"], P["f3_inf"], P["feats"], P["sketch"],
                      roi_mode=roi_mode, device=STATE.device, model_name=STATE.model_name,
                      t=float(t_val), roi_array=P["roi_array"])
    size = P["output_size"]
    res = [ab[k] if size is None else ab[k].resize(size, Image.LANCZOS)
           for k in ("baseline", "plus_sketch", "plus_roi", "plus_both")]
    return (*res, gr.Tabs(selected="compare"))


def action_auto_sketch_timeline(state, pair_value, t_val, infer_max_side):
    if STATE.fp is None or STATE.fp.a2s is None:
        raise gr.Error("Anime2Sketch недоступен. Запустите с --a2s-weights.")
    _require_pair(state)
    idx = _pair_index_from_value(pair_value, state)
    f1, f3 = state["keyframes"][idx], state["keyframes"][idx + 1]
    f1_inf, osz = _maybe_downscale(f1, int(infer_max_side))
    f3_inf, _ = _maybe_downscale(f3, int(infer_max_side))
    feats = STATE.fp.get_features(f1_inf, f3_inf)
    sk = _build_auto_sketch_v2(f1_inf, f3_inf, feats, float(t_val), STATE.fp.a2s)
    if sk is None:
        raise gr.Error("Не удалось построить авто-скетч.")
    if osz is not None:
        sk = cv2.resize(sk, f1.size, interpolation=cv2.INTER_LINEAR)
    return _sketch_editor_value_from_mask(np.clip(sk, 0, 1))


def action_clear_sketch(state, pair_value):
    if not state or len(state.get("keyframes", [])) < 2:
        return _empty_sketch_editor_value(None)
    return _empty_sketch_editor_value(state["keyframes"][_pair_index_from_value(pair_value, state)])


def action_preset_roi_timeline(state, pair_value, preset, infer_max_side):
    if not state or len(state.get("keyframes", [])) < 2:
        return _empty_roi_editor_value(None)
    idx = _pair_index_from_value(pair_value, state)
    f1, f3 = state["keyframes"][idx], state["keyframes"][idx + 1]
    if preset in ("none", "full") or STATE.fp is None:
        return _empty_roi_editor_value(f1)
    # Признаки на том же разрешении, что и инференс: попадают в кэш.
    f1_inf, _ = _maybe_downscale(f1, int(infer_max_side))
    f3_inf, _ = _maybe_downscale(f3, int(infer_max_side))
    mode = {"character": "seg", "flow": "flow_mag"}.get(preset, preset)
    mask = build_roi_mask(f1_inf, f3_inf, mode, STATE.fp.get_features(f1_inf, f3_inf))
    return _roi_editor_value_from_mask(f1, mask)


def action_reset_roi(state, pair_value):
    if not state or len(state.get("keyframes", [])) < 2:
        return _empty_roi_editor_value(None)
    return _empty_roi_editor_value(state["keyframes"][_pair_index_from_value(pair_value, state)])


def action_pick_sequence_frame(seq, evt: gr.SelectData):
    """Клик по миниатюре → кадр в большое окно и превью."""
    idx = evt.index if evt is not None else None
    if not seq or idx is None or not 0 <= idx < len(seq):
        return gr.update(), gr.update()
    return seq[idx], seq[idx]


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════

CSS_TIMELINE = """
:root {
  --bg-0: #faf9f5; --bg-1: #ffffff; --bg-2: #f4f3ee; --bg-3: #ebebe4; --bg-4: #e0e0d8;
  --border: #e4e3dc; --border-hi: #c8c7c0;
  --text-0: #1a1a1a; --text-1: #4a4a4d; --text-2: #7a7a7e; --text-3: #a5a5a8;
  --accent: #ff6a3d; --accent-hover: #e65a30; --accent-soft: #fff0eb; --accent-border: #ffd8c7;
  --sketch: #2563eb; --sketch-soft: #ebf2ff; --roi: #dc2626; --roi-soft: #fdecec;
  --shadow-sm: 0 1px 2px rgba(20,20,20,.04); --shadow-md: 0 4px 12px rgba(20,20,20,.06);
}

body, .gradio-container { background: var(--bg-0) !important; color: var(--text-0) !important; }
.gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", system-ui, sans-serif; }

#topbar {
  padding: 12px 18px !important; border-bottom: 1px solid var(--border);
  background: var(--bg-1); align-items: center; gap: 14px; box-shadow: var(--shadow-sm);
}
#topbar .gr-markdown { margin: 0 !important; }
.topbar-title { font-size: 16px; font-weight: 700; color: var(--text-0); line-height: 1.15; margin: 0; letter-spacing: -.01em; }
.topbar-sub { font-size: 11.5px; color: var(--text-2); margin: 2px 0 0 0; font-weight: 500; }

.section-title, .section-title-accent {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .09em;
  margin: 14px 0 6px 0; font-weight: 700;
}
.section-title { color: var(--text-2); }
.section-title-accent { color: var(--accent); }
.info-inline {
  color: var(--text-2) !important; font-size: 11.5px !important; margin: 0 !important;
  padding: 6px 10px !important; background: var(--bg-2); border-radius: 4px;
  border-left: 2px solid var(--border-hi);
}

.metric-box {
  background: var(--accent-soft) !important; border-left: 3px solid var(--accent) !important;
  padding: 12px 14px !important; border-radius: 6px;
}
.metric-box * { color: var(--text-0) !important; }
.best-card { box-shadow: 0 0 0 2px var(--accent), var(--shadow-md) !important; border-radius: 6px; }

button.primary, .primary button {
  background: var(--accent) !important; border-color: var(--accent) !important;
  color: #fff !important; font-weight: 600 !important; box-shadow: var(--shadow-sm);
}
button.primary:hover, .primary button:hover { background: var(--accent-hover) !important; border-color: var(--accent-hover) !important; }
button.secondary, .secondary button {
  background: var(--bg-1) !important; border: 1px solid var(--border-hi) !important;
  color: var(--text-0) !important; font-weight: 500 !important;
}
button.secondary:hover, .secondary button:hover { background: var(--bg-2) !important; border-color: var(--text-2) !important; }

.tabs > .tab-nav { background: var(--bg-1) !important; border-bottom: 1px solid var(--border) !important; padding: 0 8px; }
.tabs > .tab-nav > button {
  background: transparent !important; color: var(--text-1) !important; font-weight: 500 !important;
  border: none !important; border-bottom: 2px solid transparent !important;
  border-radius: 0 !important; padding: 10px 16px !important;
}
.tabs > .tab-nav > button.selected { color: var(--accent) !important; border-bottom-color: var(--accent) !important; background: transparent !important; }
.tab-badge-sketch::before { content: "● "; color: var(--sketch); margin-right: 2px; }
.tab-badge-roi::before { content: "● "; color: var(--roi); margin-right: 2px; }
.tab-badge-output::before { content: "▶ "; color: var(--accent); margin-right: 2px; }

.timeline-panel { border-top: 1px solid var(--border); background: var(--bg-1); padding: 10px 14px 14px 14px; }
.timeline-gallery .grid-wrap, .sequence-gallery .grid-wrap {
  background: var(--bg-2) !important; border: 1px solid var(--border) !important; border-radius: 6px;
}
.timeline-gallery img, .sequence-gallery img { object-fit: contain !important; background: var(--bg-1); border-radius: 4px; }
.sequence-gallery img { cursor: pointer; transition: transform .08s ease; }
.sequence-gallery img:hover { transform: scale(1.04); }
.timeline-gallery .thumbnail-item.selected, .sequence-gallery .thumbnail-item.selected { box-shadow: 0 0 0 2px var(--accent) !important; }

.gradio-container .image-editor, .image-editor { border: 1px solid var(--border) !important; border-radius: 8px; background: var(--bg-1) !important; }

.block, .form, .panel, .gradio-group { background: var(--bg-1) !important; border-color: var(--border) !important; }
.gr-input, .gr-textarea, .gr-dropdown, input, textarea, select {
  background: var(--bg-1) !important; border-color: var(--border-hi) !important; color: var(--text-0) !important;
}
label, .label, .gr-input-label { color: var(--text-1) !important; font-weight: 500 !important; font-size: 12.5px !important; }

.gradio-accordion { border: 1px solid var(--border) !important; border-radius: 6px !important; background: var(--bg-1) !important; margin-bottom: 8px; }
.gradio-accordion summary { font-weight: 600 !important; font-size: 12.5px !important; color: var(--text-0) !important; padding: 8px 12px !important; }

.tool-action-row { background: var(--bg-2); border-radius: 6px; padding: 8px 10px; border: 1px solid var(--border); margin-bottom: 6px; }
.tool-action-row button { font-size: 12px !important; }
"""


def _theme():
    return gr.themes.Default(
        primary_hue="orange",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#faf9f5", body_background_fill_dark="#faf9f5",
        background_fill_primary="#ffffff", background_fill_secondary="#f4f3ee",
        border_color_primary="#e4e3dc",
        block_background_fill="#ffffff", block_border_color="#e4e3dc",
        block_label_text_color="#4a4a4d", block_title_text_color="#1a1a1a",
        body_text_color="#1a1a1a", body_text_color_subdued="#7a7a7e",
        button_primary_background_fill="#ff6a3d",
        button_primary_background_fill_hover="#e65a30",
        button_primary_text_color="#ffffff", button_primary_border_color="#ff6a3d",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#f4f3ee",
        button_secondary_text_color="#1a1a1a", button_secondary_border_color="#c8c7c0",
    )


# В Gradio 6 theme и css передаются в launch(), в 4–5 — в конструктор Blocks.
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6


def _style_kwargs():
    return {"theme": _theme(), "css": CSS_TIMELINE}


def build_ui():
    blocks_kwargs = {} if GRADIO_6 else _style_kwargs()
    with gr.Blocks(title="AnimeIFNet · Timeline", **blocks_kwargs) as app:
        timeline_state = gr.State({"keyframes": [], "generated": {}, "selected_pair": 0})
        seq_state = gr.State([])  # PIL-кадры в порядке sequence_gallery

        # Topbar -------------------------------------------------
        with gr.Row(elem_id="topbar"):
            with gr.Column(scale=2, min_width=300):
                gr.Markdown(
                    "<p class='topbar-title'>AnimeIFNet · Timeline</p>"
                    "<p class='topbar-sub'>ключевые кадры → промежуточные · скетч · ROI · экспорт</p>"
                )
                gr.HTML(value=STATE.status_html())
            with gr.Column(scale=3):
                with gr.Row():
                    generate_selected_btn = gr.Button("🎬 Сгенерировать выбранную пару",
                                                      variant="primary", scale=0)
                    generate_all_btn = gr.Button("🎞 Сгенерировать всю шкалу",
                                                 variant="secondary", scale=0)

        with gr.Row():
            # ── Left column ────────────────────────────────────
            with gr.Column(scale=2, min_width=280):
                gr.Markdown("<div class='section-title'>Ключевые кадры</div>")
                keyframe_files = gr.Files(label="Загрузить N ключевых кадров",
                                          file_types=["image"], file_count="multiple",
                                          type="filepath")
                with gr.Row():
                    load_btn = gr.Button("⬆ Загрузить кадры", size="sm", variant="secondary")
                    demo_btn = gr.Button("📁 Демо", size="sm", variant="secondary")
                pair_dropdown = gr.Dropdown(label="Выбранная пара", choices=[], value=None)
                with gr.Row():
                    frame1 = gr.Image(label="K_i", type="numpy", height=110)
                    frame3 = gr.Image(label="K_{i+1}", type="numpy", height=110)
                gt = gr.Image(label="Эталон (необязательно)", type="numpy", height=100)

                gr.Markdown("<div class='section-title'>Время / движение</div>")
                t_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="t для выбранной пары")
                easing = gr.Dropdown(["linear", "ease_in", "ease_out", "ease_in_out"],
                                     value="linear", label="Сглаживание")
                num_frames = gr.Slider(1, 16, value=1, step=1, label="Промежуточных на пару")

                gr.Markdown("<div class='section-title'>Режимы</div>")
                with gr.Row():
                    recursive = gr.Checkbox(label="Рекурсивно", value=False)
                    onion = gr.Checkbox(label="Калька (onion)", value=False)
                with gr.Row():
                    tta = gr.Checkbox(label="TTA", value=False)
                    scene = gr.Checkbox(label="Предупреждать о смене сцены", value=True)

                gr.Markdown("<div class='section-title'>Производительность</div>")
                infer_max_side = gr.Dropdown(
                    choices=[("Полное разрешение (медленно)", 0), ("1280 хорошее качество", 1280),
                             ("960 баланс ⭐", 960), ("768 быстро", 768), ("512 предпросмотр", 512)],
                    value=960, label="Макс. сторона при инференсе",
                )

                gr.Markdown("<div class='section-title'>Постобработка</div>")
                do_post = gr.Checkbox(label="Добавление деталей + очистка", value=False,
                                      interactive=PostProcessor is not None)
                with gr.Group(visible=False) as post_group:
                    detail_str = gr.Slider(0, 1, value=0.6, step=0.05, label="Детали")
                    ghost_str = gr.Slider(0, 1, value=0.5, step=0.05, label="Устранение двоения")
                    color_str = gr.Slider(0, 1, value=0.4, step=0.05, label="Цвет")
                    sharpen_str = gr.Slider(0, 0.5, value=0.2, step=0.02, label="Резкость")

            # ── Center column ──────────────────────────────────
            with gr.Column(scale=5, min_width=560):
                with gr.Tabs(selected="sketch") as canvas_tabs:
                    with gr.Tab("Скетч-подсказка", id="sketch", elem_classes=["tab-badge-sketch"]):
                        with gr.Row(elem_classes=["tool-action-row"]):
                            auto_sketch_chk = gr.Checkbox(
                                label="Использовать авто-скетч v2, если холст пуст",
                                value=False, scale=2)
                            auto_sketch_btn = gr.Button("✨ Построить авто-скетч", size="sm",
                                                        variant="secondary", scale=1)
                            clear_sketch_btn = gr.Button("🧹 Очистить холст", size="sm",
                                                         variant="secondary", scale=1)
                        gr.Markdown(
                            "Нарисуйте промежуточную позу для выбранной пары. Авто-скетч "
                            "кладётся в редактируемый слой — его можно дополнить кистью "
                            "или стереть ластиком. При генерации всей шкалы ручной скетч "
                            "применяется только к выбранной паре.",
                            elem_classes=["info-inline"],
                        )
                        sketch_editor = _safe_image_editor(
                            label="Холст скетча", type="numpy", height=520,
                            brush=_safe_brush(colors=["#1a1a1a", "#2563eb", "#dc2626", "#16a34a"],
                                              default_size=4),
                            eraser=_safe_eraser(default_size=16),
                            layers=False, show_label=False,
                            sources=("upload", "clipboard"), transforms=("crop",),
                        )

                    with gr.Tab("Маска ROI", id="roi", elem_classes=["tab-badge-roi"]):
                        with gr.Row(elem_classes=["tool-action-row"]):
                            roi_preset = gr.Radio(choices=["full", "character", "motion", "flow"],
                                                  value="full", label="Пресет ROI", scale=3)
                            reset_roi_btn = gr.Button("🧹 Очистить ROI", size="sm",
                                                      variant="secondary", scale=1)
                        gr.Markdown(
                            "Выберите пресет — маска появится как **редактируемый слой** "
                            "поверх кадра. Расширьте её кистью или подчистите ластиком. "
                            "Если слой пустой, используется выбранный пресет.",
                            elem_classes=["info-inline"],
                        )
                        roi_editor = _safe_image_editor(
                            label="Холст ROI", type="numpy", height=520,
                            brush=_safe_brush(colors=["#dc2626", "#ff6a3d", "#2563eb"],
                                              default_size=40),
                            eraser=_safe_eraser(default_size=40),
                            layers=False, show_label=False,
                            sources=("upload",), transforms=("crop",),
                        )

                    with gr.Tab("Результат", id="output", elem_classes=["tab-badge-output"]):
                        pred_out = gr.Image(label="Выбранный результат", type="pil", height=460)
                        sequence_gallery = gr.Gallery(
                            label="Сгенерированная последовательность / выбранная пара",
                            columns=8, rows=1, height=140, object_fit="contain",
                            elem_classes=["sequence-gallery"], allow_preview=False,
                        )
                        gr.Markdown("<div class='section-title'>Отладочные карты</div>")
                        with gr.Row():
                            debug_flow = gr.Image(label="flow t→0", type="numpy", height=160)
                            debug_disocc = gr.Image(label="дизокклюзия", type="numpy", height=160)
                            debug_resid = gr.Image(label="невязка", type="numpy", height=160)

                    with gr.Tab("Сравнение ×4", id="compare"):
                        with gr.Row(elem_classes=["tool-action-row"]):
                            compare_btn = gr.Button("⚖ Сравнить для выбранной пары",
                                                    size="sm", variant="secondary")
                        gr.Markdown(
                            "Один и тот же кадр в четырёх режимах. Скетч и ROI берутся "
                            "с холстов; если ROI пуст и пресет `full`, используется маска персонажа.",
                            elem_classes=["info-inline"],
                        )
                        with gr.Row():
                            ab_baseline = gr.Image(label="базовый", type="pil", height=220)
                            ab_sketch = gr.Image(label="+ sketch", type="pil", height=220)
                        with gr.Row():
                            ab_roi = gr.Image(label="+ ROI", type="pil", height=220)
                            ab_both = gr.Image(label="+ both", type="pil", height=220,
                                               elem_classes=["best-card"])

            # ── Right column ───────────────────────────────────
            with gr.Column(scale=2, min_width=280):
                with gr.Accordion("Предпросмотр", open=True):
                    preview_out = gr.Image(type="pil", height=220, show_label=False)
                with gr.Accordion("Метрики / статус", open=True):
                    metrics_md = gr.Markdown("_Загрузите кадры, чтобы начать._",
                                             elem_classes=["metric-box"])
                with gr.Accordion("Экспорт", open=True):
                    png_out = gr.File(label="📄 pred.png", interactive=False)
                    gif_out = gr.File(label="🎞 sequence.gif", interactive=False)
                    zip_out = gr.File(label="🗜 frames.zip", interactive=False)

        # Bottom timeline bar -----------------------------------
        with gr.Column(elem_classes=["timeline-panel"]):
            gr.Markdown("<div class='section-title-accent'>Шкала времени · ключевые кадры "
                        "и сгенерированные промежуточные</div>")
            timeline_gallery = gr.Gallery(label="Шкала времени", columns=12, rows=1, height=145,
                                          object_fit="contain", elem_classes=["timeline-gallery"])

        # ── Events ─────────────────────────────────────────────
        do_post.change(lambda x: gr.update(visible=x), inputs=do_post, outputs=post_group)

        load_outputs = [timeline_state, timeline_gallery, pair_dropdown,
                        frame1, frame3, sketch_editor, roi_editor, metrics_md]
        load_btn.click(action_load_keyframes, inputs=[keyframe_files], outputs=load_outputs)
        demo_btn.click(action_load_demo, inputs=[], outputs=load_outputs)

        pair_dropdown.change(
            action_select_pair, inputs=[pair_dropdown, timeline_state],
            outputs=[timeline_state, frame1, frame3, sketch_editor, roi_editor, timeline_gallery],
        )

        auto_sketch_btn.click(action_auto_sketch_timeline,
                              inputs=[timeline_state, pair_dropdown, t_slider, infer_max_side],
                              outputs=[sketch_editor])
        clear_sketch_btn.click(action_clear_sketch, inputs=[timeline_state, pair_dropdown],
                               outputs=[sketch_editor])

        roi_preset.change(action_preset_roi_timeline,
                          inputs=[timeline_state, pair_dropdown, roi_preset, infer_max_side],
                          outputs=[roi_editor])
        reset_roi_btn.click(action_reset_roi, inputs=[timeline_state, pair_dropdown],
                            outputs=[roi_editor])

        gen_inputs = [timeline_state, pair_dropdown, gt, t_slider, easing, num_frames,
                      sketch_editor, auto_sketch_chk, roi_editor, roi_preset,
                      recursive, onion, tta, scene, infer_max_side,
                      do_post, detail_str, ghost_str, color_str, sharpen_str]
        gen_outputs = [timeline_state, pred_out, preview_out, metrics_md,
                       debug_flow, debug_disocc, debug_resid,
                       sequence_gallery, timeline_gallery,
                       png_out, gif_out, zip_out, canvas_tabs, seq_state]
        generate_selected_btn.click(action_generate_selected, inputs=gen_inputs, outputs=gen_outputs)
        generate_all_btn.click(action_generate_all, inputs=gen_inputs, outputs=gen_outputs)

        compare_btn.click(
            action_compare,
            inputs=[timeline_state, pair_dropdown, t_slider, infer_max_side,
                    sketch_editor, auto_sketch_chk, roi_editor, roi_preset],
            outputs=[ab_baseline, ab_sketch, ab_roi, ab_both, canvas_tabs],
        )

        sequence_gallery.select(action_pick_sequence_frame, inputs=[seq_state],
                                outputs=[pred_out, preview_out])

    return app


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="Gradio Timeline UI for AnimeIFNet")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="v8_2", choices=list(MODEL_REGISTRY))
    p.add_argument("--a2s-weights", default=None)
    p.add_argument("--a2s-repo", default=None)
    p.add_argument("--isnet-weights", default=None)
    p.add_argument("--dis-repo", default=None)
    p.add_argument("--raft-size", default="large", choices=["large", "small"])
    p.add_argument("--raft-max-side", type=int, default=0,
                   help="RAFT max side. 0 = полное разрешение; --fast ставит 768.")
    p.add_argument("--raft-iters", type=int, default=20)
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--compile", dest="compile_model", action="store_true")
    p.add_argument("--gpu-nedt", action="store_true", default=False)
    p.add_argument("--fast", action="store_true",
                   help="raft-max-side 768, raft-iters 12, gpu-nedt")
    p.add_argument("--cache-size", type=int, default=4,
                   help="Сколько пар кадров держать в кэше признаков.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p


def main(argv=None):
    args = pl.apply_fast_preset(_build_argparser().parse_args(argv))
    pl.configure(amp=args.amp, amp_dtype=args.amp_dtype, raft_iters=args.raft_iters,
                 feature_cache=True, gpu_nedt=args.gpu_nedt)
    STATE.init(
        checkpoint=args.checkpoint, model_name=args.model,
        a2s_weights=args.a2s_weights, a2s_repo=args.a2s_repo,
        isnet_weights=args.isnet_weights, dis_repo=args.dis_repo,
        raft_size=args.raft_size, raft_max_side=args.raft_max_side,
        compile_model=args.compile_model, cache_size=args.cache_size,
    )
    launch_kwargs = _style_kwargs() if GRADIO_6 else {}
    build_ui().launch(server_name=args.host, server_port=args.port,
                      share=args.share, debug=args.debug, **launch_kwargs)


if __name__ == "__main__":
    main()
