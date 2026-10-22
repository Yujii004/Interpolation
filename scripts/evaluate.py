"""
Оценка AnimeIFNet на тестовых сценах ATD-12K (бывший test_animeifnet_v8_2.py).

Каждая сцена прогоняется в четырёх режимах:
    baseline  без подсказок                    — основная метрика
    sketch    скетч эталона (a2s_gt.npy)
    roi       ROI персонажа (маска ISNet)      — вне ROI берётся coarse
    both      скетч + ROI

Отчитываются PSNR/SSIM по кадру — сырые и после постобработки sharpen_lines
(отдельно, это не часть модели), без сцен со сменой кадра, а также PSNR внутри
и снаружи ROI.

Результаты (в --output):
    results.json   сводка + метаданные: чекпоинт, число сцен, пропуски, NaN
    metrics.csv    метрики по каждой сцене
    scenes.txt     список оценённых сцен — для воспроизведения подмножества
    worst_*.png, best_*.png   сравнение режимов на худших/лучших сценах

Примеры:
    # подмножество из 1134 сцен (первые по алфавиту), только baseline
    python scripts/evaluate.py --data /data/ATD-12K/test_2k_original \\
        --checkpoint checkpoints/v8_2/best.pth --output results/v8_2 \\
        --max-scenes 1134 --modes baseline

    # повторить в точности тот же набор сцен
    python scripts/evaluate.py ... --scene-list results/v8_2/scenes.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animeifnet.data import MODEL_INPUT_KEYS, AnimeTripletTestDataset, unpad_tensor  # noqa: E402
from animeifnet.metrics import (calc_psnr, calc_ssim, is_scene_change,  # noqa: E402
                                masked_psnr, seg_roi_mask, sharpen_lines)
from animeifnet.models import AnimeIFNet  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # tqdm необязателен
    def tqdm(x, **_):
        return x

MODES = ("baseline", "sketch", "roi", "both")
NEEDS_SKETCH = {"sketch", "both"}
NEEDS_ROI = {"roi", "both"}


# ═══════════════════════════════════════════════════════════════
# Модель
# ═══════════════════════════════════════════════════════════════

def load_model(path: str, device, residual_scale=0.7, corr_levels=4, corr_radius=5):
    """
    Чекпоинт V8.2 (134 входных канала RefineNet) грузится напрямую.
    Чекпоинт V8 (132) — через from_v8_checkpoint: новые каналы нулевые,
    без подсказок модель ведёт себя как V8.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model", ck)
    kwargs = dict(residual_scale=residual_scale, corr_levels=corr_levels,
                  corr_radius=corr_radius, pretrained=False)
    in_ch = sd["refine.e1.0.weight"].shape[1]
    if in_ch == 134:
        model, kind = AnimeIFNet(**kwargs).to(device), "v8_2"
        model.load_state_dict(sd, strict=True)
    else:
        model, kind = AnimeIFNet.from_v8_checkpoint(path, device=device, **kwargs), "v8"
    model.eval()
    meta = {"file": Path(path).name, "kind": kind, "epoch": ck.get("epoch"),
            "val_psnr": ck.get("val_psnr")}
    print(f"[model] {kind} | epoch={meta['epoch']} | "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    return model, meta


# ═══════════════════════════════════════════════════════════════
# Выбор сцен
# ═══════════════════════════════════════════════════════════════

def select_scenes(args) -> Optional[List[str]]:
    if args.scene_list:
        names = [l.strip() for l in open(args.scene_list, encoding="utf-8") if l.strip()]
        print(f"[scenes] {len(names)} from {args.scene_list}")
        return names
    if not args.max_scenes:
        return None
    available = AnimeTripletTestDataset(args.data, verbose=False).available
    if args.max_scenes >= len(available):
        return None
    if args.select == "random":
        names = sorted(random.Random(args.seed).sample(available, args.max_scenes))
    else:
        names = available[:args.max_scenes]
    print(f"[scenes] {len(names)} of {len(available)} ({args.select}"
          + (f", seed={args.seed}" if args.select == "random" else "") + ")")
    return names


# ═══════════════════════════════════════════════════════════════
# Прогон одной сцены
# ═══════════════════════════════════════════════════════════════

class SceneRunner:
    def __init__(self, model, device, amp: bool, sharp_strength: float, sharp_dilate: int):
        self.model, self.device = model, device
        self.amp = amp and device.type == "cuda"
        self.sharp = (sharp_strength, sharp_dilate)

    def prepare(self, data):
        pi = data["pad_info"].tolist()
        x = {k: data[k][None].to(self.device, non_blocking=True)
             for k in MODEL_INPUT_KEYS + ["it", "sketch"]}
        # ROI строится по кадру без padding, затем дополняется как остальные входы
        roi = seg_roi_mask(unpad_tensor(data["seg0"], pi), unpad_tensor(data["seg1"], pi))
        ph, pw = pi[0], pi[1]
        roi_pad = F.pad(roi[None], (0, pw, 0, ph), mode="replicate") if (ph or pw) else roi[None]
        return pi, x, roi, roi_pad.to(self.device)

    @torch.no_grad()
    def forward(self, x, roi_pad, mode):
        kw = {}
        if mode in NEEDS_SKETCH:
            kw["sketch_hint"] = x["sketch"]
        if mode in NEEDS_ROI:
            kw["roi_mask"] = roi_pad
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            return self.model(*(x[k] for k in MODEL_INPUT_KEYS), t=0.5, **kw)

    def metrics(self, out, x, pi, roi):
        pred = out["pred"][0].float()
        n_bad = int((~torch.isfinite(pred)).sum())
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        pc = unpad_tensor(pred, pi).cpu()
        gt = unpad_tensor(x["it"][0], pi).float().cpu()
        ps = sharpen_lines(pc, *self.sharp)
        m = {"psnr": calc_psnr(pc, gt), "ssim": calc_ssim(pc, gt),
             "psnr_sharp": calc_psnr(ps, gt), "ssim_sharp": calc_ssim(ps, gt),
             "psnr_in": None, "psnr_out": None, "nonfinite_px": n_bad}
        inside = (roi > 0.5).float()
        if inside.mean() < 0.99:  # ROI-метрики осмысленны только для частичной маски
            m["psnr_in"] = masked_psnr(pc, gt, inside)
            m["psnr_out"] = masked_psnr(pc, gt, 1 - inside)
        return m, pc, gt


# ═══════════════════════════════════════════════════════════════
# Сводка
# ═══════════════════════════════════════════════════════════════

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def summarize(rows: List[dict], modes) -> Dict[str, dict]:
    summary = {}
    for mode in modes:
        ok = [r[mode] for r in rows if r.get(mode)]
        no_sc = [r[mode] for r in rows if r.get(mode) and not r["scene_change"]]
        s = {"n_scenes": len(ok)}
        for key in ("psnr", "ssim", "psnr_sharp", "ssim_sharp"):
            s[key] = _mean([m[key] for m in ok])[0]
        s["psnr_no_scene_change"], s["n_no_scene_change"] = _mean([m["psnr"] for m in no_sc])
        s["ssim_no_scene_change"] = _mean([m["ssim"] for m in no_sc])[0]
        s["psnr_in_roi"], s["n_partial_roi"] = _mean([m["psnr_in"] for m in ok])
        s["psnr_out_roi"] = _mean([m["psnr_out"] for m in ok])[0]
        s["scenes_with_nonfinite"] = sum(m["nonfinite_px"] > 0 for m in ok)
        summary[mode] = s
    return summary


def print_summary(summary, meta):
    fmt = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"
    print(f"\n{'═' * 84}")
    print(f"  scenes={meta['n_scenes']} (available {meta['n_available']}, "
          f"skipped w/o features {meta['skipped_no_features']}) | "
          f"scene change={meta['n_scene_change']} | sketch GT={meta['n_with_sketch']}")
    print(f"{'═' * 84}")
    print(f"  {'mode':<9}|{'PSNR':>7} {'SSIM':>7} |{'PSNR sh':>8} {'SSIM sh':>8} |"
          f"{'PSNR noSC':>10} |{'in ROI':>8} {'out ROI':>8} |{'NaN':>4}")
    print("  " + "-" * 82)
    for mode, s in summary.items():
        print(f"  {mode:<9}|{fmt(s['psnr']):>7} {fmt(s['ssim'], 4):>7} |"
              f"{fmt(s['psnr_sharp']):>8} {fmt(s['ssim_sharp'], 4):>8} |"
              f"{fmt(s['psnr_no_scene_change']):>10} |"
              f"{fmt(s['psnr_in_roi']):>8} {fmt(s['psnr_out_roi']):>8} |"
              f"{s['scenes_with_nonfinite']:>4}")
    if "baseline" in summary and "roi" in summary and summary["roi"]["psnr_in_roi"] is not None:
        b, r = summary["baseline"], summary["roi"]
        print(f"\n  ROI vs baseline: внутри {r['psnr_in_roi'] - b['psnr_in_roi']:+.2f} dB, "
              f"снаружи {r['psnr_out_roi'] - b['psnr_out_roi']:+.2f} dB")
    if any(s["scenes_with_nonfinite"] for s in summary.values()):
        print("  ⚠ Есть сцены с NaN/Inf в выходе (заменены на конечные значения) — "
              "попробуй --no-amp.")
    print(f"{'═' * 84}\n")


def write_csv(rows, modes, path: Path):
    keys = ("psnr", "ssim", "psnr_sharp", "ssim_sharp", "psnr_in", "psnr_out", "nonfinite_px")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "scene_change", "has_sketch"]
                   + [f"{m}_{k}" for m in modes for k in keys])
        for r in rows:
            row = [r["scene"], int(r["scene_change"]), int(r["has_sketch"])]
            for m in modes:
                d = r.get(m)
                row += ["" if d is None or d[k] is None else
                        (d[k] if isinstance(d[k], int) else f"{d[k]:.4f}") for k in keys]
            w.writerow(row)


# ═══════════════════════════════════════════════════════════════
# Визуализация
# ═══════════════════════════════════════════════════════════════

def _np_img(t):
    return ((t.float().clamp(-1, 1) + 1) * .5).permute(1, 2, 0).cpu().numpy()


def save_comparison(path, scene, sc, i0, i1, gt, roi, outs, mets, modes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(modes) + 1, 4, figsize=(20, 5 * len(modes) + 5),
                             facecolor="#0d0d14", squeeze=False)
    top = [(_np_img(i0), "frame1"), (_np_img(i1), "frame3"), (_np_img(gt), "GT (frame2)"),
           (roi.squeeze().numpy(), "ROI (seg)")]
    for ax, (img, title) in zip(axes[0], top):
        ax.imshow(img, cmap="cividis" if img.ndim == 2 else None, vmin=0, vmax=1)
        ax.set_title(title, color="#c8c4bb", fontsize=10)
    for row, mode in zip(axes[1:], modes):
        out, m = outs[mode], mets[mode]
        pred = out["pred"]
        err = ((pred - gt).abs().mean(0) * 1.5).clamp(0, 1).numpy()
        res = out["residual"].abs().mean(0)
        res = (res / res.max().clamp_min(1e-6)).numpy()
        roi_txt = (f" | in {m['psnr_in']:.2f} / out {m['psnr_out']:.2f}"
                   if m["psnr_in"] is not None else "")
        panels = [(_np_img(pred), f"{mode}: {m['psnr']:.2f} dB / {m['ssim']:.4f}{roi_txt}"),
                  (err, "error"), (_np_img(out["coarse"]), "coarse"), (res, "|residual|")]
        for ax, (img, title) in zip(row, panels):
            ax.imshow(img, cmap="inferno" if img.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_title(title, color="#c8c4bb", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"AnimeIFNet | {scene}" + (" [SCENE CHANGE]" if sc else ""),
                 color="#ff6b6b" if sc else "#e8e4d9", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def evaluate(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    modes = [m for m in MODES if m in args.modes]

    scenes = select_scenes(args)
    ds = AnimeTripletTestDataset(args.data, crop_size=args.crop,
                                 load_sketch=bool(NEEDS_SKETCH & set(modes)), scenes=scenes)
    if not len(ds):
        sys.exit("No scenes with features found")
    loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")

    model, model_meta = load_model(args.checkpoint, device)
    runner = SceneRunner(model, device, args.amp, args.sharp_strength, args.sharp_dilate)

    rows, t0 = [], time.time()
    for data in tqdm(loader, desc="evaluate", total=len(ds)):
        pi, x, roi, roi_pad = runner.prepare(data)
        row = {"scene": data["scene_name"], "has_sketch": data["has_sketch"],
               "scene_change": is_scene_change(x["i0"], x["i1"], args.sc_mse, args.sc_hist)}
        for mode in modes:
            if mode in NEEDS_SKETCH and not row["has_sketch"]:
                row[mode] = None  # нет a2s_gt.npy — режим для этой сцены не считается
                continue
            row[mode] = runner.metrics(runner.forward(x, roi_pad, mode), x, pi, roi)[0]
        rows.append(row)

    summary = summarize(rows, modes)
    meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": Path(args.data).name, "checkpoint": model_meta,
        "device": (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        "n_scenes": len(rows), "n_available": len(ds.available),
        "skipped_no_features": ds.skipped["base"] + ds.skipped["a2s"],
        "skipped_detail": {"no_raft_isnet": ds.skipped["base"], "no_anime2sketch": ds.skipped["a2s"],
                           "no_frames": ds.skipped["frames"]},
        "selection": ("list:" + Path(args.scene_list).name if args.scene_list else
                      f"{args.select}:{args.max_scenes}" if args.max_scenes else "all"),
        "n_scene_change": sum(r["scene_change"] for r in rows),
        "n_with_sketch": sum(r["has_sketch"] for r in rows),
        "settings": {"crop": args.crop, "amp_bf16": runner.amp,
                     "sharpen": {"strength": args.sharp_strength, "dilate": args.sharp_dilate},
                     "scene_change": {"mse": args.sc_mse, "hist": args.sc_hist}},
        "seconds": round(time.time() - t0, 1),
    }
    print_summary(summary, meta)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary}, f, indent=2, ensure_ascii=False)
    write_csv(rows, modes, out_dir / "metrics.csv")
    (out_dir / "scenes.txt").write_text("\n".join(r["scene"] for r in rows) + "\n",
                                        encoding="utf-8")
    print(f"→ {out_dir / 'results.json'}, metrics.csv, scenes.txt")

    if args.top_k > 0:
        visualize(rows, ds, runner, modes, out_dir, args.top_k)


def visualize(rows, ds, runner, modes, out_dir, top_k):
    """Худшие и лучшие сцены по baseline PSNR — все режимы на одной картинке."""
    def score(i):
        m = next((rows[i][k] for k in modes if rows[i].get(k)), None)
        return m["psnr"] if m else float("inf")
    order = sorted((i for i in range(len(rows)) if score(i) != float("inf")), key=score)
    picks = [("worst", r + 1, i) for r, i in enumerate(order[:top_k])]
    picks += [("best", r + 1, i) for r, i in enumerate(order[::-1][:top_k])]
    for cat, rank, idx in tqdm(picks, desc="visualize"):
        data = ds[idx]
        pi, x, roi, roi_pad = runner.prepare(data)
        outs, mets = {}, {}
        for mode in modes:
            if rows[idx].get(mode) is None:
                continue
            out = runner.forward(x, roi_pad, mode)
            mets[mode], pred, gt = runner.metrics(out, x, pi, roi)
            outs[mode] = {"pred": pred,
                          "coarse": unpad_tensor(out["coarse"][0], pi).float().cpu(),
                          "residual": unpad_tensor(out["residual"][0], pi).float().cpu()}
        shown = [m for m in modes if m in outs]
        save_comparison(out_dir / f"{cat}_{rank:02d}_{mets[shown[0]]['psnr']:.1f}dB.png",
                        data["scene_name"], rows[idx]["scene_change"],
                        unpad_tensor(data["i0"], pi), unpad_tensor(data["i1"], pi),
                        gt, roi, outs, mets, shown)


def build_argparser():
    p = argparse.ArgumentParser(description="Evaluate AnimeIFNet on ATD-12K test scenes",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--data", required=True, help="папка тестовых сцен")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="results/eval")
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)

    g = p.add_argument_group("выбор сцен")
    g.add_argument("--max-scenes", type=int, default=0, help="0 — все сцены")
    g.add_argument("--select", choices=["first", "random"], default="first",
                   help="first — первые по имени, random — случайные с --seed")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--scene-list", default=None,
                   help="файл с именами сцен (например, scenes.txt прошлого запуска)")

    p.add_argument("--crop", type=int, default=0, help="центральный кроп; 0 — полный кадр")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="bf16 autocast на CUDA")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--top-k", type=int, default=5, help="сколько худших/лучших сцен рисовать")
    p.add_argument("--sharp-strength", type=float, default=0.3)
    p.add_argument("--sharp-dilate", type=int, default=3)
    p.add_argument("--sc-mse", type=float, default=0.35, help="порог MSE для смены сцены")
    p.add_argument("--sc-hist", type=float, default=0.5,
                   help="порог расстояния Бхаттачарьи для смены сцены")
    return p


if __name__ == "__main__":
    evaluate(build_argparser().parse_args())
