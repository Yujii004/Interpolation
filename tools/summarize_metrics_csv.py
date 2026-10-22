"""
Сводка по CSV с метриками по сценам (формат старого test-скрипта и scripts/evaluate.py).

    python tools/summarize_metrics_csv.py results/v8_2_disney1131/metrics.csv

Пишет рядом summary.json (среднее, погрешность, разница режимов с baseline
по тем же сценам) и psnr_hist.png (распределение PSNR по сценам).
"""

import csv
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

MODES = ("baseline", "sketch", "roi", "both")


def _f(v):
    return float(v) if v not in ("", None) else None


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("scene") and r["scene"] != "MEAN"]


def mean_se(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    se = st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return round(st.mean(vals), 4), round(se, 4), len(vals)


def summarize(rows):
    out = {"n_scenes": len(rows),
           "sources": dict(Counter(r["scene"].split("_")[0] for r in rows)),
           "modes": {}}
    for m in MODES:
        if f"{m}_psnr" not in rows[0]:
            continue
        s = {}
        for key in ("psnr", "ssim", "psnr_sharp", "psnr_in", "psnr_out"):
            mean, se, n = mean_se([_f(r[f"{m}_{key}"]) for r in rows])
            s[key] = {"mean": mean, "se": se, "n": n}
        if m != "baseline":  # парная разница с baseline на тех же сценах
            for key in ("psnr", "ssim", "psnr_in", "psnr_out"):
                d = [_f(r[f"{m}_{key}"]) - _f(r[f"baseline_{key}"]) for r in rows
                     if _f(r[f"{m}_{key}"]) is not None and _f(r[f"baseline_{key}"]) is not None]
                mean, se, n = mean_se(d)
                s[f"delta_{key}"] = {"mean": mean, "se": se, "n": n}
        out["modes"][m] = s
    return out


def plot_hist(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = [_f(r["baseline_psnr"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    ax.hist(v, bins=40, color="#e8734a", edgecolor="white", linewidth=0.4)
    ax.axvline(st.mean(v), color="#1f2937", linewidth=1.2)
    ax.text(st.mean(v) + 0.3, ax.get_ylim()[1] * 0.9, f"mean {st.mean(v):.2f} dB",
            color="#1f2937", fontsize=9)
    ax.set_xlabel("PSNR, dB (per scene)")
    ax.set_ylabel("scenes")
    ax.set_title(f"AnimeIFNet, {len(v)} test scenes", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(csv_path):
    csv_path = Path(csv_path)
    rows = load_rows(csv_path)
    summary = summarize(rows)
    (csv_path.parent / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_hist(rows, csv_path.parent / "psnr_hist.png")
    b = summary["modes"]["baseline"]
    print(f"{summary['n_scenes']} scenes {summary['sources']} | "
          f"PSNR {b['psnr']['mean']:.2f} ± {b['psnr']['se']:.2f} | SSIM {b['ssim']['mean']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
