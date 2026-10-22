"""Тесты датасета, аугментаций и обучения на синтетических сценах.

Запуск:  python -m pytest tests/ -q
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from animeifnet.data import (MODEL_INPUT_KEYS, AnimeTripletDataset,
                             AnimeTripletTestDataset, split_indices, unpad_tensor)
from animeifnet.data.augment import flip_flow_h, flip_flow_v, rotate_flow, rotate_spatial

H, W = 70, 90  # нечётный размер — проверяем padding


def _write_scene(d, rng, with_sketch=True):
    d.mkdir(parents=True)
    for n in (1, 2, 3):
        Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(d / f"frame{n}.png")
    arrays = {"raft_fwd": rng.standard_normal((2, H, W)) * 3,
              "raft_bwd": rng.standard_normal((2, H, W)) * 3}
    for name in ("a2s_edge0", "a2s_edge1", "a2s_dt0", "a2s_dt1"):
        arrays[name] = rng.random((1, H, W))
    for name, x0 in (("seg0", 10), ("seg1", 16)):  # «персонаж» — прямоугольник
        seg = np.zeros((1, H, W))
        seg[:, 15:45, x0:x0 + 30] = 1.0
        arrays[name] = seg
    if with_sketch:
        arrays["a2s_gt"] = rng.random((1, H, W))
    for name, a in arrays.items():
        np.save(d / f"{name}.npy", a.astype(np.float16))


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("atd")
    rng = np.random.default_rng(0)
    for i in range(10):
        _write_scene(root / f"scene_{i:02d}", rng, with_sketch=i % 2 == 0)
    (root / "broken").mkdir()  # без кадров — должна пропускаться
    return root


# ── Геометрия потока ─────────────────────────────────────────

def _shifted_pair(dx, dy, size=32):
    """i1 — это i0, сдвинутый на (dx, dy); поток 0→1 постоянен и равен (dx, dy)."""
    i0 = torch.rand(1, size, size)
    i1 = torch.roll(i0, shifts=(dy, dx), dims=(-2, -1))
    flow = torch.zeros(2, size, size)
    flow[0], flow[1] = dx, dy
    return i0, i1, flow


def _flow_explains_motion(i0, i1, flow):
    """i1[p + flow(p)] == i0[p] для всех p (поток целочисленный, сдвиг циклический)."""
    size = i0.shape[-1]
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    ty = (ys + flow[1].round().long()) % size
    tx = (xs + flow[0].round().long()) % size
    return torch.equal(i1[0, ty, tx], i0[0])


@pytest.mark.parametrize("op", ["flip_h", "flip_v", 90, 180, 270])
def test_flow_augmentation_matches_image_augmentation(op):
    i0, i1, flow = _shifted_pair(dx=3, dy=-2)
    assert _flow_explains_motion(i0, i1, flow)
    if op == "flip_h":
        i0, i1, flow = i0.flip(-1), i1.flip(-1), flip_flow_h(flow)
    elif op == "flip_v":
        i0, i1, flow = i0.flip(-2), i1.flip(-2), flip_flow_v(flow)
    else:
        i0, i1, flow = rotate_spatial(i0, op), rotate_spatial(i1, op), rotate_flow(flow, op)
    assert _flow_explains_motion(i0, i1, flow)


def test_rotation_matches_pil():
    img = np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    t = torch.from_numpy(img).permute(2, 0, 1)
    for angle in (90, 180, 270):
        pil = np.array(Image.fromarray(img).rotate(angle))
        assert torch.equal(rotate_spatial(t, angle), torch.from_numpy(pil).permute(2, 0, 1))


# ── Датасет ──────────────────────────────────────────────────

def test_train_sample_shapes_and_ranges(data_root):
    ds = AnimeTripletDataset(str(data_root), crop_size=64, augment=True, verbose=False)
    assert len(ds) == 10
    s = ds[0]
    assert set(s) == set(MODEL_INPUT_KEYS) | {"it", "sketch", "roi"}
    for k in ("i0", "i1", "it"):
        assert s[k].shape == (3, 64, 64) and s[k].min() >= -1 and s[k].max() <= 1
    for k in ("edge0", "dt1", "seg0", "sketch", "roi"):
        assert s[k].shape == (1, 64, 64) and 0 <= s[k].min() and s[k].max() <= 1
    assert s["flow_fwd"].shape == (2, 64, 64)


def test_upscale_when_frame_smaller_than_crop(data_root):
    ds = AnimeTripletDataset(str(data_root), crop_size=96, augment=False, verbose=False)
    s = ds[0]
    assert s["i0"].shape == (3, 96, 96)


def test_validation_is_deterministic(data_root):
    ds = AnimeTripletDataset(str(data_root), crop_size=48, augment=False, verbose=False)
    a, b = ds[3], ds[3]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert torch.equal(a["roi"], torch.ones_like(a["roi"]))


def test_split_is_disjoint():
    train_idx, val_idx = split_indices(100, 0.1, seed=42)
    assert len(val_idx) == 10 and len(train_idx) == 90
    assert not set(train_idx) & set(val_idx)


def test_test_dataset_pads_and_unpads(data_root):
    ds = AnimeTripletTestDataset(str(data_root), verbose=False)
    s = ds[0]
    assert s["i0"].shape[-2:] == (96, 96)  # 70×90 → кратно 32
    assert unpad_tensor(s["i0"], s["pad_info"]).shape[-2:] == (H, W)
    assert s["sketch"].abs().sum() == 0


# ── Обучение ─────────────────────────────────────────────────

def test_curriculum_phases():
    from animeifnet.training import HintCurriculum
    c = HintCurriculum(2, 4, 6)
    sketch, roi = torch.ones(64, 1, 4, 4), torch.zeros(64, 1, 4, 4)
    sk, r = c.apply(sketch, roi, epoch=1)
    assert sk.sum() == 0 and torch.equal(r, torch.ones_like(r))       # A
    sk, r = c.apply(sketch, roi, epoch=3)
    assert 0 < sk.sum() < sketch.sum() and r.min() == 1                 # B
    _, r = c.apply(sketch, roi, epoch=5)
    assert torch.equal(r, roi)                                          # C


def test_training_smoke_and_resume(data_root, tmp_path):
    from animeifnet.training import TrainConfig, train
    from animeifnet.losses import LossConfig
    import torchvision.models as tvm

    # без скачивания ImageNet-весов
    orig = {n: getattr(tvm, n) for n in ("efficientnet_b0", "resnet50")}
    for n, fn in orig.items():
        setattr(tvm, n, lambda weights=None, _fn=fn, **kw: _fn(weights=None, **kw))
    try:
        cfg = TrainConfig(data=str(data_root), save_dir=str(tmp_path), epochs=3,
                          batch=2, crop=64, workers=0, device="cpu", val_fraction=0.2,
                          phase_a_end=1, phase_b_end=2, phase_c_end=3,
                          limit_train_batches=2, limit_val_batches=1, save_every=0)
        # ROI-лоссы и все остальные — с первых эпох, чтобы их проверить
        loss_cfg = LossConfig(coarse_start=1, ssim_start=1, warp_start=1, lap_start=1,
                              percep_start=99, w_percep=0.0, roi_start=1)
        train(cfg, loss_cfg)
        log = [json.loads(l) for l in open(tmp_path / "train_log.jsonl")]
        assert [r["phase"] for r in log] == ["A", "B", "C"]
        assert all(np.isfinite(r["val_psnr"]) for r in log)

        cfg.epochs, cfg.resume = 4, str(tmp_path / "last.pth")
        train(cfg, loss_cfg)
        log = [json.loads(l) for l in open(tmp_path / "train_log.jsonl")]
        assert log[-1]["epoch"] == 4 and log[-1]["phase"] == "D"
    finally:
        for n, fn in orig.items():
            setattr(tvm, n, fn)


# ── Оценка ───────────────────────────────────────────────────

def test_evaluate_subset_is_reproducible(data_root, tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate as ev
    from animeifnet.models import AnimeIFNet

    torch.manual_seed(0)
    ckpt = tmp_path / "model.pth"
    torch.save({"model": AnimeIFNet(pretrained=False).state_dict(), "epoch": 7}, ckpt)

    def run(out, *extra):
        args = ev.build_argparser().parse_args(
            ["--data", str(data_root), "--checkpoint", str(ckpt), "--output", str(out),
             "--device", "cpu", "--workers", "0", "--top-k", "1", *extra])
        ev.evaluate(args)
        return json.load(open(out / "results.json"))

    r1 = run(tmp_path / "a", "--max-scenes", "4", "--select", "random", "--seed", "3")
    assert r1["meta"]["n_scenes"] == 4 and r1["meta"]["n_available"] == 10
    assert r1["meta"]["checkpoint"]["epoch"] == 7
    s = r1["summary"]
    assert s["baseline"]["n_scenes"] == 4
    assert s["sketch"]["n_scenes"] == sum(  # скетч есть только у чётных сцен
        int(n.split("_")[1]) % 2 == 0 for n in (tmp_path / "a/scenes.txt").read_text().split())
    assert s["roi"]["psnr_in_roi"] is not None
    assert len(list((tmp_path / "a").glob("worst_*.png"))) == 1

    r2 = run(tmp_path / "b", "--scene-list", str(tmp_path / "a/scenes.txt"), "--modes", "baseline")
    assert (tmp_path / "a/scenes.txt").read_text() == (tmp_path / "b/scenes.txt").read_text()
    assert r2["summary"]["baseline"]["psnr"] == pytest.approx(s["baseline"]["psnr"])
    assert list(r2["summary"]) == ["baseline"]
