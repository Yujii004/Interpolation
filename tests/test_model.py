"""Smoke-тесты AnimeIFNet: формы выходов и четыре режима подсказок.

Запуск:  python -m pytest tests/ -q
"""

import torch

from animeifnet.models import AnimeIFNet


def _inputs(B=1, H=128, W=128):
    return dict(
        i0=torch.rand(B, 3, H, W) * 2 - 1, i1=torch.rand(B, 3, H, W) * 2 - 1,
        edge0=torch.rand(B, 1, H, W), edge1=torch.rand(B, 1, H, W),
        dt0=torch.rand(B, 1, H, W), dt1=torch.rand(B, 1, H, W),
        flow_fwd=torch.randn(B, 2, H, W), flow_bwd=torch.randn(B, 2, H, W),
        seg0=torch.rand(B, 1, H, W), seg1=torch.rand(B, 1, H, W),
    )


def _model():
    torch.manual_seed(0)
    return AnimeIFNet(pretrained=False).eval()


@torch.no_grad()
def test_all_hint_modes():
    model, x = _model(), _inputs()
    B, _, H, W = x["i0"].shape
    sketch = torch.rand(B, 1, H, W)
    roi = (torch.rand(B, 1, H, W) > 0.5).float()

    for kw in ({}, {"sketch_hint": sketch}, {"roi_mask": roi},
               {"sketch_hint": sketch, "roi_mask": roi}):
        out = model(**x, t=0.5, **kw)
        assert out["pred"].shape == (B, 3, H, W)
        assert out["pred"].abs().max() <= 1.0

    # вне ROI модель не должна трогать coarse
    out = model(**x, roi_mask=roi)
    outside = (out["pred"] - out["coarse"]) * (1 - roi)
    assert outside.abs().max() < 1e-6


@torch.no_grad()
def test_multi_t_matches_single_forward():
    model, x = _model(), _inputs()
    multi = model.generate_multi_t(**x, t_values=(0.25, 0.75))
    for res in multi:
        single = model(**x, t=res["t"])
        assert torch.allclose(res["pred"], single["pred"], atol=1e-5)


@torch.no_grad()
def test_per_pixel_t_map():
    model, x = _model(), _inputs()
    t_map = torch.rand(1, 1, 128, 128)
    assert model(**x, t=t_map)["pred"].shape == (1, 3, 128, 128)
