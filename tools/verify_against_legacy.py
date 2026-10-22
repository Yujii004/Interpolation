"""Сравнение старой цепочки v6→v8.2 (и v9_geom) с новым пакетом.
Запуск: python tools/verify_against_legacy.py <папка, где лежат source/core/model и model/>
Веса старой модели грузятся в новую с strict=True, выходы сравниваются."""
import sys, torch, torchvision.models as tvm
LEGACY = sys.argv[1] if len(sys.argv) > 1 else "legacy"  # папка с source/core/model и model/
sys.path[:0] = [LEGACY, "."]

# офлайн: ImageNet-веса не скачиваем (сравниваем всё равно одни и те же веса)
for fn in ("efficientnet_b0", "resnet50"):
    orig = getattr(tvm, fn)
    setattr(tvm, fn, (lambda o: lambda weights=None, **kw: o(weights=None, **kw))(orig))

from source.core.model.animeIFNet_v8_2 import AnimeIFNetV8_2 as Old
from model.animeIFNet_v9_geom import AnimeIFNetV9Geom as OldV9
from animeifnet.models import AnimeIFNet as New
from animeifnet.models.animeifnet_v9_geom import AnimeIFNetV9Geom as NewV9

torch.manual_seed(0)
B, H, W = 2, 128, 160
x = dict(i0=torch.rand(B,3,H,W)*2-1, i1=torch.rand(B,3,H,W)*2-1,
         edge0=torch.rand(B,1,H,W), edge1=torch.rand(B,1,H,W),
         dt0=torch.rand(B,1,H,W), dt1=torch.rand(B,1,H,W),
         flow_fwd=torch.randn(B,2,H,W)*4, flow_bwd=torch.randn(B,2,H,W)*4,
         seg0=(torch.rand(B,1,H,W)>.5).float(), seg1=(torch.rand(B,1,H,W)>.5).float())
sk = torch.rand(B,1,H,W); roi = (torch.rand(B,1,H,W)>.5).float()
tmap = torch.rand(B,1,H,W)

def compare(old, new, name, **kw):
    old.eval(); new.eval()
    with torch.no_grad():
        a, b = old(**x, **kw), new(**x, **kw)
    worst = max((a[k]-b[k]).abs().max().item() for k in a if torch.is_tensor(a[k]))
    print(f"{name:28s} keys_equal={set(a)==set(b)}  max|Δ|={worst:.2e}")
    assert worst < 1e-5, name

old = Old()
new = New(pretrained=False)
assert list(old.state_dict()) == list(new.state_dict()) or set(old.state_dict()) == set(new.state_dict())
missing = new.load_state_dict(old.state_dict(), strict=True)
print("v8_2 state_dict: strict load OK,", len(old.state_dict()), "tensors")
print("  same key order:", list(old.state_dict()) == list(new.state_dict()))

compare(old, new, "v8_2 baseline t=0.5")
compare(old, new, "v8_2 sketch+roi t=0.3", t=0.3, sketch_hint=sk, roi_mask=roi)
compare(old, new, "v8_2 t-map", t=tmap)

with torch.no_grad():
    ra = old.generate_multi_t(**x, t_values=(0.25, 0.75))
    rb = new.generate_multi_t(**x, t_values=(0.25, 0.75))
print(f"{'v8_2 generate_multi_t':28s} max|Δ|={max((p['pred']-q['pred']).abs().max().item() for p,q in zip(ra,rb)):.2e}")

ov9 = OldV9(); nv9 = NewV9()
nv9.load_state_dict(ov9.state_dict(), strict=True)
print("v9_geom state_dict: strict load OK,", len(ov9.state_dict()), "tensors")
compare(ov9, nv9, "v9_geom sketch+roi", sketch_hint=sk, roi_mask=roi)
print("ALL CHECKS PASSED")
