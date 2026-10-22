"""AnimeIFNet models.

    from animeifnet.models import AnimeIFNet
    model = AnimeIFNet.from_checkpoint("weights/best.pth", device="cuda")
"""

from .animeifnet import AnimeIFNet, IFNet, RefineNet

# Имена из старых файлов — чтобы не ломать скрипты, которые их импортируют.
AnimeIFNetV8_2 = AnimeIFNet
IFNetV7_3 = IFNet
RefineNetV8_2 = RefineNet


def __getattr__(name):
    # Экспериментальная ветка грузится лениво: она не нужна для инференса V8.2.
    if name in ("AnimeIFNetV9Geom", "IFNetSketchGRU"):
        from . import animeifnet_v9_geom
        return getattr(animeifnet_v9_geom, name)
    raise AttributeError(name)


MODEL_REGISTRY = {
    "v8_2": "animeifnet.models.animeifnet:AnimeIFNet",
    "v9_geom": "animeifnet.models.animeifnet_v9_geom:AnimeIFNetV9Geom",
}

__all__ = ["AnimeIFNet", "IFNet", "RefineNet", "AnimeIFNetV8_2", "MODEL_REGISTRY"]
