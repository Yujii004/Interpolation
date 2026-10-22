"""
Обучение AnimeIFNet.

Рекомендуемый запуск — дообучение из чекпоинта V8:

    python scripts/train.py --data /data/ATD-12K/train \\
        --save-dir checkpoints/v8_2 --init-v8 checkpoints/v8/best.pth \\
        --epochs 40 --lr 5e-5

Продолжить прерванное обучение:

    python scripts/train.py --data /data/ATD-12K/train \\
        --save-dir checkpoints/v8_2 --resume checkpoints/v8_2/last.pth

Быстрая проверка, что всё запускается (пара батчей на эпоху):

    python scripts/train.py --data ... --epochs 2 --limit-train-batches 2 \\
        --limit-val-batches 1 --workers 0
"""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animeifnet.losses import LossConfig  # noqa: E402
from animeifnet.training import TrainConfig, train  # noqa: E402


def _add_dataclass_args(parser, cls, help_prefix=""):
    """Флаг --kebab-case для каждого поля dataclass, с его значением по умолчанию."""
    for f in fields(cls):
        flag = "--" + f.name.replace("_", "-")
        default = f.default
        if isinstance(default, bool):
            parser.add_argument(flag, dest=f.name, default=default,
                                action=argparse.BooleanOptionalAction)
        else:
            typ = type(default) if default is not None else str
            parser.add_argument(flag, dest=f.name, type=typ, default=default,
                                help=f"{help_prefix}(default: {default})")


def main(argv=None):
    p = argparse.ArgumentParser(description="Train AnimeIFNet",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    _add_dataclass_args(p, TrainConfig)
    loss_group = p.add_argument_group("loss weights and schedule")
    _add_dataclass_args(loss_group, LossConfig)
    args = vars(p.parse_args(argv))

    if not args["data"]:
        p.error("--data is required")
    train_cfg = TrainConfig(**{f.name: args[f.name] for f in fields(TrainConfig)})
    loss_cfg = LossConfig(**{f.name: args[f.name] for f in fields(LossConfig)})
    train(train_cfg, loss_cfg)


if __name__ == "__main__":
    main()
