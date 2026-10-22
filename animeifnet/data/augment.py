"""
Геометрические аугментации, согласованные для изображений и оптического потока.

Поток хранится как [2, H, W] = (dx, dy) в пикселях. При отражении/повороте
кадра нужно не только переставить пиксели поля, но и повернуть сами векторы.
"""

from __future__ import annotations

import torch

_ROT_K = {90: 1, 180: 2, 270: 3}


def flip_flow_h(flow: torch.Tensor) -> torch.Tensor:
    """Горизонтальное отражение: зеркалим поле и меняем знак dx."""
    flow = flow.flip(-1).clone()
    flow[0] = -flow[0]
    return flow


def flip_flow_v(flow: torch.Tensor) -> torch.Tensor:
    """Вертикальное отражение: зеркалим поле и меняем знак dy."""
    flow = flow.flip(-2).clone()
    flow[1] = -flow[1]
    return flow


def rotate_spatial(t: torch.Tensor, angle: int) -> torch.Tensor:
    """Поворот [C, H, W] на 90/180/270° против часовой стрелки (как PIL.rotate)."""
    return torch.rot90(t, k=_ROT_K[angle], dims=[-2, -1])


def rotate_flow(flow: torch.Tensor, angle: int) -> torch.Tensor:
    """
    Поворот поля потока на 90/180/270° против часовой стрелки.
    Пространственно — как rotate_spatial, векторы (dx, dy) поворачиваются вместе
    с кадром (ось y направлена вниз).
    """
    dx, dy = rotate_spatial(flow, angle)
    if angle == 90:
        return torch.stack([dy, -dx])
    if angle == 180:
        return torch.stack([-dx, -dy])
    if angle == 270:
        return torch.stack([-dy, dx])
    raise ValueError(f"Unsupported angle: {angle}")
