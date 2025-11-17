from __future__ import annotations

import torch


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))  # type: ignore
