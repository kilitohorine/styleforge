"""Unit prices in CNY. Look is free; unknown paid models use a conservative default."""

from __future__ import annotations

LOOK_COST_CNY = 0.0
# Conservative: Kolors-class ~0.1 CNY/image. Overestimate to trip the fuse early.
SILICONFLOW_T2I_CNY = 0.12
WANXIANG_T2I_CNY = 0.20
UNKNOWN_PAID_CNY = 0.5


def estimate_cny(modality: str, backend: str = "") -> float:
    if modality == "image.photo_look":
        return LOOK_COST_CNY
    if modality == "image.2d":
        if backend == "mock":
            return 0.0
        if backend in {"dashscope", "dashscope_wanxiang", "wanxiang"}:
            return WANXIANG_T2I_CNY
        return SILICONFLOW_T2I_CNY
    return UNKNOWN_PAID_CNY
