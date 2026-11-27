#!/usr/bin/env python3
"""探查高度图分布与候选等值线长度（一次性分析脚本）。"""
import sys
import numpy as np

BIN = sys.argv[1]
N = 128
h = np.fromfile(BIN, dtype="<f8").reshape(N, N)
print("shape", h.shape, "min", h.min(), "max", h.max(), "mean", h.mean(), "std", h.std())
qs = np.quantile(h, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
print("quantiles", dict(zip([5, 10, 25, 50, 75, 90, 95], np.round(qs, 4))))


def contour_pixels(level: float) -> np.ndarray:
    """边缘穿越法：网格边两端跨越 level 则标记该边相邻的栅格像素。"""
    above = h >= level
    mask = np.zeros((N, N), dtype=bool)
    # 水平边（沿 x 方向相邻格点）
    cross_x = above[:, :-1] != above[:, 1:]
    mask[:, :-1] |= cross_x
    mask[:, 1:] |= cross_x
    # 垂直边（沿 y 方向相邻格点）
    cross_y = above[:-1, :] != above[1:, :]
    mask[:-1, :] |= cross_y
    mask[1:, :] |= cross_y
    return mask


for lv in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    m = contour_pixels(lv)
    ncomp = None
    print(f"level={lv:4.2f}  contour_pixels={m.sum():4d}")
