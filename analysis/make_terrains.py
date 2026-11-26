"""生成两块解析地形（B1 平行正弦山脊、B2 弯曲山脊），归一化后写 PGM。

不使用噪声库，纯解析式，便于与公式逐项对照。坐标系 x 向右、y 向下。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgmio import write_pgm

SIZE = 256


def normalize(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo)


def b1_parallel_ridges(size=SIZE):
    """沿传播方位 theta_k=55° 的一维正弦，k 个周期；等值线（山脊走向）与传播方向垂直。"""
    theta = np.deg2rad(55.0)
    periods = 10.0
    x = np.arange(size, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)
    phase = 2 * np.pi * periods * (xx * np.cos(theta) + yy * np.sin(theta)) / size
    return normalize(np.sin(phase))


def b2_meandering_ridge(size=SIZE):
    """中心线 x = cx + A sin(2π y / λ) 的弯曲山脊，横截面为高斯脊。"""
    cx = size / 2.0
    amp = size * 0.18          # 中心线摆幅
    wavelength = size / 3.0    # 三个弯曲
    sigma = size * 0.045       # 脊宽
    ridge_h = 1.0
    x = np.arange(size, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)
    center = cx + amp * np.sin(2 * np.pi * yy / wavelength)
    d = xx - center
    h = ridge_h * np.exp(-(d * d) / (2 * sigma * sigma))
    return normalize(h)


def main():
    write_pgm("../artifacts/heightmaps/b1_sinusoid.pgm", b1_parallel_ridges())
    write_pgm("../artifacts/heightmaps/b2_meander.pgm", b2_meandering_ridge())
    print("wrote b1_sinusoid.pgm, b2_meander.pgm")


if __name__ == "__main__":
    main()
