"""合成解析场自检：梯度加权直方图峰值能否回收已知脊角 atan(r)。

H(x,y)=sin(3(x+r·y_north)) 的梯度恒正比于 (1, r)，梯度角解析为 atan(r)。
用与 analyze.py 完全相同的梯度/直方图/峰值流程估计，验证估计器与公式同口径。
运行：.venv/bin/python -I synthetic_check.py
"""
import math

import numpy as np

from analyze import gradient_field, weighted_histogram, parabolic_peak

SIZE = 600
NBINS = 180


def main():
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    x = cols * 0.05
    y_north = (SIZE - 1 - rows) * 0.05  # 原生行序：第 0 行为北边缘
    print(f"{'r':>4} {'直方图峰值':>10} {'解析 atan(r)':>12} {'偏差':>8}")
    worst = 0.0
    for r in (0.5, 1.0, 2.0):
        h = np.sin(3.0 * (x + r * y_north))
        gx, gy, mag = gradient_field(h)
        centers, counts = weighted_histogram(gx, gy, mag)
        measured = parabolic_peak(centers, counts)
        truth = math.degrees(math.atan(r))
        diff = abs(measured - truth)
        diff = min(diff, 180.0 - diff)
        worst = max(worst, diff)
        print(f"{r:>4} {measured:>10.2f} {truth:>12.2f} {diff:>8.2f}")
    assert worst <= 3.0, f"合成场自检超差: {worst}"
    print(f"\n最大偏差 {worst:.2f}° ≤ 3°（1° 桶 + 离散化），估计器可正确回收 atan(r)。")


if __name__ == "__main__":
    main()
