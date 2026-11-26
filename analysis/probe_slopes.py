"""探查高度图坡度分布，为仰角选择提供依据（渲染读取的是量化后的 PGM）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgmio import read_pgm


def gradients(h):
    """与 Go shade.Render 完全一致的中心差分梯度。"""
    gx = np.zeros_like(h)
    gy = np.zeros_like(h)
    gx[:, 1:-1] = h[:, 2:] - h[:, :-2]
    gx[:, 0] = h[:, 1] - h[:, 0]
    gx[:, -1] = h[:, -1] - h[:, -2]
    gy[1:-1, :] = h[2:, :] - h[:-2, :]
    gy[0, :] = h[1, :] - h[0, :]
    gy[-1, :] = h[-1, :] - h[-2, :]
    return gx, gy


def main():
    for name in ("fbm_s7", "b1_sinusoid", "b2_meander"):
        size, h = read_pgm(f"../artifacts/heightmaps/{name}.pgm")
        gx, gy = gradients(h)
        m = np.hypot(gx, gy)
        flat = m < 1e-9
        alpha = np.where(flat, 0.0, np.arctan2(gx, gy)) % (2 * np.pi)
        print(f"== {name} (size={size}) ==")
        qs = [0, 50, 90, 95, 99, 100]
        print("  slope m quantiles:", {q: round(float(np.percentile(m, q)), 4) for q in qs})
        print(f"  flat pixels: {flat.sum()} ({100*flat.mean():.1f}%)")
        for eps_deg in (10, 12, 15, 20, 30):
            eps = np.deg2rad(eps_deg)
            # 单灯在朝向 phi 时被钳制：n·l<0 -> m cos(alpha-phi) > tan eps
            # 取与梯度反向（最坏情况）即 m>tan eps 时存在某个朝向会被钳制
            frac = float((m > np.tan(eps)).mean())
            print(f"  eps={eps_deg:>2d}°  tan(eps)={np.tan(eps):.4f}  可被某朝向钳制的像素占比={100*frac:.1f}%")
        print(f"  alpha 范围: {np.rad2deg(alpha.min()):.1f} .. {np.rad2deg(alpha.max()):.1f} deg")


if __name__ == "__main__":
    main()
