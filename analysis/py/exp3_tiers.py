"""实验三：三档层数并排——基础频率上限（反解）与实测条纹对比度。

几何：细网格 8 倍超采样（Δ=8），细网格奈奎斯特 0.5，远高于所有倍频层，
其 DFT 是无混叠参考；粗网格每 8 点取一个，奈奎斯特 fN=0.5/Δ。
全部频率以细网格单位表示，像素当量（Δ=1）= 细单位 ×Δ。

两套上限：
  hard cap  : f0_max = fN / L^(K-1)
  阈值反解  : 数值求解 C_alias(f0; K) = C*（各层振幅加权的方形奈奎斯特尾能量）
实测对比度：细网格 2D DFT 方形奈奎斯特区（粗网格带限）之外的功率分数 E2。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import RAW, alias_energy_2d, variance_analytic, read_raw

F0_TEST = 0.025                        # 细网格单位（像素当量 0.2）
L = 2.0
G = 0.5
STEP = 8
DELTA = float(STEP)
FN = 0.5 / DELTA                       # 粗网格奈奎斯特（细单位）=0.0625
KS = (3, 4, 5)
NF = 1024
NC = 128
C_STAR = 0.001                         # 条纹对比度阈值（0.1%）


def fold_frequency(f: float) -> float:
    """折叠进粗网格主奈奎斯特区 [-fN,fN)。"""
    fs = 2 * FN
    return ((f + FN) % fs) - FN


def analytic_contrast(f0: float, K: int) -> float:
    """解析 2D 混叠对比度：各层归一化振幅平方 × 该层方形奈奎斯特尾能量占比。"""
    norm = sum(G ** k for k in range(K))
    total = 0.0
    for k in range(K):
        fk = f0 * L ** k
        if fk > FN:
            w = (G ** k / norm) ** 2
            total += w * alias_energy_2d(fk, DELTA) / variance_analytic(fk)
    return total


def invert_f0(K: int, c_star: float) -> float:
    hard = FN / L ** (K - 1)
    if analytic_contrast(hard, K) <= c_star:
        return hard
    lo, hi = 1e-7, hard
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if analytic_contrast(mid, K) < c_star:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


NSEG = 256


def measured_contrast(K: int) -> float:
    """实测混叠 E2：对细网格全部行做 1D Welch（汉宁窗），得行方向尾能量 E1，
    再由值噪声可分离性 E2 = 2E1 - E1² 推出 2D 方形奈奎斯特尾。

    2D 全场 DFT 的方形边界会被强低频泄漏主导，无法分辨千分之一量级的尾；
    1D Welch 用上千个独立段平均，可在该量级稳定测量（与实验一同口径）。
    """
    z = read_raw(os.path.join(RAW, f"fbm_k{K}.fine.raw"), NF)
    z = z - z.mean(axis=1, keepdims=True)
    win = np.hanning(NSEG)
    wnorm = (win ** 2).sum()
    nseg = NF // NSEG
    segs = z[:, : nseg * NSEG].reshape(NF * nseg, NSEG)
    spec = np.abs(np.fft.fft(segs * win, axis=1)) ** 2 / wnorm
    p = spec.mean(axis=0)
    k_cut = int(round(NSEG * FN))                   # 粗网格 fN（细单位）对应的 bin
    inband = np.r_[0:k_cut + 1, NSEG - k_cut:NSEG]
    e1 = float(1.0 - p[inband].sum() / p.sum())
    return 2 * e1 - e1 ** 2


def main():
    rows = []
    print(f"条纹阈值 C*={C_STAR:.2%}，粗网格奈奎斯特 fN={FN:.4f}（细单位），"
          f"测试 f0={F0_TEST}（像素当量 {F0_TEST*STEP:.2f}）")
    print(f"{'K':>2} {'f_top':>6} {'硬上限':>9} {'反解上限':>9} "
          f"{'解析C':>9} {'实测C':>9} {'超限':>4} {'折叠fa':>7} {'周期px':>7}")
    for K in KS:
        f_top = F0_TEST * L ** (K - 1)
        hard = FN / L ** (K - 1)
        f0_cap = invert_f0(K, C_STAR)
        c_an = analytic_contrast(F0_TEST, K)
        c_meas = measured_contrast(K)
        over = f_top > FN
        f_alias = abs(fold_frequency(f_top)) if over else 0.0
        # 周期用像素当量：粗像素频率 = 细单位 ×Δ
        period = 1 / (f_alias * DELTA) if over and f_alias > 0 else float("inf")
        rows.append(dict(K=K, f_top=f_top, f_top_px=f_top * DELTA,
                         hard_cap=hard, hard_cap_px=hard * DELTA,
                         f0_cap=f0_cap, f0_cap_px=f0_cap * DELTA,
                         c_analytic=c_an, c_measured=c_meas, over=over,
                         alias_freq=f_alias, period_px=period))
        print(f"{K:>2} {f_top:>6.3f} {hard:>9.5f} {f0_cap:>9.5f} "
              f"{c_an:>9.5f} {c_meas:>9.5f} {('是' if over else '否'):>4} "
              f"{f_alias:>7.3f} {period:>7.2f}")

    print("\n像素当量（×Δ=8）基础频率上限：")
    for r in rows:
        print(f"  K={r['K']}: 硬上限 f0≤{r['hard_cap_px']:.4f}/px  "
              f"反解 f0≤{r['f0_cap_px']:.4f}/px  最高层 f_top={r['f_top_px']:.2f}/px")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp3_rows.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
