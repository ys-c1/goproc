"""实验一（1D 长序列）：证明混叠能量与频率超出量 ξ=f_c-fN 成正比，并核对 Δ 缩放。

为什么用 1D 长序列：值噪声可分离，2D 场的 1D 切片就是合法 1D 值噪声；
65536 点 × 64 条独立序列给出数百个噪声格点周期，Welch 周期图统计充分。
2D 混叠能量由可分离性 E2 = 2E1 - E1² 推出（解析侧直接积分 2D 核）。

度量：
  细格距 1、粗格距 Δ；细网格单位 ffine=f_c/Δ，粗网格奈奎斯特 fN=1/(2Δ)。
  细网格 DFT 无混叠，其在粗网格奈奎斯特对应频率（bin = nf/(2Δ)）之外的
  功率分数，即粗采样后折叠进带内的混叠能量 E1。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import (RAW, variance_analytic, alias_energy_1d, alias_energy_2d)

NF = 65536
NLINES = 64
QS = (0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
STEPS = (2, 4, 8)
NPERSEG = 4096


def welch_tail(tag: str, step: int) -> float:
    """细网格 1D 序列的 Welch 周期图，在粗网格奈奎斯特之外的功率分数。"""
    z = np.fromfile(os.path.join(RAW, f"{tag}.fine.raw"),
                    dtype="<f8").reshape(NLINES, NF)
    z = z - z.mean(axis=1, keepdims=True)
    # 简单 Welch：每条序列切成 NPERSEG 段，汉宁窗抑制泄漏
    nseg = NF // NPERSEG
    win = np.hanning(NPERSEG)
    wnorm = (win ** 2).sum()
    segs = z[:, : nseg * NPERSEG].reshape(NLINES * nseg, NPERSEG)
    spec = np.abs(np.fft.fft(segs * win, axis=1)) ** 2 / wnorm
    p = spec.mean(axis=0)
    k_cut = NPERSEG // (2 * step * step)   # 粗网格 fN 换算到细网格单位 1/(2Δ²) 的 bin
    inband = np.r_[0:k_cut + 1, NPERSEG - k_cut:NPERSEG]
    return float(1.0 - p[inband].sum() / p.sum())


def main():
    rows = []
    for step in STEPS:
        delta = float(step)
        fN = 1.0 / (2 * delta)
        for q in QS:
            fc = q * fN
            tag = f"l1_d{step}_q{q:g}"
            xi = max(fc - fN, 0.0)
            e1_meas = welch_tail(tag, step)
            e1_an = alias_energy_1d(fc, delta) / variance_analytic(fc) ** 0.5
            e2_an = alias_energy_2d(fc, delta) / variance_analytic(fc)
            # 可分离性：2D 混叠分数 = 1-(1-E1)^2
            e2_meas = 1 - (1 - e1_meas) ** 2
            rows.append(dict(step=step, delta=delta, q=q, fc=fc, fN=fN, xi=xi,
                             e1_meas=e1_meas, e1_an=e1_an,
                             e2_meas=e2_meas, e2_an=e2_an))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp1_rows.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)

    print(f"{'Δ':>3} {'q':>5} {'f_c':>9} {'ξ':>9} "
          f"{'E1实测':>9} {'E1解析':>9} {'E2实测':>9} {'E2解析':>9}")
    for r in rows:
        print(f"{r['delta']:>3.0f} {r['q']:>5.2f} {r['fc']:>9.5f} {r['xi']:>9.5f} "
              f"{r['e1_meas']:>9.4f} {r['e1_an']:>9.4f} "
              f"{r['e2_meas']:>9.4f} {r['e2_an']:>9.4f}")

    print("\n近阈值线性回归 E1 ≈ a·ξ （1<q≤1.5，ξ 单位：周期/粗像素）:")
    for step in STEPS:
        sel = [r for r in rows if r["step"] == step and 1.0 < r["q"] <= 1.5]
        x = np.array([r["xi"] for r in sel])
        for label, key in (("实测", "e1_meas"), ("解析", "e1_an")):
            y = np.array([r[key] for r in sel])
            a = np.dot(x, y) / np.dot(x, x)
            r2 = 1 - np.sum((y - a * x) ** 2) / np.sum((y - y.mean()) ** 2)
            print(f"  Δ={step:>2} {label}: a={a:9.3f}  R²={r2:.5f}")

    print("\nΔ 缩放核对（等 q 下 E1 只取决于 q=f_c/fN，与 Δ 无关）:")
    for q in (1.1, 1.25, 1.5, 2.0):
        vm = [next(r for r in rows if r["step"] == d and r["q"] == q)["e1_meas"]
              for d in STEPS]
        va = [next(r for r in rows if r["step"] == d and r["q"] == q)["e1_an"]
              for d in STEPS]
        print(f"  q={q:>4} 实测: " + " ".join(f"{v:.4f}" for v in vm)
              + f"  极差/均值={(max(vm)-min(vm))/np.mean(vm):.3%}"
              + "  | 解析: " + " ".join(f"{v:.4f}" for v in va))


if __name__ == "__main__":
    main()
