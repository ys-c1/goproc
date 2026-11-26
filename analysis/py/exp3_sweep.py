"""实验三补充：扫描基础频率 f0，验证实测对比度曲线与解析预测、阈值反解一致。

对每个 K 生成若干 f0 的 fBm 细网格场，用同一 Welch 流程测 E2，
叠加解析 C(f0) 曲线，标出 C*=0.1% 的解析反解点与实测穿越点。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import alias_energy_2d, variance_analytic

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L, G, STEP, NF, NSEG = 2.0, 0.5, 8, 1024, 256
FN = 0.5 / STEP
C_STAR = 0.001
win = np.hanning(NSEG); wnorm = (win ** 2).sum()
k_cut = 16
INBAND = np.r_[0:k_cut + 1, NSEG - k_cut:NSEG]


def analytic_c(f0, K):
    norm = sum(G ** k for k in range(K))
    t = 0.0
    for k in range(K):
        fk = f0 * L ** k
        if fk > FN:
            t += (G ** k / norm) ** 2 * alias_energy_2d(fk, STEP) / variance_analytic(fk)
    return t


def measure(f0, K):
    fine = "/tmp/scan.fine.raw"
    subprocess.run(["go", "run", "./analysis/cmd/aliasdump", "--mode", "fbm",
                    "--seed", "7", "--nf", str(NF), "--nc", "128",
                    "--step", str(STEP), "--ffine", str(f0),
                    "--octaves", str(K), "--fine", fine, "--coarse", "/tmp/scan.c.raw"],
                   cwd=REPO, check=True, capture_output=True)
    z = np.fromfile(fine, dtype="<f8").reshape(NF, NF)
    z = z - z.mean(axis=1, keepdims=True)
    nseg = NF // NSEG
    segs = z[:, : nseg * NSEG].reshape(NF * nseg, NSEG)
    p = (np.abs(np.fft.fft(segs * win, axis=1)) ** 2 / wnorm).mean(axis=0)
    e1 = 1 - p[INBAND].sum() / p.sum()
    return 2 * e1 - e1 ** 2


def main():
    f0s = [0.005, 0.008, 0.012, 0.015625, 0.02, 0.025, 0.03, 0.04]
    out = {}
    for K in (3, 4, 5):
        hard = FN / L ** (K - 1)
        print(f"\nK={K}  硬上限 f0={hard:.5f}（细单位，像素当量 {hard*STEP:.4f}）")
        print(f"{'f0':>9} {'f0/px':>8} {'解析C':>9} {'实测E2':>9}")
        rec = []
        for f0 in f0s:
            cm = analytic_c(f0, K)
            me = measure(f0, K)
            rec.append((f0, cm, me))
            print(f"{f0:>9.5f} {f0*STEP:>8.4f} {cm:>9.5f} {me:>9.5f}")
        out[K] = rec
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "exp3_sweep.json"), "w") as fh:
        json.dump({str(k): v for k, v in out.items()}, fh, indent=1)


if __name__ == "__main__":
    main()
