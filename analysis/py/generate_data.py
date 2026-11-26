"""生成全部实验数据：调用 Go 导出器（真实 noise 实现），产物落在 analysis/raw/。

空间约定：细格距 1，粗格距 Δ。粗网格单位下噪声基频 f_c = q·fN = q/(2Δ)，
对应细网格单位 ffine = f_c/Δ = q/(2Δ²)。细网格奈奎斯特 0.5 远高于 f_c/Δ，
保证细网格 DFT 是连续场谱的无混叠测量。
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
DUMPER = "./analysis/cmd/aliasdump"


def run(mode: str, tag: str, ffine: float, step: int, octaves: int = 1,
        nf: int = 1024, nc: int = 128, seed: int = 7):
    os.makedirs(RAW, exist_ok=True)
    fine = os.path.join(RAW, f"{tag}.fine.raw")
    coarse = os.path.join(RAW, f"{tag}.coarse.raw")
    cmd = ["go", "run", DUMPER,
           "--mode", mode, "--seed", str(seed),
           "--nf", str(nf), "--nc", str(nc), "--step", str(step),
           "--ffine", str(ffine), "--fine", fine, "--coarse", coarse,
           "--octaves", str(octaves)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=os.path.dirname(ROOT), check=True)


def main():
    # 实验一（1D 长序列）：三档格距 Δ，扫 q=f_c/fN。ffine=q/(2Δ²)。
    # 每条序列 nf 点、共 nlines 条独立序列，保证周期图统计充分。
    nf, nlines = 65536, 64
    for step in (2, 4, 8):
        ncf = nf // step
        for q in (0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0):
            ffine = q / (2 * step * step)
            tag = f"l1_d{step}_q{q:g}"
            os.makedirs(RAW, exist_ok=True)
            fine = os.path.join(RAW, f"{tag}.fine.raw")
            coarse = os.path.join(RAW, f"{tag}.coarse.raw")
            cmd = ["go", "run", DUMPER, "--mode", "line", "--seed", "7",
                   "--nf", str(nf), "--step", str(step), "--ffine", str(ffine),
                   "--nlines", str(nlines), "--fine", fine, "--coarse", coarse]
            print("+", " ".join(cmd))
            subprocess.run(cmd, cwd=os.path.dirname(ROOT), check=True)

    # 实验二/三：fBm 2D，Δ=1（粗格即像素格），三档层数，f0=0.2 周期/像素，
    # 三档最高层频率 0.8/1.6/3.2 均越过奈奎斯特 0.5。
    for k in (3, 4, 5):
        tag = f"fbm_k{k}"
        run("fbm", tag, ffine=0.2, step=1, octaves=k, nf=1024, nc=128)


if __name__ == "__main__":
    sys.exit(main())
