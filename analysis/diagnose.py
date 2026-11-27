#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断: 邻域方差分布随 k 的变化 + 候选阈值余量 + 超采样基准对比。"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_f64, local_variance, octave_centers, NYQUIST, fold  # noqa: E402

MAPS = "artifacts/maps"
SS = "artifacts/maps_ss"
SIZE = 256


def find(d, k):
    h = glob.glob(os.path.join(d, f"map_k{k:06.3f}.f64"))
    return h[0] if h else None


def main():
    ks = [2.0, 2.5, 2.75, 3.0, 3.125, 3.25, 3.5, 4.0, 5.0, 6.25, 8.0]
    print("== 邻域方差分位数 (3x3, 内部像素) ==")
    print(f"{'k':>7}{'max':>11}{'q99.9':>11}{'q99.5':>11}{'q99':>11}{'mean':>11}")
    stats = {}
    for k in ks:
        img = load_f64(find(MAPS, k), SIZE)
        v = local_variance(img)
        stats[k] = v
        print(f"{k:>7.3f}{v.max():>11.3e}{np.quantile(v,.999):>11.3e}"
              f"{np.quantile(v,.995):>11.3e}{np.quantile(v,.99):>11.3e}{v.mean():>11.3e}")

    # 候选阈值: 用干净图 k=2 的极值/高分位标定, 看其它图的越线像素率
    v2 = stats[2.0]
    print("\n== 候选阈值下各图纹路像素占比 (%) ==")
    cands = {"max(k=2)": v2.max(), "q99.9(k=2)": np.quantile(v2, .999),
             "q99.5(k=2)": np.quantile(v2, .995), "q99(k=2)": np.quantile(v2, .99)}
    print(f"{'k':>7}" + "".join(f"{n:>13}" for n in cands))
    for k in ks:
        v = stats[k]
        row = f"{k:>7.3f}"
        for n, T in cands.items():
            row += f"{100*np.count_nonzero(v>T)/v.size:>12.3f}%"
        print(row)

    # 超采样基准对比 (若已生成)
    print("\n== 直采 vs 4x超采样盒滤波基准 ==")
    if os.path.isdir(SS):
        print(f"{'k':>7}{'直采-超采样 RMS':>16}{'相关系数':>10}")
        for k in ks:
            p1, p2 = find(MAPS, k), find(SS, k)
            if not p2:
                continue
            a = load_f64(p1, SIZE)
            b = load_f64(p2, SIZE)
            rms = np.sqrt(np.mean((a - b) ** 2))
            corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            print(f"{k:>7.3f}{rms:>16.4f}{corr:>10.4f}")
    else:
        print("(未找到超采样目录, 跳过)")

    # 倍频程分带能量: 验证最高层在 k>3.125 后折叠
    print("\n== FFT 倍频程分带 (能量占比 %) ==")
    print(f"{'k':>7}" + "".join(f"{'oct'+str(i):>9}" for i in range(4)) +
          f"{'顶层是否混叠':>12}{'折叠到':>9}")
    for k in ks:
        img = load_f64(find(MAPS, k), SIZE)
        spec = np.fft.fftshift(np.abs(np.fft.fft2(img - img.mean())) ** 2)
        f = np.fft.fftshift(np.fft.fftfreq(SIZE, d=1.0))
        xx, yy = np.meshgrid(f, f)
        rad = np.sqrt(xx ** 2 + yy ** 2)
        centers = octave_centers(k)
        total = spec.sum()
        row = f"{k:>7.3f}"
        for i, fc in enumerate(centers):
            lo, hi = fc / np.sqrt(2), fc * np.sqrt(2)
            m = (rad >= lo) & (rad < min(hi, NYQUIST))
            row += f"{100*spec[m].sum()/total:>9.2f}"
        alias = centers[-1] > NYQUIST
        row += f"{('是' if alias else '否'):>12}"
        row += f"{fold(centers[-1]):>9.4f}" if alias else f"{'-':>9}"
        print(row)


if __name__ == "__main__":
    main()
