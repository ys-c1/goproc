#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""摩尔纹对照分析。

量定义:
  fs          采样率 = 1 采样/像素, Nyquist = 0.5 周/像素
  freq0(k)    基础频率 = k * 0.02 (缩放比 1:k, 拉远 k 倍频率升 k 倍)
  fm(k)       最高 FBM 层主频 = freq0 * 2^(octaves-1) = 0.16k (首个越过 Nyquist 的层)
  r(k)        主频/采样率 = fm / fs = fm (fs=1)
  纹路像素     3x3 邻域样本方差 > T 的内部像素;
              T 用干净对照点 (k=2) 邻域方差的 99.9 分位标定, 全程固定。
混叠判据 (采样定理): r > 0.5 即欠采样, 表观频率 fold(f)=|((f+0.5) mod 1)-0.5|。
"""
import argparse
import glob
import os
import sys

import numpy as np

BASE_FREQ = 0.02
OCTAVES = 4
LAC = 2
FS = 1.0
NYQUIST = 0.5
TILE = 3  # 邻域边长


def load_f64(path, size):
    a = np.fromfile(path, dtype="<f8")
    if a.size != size * size:
        raise ValueError(f"{path}: 期望 {size*size} 个浮点, 实际 {a.size}")
    return a.reshape(size, size)


def find_map(maps_dir, k):
    hits = glob.glob(os.path.join(maps_dir, f"map_k{k:06.3f}.f64"))
    if not hits:
        raise FileNotFoundError(f"缺少 k={k} 的数据, 先用 genmap 生成")
    return hits[0]


def octave_centers(k):
    f0 = k * BASE_FREQ
    return [f0 * LAC ** i for i in range(OCTAVES)]


def local_variance(img):
    """3x3 邻域样本方差 (ddof=1), 仅内部像素, 形状 (N-2, N-2)。"""
    x = img.astype(np.float64)
    k = np.ones((TILE, TILE), dtype=np.float64)
    n = k.size
    s = _conv2d(x, k)
    sq = _conv2d(x * x, k)
    return np.maximum((sq - s * s / n) / (n - 1), 0.0)


def _conv2d(x, k):
    h, w = x.shape
    kh, kw = k.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * x[i:i + h - kh + 1, j:j + w - kw + 1]
    return out


def fold(f):
    """采样折叠: 混叠后落到基带的表观频率。"""
    return abs(((f + NYQUIST) % 1.0) - NYQUIST)


def top_octave_band_share(img, k):
    """最高 FBM 层在其原生倍频程频带内的能量占比 (混叠后该值塌缩)。"""
    n = img.shape[0]
    spec = np.fft.fftshift(np.abs(np.fft.fft2(img - img.mean())) ** 2)
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
    xx, yy = np.meshgrid(f, f)
    rad = np.sqrt(xx ** 2 + yy ** 2)
    fc = octave_centers(k)[-1]
    m = (rad >= fc / np.sqrt(2)) & (rad < min(fc * np.sqrt(2), NYQUIST))
    return 100.0 * spec[m].sum() / spec.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default="artifacts/maps")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--clean-k", type=float, default=2.0)
    ap.add_argument("--moire-k", type=float, default=8.0)
    ap.add_argument("--in-k", type=float, default=4.0)
    ap.add_argument("--out-k", type=float, default=2.5)
    ap.add_argument("--quantile", type=float, default=0.999)
    ap.add_argument("--scan", default="")
    args = ap.parse_args()

    def img(k):
        return load_f64(find_map(args.maps_dir, k), args.size)

    clean = img(args.clean_k)
    T = float(np.quantile(local_variance(clean), args.quantile))

    def report(k):
        im = img(k)
        fm = octave_centers(k)[-1]
        var = local_variance(im)
        cnt = int(np.count_nonzero(var > T))
        return dict(k=k, freq0=k * BASE_FREQ, fm=fm, r=fm / FS,
                    fold=fold(fm), alias=fm > NYQUIST,
                    band=top_octave_band_share(im, k),
                    cnt=cnt, total=var.size, pct=100.0 * cnt / var.size)

    keys = [args.clean_k, args.moire_k, args.in_k, args.out_k]
    roles = {args.clean_k: "干净对照", args.moire_k: "纹路对照",
             args.in_k: "区间内验证", args.out_k: "区间外验证"}
    rows = {k: report(k) for k in keys}

    print(f"采样率 fs={FS} 采样/像素, Nyquist={NYQUIST} 周/像素, 邻域={TILE}x{TILE}")
    print(f"阈值 T = 干净对照(k={args.clean_k})邻域方差 {args.quantile*100:.1f} 分位 = {T:.6e}\n")
    print(f"{'角色':<10}{'1:k':>7}{'freq0':>8}{'主频fm':>8}{'r=fm/fs':>9}"
          f"{'是否混叠':>8}{'折叠表观':>8}{'顶层带能量%':>11}{'纹路像素':>9}{'占比%':>8}")
    for k in keys:
        d = rows[k]
        print(f"{roles[k]:<10}{d['k']:>7.3f}{d['freq0']:>8.4f}{d['fm']:>8.4f}"
              f"{d['r']:>9.4f}{('是' if d['alias'] else '否'):>8}"
              f"{d['fold']:>8.4f}{d['band']:>11.3f}{d['cnt']:>9d}{d['pct']:>8.2f}")

    k_star = NYQUIST / (BASE_FREQ * LAC ** (OCTAVES - 1))
    r_lo, r_hi = NYQUIST, rows[args.moire_k]["r"]
    print("\n== 由两个对照比值反解纹路区间 ==")
    print(f"干净对照 r={rows[args.clean_k]['r']:.4f} (<0.5, 不混叠); "
          f"纹路对照 r={rows[args.moire_k]['r']:.4f} (>0.5, 混叠)")
    print(f"临界缩放 1:k* = 1:{k_star:.4f}, 即 k* = 0.5/(0.02*2^3) = {k_star:.4f}")
    print(f"=> 比值区间 r ∈ [{r_lo:.4f}, {r_hi:.4f}] 周/像素")
    print(f"=> 对应缩放区间 1:k ∈ [1:{k_star:.4f}, 1:8.0000] (k ∈ [{k_star:.4f}, 8])")
    print(f"区间内验证 1:{args.in_k:g}: r={rows[args.in_k]['r']:.4f}, "
          f"纹路像素 {rows[args.in_k]['cnt']}")
    print(f"区间外验证 1:{args.out_k:g}: r={rows[args.out_k]['r']:.4f}, "
          f"纹路像素 {rows[args.out_k]['cnt']}")

    if args.scan:
        ks = [float(t) for t in
              (open(args.scan).read().split() if os.path.exists(args.scan)
               else args.scan.split(","))]
        print("\n== 密集扫描 (阈值固定, 步长 0.25) ==")
        print(f"{'1:k':>8}{'r':>8}{'混叠':>6}{'顶层带能量%':>11}{'纹路像素':>9}{'占比%':>8}")
        for k in ks:
            try:
                d = report(k)
            except FileNotFoundError:
                continue
            print(f"{d['k']:>8.3f}{d['r']:>8.4f}{('是' if d['alias'] else '否'):>6}"
                  f"{d['band']:>11.3f}{d['cnt']:>9d}{d['pct']:>8.2f}")


if __name__ == "__main__":
    sys.exit(main())
