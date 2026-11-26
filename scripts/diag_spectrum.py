#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断: 谱斜率换算关系 + fBm 场的标度区定位。
用谱合成生成精确 S2D(k)~k^beta2D 的实随机场, 检查径向 2D 斜率与 1D 切片斜率。"""
import numpy as np

rng = np.random.default_rng(0)
N = 1024


def synth_field(beta2d, n=N):
    """在未移位的 FFT 网格上直接构造 Hermitian 谱, S2D~k^beta2d。"""
    G = np.zeros((n, n), np.complex128)
    yy, xx = np.mgrid[0:n, 0:n]
    kx = np.where(xx > n // 2, xx - n, xx)
    ky = np.where(yy > n // 2, yy - n, yy)
    k = np.sqrt(kx ** 2 + ky ** 2)
    with np.errstate(divide="ignore"):
        amp = np.where(k > 0, k ** (beta2d / 2.0), 0.0)
    re = rng.normal(0, 1, (n, n)) * amp
    im = rng.normal(0, 1, (n, n)) * amp
    G = re + 1j * im
    G[0, 0] = 0.0
    # Hermitian: G[-i,-j] = conj(G[i,j]), 只填独立半平面后镜像
    for i in range(n):
        for j in range(1, n):
            G[(n - i) % n, (n - j) % n] = np.conj(G[i, j])
    # Nyquist 行/列取实部
    G[0, n // 2] = G[0, n // 2].real
    G[n // 2, 0] = G[n // 2, 0].real
    G[n // 2, n // 2] = G[n // 2, n // 2].real
    return np.fft.ifft2(G).real


def radial_slope(f, k_lo, k_hi, window=True):
    n = f.shape[0]
    win = np.hanning(n) if window else np.ones(n)
    x = (f - f.mean()) * np.outer(win, win)
    F = np.fft.fftshift(np.fft.fft2(x))
    psd = (F.real ** 2 + F.imag ** 2) / (n * n)
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rmax = r.max() + 1
    acc, cnt = np.zeros(rmax), np.zeros(rmax)
    np.add.at(acc, r.ravel(), psd.ravel())
    np.add.at(cnt, r.ravel(), 1.0)
    k = np.arange(1, n // 2)
    p = acc[k] / cnt[k]
    m = (k >= k_lo) & (k <= k_hi)
    b, _ = np.polyfit(np.log10(k[m]), np.log10(p[m]), 1)
    return b


def slice_slope(f, k_lo, k_hi, window=True):
    n = f.shape[0]
    win = np.hanning(n) if window else np.ones(n)
    x = (f - f.mean(axis=1, keepdims=True)) * win
    P = np.fft.rfft(x, axis=1)
    p = (P.real ** 2 + P.imag ** 2).mean(axis=0) / n
    k = np.arange(p.size)
    m = (k >= k_lo) & (k <= k_hi)
    b, _ = np.polyfit(np.log10(k[m]), np.log10(p[m]), 1)
    return b


print("== 谱合成精确幂律场 (Hann window), 拟合 k in [3,80] ==")
for b2 in [-2.5, -3.0, -3.5, -4.0]:
    f = synth_field(b2)
    m2 = radial_slope(f, 3, 80)
    m1 = slice_slope(f, 3, 80)
    print(f"target beta2D={b2:5.1f}  measured beta2D={m2:7.3f}  beta1D={m1:7.3f}  "
          f"beta1D-beta2D={m1-m2:+.3f} (theory +1)")

print("\n== Go fBm 场 (8 octaves), 不同拟合带 (Hann window) ==")
for tag, fn in [("g057", "artifacts/coastline/data/t1_g057.f64"),
                ("g060", "artifacts/coastline/data/t2_g060.f64"),
                ("g062", "artifacts/coastline/data/t3_g062.f64")]:
    f = np.fromfile(fn, dtype="<f8").reshape(1024, 1024)
    for lo, hi in [(2, 12), (3, 16), (4, 24), (8, 40), (16, 64), (32, 128), (64, 256)]:
        m2 = radial_slope(f, lo, hi)
        m1 = slice_slope(f, lo, hi)
        print(f"{tag} k[{lo:3d},{hi:3d}] b2D={m2:7.3f} b1D={m1:7.3f} diff={m1-m2:+.3f}")
    print()

print("== 局部对数斜率 (径向 PSD, 每 8 个波数一档) ==")
for tag, fn in [("g057", "artifacts/coastline/data/t1_g057.f64"),
                ("g060", "artifacts/coastline/data/t2_g060.f64"),
                ("g062", "artifacts/coastline/data/t3_g062.f64")]:
    f = np.fromfile(fn, dtype="<f8").reshape(1024, 1024)
    n = 1024
    win = np.hanning(n)
    x = (f - f.mean()) * np.outer(win, win)
    F = np.fft.fftshift(np.fft.fft2(x))
    psd = (F.real ** 2 + F.imag ** 2) / (n * n)
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rmax = r.max() + 1
    acc, cnt = np.zeros(rmax), np.zeros(rmax)
    np.add.at(acc, r.ravel(), psd.ravel())
    np.add.at(cnt, r.ravel(), 1.0)
    k = np.arange(1, n // 2)
    p = acc[k] / cnt[k]
    lk, lp = np.log10(k.astype(float)), np.log10(p)
    print(tag, end="  ")
    for lo in [2, 4, 8, 16, 32, 64, 128, 256]:
        m = (k >= lo) & (k < lo * 2)
        if m.sum() > 2:
            b, _ = np.polyfit(lk[m], lp[m], 1)
            print(f"k[{lo:3d},{2*lo:3d}]={b:6.2f}", end=" ")
    print()
