#!/usr/bin/env python3
"""各向异性奈奎斯特采样实验分析。

输入 cmd/exp 落盘的 float64 原始场与 meta.json，输出：
  1. 频谱椭圆轴比（FFT 边际功率谱的高斯二阶矩宽度，长/短轴比）；
  2. 空间自相关椭圆轴比（1/e 相关宽度，交叉验证，理论上同为 r）；
  3. 短轴方向实测局部频率与 r*f_base 的对照；
  4. 朴素网格与奈奎斯特修正网格相对过采样参考场的细节保留度。

关键方法学：被测场一律上采样到参考网格后再比较，绝不能把参考场降采样到
被测网格——后者会先丢弃被测网格本就缺失的高频，使混叠不可见。

运行：python3 -I analysis/analyze.py artifacts/exp
"""
import json
import math
import os
import sys

import numpy as np


def load_raw(path, nx, ny):
    return np.fromfile(path, dtype="<f8").reshape(ny, nx)


def freq_axes(nx, ny, L):
    # 物理域为 [0,L)×[0,L)，故两轴频率分辨率都是 1/L（周/单位距离）。
    fx = np.fft.fftfreq(nx, d=L / nx)
    fy = np.fft.fftfreq(ny, d=L / ny)
    return np.fft.fftshift(fx), np.fft.fftshift(fy)


def psd(field):
    ny, nx = field.shape
    win = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    spec = np.fft.fftshift(np.fft.fft2(field * win))
    return (np.abs(spec) ** 2) / (win**2).sum()


def marginal_widths(p, fx, fy):
    """边际功率谱相对零频的二阶矩宽度（高斯 sigma 估计量）。"""
    mx = p.sum(axis=0)
    my = p.sum(axis=1)
    sx = math.sqrt(np.clip((fx**2 * mx).sum() / mx.sum(), 0, None))
    sy = math.sqrt(np.clip((fy**2 * my).sum() / my.sum(), 0, None))
    return sx, sy


def spatial_widths(field, L):
    """沿两轴的自相关 1/e 宽度（物理单位），取首次跌破 1/e 的滞后。"""
    ny, nx = field.shape
    p = np.abs(np.fft.fft2(field)) ** 2
    ac = np.fft.ifftshift(np.fft.ifft2(p).real)
    ac = ac / ac[ny // 2, nx // 2]

    def width1e(prof, step):
        idx = np.flatnonzero(prof <= 1.0 / math.e)
        return float((idx[0] if idx.size else prof.size - 1) * step)

    wx = width1e(ac[ny // 2, nx // 2:], L / nx)
    wy = width1e(ac[ny // 2:, nx // 2], L / ny)
    return wx, wy


def alias_fraction(field, L, hx, hy, f_base):
    """被测场功率谱中落在该网格奈奎斯特椭圆外的能量占比。

    奈奎斯特椭圆为 (fx*hx)^2 + (fy*hy)^2 = (1/2)^2；理想采样下椭圆外
    只应有数值泄漏，混叠会把该比例显著抬高。f_base 仅用于结果注释。
    """
    ny, nx = field.shape
    fx, fy = freq_axes(nx, ny, L)
    p = psd(field)
    outside = (fx[None, :] * hx) ** 2 + (fy[:, None] * hy) ** 2 > 0.25
    return float(p[outside].sum() / p.sum())


def resample_bilinear(field, nx_dst, ny_dst):
    """把场从自身网格双线性重采样到同物理域的 dst 网格（可上可下采样）。"""
    ny_src, nx_src = field.shape
    ys = (np.arange(ny_dst) + 0.0) * ny_src / ny_dst
    xs = (np.arange(nx_dst) + 0.0) * nx_src / nx_dst
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]
    y1 = (y0 + 1) % ny_src
    x1 = (x0 + 1) % nx_src
    a = field[y0[:, None], x0[None, :]]
    b = field[y0[:, None], x1[None, :]]
    c = field[y1[:, None], x0[None, :]]
    d = field[y1[:, None], x1[None, :]]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def highpass(field, L, kc):
    """物理频率坐标下的各向同性高斯高通，kc 为过渡中心（周/单位距离）。"""
    ny, nx = field.shape
    fx, fy = freq_axes(nx, ny, L)
    k2 = fy[:, None] ** 2 + fx[None, :] ** 2
    mask = 1.0 - np.exp(-k2 / (2 * kc**2))
    spec = np.fft.fftshift(np.fft.fft2(field)) * mask
    return np.fft.ifft2(np.fft.ifftshift(spec)).real


def retention(test, ref, L, kc):
    """被测场上采样到参考网格，同掩码高通后比较细节能量与相关性。

    D_energy: 被测场保留的高频能量占参考场的比例；
    rho:      高通细节场的皮尔逊相关，混叠表现为低相关而非高能量。
    """
    up = resample_bilinear(test, ref.shape[1], ref.shape[0])
    ht = highpass(up, L, kc)
    hr = highpass(ref, L, kc)
    return float(np.var(ht) / np.var(hr)), float(np.corrcoef(ht.ravel(), hr.ravel())[0, 1])


def main():
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/exp"
    with open(os.path.join(exp_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    L = meta["domain_L"]
    f_base = meta["f_base"]
    # 高通带中心取各向同性特征频率 sigma0 附近；r=1 时该带内仍有可观能量。
    kc = f_base / 2.0

    rows = []
    iso_ret = None
    for rm in meta["ratios"]:
        r = rm["ratio"]
        ref = load_raw(os.path.join(exp_dir, rm["ref_raw"]), rm["ref"]["nx"], rm["ref"]["ny"])
        naive = load_raw(os.path.join(exp_dir, rm["naive_raw"]),
                         rm["naive"]["nx"], rm["naive"]["ny"])
        fixed = load_raw(os.path.join(exp_dir, rm["fixed_raw"]),
                         rm["fixed"]["nx"], rm["fixed"]["ny"])

        p = psd(ref)
        fx, fy = freq_axes(rm["ref"]["nx"], rm["ref"]["ny"], L)
        skx, sky = marginal_widths(p, fx, fy)
        spec_ratio = max(skx, sky) / min(skx, sky)

        wx, wy = spatial_widths(ref, L)
        spat_ratio = max(wx, wy) / min(wx, wy)

        d_naive, rho_naive = retention(naive, ref, L, kc)
        d_fixed, rho_fixed = retention(fixed, ref, L, kc)
        af_naive = alias_fraction(naive, L, rm["naive"]["hx"], rm["naive"]["hy"], f_base)
        af_fixed = alias_fraction(fixed, L, rm["fixed"]["hx"], rm["fixed"]["hy"], f_base)
        if r == 1:
            iso_ret = d_fixed
            iso_rho = rho_fixed

        rows.append({
            "r": r,
            "h_short_axis": rm["fixed"]["hy"],
            "n_short_axis": rm["fixed"]["ny"],
            "spec_sigma_x": skx,
            "spec_sigma_y": sky,
            "spec_axis_ratio": spec_ratio,
            "spat_width_x": wx,
            "spat_width_y": wy,
            "spat_axis_ratio": spat_ratio,
            "f_hi_measured": max(skx, sky),
            "f_hi_predicted": r * f_base,
            "ret_naive_energy": d_naive,
            "ret_fixed_energy": d_fixed,
            "rho_naive": rho_naive,
            "rho_fixed": rho_fixed,
            "alias_frac_naive": af_naive,
            "alias_frac_fixed": af_fixed,
        })

    out = {
        "f_base": f_base,
        "kc_highpass": kc,
        "isotropic": {"retention_energy": iso_ret, "rho": iso_rho},
        "rows": rows,
        "deltas_fixed_minus_iso": [x["ret_fixed_energy"] - iso_ret for x in rows],
        "deltas_naive_minus_iso": [x["ret_naive_energy"] - iso_ret for x in rows],
    }
    with open(os.path.join(exp_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    hdr = ("r", "h_short", "N_short", "D_naive", "D_fixed", "rho_naive",
           "rho_fixed", "alias_naive", "alias_fixed", "spec_ratio",
           "spat_ratio", "f_hi_meas/pred")
    print("\t".join(hdr))
    for x in rows:
        print(f"{x['r']:g}\t{x['h_short_axis']:.4f}\t{x['n_short_axis']}\t"
              f"{x['ret_naive_energy']:.4f}\t{x['ret_fixed_energy']:.4f}\t"
              f"{x['rho_naive']:.4f}\t{x['rho_fixed']:.4f}\t"
              f"{x['alias_frac_naive']:.4f}\t{x['alias_frac_fixed']:.4f}\t"
              f"{x['spec_axis_ratio']:.3f}\t{x['spat_axis_ratio']:.3f}\t"
              f"{x['f_hi_measured']:.3f}/{x['f_hi_predicted']:.3g}")
    print(f"\n各向同性基线 D_iso={iso_ret:.4f} rho_iso={iso_rho:.4f}")
    print("修正网格 Δ(D_fixed-D_iso):", " ".join(f"{d:+.4f}" for d in out["deltas_fixed_minus_iso"]))
    print("朴素网格 Δ(D_naive-D_iso):", " ".join(f"{d:+.4f}" for d in out["deltas_naive_minus_iso"]))
    print(f"结果 -> {os.path.join(exp_dir, 'results.json')}")


if __name__ == "__main__":
    main()
