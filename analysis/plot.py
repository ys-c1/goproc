#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""摩尔纹分析可视化: 扫描曲线 + 四点对照 + 纹路掩膜 + 径向频谱。"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

_CJK = os.path.expanduser("~/.local/share/fonts/NotoSansCJKsc-Regular.ttf")
if os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
    CJK_NAME = font_manager.FontProperties(fname=_CJK).get_name()
else:
    CJK_NAME = "DejaVu Sans"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import (load_f64, local_variance, octave_centers, NYQUIST,  # noqa: E402
                     BASE_FREQ, OCTAVES, LAC, TILE)

MAPS = "artifacts/maps"
SIZE = 256
BLUE, ORANGE, AQUA, RED, INK, MUTED, GRID = (
    "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#0b0b0b", "#52514e", "#e1e0d9")

plt.rcParams.update({
    "font.family": CJK_NAME,
    "font.size": 10,
    "axes.unicode_minus": False,
    "text.color": INK, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def find(k):
    return glob.glob(os.path.join(MAPS, f"map_k{k:06.3f}.f64"))[0]


def radial_spectrum(img):
    n = img.shape[0]
    spec = np.fft.fftshift(np.abs(np.fft.fft2(img - img.mean())) ** 2)
    f = np.fft.fftshift(np.fft.fftfreq(n, 1.0))
    xx, yy = np.meshgrid(f, f)
    rad = np.sqrt(xx ** 2 + yy ** 2)
    bins = np.linspace(0, NYQUIST, n // 2 + 1)
    idx = np.digitize(rad, bins)
    out = np.zeros(len(bins) - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for b in range(1, len(bins)):
        m = idx == b
        if m.any():
            out[b - 1] = spec[m].mean()
    return centers, out


def main():
    ks = np.arange(2.0, 8.0 + 1e-9, 0.25)
    T = np.quantile(local_variance(load_f64(find(2.0), SIZE)), 0.999)

    cnts, rs, alias_flags = [], [], []
    for k in ks:
        img = load_f64(find(round(float(k), 3)), SIZE)
        v = local_variance(img)
        cnts.append(np.count_nonzero(v > T))
        r = octave_centers(round(float(k), 3))[-1]
        rs.append(r)
        alias_flags.append(r > NYQUIST)
    cnts = np.array(cnts)
    rs = np.array(rs)
    alias_flags = np.array(alias_flags)

    fig = plt.figure(figsize=(13.5, 11.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.18,
                          height_ratios=[1.05, 0.95])

    # ---- 面板1: 扫描曲线 ----
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ks, cnts, color=BLUE, lw=2, zorder=3, label="纹路像素数")
    ax1.scatter(ks[~alias_flags], cnts[~alias_flags], s=42, color=BLUE,
                marker="o", zorder=4, label="r ≤ 0.5（未混叠）")
    ax1.scatter(ks[alias_flags], cnts[alias_flags], s=52, color=ORANGE,
                marker="^", zorder=4, label="r > 0.5（混叠）")
    k_star = NYQUIST / (BASE_FREQ * LAC ** (OCTAVES - 1))
    ax1.axvline(k_star, color=RED, ls="--", lw=1.8, zorder=2)
    ax1.annotate(f"临界 1:{k_star:g}\nr=0.5 Nyquist", xy=(k_star, cnts.max() * 0.55),
                 xytext=(k_star + 0.35, cnts.max() * 0.62), color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.4), fontsize=9.5)
    for k, lab, dy in [(2.0, "干净对照 1:2", 3200), (8.0, "纹路对照 1:8", -3200),
                       (4.0, "区间内 1:4", 2200), (2.5, "区间外 1:2.5", 3800)]:
        c = cnts[np.argmin(np.abs(ks - k))]
        ax1.annotate(f"{lab}\n{c} 像素", xy=(k, c), xytext=(k, c + dy),
                     ha="center", fontsize=9, color=INK,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax1.set_xlabel("缩放比分母 k（缩放比 1:k）")
    ax1.set_ylabel(f"纹路像素数（3×3 邻域方差 > T，内部 {SIZE-2}×{SIZE-2}）")
    ax1.set_title("纹路像素随缩放比单调上升，混叠边界处频谱开始折叠", loc="left", fontsize=12)
    ax1.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)

    # ---- 面板2: 四点高度图对照 ----
    ax2 = fig.add_subplot(gs[1, 0])
    panel_ks = [2.0, 2.5, 4.0, 8.0]
    titles = ["干净对照 1:2\nr=0.32", "区间外 1:2.5\nr=0.40",
              "区间内 1:4\nr=0.64", "纹路对照 1:8\nr=1.28"]
    montage = np.hstack([load_f64(find(k), SIZE) for k in panel_ks])
    ax2.imshow(montage, cmap="gray", vmin=0, vmax=1, aspect="equal")
    for i, t in enumerate(titles):
        ax2.text(i * SIZE + 6, 22, t, ha="left", va="top", fontsize=9,
                 color="#ffffff",
                 bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.55),
                           ec="none"))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(False)
    ax2.set_title("高度图对照（同种子、同归一化）", loc="left", fontsize=12, pad=10)

    # ---- 面板3: 纹路掩膜对照 ----
    ax3 = fig.add_subplot(gs[1, 1])
    masks = []
    for k in panel_ks:
        v = local_variance(load_f64(find(k), SIZE))
        m = np.pad(v > T, 1, constant_values=False)
        masks.append(m)
    mm = np.hstack(masks)
    ax3.imshow(mm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "m", ["#fcfcfb", ORANGE]), aspect="equal")
    for i, k in enumerate(panel_ks):
        c = int(masks[i].sum())
        ax3.text(i * SIZE + 6, 22, f"1:{k:g}\n{c} 像素", ha="left", va="top",
                 fontsize=9, color=INK,
                 bbox=dict(boxstyle="round,pad=0.25",
                           fc=(252 / 255, 252 / 255, 251 / 255, 0.82),
                           ec="#c3c2b7", lw=0.6))
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.grid(False)
    ax3.set_title(f"纹路掩膜（橙色=方差>T，T={T:.2e}）", loc="left",
                  fontsize=12, pad=10)

    fig.savefig("artifacts/moire_analysis.png", dpi=150, bbox_inches="tight")
    print("wrote artifacts/moire_analysis.png")

    # ---- 第二张图: 径向频谱 ----
    fig2, ax = plt.subplots(figsize=(10, 5.4), dpi=150)
    ref_f, ref_p = radial_spectrum(load_f64(find(2.0), SIZE))
    curves = [(3.125, AQUA, "s"), (4.0, ORANGE, "^"), (8.0, RED, "D")]
    for k, color, marker in curves:
        f, p = radial_spectrum(load_f64(find(k), SIZE))
        ax.plot(f, 10 * np.log10(p + 1e-12), color=color, lw=1.8,
                label=f"1:{k:g}, r={octave_centers(k)[-1]:.3f}")
    ax.plot(ref_f, 10 * np.log10(ref_p + 1e-12), color=BLUE, lw=2.0,
            label=f"1:2, r={octave_centers(2.0)[-1]:.3f}（干净对照）")
    # 相对干净对照的能量差: 混叠使基带隆起、高频端下凹
    f8, p8 = radial_spectrum(load_f64(find(8.0), SIZE))
    ax.fill_between(f8, 10 * np.log10(ref_p + 1e-12), 10 * np.log10(p8 + 1e-12),
                    where=(f8 <= 0.32), color=RED, alpha=0.12,
                    label="1:8 相对 1:2 的能量差（基带隆起）")
    ax.axvline(NYQUIST, color=RED, ls="--", lw=1.8)
    ax.annotate("Nyquist = 0.5 周/像素\n（采样率的一半）",
                xy=(0.5, ax.get_ylim()[0] + 12),
                xytext=(0.30, ax.get_ylim()[0] + 34), color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4), fontsize=9.5)
    ax.set_xlabel("径向频率（周/像素）")
    ax.set_ylabel("径向平均功率（dB）")
    ax.set_title("混叠把顶层能量折叠回基带：r>0.5 后高频端被掏空、低频端隆起",
                 loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig2.tight_layout()
    fig2.savefig("artifacts/moire_spectrum.png", bbox_inches="tight")
    print("wrote artifacts/moire_spectrum.png")


if __name__ == "__main__":
    main()
