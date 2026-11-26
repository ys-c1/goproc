"""生成全部正式图（matplotlib，遵循 dataviz 规范：单坐标、2px 线、直接标注）。"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
font_manager.fontManager.addfont(
    "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf")
import matplotlib.pyplot as plt
import numpy as np

from common import RAW, read_raw

plt.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "font.size": 10,
    "axes.edgecolor": "#c3c2b7",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "text.color": "#0b0b0b",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
os.makedirs(FIGS, exist_ok=True)


def load(name):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name)) as fh:
        return json.load(fh)


def fig_exp1():
    rows = load("exp1_rows.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for step, color in ((2, BLUE), (4, ORANGE), (8, GREEN)):
        r = [x for x in rows if x["step"] == step and x["q"] >= 1.0]
        xi = [x["xi"] for x in r]
        ax1.plot(xi, [x["e1_an"] for x in r], "-", color=color, lw=2,
                 label=f"解析 Δ={step}")
        ax1.plot(xi, [x["e1_meas"] for x in r], "o", color=color, ms=6,
                 mec="#fcfcfb", mew=1.5, label=f"实测 Δ={step}")
    ax1.set_xlabel("频率超出量 ξ = f_c − f_N（周期/粗像素）")
    ax1.set_ylabel("混叠能量分数 E1")
    ax1.set_title("混叠能量 ∝ 频率超出量（近阈值线性）")
    ax1.legend(fontsize=8, frameon=False, ncol=2)
    ax1.set_ylim(bottom=0)

    # Δ 缩放：等 q 下 E1 与 Δ 无关
    qs = sorted({x["q"] for x in rows if x["q"] >= 1.1})
    width = 0.25
    xidx = np.arange(len(qs))
    for i, step in enumerate((2, 4, 8)):
        vals = [next(x["e1_meas"] for x in rows if x["step"] == step and x["q"] == q)
                for q in qs]
        ax2.bar(xidx + (i - 1) * width, vals, width, color=[BLUE, ORANGE, GREEN][i],
                label=f"Δ={step}")
    ax2.set_xticks(xidx)
    ax2.set_xticklabels([f"q={q:g}" for q in qs])
    ax2.set_ylabel("实测混叠能量 E1")
    ax2.set_title("Δ 缩放：等 q=f_c/f_N 下 E1 与格距无关")
    ax2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_linearity.png"), dpi=130)
    plt.close(fig)


def fig_exp2():
    rows = load("exp2_rows.json")
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    Ks = [r["K"] for r in rows]
    x = np.arange(len(Ks))
    ax.bar(x - 0.2, [r["total"] for r in rows], 0.38, color=BLUE,
           label="全部倍频层混叠能量")
    ax.bar(x + 0.2, [r["top"] for r in rows], 0.38, color=ORANGE,
           label="仅最高层贡献")
    for i, r in enumerate(rows):
        ax.text(i, r["total"] * 1.08, f"最高层占比 {r['top_fraction']:.0%}",
                ha="center", fontsize=9, color="#0b0b0b")
    ax.set_ylim(0, max(r["total"] for r in rows) * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}\nf_top={r['f_top']:.2f}" for k, r in zip(Ks, rows)])
    ax.set_ylabel("混叠能量")
    ax.set_title("多倍频叠加：混叠由最高层频率主导，而非基础频率")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_dominant.png"), dpi=130)
    plt.close(fig)


def fig_exp3():
    rows = load("exp3_rows.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    Ks = [r["K"] for r in rows]
    x = np.arange(len(Ks))
    w = 0.36
    # 左：基础频率上限（单坐标，分组柱）
    b1 = ax1.bar(x - w / 2, [r["hard_cap_px"] for r in rows], w, color=BLUE,
                 label="硬上限 f0（/px）")
    b2 = ax1.bar(x + w / 2, [r["f0_cap_px"] for r in rows], w, color=GREEN,
                 label="阈值反解 f0（/px）")
    ax1.set_ylabel("基础频率上限 f0（周期/像素）")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"K={k}" for k in Ks])
    ax1.set_title("三档层数的基础频率上限（C*=0.1%）")
    ax1.legend(fontsize=9, frameon=False)
    for rect, r in zip(b1, rows):
        ax1.text(rect.get_x() + rect.get_width() / 2, r["hard_cap_px"] + 0.003,
                 f"{r['hard_cap_px']:.3f}", ha="center", fontsize=8)
    for rect, r in zip(b2, rows):
        ax1.text(rect.get_x() + rect.get_width() / 2, r["f0_cap_px"] + 0.003,
                 f"{r['f0_cap_px']:.3f}", ha="center", fontsize=8)
    ax1.set_ylim(0, 0.15)

    # 右：条纹对比度（单坐标，解析 vs 实测）
    ax2.bar(x - w / 2, [r["c_analytic"] for r in rows], w, color=ORANGE,
            label="解析对比度 C")
    ax2.bar(x + w / 2, [r["c_measured"] for r in rows], w, color=RED,
            label="实测对比度 E2")
    ax2.axhline(0.001, color="#898781", lw=1.2)
    ax2.text(2.4, 0.0012, "C*=0.1%", fontsize=8, color="#52514e", ha="right")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"K={k}" for k in Ks])
    ax2.set_ylabel("条纹对比度")
    ax2.set_title("三档层数的条纹对比度（f0=0.2/px，均超限）")
    ax2.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_tiers.png"), dpi=130)
    plt.close(fig)


def fig_table():
    """最高层频率、折叠位置、条纹周期单列成表（PNG）。"""
    rows = load("exp3_rows.json")
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.axis("off")
    head = ["层数 K", "最高层频率\nf_top (/px)", "硬上限 f0\n(/px)", "反解上限 f0\n(/px)",
            "折叠 fa\n(/px)", "条纹周期\n(px)", "解析 C", "实测 E2", "超限"]
    cell = [[str(r["K"]), f"{r['f_top_px']:.2f}", f"{r['hard_cap_px']:.3f}",
             f"{r['f0_cap_px']:.3f}", f"{r['alias_freq']*8:.2f}",
             f"{r['period_px']:.1f}", f"{r['c_analytic']:.4f}",
             f"{r['c_measured']:.4f}", "是" if r["over"] else "否"] for r in rows]
    t = ax.table(cellText=cell, colLabels=head, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1, 1.9)
    for key, c in t.get_celld().items():
        c.set_edgecolor("#e1e0d9")
        c.set_facecolor("#eef3fb" if key[0] == 0 else "#fcfcfb")
    ax.set_title("三档层数并排：上限、最高层频率、条纹位置与对比度", pad=12, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3b_table.png"), dpi=130)
    plt.close(fig)


def fig_sweep():
    data = load("exp3_sweep.json")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = {3: BLUE, 4: ORANGE, 5: GREEN}
    for K in ("3", "4", "5"):
        rec = np.array(data[K])
        fpx = rec[:, 0] * 8
        ax.plot(fpx, rec[:, 1], "-", color=colors[int(K)], lw=2,
                label=f"K={K} 解析")
        ax.plot(fpx, rec[:, 2], "o", color=colors[int(K)], ms=6,
                mec="#fcfcfb", mew=1.5, label=f"K={K} 实测")
    ax.axhline(0.001, color="#898781", lw=1.2)
    ax.text(0.30, 0.0013, "C*=0.1%", fontsize=9, color="#52514e", ha="right")
    ax.set_xlabel("基础频率 f0（周期/像素）")
    ax.set_ylabel("条纹对比度")
    ax.set_title("对比度随基础频率上升：解析与实测同序同量级、单调一致\n"
                 "（实测偏高来自强低频层在阈值边界的频谱泄漏，单倍频核对已证实）")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_ylim(0, 0.022)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_sweep.png"), dpi=130)
    plt.close(fig)


def fig_maps():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    L, FN, FS, DELTA = 2.0, 0.0625, 0.125, 8.0
    f0 = 0.025
    for ax, K in zip(axes, (3, 4, 5)):
        z = read_raw(os.path.join(RAW, f"fbm_k{K}.coarse.raw"), 128)
        g = (z - z.min()) / np.ptp(z)
        ax.imshow(g, cmap="gray", interpolation="nearest", origin="upper")
        f_top = f0 * L ** (K - 1)
        fa = abs(((f_top + FN) % FS) - FN)
        period = 1 / (fa * DELTA)
        # 全图均匀的淡指示线（预测条纹周期），不遮挡地形
        for xx in np.arange(period / 2, 128, period):
            ax.axvline(xx, color=RED, lw=0.5, alpha=0.3)
        ax.set_title(f"K={K}  f_top={f_top*DELTA:.2f}/px\n"
                     f"预测混叠折叠周期≈{period:.1f}px（淡红线）", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("fBm 粗采样地形（f0=0.2/px，奈奎斯特 0.5/px）三档均超限")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_maps.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_exp1()
    fig_exp2()
    fig_exp3()
    fig_table()
    fig_sweep()
    fig_maps()
    print("figures in", FIGS)
    print(sorted(os.listdir(FIGS)))
