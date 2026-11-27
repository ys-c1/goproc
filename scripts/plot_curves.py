#!/usr/bin/env python3
"""读取标定 JSON，绘制雾化曲线（线性坐标 + log-log 双面板）。

数据全部来自 Go 标定管线的产物，本脚本只做可视化，不参与数值计算。
运行：python3 -I scripts/plot_curves.py artifacts/calib/calibration.json artifacts/calib
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 显式注册系统里的 Noto CJK，避免 rcParams 名称匹配失败回退 DejaVu。
_CJK_PATH = "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf"
try:
    font_manager.fontManager.addfont(_CJK_PATH)
    _CJK_NAME = font_manager.FontProperties(fname=_CJK_PATH).get_name()
except OSError:
    _CJK_NAME = "DejaVu Sans"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d3"
SURFACE = "#fcfcfb"

COLORS = {"box": BLUE, "triangle": ORANGE}
LABELS = {"box": "盒形核 box（支撑半径 0.5/s）", "triangle": "三角核 triangle（支撑半径 1/s）"}


def piecewise_curve(curve, dense_s):
    """复刻 Go 端 log-log 分段线性预判，保证图与 CLI 完全一致。"""
    pts = sorted(curve["points"], key=lambda p: p["Scale"])
    s_pts = np.array([p["Scale"] for p in pts])
    r_pts = np.array([p["R"] for p in pts])
    out = np.empty_like(dense_s)
    s_min = s_pts[0]
    beta_lo = (np.log(r_pts[1]) - np.log(r_pts[0])) / (np.log(s_pts[1]) - np.log(s_pts[0]))
    for i, s in enumerate(dense_s):
        if s >= s_pts[-1]:
            out[i] = 1.0
        elif s <= s_min:
            out[i] = np.exp(np.log(r_pts[0]) + beta_lo * (np.log(s) - np.log(s_pts[0])))
        else:
            j = np.searchsorted(s_pts, s) - 1
            t = (np.log(s) - np.log(s_pts[j])) / (np.log(s_pts[j + 1]) - np.log(s_pts[j]))
            out[i] = np.exp(np.log(r_pts[j]) + t * (np.log(r_pts[j + 1]) - np.log(r_pts[j])))
    return out


def main():
    calib_path, outdir = sys.argv[1], sys.argv[2]
    with open(calib_path, encoding="utf-8") as f:
        calib = json.load(f)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [_CJK_NAME, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    dense_s = np.linspace(0.10, 1.0, 450)

    for ax in axes:
        ax.axvspan(0.10, 0.25, color=GRID, alpha=0.45, zorder=0)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_xlabel("缩放比 s = 缩略图边长 / 原图边长")

    for curve in calib["curves"]:
        name = curve["kernel"]
        color = COLORS[name]
        pts = sorted(curve["points"], key=lambda p: p["Scale"])
        s_pts = [p["Scale"] for p in pts]
        r_pts = [p["R"] for p in pts]
        f_pts = [p["F"] for p in pts]
        r_dense = piecewise_curve(curve, dense_s)
        f_dense = 1 - r_dense

        # 左图：线性坐标，雾化 F
        ax = axes[0]
        ax.plot(dense_s, f_dense, color=color, linewidth=2, label=LABELS[name], zorder=3)
        ax.scatter(s_pts, f_pts, color=color, s=55, zorder=4,
                   edgecolor=SURFACE, linewidth=1.2)
        # 右图：log-log，方差比 R
        ax = axes[1]
        ax.plot(dense_s, r_dense, color=color, linewidth=2, zorder=3)
        ax.scatter(s_pts, r_pts, color=color, s=55, zorder=4,
                   edgecolor=SURFACE, linewidth=1.2)
        # 紧凑幂律参考线（虚线）
        pl = curve["A"] * dense_s ** curve["gamma"]
        ax.plot(dense_s, pl, color=color, linewidth=1.4, linestyle="--",
                alpha=0.7, zorder=2,
                label=f"{LABELS[name].split('（')[0]} 幂律 γ={curve['gamma']:.3f}, R²={curve['rSquared']:.4f}")

    axes[0].set_ylabel("雾化程度 F = 1 − R（R 为局部方差比）")
    axes[0].set_title("雾化曲线（线性坐标）", fontsize=12)
    axes[0].set_xlim(0.10, 1.0)
    axes[0].set_ylim(0, 1.0)
    axes[0].annotate("灰色带：标定区间外\n（端段斜率外推）", xy=(0.105, 0.06),
                     fontsize=8.5, color=MUTED, va="bottom")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    axes[1].set_ylabel("局部方差比 R")
    axes[1].set_title("log-log 曲线（实线：分段标定；虚线：幂律）", fontsize=12)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(0.10, 1.0)
    axes[1].set_ylim(0.04, 1.1)
    axes[1].annotate("五个标定取点：s = 2^(−k/2), k=0..4\n(1.000, 0.707, 0.500, 0.354, 0.250)\n灰带为区间外外推",
                     xy=(0.115, 0.55), fontsize=8.5, color=MUTED)
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle(
        f"缩略图雾化标定曲线（原图 {calib['size']}×{calib['size']}，"
        f"局部方差窗 W={calib['localVarianceWindow']}，圆点为五档实测点）",
        fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"{outdir}/fog_curves.png"
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
