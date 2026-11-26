"""绘制两组各向异性地形的梯度方向直方图（每组一张）。

图中标注双阈值：高度阈值横线、被采纳的峰、被高度/间距否决的候选。
用法: python3 -I make_plots.py
"""
import json
import os
import sys

# -I 隔离模式不自动加入用户 site-packages，这里显式加入（不引入 cwd，保持隔离）
for _p in ("/home/ys/.local/lib/python3.14/site-packages",
           "/usr/local/lib/python3.14/dist-packages",
           "/usr/lib/python3/dist-packages"):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CJK = "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf"
if os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
    _cjk_name = font_manager.FontProperties(fname=_CJK).get_name()
    plt.rcParams["font.family"] = [_cjk_name, "DejaVu Sans"]
else:
    plt.rcParams["font.family"] = "DejaVu Sans"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_core import load_raw, hog_parts, pick_peaks

SIZE = 512
TERRAIN_DIR = "artifacts/terrains"
OUT_DIR = "artifacts/results"
HOG_H, HOG_SEP = 1.25, 15.0

# dataviz 参考调色板（light）
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#8a8a83"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"

plt.rcParams.update({
    "text.color": INK, "axes.edgecolor": "#c9c8c2",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.unicode_minus": False,
})


def plot_one(name, title):
    z = load_raw(os.path.join(TERRAIN_DIR, f"terrain_{name}.raw"), SIZE)
    h_mean, h_max, _, _ = hog_parts(z)
    accepted, rej_h, rej_s = pick_peaks(h_mean, HOG_H, HOG_SEP)

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=140)
    ax.bar(np.arange(180), h_mean, width=1.0, color=BLUE, alpha=0.85,
           label="梯度方向占比（峰/均值归一）")
    ax.axhline(HOG_H, color=ORANGE, lw=1.8, ls="--",
               label=f"高度阈值 {HOG_H}")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":", label="均值水平 1.0")

    ytop = float(h_mean.max())
    for p in accepted:
        ax.axvline(p["angle"], color="#0b5a2e", lw=1.6, alpha=0.9)
        # 标注放在峰上方的图顶区，左右错开避免压图例
        side = -1 if p["angle"] > 120 else 1
        ax.annotate(f"采纳峰 {p['angle']:.0f}°（h={p['height']:.2f}）",
                    xy=(p["angle"], p["height"]),
                    xytext=(p["angle"] + side * 26, ytop * 1.06),
                    fontsize=9, color="#0b5a2e", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#0b5a2e", lw=1))
    for r in rej_h:
        ax.annotate(f"高度否决 {r['angle']:.0f}°（h={r['height']:.2f}<{HOG_H}）",
                    xy=(r["angle"], r["height"]),
                    xytext=(max(r["angle"] - 45, 2), r["height"] + 0.28),
                    fontsize=8.5, color=ORANGE,
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))
    for r in rej_s:
        ax.annotate(f"间距否决 {r['angle']:.0f}°（与邻峰<{HOG_SEP}°）",
                    xy=(r["angle"], r["height"]),
                    xytext=(min(r["angle"] + 5, 90), r["height"] + 0.28),
                    fontsize=8.5, color="#b03a2e",
                    arrowprops=dict(arrowstyle="->", color="#b03a2e", lw=0.9))

    ax.set_xlim(0, 180)
    ax.set_ylim(0, ytop * 1.18)
    ax.set_xlabel("梯度方向角（度，模 180°）")
    ax.set_ylabel("幅度加权占比 / 均值")
    ax.set_title(title, fontsize=12, color=INK, pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"hog_{name}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for n, t in (
        ("A", "强各向异性地形 A：梯度方向直方图（谱预判 2 峰，HOG 实测 1 峰）"),
        ("B", "弱各向异性地形 B：梯度方向直方图（谱预判 2 峰，HOG 实测 2 峰）"),
    ):
        print("wrote", plot_one(n, t))
