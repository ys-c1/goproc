#!/usr/bin/env python3
"""根据 results.json 生成技术报告配图。

配色（dataviz 规范）：
- 等值线层级是有序量 → 单色蓝顺序 ramp 五级；
- 地形用中性灰底，不与等值线抢色相；
- 方法对比用已验证的分类色前三槽（蓝/橙/青），图内直接标注数值。
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

_CJK_CANDIDATES = [
    "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
_CJK_FONT = next((p for p in _CJK_CANDIDATES if Path(p).exists()), None)
if _CJK_FONT:
    font_manager.fontManager.addfont(_CJK_FONT)
    CJK_NAME = font_manager.FontProperties(fname=_CJK_FONT).get_name()
else:
    CJK_NAME = "DejaVu Sans"

BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
CONTOUR_BLUES = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, GRID, INK, MUTED = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": [CJK_NAME, "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.unicode_minus": False,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

N = 128
BOX_W, BOX_H = 9, 5


def gaussian_blur(a, sigma):
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()

    def conv(v, axis):
        pad = [(0, 0)] * v.ndim
        pad[axis] = (r, r)
        p = np.pad(v, pad, mode="reflect")
        out = np.zeros_like(v)
        sl = [slice(None)] * v.ndim
        for i, w in enumerate(k):
            sl[axis] = slice(i, i + v.shape[axis])
            out += w * p[tuple(sl)]
        return out

    return conv(conv(a, 0), 1)


def contour_mask(h, level):
    above = h >= level
    m = np.zeros_like(above)
    cx = above[:, :-1] != above[:, 1:]
    m[:, :-1] |= cx
    m[:, 1:] |= cx
    cy = above[:-1, :] != above[1:, :]
    m[:-1, :] |= cy
    m[1:, :] |= cy
    return m


def main():
    root = Path(sys.argv[1])
    work, figs = root / "work", root / "figures"
    res = json.loads((work / "results.json").read_text())
    levels = res["params"]["levels"]
    cands = res["candidates"]
    h = np.fromfile(work / "heightmap.bin", dtype="<f8").reshape(N, N)
    hs = gaussian_blur(h, 1.2)
    masks = {lv: contour_mask(h, lv) for lv in levels}

    cmap_blue = LinearSegmentedColormap.from_list("blue-ramp", BLUE_RAMP)
    cmap_gray = LinearSegmentedColormap.from_list("terrain-gray", ["#f4f3f0", "#6f6e6a"])

    # ---------- 图 1：地形 + 等值线 + 候选标签框 ----------
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    ax.grid(False)
    ax.imshow(hs, cmap=cmap_gray, origin="upper", extent=[0, N, N, 0], interpolation="nearest")
    for lv, cc in zip(levels, CONTOUR_BLUES):
        ys, xs = np.nonzero(masks[lv])
        ax.scatter(xs, ys, s=2.4, color=cc, label=f"等值线 {lv:.2f}")
    for c in cands:
        rect = mpatches.Rectangle((c["x0"], c["y0"]), BOX_W, BOX_H,
                                  linewidth=1.0, edgecolor="#111111", facecolor="white", alpha=0.82)
        ax.add_patch(rect)
        ax.text(c["x0"] + BOX_W / 2, c["y0"] + BOX_H / 2, str(c["idx"]),
                ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_title("地形、5 条等值线与 20 个候选标签框（框内编号＝代价升序名次）")
    ax.set_xlim(0, N)
    ax.set_ylim(N, 0)
    ax.set_xlabel("x（像素）")
    ax.set_ylabel("y（像素）")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.95,
              fontsize=9, markerscale=5, title="等值线层级")
    fig.tight_layout()
    fig.savefig(figs / "fig1_candidates.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 2：两组 k 的贪心 vs 两种穷举（双面板，各自单轴）----------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    width = 0.26
    for ax, metric, title in zip(
        axes,
        ["pixels", "cost"],
        ["遮挡等值线像素数（个）", "加权代价  Σ bᵢ·κᵢ"],
    ):
        x = np.arange(len(res["summary"]))
        gv = [s["greedy"][metric] for s in res["summary"]]
        ev = [s["exhaust_cost_opt"][metric] for s in res["summary"]]
        pv = [s["exhaust_pixel_opt"][metric] for s in res["summary"]]
        b1 = ax.bar(x - width, gv, width, label="贪心（按代价序）", color=C_BLUE)
        b2 = ax.bar(x, ev, width, label="穷举·代价最优", color=C_AQUA)
        b3 = ax.bar(x + width, pv, width, label="穷举·像素最优", color=C_ORANGE)
        for bars in (b1, b2, b3):
            for r in bars:
                txt = f"{r.get_height():.2f}" if metric == "cost" else f"{int(r.get_height())}"
                ax.annotate(txt, (r.get_x() + r.get_width() / 2, r.get_height()),
                            ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={s['k']}" for s in res["summary"]])
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8.5, framealpha=0.92)
        ax.set_ylim(0, max(max(gv), max(ev), max(pv)) * 1.18)
    fig.suptitle("贪心与穷举对照：同一候选集上的两种目标", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(figs / "fig2_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 3：候选代价的像素—曲率构成 ----------
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    bs = np.array([c["b"] for c in cands])
    ks = np.array([c["kappa"] for c in cands])
    cs = np.array([c["cost"] for c in cands])
    sc = ax.scatter(bs, ks, s=78, c=cs, cmap=cmap_blue, edgecolor=INK, linewidth=0.6, zorder=3)
    for c in cands:
        ax.annotate(str(c["idx"]), (c["b"], c["kappa"]),
                    textcoords="offset points", xytext=(7, 3), fontsize=8.5, color=MUTED)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("代价  bᵢ·κᵢ")
    ax.set_xlabel("框内遮挡等值线像素数  bᵢ")
    ax.set_ylabel("锚点曲率绝对值  |κᵢ|")
    ax.set_title("候选点权衡：像素数与曲率不共序（点编号同图 1）")
    fig.tight_layout()
    fig.savefig(figs / "fig3_tradeoff.png", dpi=160)
    plt.close(fig)

    print("wrote", sorted(p.name for p in figs.glob("*.png")))


if __name__ == "__main__":
    main()
