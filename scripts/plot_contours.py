#!/usr/bin/env python3
"""绘制两个等高距的等高线图，分别存档到 artifacts/contour/figures/。

设计（遵循 dataviz 规范）：
  - 高度底图用单色灰阶；等高线用单色调蓝阶（浅→深，随高度递增），不用彩虹色；
  - 开弧与闭合环用线型区分（实线=闭合环，虚线=开弧），不依赖颜色 alone；
  - 边界跨越点用橙色状态色标记，配图例；
  - 等高线层级较多，图例只给线型/标记语义，色阶用右侧 colorbar 表达高度。

输入：artifacts/contour/heightmap.f64 与 curves_d*.json（Go 生成）。
"""
import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

# 注册本机 CJK 字体，避免中文显示为方框。
_CJK = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.ttf"
if _CJK.exists():
    font_manager.fontManager.addfont(str(_CJK))
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "contour"
FIG = OUT / "figures"

# 单色调蓝阶（顺序数据，单色相浅→深）。
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "contour_blues",
    ["#cfe0f5", "#9dbfec", "#6b9de0", "#3f7cd2", "#2a5fb0", "#173f83"],
)
BOUNDARY_MARK = "#eb6834"  # 状态橙，仅用于边界跨越点
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"


def load_heightmap():
    raw = (OUT / "heightmap.f64").read_bytes()
    n = int(round((len(raw) / 8) ** 0.5))
    vals = struct.unpack(f"<{n*n}d", raw)
    return n, vals


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    n, vals = load_heightmap()
    hmin, hmax = min(vals), max(vals)
    norm = Normalize(vmin=hmin, vmax=hmax)

    for tag, title in (("d0.1", "等高距 Δ=0.10"), ("d0.07", "等高距 Δ=0.07")):
        doc = json.loads((OUT / f"curves_{tag}.json").read_text())
        fig, ax = plt.subplots(figsize=(9, 9), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.imshow(
            [vals[r * n:(r + 1) * n] for r in range(n)],
            cmap="gray", alpha=0.18, origin="upper", extent=(0, n - 1, n - 1, 0),
        )

        for level in doc["levels"]:
            color = BLUE_RAMP(norm(level["h"]))
            for curve in level["curves"]:
                xs = [p[0] for p in curve["pts"]]
                ys = [p[1] for p in curve["pts"]]
                if curve["open"]:
                    ax.plot(xs, ys, linestyle=(0, (5, 3)), linewidth=1.3,
                            color=color, alpha=0.95)
                else:
                    ax.plot(xs, ys, linestyle="-", linewidth=1.6,
                            color=color, alpha=0.95)
            # 标记该档的边界跨越点（开弧端点）。
            for curve in level["curves"]:
                if not curve["open"]:
                    continue
                for p in (curve["pts"][0], curve["pts"][-1]):
                    if p[0] in (0, n - 1) or p[1] in (0, n - 1):
                        ax.plot(p[0], p[1], marker="o", markersize=3.5,
                                markerfacecolor=BOUNDARY_MARK,
                                markeredgecolor=INK, markeredgewidth=0.4,
                                linestyle="none", zorder=5)

        ax.set_title(f"{title}：等高线（{len(doc['levels'])} 档，种子 7）",
                     color=INK, fontsize=13)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xlabel("x（格点坐标）", color=MUTED)
        ax.set_ylabel("y（格点坐标）", color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#d8d6d0")

        sm = plt.cm.ScalarMappable(cmap=BLUE_RAMP, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("等值高度 h", color=MUTED)
        cbar.ax.tick_params(colors=MUTED, labelsize=8)

        legend_handles = [
            Line2D([0], [0], color="#2a5fb0", linestyle="-", linewidth=1.6,
                   label="闭合环（不触边界）"),
            Line2D([0], [0], color="#2a5fb0", linestyle=(0, (5, 3)), linewidth=1.3,
                   label="开弧（两端触边界）"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=BOUNDARY_MARK, markeredgecolor=INK,
                   markeredgewidth=0.4, label="边界跨越点"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=True,
                  facecolor=SURFACE, edgecolor="#d8d6d0", fontsize=9)

        out = FIG / f"contours_{tag}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
