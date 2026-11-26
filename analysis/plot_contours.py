"""绘制两组地形 × 两个等高距的等高线图。

复用 contour_analysis 的 marching-squares 链与数字阶梯长度，保证图上
所见与实测总长同源。每个地形一张图，左右两个面板分别对应两个等高距：
dh=0.10 用蓝、dh=0.05 用橙（经 dataviz 校验的 categorical 相邻对），
底图为灰度高度场，坐标轴等比例。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contour_analysis import (  # noqa: E402
    load_heightmap,
    measure_contours,
)

# 中文字体（系统已装 Noto Sans CJK SC），按文件注册避免缓存未收录
_FONT_CANDIDATES = [
    "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for _p in _FONT_CANDIDATES:
    if Path(_p).exists():
        fm.fontManager.addfont(_p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
        break
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"


def plot_terrain(h: np.ndarray, title: str, out_path: Path) -> None:
    n = h.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), facecolor=SURFACE)
    fig.suptitle(title, fontsize=15, color=INK, fontweight="bold")
    specs = [(0.10, BLUE), (0.05, ORANGE)]
    for ax, (dh, color) in zip(axes, specs):
        ax.set_facecolor(SURFACE)
        ax.imshow(h, cmap="gray", origin="upper", extent=(0, n, n, 0),
                  vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
        m = measure_contours(h, dh)
        # measure_contours 内部剥离了链，这里重新取链用于绘制
        from contour_analysis import contour_level, levels_for, trace_chains
        for lv in levels_for(dh):
            r = contour_level(h, float(lv))
            for chain in trace_chains(r["edges"], n):
                xs = [p[0] for p in chain]
                ys = [p[1] for p in chain]
                ax.plot(xs, ys, color=color, linewidth=0.9, solid_joinstyle="miter")
        ax.set_title(
            f"等高距 dh={dh:.2f}｜{m['n_levels']} 层｜"
            f"实测总长 {m['total_length']:.1f} px\n"
            f"横段 {m['n_straight']} 条 ×1 + 对角段 {m['n_diagonal']} 条 ×√2",
            fontsize=11, color=INK)
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("x / px", color=MUTED)
        ax.set_ylabel("y / px", color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#d8d7d2")
        ax.grid(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    data_dir = Path("artifacts/data")
    fig_dir = Path("artifacts/fig")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_terrain(
        load_heightmap(data_dir / "terrain_A_oct2.bin"),
        "地形 A：octaves=2（低频、缓坡为主）",
        fig_dir / "contours_A_oct2.png")
    plot_terrain(
        load_heightmap(data_dir / "terrain_B_oct8.bin"),
        "地形 B：octaves=8（高频细节多、坡度大）",
        fig_dir / "contours_B_oct8.png")


if __name__ == "__main__":
    main()
