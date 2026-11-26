#!/usr/bin/env python3
"""绘制水平/竖直剖面曲线，并独立复核 JSON 中的符号翻转计数。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 中文字体；matplotlib 的缓存有时看不到 fc-list 已注册的用户字体，按文件显式载入。
def configure_cjk_font() -> None:
    patterns = (
        "/usr/share/fonts/**/NotoSansCJK*Regular*.ttf",
        "/usr/local/share/fonts/**/NotoSansCJK*Regular*.ttf",
        str(Path.home() / ".local/share/fonts/NotoSansCJK*Regular*.ttf"),
    )
    for pattern in patterns:
        for font_path in sorted(Path("/").glob(pattern.lstrip("/")) if pattern.startswith("/") else Path().glob(pattern)):
            if font_path.is_file():
                font_manager.fontManager.addfont(str(font_path))
                name = font_manager.FontProperties(fname=str(font_path)).get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                return
    for candidate in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
        if any(f.name == candidate for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = candidate
            return


configure_cjk_font()
plt.rcParams["axes.unicode_minus"] = False

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"


def count_sign_flips(values: np.ndarray) -> tuple[int, list[int]]:
    products = values[:-1] * values[1:]
    edges = np.flatnonzero(products < 0).tolist()
    return len(edges), edges


def plot_profile(profile: dict, out_path: Path) -> None:
    n = int(profile["n_samples"])
    x = np.arange(n)
    total = np.asarray(profile["total_raw_fbm"], dtype=float)
    low = np.asarray(profile["low"], dtype=float)

    measured, edges = count_sign_flips(total)
    assert measured == profile["measured_sign_flips"]
    assert edges == [c["edge"] for c in profile["crossings"]]

    # 独立复核增量恒等式。
    incremental = int(profile["low_crossings"])
    for step in profile["steps"][1:]:
        incremental += int(step["net_extra"])
    assert incremental == profile["measured_sign_flips"]
    assert profile["exact_increment_prediction"] == measured
    assert profile["exact_edge_false_alarms"] == 0
    assert profile["exact_edge_misses"] == 0

    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(x, total, 0, where=total >= 0, interpolate=True,
                    color=BLUE, alpha=0.12, linewidth=0)
    ax.fill_between(x, total, 0, where=total < 0, interpolate=True,
                    color=RED, alpha=0.10, linewidth=0)
    ax.axhline(0, color=BASELINE, linewidth=1.2, zorder=1)
    ax.plot(x, low, color=MUTED, linewidth=1.6, alpha=0.95,
            label="最低频层 $L_0$", zorder=2)
    ax.plot(x, total, color=BLUE, linewidth=2.2,
            label="合成 fBm 剖面", zorder=3)

    for k, crossing in enumerate(profile["crossings"], start=1):
        xz = float(crossing["x_zero_linear"])
        ax.scatter([xz], [0], s=82, marker="o", color=ORANGE,
                   edgecolor=SURFACE, linewidth=2.0, zorder=5,
                   label="实测符号翻转" if k == 1 else None)
        offset = 13 if k % 2 else -18
        va = "bottom" if offset > 0 else "top"
        ax.annotate(f"edge {crossing['edge']}", xy=(xz, 0),
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", va=va, fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=ORANGE,
                                    linewidth=0.8, shrinkA=2, shrinkB=2))

    orient_cn = "水平" if profile["orientation"] == "horizontal" else "竖直"
    fixed = profile["fixed_index"]
    ratio_pred = profile["ratio_only_prediction"]
    over = profile["ratio_overcount"]
    under = profile["ratio_undercount"]
    ratio_error = f"多算 {over}" if over else (f"漏算 {under}" if under else "一致")
    ax.set_title(
        f"{orient_cn}剖面 {profile['fixed_axis']}={fixed}：实测 {measured} 次符号翻转；"
        f"频率比预判 {ratio_pred} 次（{ratio_error}），增量式预判 {profile['exact_increment_prediction']} 次",
        color=INK, fontsize=12, fontweight="semibold", pad=12,
    )
    ax.set_xlabel(f"沿线格点 t（采样间距 {profile['spacing_cells']:g} 格）",
                  color=SECONDARY)
    ax.set_ylabel("有符号 fBm 值", color=SECONDARY)
    ax.set_xlim(0, n - 1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    legend_loc = "upper left" if profile["orientation"] == "horizontal" else "upper right"
    ax.legend(loc=legend_loc, frameon=False, fontsize=9.5, ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def write_markdown_table(data: dict, out_path: Path) -> None:
    lines = [
        "| 剖面 | 最低频 $Z_0$ | 频率比 $r$ / 总倍率 $r^{K-1}$ | 频率比预判 | 实测 | 多算 | 漏算 | 增量式预判 | 边级多算/漏算 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in data["profiles"]:
        name = "水平 " + f"{p['fixed_axis']}={p['fixed_index']}" if p["orientation"] == "horizontal" else "竖直 " + f"{p['fixed_axis']}={p['fixed_index']}"
        lines.append(
            f"| {name} | {p['low_crossings']} | {p['adjacent_frequency_ratio']:g} / {p['highest_to_lowest_ratio']:g} | "
            f"{p['ratio_only_prediction']} | **{p['measured_sign_flips']}** | "
            f"{p['ratio_overcount']} | {p['ratio_undercount']} | {p['exact_increment_prediction']} | "
            f"{p['exact_edge_false_alarms']} / {p['exact_edge_misses']} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    json_path = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/zero_profile_data.json")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {"horizontal-center": "profile_horizontal.png",
             "vertical-center": "profile_vertical.png"}
    for p in data["profiles"]:
        out = out_dir / names[p["id"]]
        plot_profile(p, out)
        print(f"wrote {out}")

    md = out_dir / "zero_profile_table.md"
    write_markdown_table(data, md)
    print(f"wrote {md}")

    total_over = sum(p["ratio_overcount"] for p in data["profiles"])
    total_under = sum(p["ratio_undercount"] for p in data["profiles"])
    print(f"independent crosscheck passed; ratio over={total_over}, under={total_under}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
