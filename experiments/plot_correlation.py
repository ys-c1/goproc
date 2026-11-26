#!/usr/bin/env python
"""读取 seedexp 的 JSON 报告, 绘制两幅相关面积分组柱状图。

图一 artifacts/figures/correlated_area_weak.png:
    弱派生基线三档种子间隔 vs 独立判据(红虚线), 超判据即扩散不足。
图二 artifacts/figures/correlated_area_hash.png:
    散列扰动各 q 档、三档间隔 vs 同一独立判据, 并标注充分条件阈值。

配色取自 dataviz 技能参考调色板(已通过 validate_palette.js)。
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 系统仅有 DejaVu Sans 不含 CJK 字形, 显式注册本地 Noto Sans CJK SC。
font_manager.fontManager.addfont(
    "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf"
)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d9d8d3"

GAP_LABELS = ["间隔 1", "间隔 16", "间隔 256"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("相关面积（逐点高度差 < τ 的像素占比）", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(GAP_LABELS, color=INK, fontsize=11)


def criterion_line(ax, value, ha="right", xpos=2.46):
    ax.axhline(value, color=RED, linestyle="--", linewidth=1.8, zorder=3)
    ax.text(xpos, value + 0.018, f"独立判据 {value:.3f}", color=RED,
            fontsize=10, ha=ha, va="bottom",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))


def label_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK)


def fig_weak(report, out_path):
    crit = report["independence"]["correlatedArea"]
    values = [g["correlatedArea"] for g in report["weak"]["gaps"]]
    entries = [g["changedEntries"] for g in report["weak"]["gaps"]]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), facecolor=SURFACE)
    x = np.arange(3)
    bars = ax.bar(x, values, width=0.56, color=BLUE, zorder=2)
    label_bars(ax, bars)
    criterion_line(ax, crit, ha="left", xpos=-0.28)
    style_axes(ax)
    # 每档在柱体下方留白处注明实际改动条目数, 表明相关面积由改动条目数决定。
    for xi, e in zip(x, entries):
        ax.text(xi, 0.04, f"改动 {e:.1f}/256 条目", ha="center",
                va="bottom", fontsize=9, color=INK,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.set_title("弱派生基线：相邻种子只改动少数置换条目，相关面积远超独立判据",
                 color=INK, fontsize=12.5, pad=12)
    fig.text(0.01, 0.01,
             "蓝柱为弱派生（固定基础置换 + 顶部 k 次种子交换）。"
             f"τ={report['config']['tau']}，每档 {report['config']['trials']} 对蒙特卡洛；"
             "三档相关面积均高于独立判据 ⇒ 扩散不足，地形可视作同一地点。",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_hash(report, out_path):
    crit = report["independence"]["correlatedArea"]
    q_levels = report["config"]["qLevels"]
    threshold = report["diffusionThresholdQ"]

    fig, ax = plt.subplots(figsize=(9.4, 5.6), facecolor=SURFACE)
    width = 0.16
    series_colors = [BLUE, ORANGE, AQUA, "#6250d6", "#e87ba4"]
    x = np.arange(3)
    for idx, scheme in enumerate(report["hash"]):
        q = scheme["q"]
        values = [g["correlatedArea"] for g in scheme["gaps"]]
        offset = (idx - 2) * width
        tag = f"q={q:g}" + ("  ≥阈值" if q >= threshold else "  <阈值")
        bars = ax.bar(x + offset, values, width=width * 0.92,
                      color=series_colors[idx], zorder=2, label=tag)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7.6, color=INK)
    criterion_line(ax, crit, ha="left", xpos=-0.34)
    style_axes(ax)
    ax.set_title("散列扰动：每条目扰动概率 q 超过阈值后，相关面积回落到独立判据",
                 color=INK, fontsize=12.5, pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=1,
              title="每条目扰动概率", title_fontsize=9.5)
    fig.text(0.01, 0.01,
             f"充分条件：q > 1 − 1/256 = {threshold:.4f}（每条目未被扰动概率小于 1/256）；"
             "q 达阈值的两档与独立判据重合，间隔 1/16/256 无差异。",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    report_path, fig1, fig2 = sys.argv[1], sys.argv[2], sys.argv[3]
    report = load(report_path)
    fig_weak(report, fig1)
    fig_hash(report, fig2)
    print("wrote", fig1, "and", fig2)


if __name__ == "__main__":
    main()
