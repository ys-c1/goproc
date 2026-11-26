#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 lodreport 生成的 lod_data.json：

1. 用独立解析公式 E_l=(W/2^l+1)^2 复核 Go 端按三角形引用统计的有效顶点、
   实际比值与偏差（不共享 Go 代码）；
2. 为每组误差阈值渲染一张“层级切换图”PNG：
   上：有效顶点随阈值档的阶梯切换；
   左下：预判比值 4^d 与实际有效顶点比逐对对比；
   右下：体密度预判偏差与 5%（半成）判定线；
3. 额外写出每组一份 Markdown 比值表。

用法: python3 analysis/lod_charts.py [lod_data.json] [输出目录]
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

# ---------- 字体 ----------
FONT_PATH = "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf"
try:
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
except Exception:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 参考调色板（dataviz reference palette, light） ----------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def fmt_t(t) -> str:
    t = float(t)
    return "0" if t == 0 else f"{t:g}"


def independent_crosscheck(data: dict) -> None:
    """纯解析重算，逐层级、逐行核对 Go 结果。"""
    w = data["config"]["side_cells"]
    eff = {}
    for lv in data["levels"]:
        l = lv["level"]
        expect = (w // (2 ** l) + 1) ** 2
        eff[l] = expect
        assert lv["effective_vertices"] == expect, (l, lv["effective_vertices"], expect)
        assert lv["effective_vertices_crosscheck"] == expect
    for s in data["sets"]:
        for r in s["rows"]:
            i, j, d = r["fine_level"], r["coarse_level"], r["level_gap"]
            assert d == j - i
            actual = eff[i] / eff[j]
            pred = 4.0 ** d
            exact = ((w / 2 ** i + 1) / (w / 2 ** j + 1)) ** 2
            dev = abs(pred - actual) / actual * 100.0
            assert math.isclose(actual, r["actual_ratio"], rel_tol=1e-12)
            assert math.isclose(pred, r["predicted_ratio_bulk"], rel_tol=1e-12)
            assert math.isclose(exact, r["predicted_ratio_boundary_exact"], rel_tol=1e-9)
            assert math.isclose(dev, r["deviation_bulk_pct"], rel_tol=1e-9, abs_tol=1e-9)
            assert r["exceeds_5pct"] == (dev > 5.0)
    print(f"[crosscheck] W={w}: {len(data['levels'])} 级有效顶点、全部比值行独立复核通过")


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.yaxis.label.set_color(SECONDARY)
    ax.xaxis.label.set_color(SECONDARY)
    ax.title.set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def draw_switch(ax, data, s):
    levels = {lv["level"]: lv for lv in data["levels"]}
    sel = s["selections"]
    xs = list(range(len(sel)))
    ys = [levels[x["selected_level"]]["effective_vertices"] for x in sel]
    ls = [x["selected_level"] for x in sel]

    ax.step(xs, ys, where="post", color=BLUE, linewidth=2.2, zorder=3)
    ax.scatter(xs, ys, s=74, color=BLUE, edgecolor="white", linewidth=1.6, zorder=4)

    # 层级与顶点数直接放进 x 轴刻度（避免与切换徽标碰撞）
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"阈值 {fmt_t(x['threshold'])}\nL{l} · {y:,}" for x, l, y in zip(sel, ls, ys)],
        fontsize=9,
    )

    # 每次切换的实际/预判跌幅徽标，放在竖降段右侧空白带
    rowmap = {(r["fine_level"], r["coarse_level"]): r for r in s["rows"]}
    for k in range(len(xs) - 1):
        i, j = ls[k], ls[k + 1]
        r = rowmap.get((i, j))
        if r is None:
            continue
        ym = math.sqrt(ys[k] * ys[k + 1])
        last = (k + 1 == len(xs) - 1)
        ax.text(
            xs[k + 1] - 0.10 if last else xs[k + 1] + 0.10, ym,
            f"实际跌 {r['actual_jump_drop_pct']:.1f}%\n预判跌 {r['predicted_jump_drop_pct']:.1f}%",
            ha="right" if last else "left", va="center", fontsize=8, color=SECONDARY,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor=AXIS, linewidth=0.8),
            zorder=6,
        )

    ax.set_yscale("log")
    ax.set_xlim(-0.55, len(xs) - 1 + 0.55)
    ax.set_ylim(min(ys) / 5.0, max(ys) * 2.6)
    ax.set_ylabel("有效顶点数（对数轴）")
    ax.set_title("① 层级切换阶梯：有效顶点随误差阈值档放宽而突降",
                 fontsize=13, loc="left", pad=12, color=INK)
    style_ax(ax)


def draw_ratio_bars(ax, s):
    rows = s["rows"]
    n = len(rows)
    x = list(range(n))
    w = 0.38
    pred = [r["predicted_ratio_bulk"] for r in rows]
    act = [r["actual_ratio"] for r in rows]
    kinds = [r["row_kind"] for r in rows]

    bp = ax.bar([i - w / 2 for i in x], pred, width=w, color=BLUE, zorder=3,
                label="预判 4^d（格距平方律）")
    ba = ax.bar([i + w / 2 for i in x], act, width=w, color=ORANGE, zorder=3,
                label="实际有效顶点比")
    for k, kind in enumerate(kinds):
        if kind == "reference_adjacent":
            bp[k].set_alpha(0.28)
            ba[k].set_alpha(0.28)

    ax.set_yscale("log")
    ax.set_ylim(1, max(pred) * 2.5)
    for i, r in enumerate(rows):
        ax.text(i - w / 2, pred[i] * 1.07, f"{pred[i]:g}", ha="center", va="bottom",
                fontsize=7.5, color=BLUE, rotation=90)
        ax.text(i + w / 2, act[i] * 1.07, f"{act[i]:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=ORANGE, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['pair']}\nδ {r['deviation_bulk_pct']:.1f}%" for r in rows],
        rotation=32, ha="right", fontsize=8.5,
    )
    for tick, r in zip(ax.get_xticklabels(), rows):
        tick.set_color(CRITICAL if r["exceeds_5pct"] else GOOD)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("细级/粗级 顶点数比（对数轴）")
    ax.set_title("② 逐层级对：预判比值 vs 实际比值（刻度色＝δ 是否超 5%）",
                 fontsize=13, loc="left", pad=12, color=INK)
    style_ax(ax)
    handles = [
        Patch(facecolor=BLUE, label="预判 4^d（格距平方律）"),
        Patch(facecolor=ORANGE, label="实际有效顶点比"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=9,
                    frameon=True, framealpha=0.96)
    leg.get_frame().set_edgecolor(AXIS)
    if any(k == "reference_adjacent" for k in kinds):
        ax.text(0.0, -0.34,
                "浅色柱＝相邻级基线对（供完整比对，并非本组阈值的直接切换对）",
                transform=ax.transAxes, fontsize=8.5, color=MUTED)
    else:
        ax.text(0.0, -0.27, "x 轴刻度绿色＝δ≤5%，红色＝δ>5%（需说明的层级对）",
                transform=ax.transAxes, fontsize=8.5, color=MUTED)


def draw_deviation(ax, s):
    rows = s["rows"]
    x = list(range(len(rows)))
    dev = [r["deviation_bulk_pct"] for r in rows]
    cols = [CRITICAL if r["exceeds_5pct"] else GOOD for r in rows]
    bars = ax.bar(x, dev, color=cols, zorder=3, width=0.66)
    for k, r in enumerate(rows):
        if r["row_kind"] == "reference_adjacent":
            bars[k].set_alpha(0.38)
        ax.text(x[k], dev[k] + max(dev) * 0.02, f"{dev[k]:.1f}",
                ha="center", va="bottom", fontsize=8)

    line5 = ax.axhline(5.0, color=CRITICAL, linestyle=(0, (5, 3)), linewidth=1.6, zorder=4)
    leg3 = ax.legend(handles=[line5], labels=["半成线 5%"], loc="upper left",
                     fontsize=9.5, frameon=True, framealpha=0.96)
    leg3.get_frame().set_edgecolor(AXIS)
    ax.set_xticks(x)
    ax.set_xticklabels([r["pair"] for r in rows], rotation=32, ha="right", fontsize=8.5)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylabel("偏差 δ（%）")
    ax.set_ylim(0, max(dev) * 1.16)
    ax.set_title("③ 预判偏差与 5% 判定（超限层级对在报告中说明）",
                 fontsize=13, loc="left", pad=12, color=INK)
    style_ax(ax)


def write_markdown_table(s: dict, path: Path) -> None:
    lines = [
        f"# 误差阈值组 {s['id']}《{s['name']}》比值表",
        "",
        f"- 切换方式：{s['switch_mode']}",
        f"- 阈值档（归一化高程误差）：{', '.join(fmt_t(t) for t in s['thresholds'])}",
        f"- 选中层级链：{' → '.join('L%d' % l for l in s['chain_levels'])}",
        "- 比值方向：细级 i / 粗级 j；预判体密度律 = 4^d（d=j−i）；"
        "边界修正式 = [(W/2^i+1)/(W/2^j+1)]²；δ=|预判−实际|/实际×100%。",
        "- 有效顶点＝被三角形引用过的不同顶点。",
        "",
        "| 层级对 | 类型 | d | 格距比 | 预判 4^d | 边界修正 | 实际比值 | 有效顶点(细→粗) | δ体密度(%) | δ修正(%) | 超5% | 预判跌幅/实际跌幅(%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    kind_cn = {
        "switch_adjacent": "直接切换(相邻)",
        "switch_skip": "直接切换(跨级)",
        "reference_adjacent": "相邻基线",
    }
    for r in s["rows"]:
        lines.append(
            f"| {r['pair']} | {kind_cn.get(r['row_kind'], r['row_kind'])} | {r['level_gap']} "
            f"| {r['spacing_ratio']:g} | {r['predicted_ratio_bulk']:g} "
            f"| {r['predicted_ratio_boundary_exact']:.6f} | {r['actual_ratio']:.6f} "
            f"| {r['eff_fine']:,}→{r['eff_coarse']:,} "
            f"| {r['deviation_bulk_pct']:.3f} | {r['deviation_exact_pct']:.1e} "
            f"| {'是' if r['exceeds_5pct'] else '否'} "
            f"| {r['predicted_jump_drop_pct']:.2f}/{r['actual_jump_drop_pct']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table] {path}")


def render(data: dict, s: dict, out_dir: Path) -> None:
    plt.rcParams["figure.facecolor"] = SURFACE
    fig = plt.figure(figsize=(16, 11.6), dpi=150, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.30, 1.0], hspace=0.52, wspace=0.18,
                          left=0.055, right=0.985, top=0.845, bottom=0.115)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    draw_switch(ax1, data, s)
    draw_ratio_bars(ax2, s)
    draw_deviation(ax3, s)

    c = data["config"]
    n_bad = sum(1 for r in s["rows"]
                if r["exceeds_5pct"] and r["row_kind"] != "reference_adjacent")
    n_bad_all = sum(1 for r in s["rows"] if r["exceeds_5pct"])
    fig.text(0.055, 0.965,
             f"误差阈值组 {s['id']}《{s['name']}》  LOD 层级切换与顶点数突变预判",
             fontsize=19, color=INK, ha="left", va="top")
    fig.text(0.055, 0.925,
             f"{s['switch_mode']}。地形 {c['size']}×{c['size']} 高度图，"
             f"W={c['side_cells']} 个格距区间，种子 {c['seed']}，frequency={c['frequency']}。",
             fontsize=10.5, color=SECONDARY, ha="left", va="top")
    fig.text(0.055, 0.902,
             f"计数口径：有效顶点＝被三角形索引引用过的不同顶点；共享顶点池恒为 "
             f"{c['size']*c['size']:,}（若误按池长计数会得到“跳变为零”的假象）。"
             f"本组直接切换对超 5% 者 {n_bad} 对；表内相邻基线合计 {n_bad_all} 对超限。",
             fontsize=10.5, color=SECONDARY, ha="left", va="top")

    out_path = out_dir / s["chart_file"]
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[chart] {out_path}")
    write_markdown_table(s, out_dir / f"lod_ratio_table_{s['id'].lower()}.md")


def main() -> int:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/lod_data.json")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else json_path.parent
    data = json.loads(json_path.read_text(encoding="utf-8"))
    independent_crosscheck(data)
    for s in data["sets"]:
        render(data, s, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
