#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 extrema_analysis 的结果，生成全部标记图、统计图与 REPORT.md。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

from extrema_analysis import (
    REPO, make_perm, value_layer, extrema_masks, combined_mask,
    union_prediction,
)

OUT = REPO / "artifacts" / "extrema"
SEED = 7
N = 512

# 品牌/校验过的色板（dataviz reference palette light steps）
BLUE = "#2a78d6"
BLUE_D = "#1c5cab"
ORANGE = "#eb6834"
ORANGE_D = "#c44f1f"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d8d6d0"

FONT_PATH = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.ttf"
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def load_csv(name):
    with (OUT / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def seed7_bundle():
    perm = make_perm(SEED)
    fields, masks, maxs, mins = [], [], [], []
    comp = np.zeros((N, N))
    amp = 1.0
    norm = 0.0
    for k in range(8):
        h = value_layer(perm, N, 0.02 * 2.0 ** k)
        mx, mn = extrema_masks(h)
        fields.append(h)
        maxs.append(mx)
        mins.append(mn)
        masks.append(mx | mn)
        comp += h * amp
        norm += amp
        amp *= 0.5
    comp /= norm
    return fields, masks, maxs, mins, comp


def scatter_extrema(ax, mx, mn, n_total):
    """mx/mn 为 (N-2)^2 掩膜，坐标偏移 +1。"""
    size = 20 if n_total <= 300 else (9 if n_total <= 3000 else 2.2)
    yy, xx = np.nonzero(mx)
    ax.scatter(xx + 1, yy + 1, marker="^", s=size, c=ORANGE_D,
               linewidths=0, zorder=3, label=f"极大 {mx.sum()}")
    yy, xx = np.nonzero(mn)
    ax.scatter(xx + 1, yy + 1, marker="v", s=size, c=BLUE_D,
               linewidths=0, zorder=3, label=f"极小 {mn.sum()}")


def style_map_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
        sp.set_linewidth(0.8)


def hexrgb(hx):
    hx = hx.lstrip("#")
    return np.array([int(hx[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def gray_base(field, lo=0.84, span=0.14):
    """把 [-1,1] 场铺成浅灰 RGB 底，给彩色标记留出对比。"""
    g = lo + span * np.clip((field + 1) / 2, 0, 1)
    return np.repeat(g[..., None], 3, axis=2)


def paint_layer(field, mx, mn):
    """像素级精确着色：极大深橙、极小深蓝，避免 scatter 面积扭曲。"""
    rgb = gray_base(field)
    mf = np.zeros((N, N), bool)
    nf = np.zeros((N, N), bool)
    mf[1:-1, 1:-1] = mx
    nf[1:-1, 1:-1] = mn
    rgb[mf] = hexrgb(ORANGE_D)
    rgb[nf] = hexrgb(BLUE_D)
    return rgb


def paint_union(comp, mult):
    rgb = gray_base(comp, lo=0.88, span=0.10)
    rgb[mult == 1] = hexrgb(BLUE)
    rgb[mult >= 2] = hexrgb(ORANGE)
    return rgb


def save_layer_maps(fields, maxs, mins):
    for K in (4, 8):
        for k in range(K):
            n_total = int((maxs[k] | mins[k]).sum())
            fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
            ax.imshow(paint_layer(fields[k], maxs[k], mins[k]),
                      interpolation="nearest")
            if n_total <= 3000:  # 稀疏层叠加方向三角，密层保持像素真实比例
                scatter_extrema(ax, maxs[k], mins[k], n_total)
            ax.set_xlim(0, N - 1)
            ax.set_ylim(N - 1, 0)
            ax.set_title(f"K={K} 第 {k} 层（octave {k}，f={0.02 * 2**k:.3f}）"
                         f"  极值 {n_total}", fontsize=12)
            if n_total <= 3000:
                ax.legend(loc="upper right", fontsize=10, framealpha=0.9,
                          markerscale=1.6)
            style_map_ax(ax)
            fig.tight_layout()
            p = OUT / f"markers_K{K}_layer{k}_seed{SEED}.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            print("wrote", p.name)


def union_arrays(masks, K):
    mult = np.zeros((N, N), dtype=np.uint8)
    for k in range(K):
        mult[1:-1, 1:-1] += masks[k].astype(np.uint8)
    return mult


def save_union_maps(comp, masks):
    for K in (4, 8):
        mult = union_arrays(masks, K)
        once = mult == 1
        multi = mult >= 2
        fig, ax = plt.subplots(figsize=(7.6, 7.6), dpi=150)
        ax.imshow(paint_union(comp, mult), interpolation="nearest")
        if multi.sum() < 2000:  # K=4 重叠点少，放大加白边便于定位；K=8 保持像素比例
            yy, xx = np.nonzero(multi)
            ax.scatter(xx, yy, s=16, c=ORANGE, linewidths=0.5,
                       edgecolors="white", zorder=3)
        # 代理图例句柄（RGB 像素本身不进 legend）
        ax.scatter([], [], s=18, c=BLUE, label=f"仅单层标记 {once.sum()}")
        ax.scatter([], [], s=42 if multi.sum() < 2000 else 18, c=ORANGE,
                   edgecolors="white", linewidths=0.5,
                   label=f"≥2 层重叠 {multi.sum()}")
        ax.set_xlim(0, N - 1)
        ax.set_ylim(N - 1, 0)
        ax.set_title(f"K={K} 极值标记并集（seed={SEED}）："
                     f"不同像素 {int((mult >= 1).sum())}，重复计数修正 "
                     f"{int(mult.sum() - (mult >= 1).sum())}"
                     f"（≥2 层像素 {int(multi.sum())}）",
                     fontsize=12)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.92,
                  markerscale=1.4)
        style_map_ax(ax)
        fig.tight_layout()
        p = OUT / f"markers_K{K}_union_seed{SEED}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p.name)


def _layer_panel(ax, fields, maxs, mins, k):
    n = int((maxs[k] | mins[k]).sum())
    ax.imshow(paint_layer(fields[k], maxs[k], mins[k]),
              interpolation="nearest")
    if n <= 3000:
        scatter_extrema(ax, maxs[k], mins[k], n)
    ax.set_xlim(0, N - 1)
    ax.set_ylim(N - 1, 0)
    ax.set_title(f"第 {k} 层 f={0.02 * 2**k:.3f} · {n} 点", fontsize=10.5)
    style_map_ax(ax)


def _crop_around(mult, half=64):
    """选 (2half)^2 窗内 ≥2 层重叠像素最多的窗口。"""
    ov = (mult >= 2).astype(np.int64)
    S = np.pad(ov, 1).cumsum(0).cumsum(1)
    w = 2 * half
    best, score = (N // 2, N // 2), -1
    for y0 in range(0, N - w + 1, 8):
        for x0 in range(0, N - w + 1, 8):
            cnt = (S[y0 + w, x0 + w] - S[y0, x0 + w]
                   - S[y0 + w, x0] + S[y0, x0])
            if cnt > score:
                score, best = cnt, (y0 + half, x0 + half)
    cy, cx = best
    cy = int(np.clip(cy, half, N - half))
    cx = int(np.clip(cx, half, N - half))
    return cx - half, cx + half, cy - half, cy + half


def save_overviews(fields, maxs, mins, masks, comp):
    for K in (4, 8):
        nrows = 3 if K == 8 else 2
        fig = plt.figure(figsize=(16, 4.1 * nrows + 0.4), dpi=140)
        gs = fig.add_gridspec(nrows, 4, hspace=0.18, wspace=0.06)
        if K == 4:
            for k in range(4):
                ax = fig.add_subplot(gs[0, k])
                _layer_panel(ax, fields, maxs, mins, k)
            bottom = 1
        else:
            for k in range(8):
                ax = fig.add_subplot(gs[k // 4, k % 4])
                _layer_panel(ax, fields, maxs, mins, k)
            bottom = 2
        mult = union_arrays(masks, K)
        union = mult >= 1
        overlap = mult >= 2

        ax = fig.add_subplot(gs[bottom, 0])
        ax.imshow(comp, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"fBm 叠加场（{K} 层振幅归一）", fontsize=10.5)
        style_map_ax(ax)

        ax = fig.add_subplot(gs[bottom, 1])
        ax.imshow(paint_union(comp, mult), interpolation="nearest")
        if overlap.sum() < 2000:
            yy, xx = np.nonzero(overlap)
            ax.scatter(xx, yy, s=10, c=ORANGE, edgecolors="white",
                       linewidths=0.4, zorder=3)
        ax.set_title(f"标记并集 · {union.sum()} 点（橙=重叠）", fontsize=10.5)
        style_map_ax(ax)

        ax = fig.add_subplot(gs[bottom, 2])
        ov_rgb = np.ones((N, N, 3))
        ov_rgb[overlap] = hexrgb(ORANGE_D)
        ax.imshow(ov_rgb, interpolation="nearest")
        if overlap.sum() < 2000:
            yy, xx = np.nonzero(overlap)
            ax.scatter(xx, yy, s=8, c=ORANGE_D, linewidths=0)
        ax.set_title(f"重叠像素（≥2 层）· {overlap.sum()} 点", fontsize=10.5)
        style_map_ax(ax)

        x0, x1, y0, y1 = _crop_around(mult)
        ax = fig.add_subplot(gs[bottom, 3])
        zoom_rgb = paint_union(comp, mult)[y0:y1, x0:x1]
        ax.imshow(zoom_rgb, interpolation="nearest")
        ax.set_title(f"最高多重叠区放大 {x1 - x0}×{y1 - y0}（像素方块）",
                     fontsize=10.5)
        style_map_ax(ax)

        fig.suptitle(f"{K} 层 fBm 极值标记总览 · seed={SEED} · N={N}",
                     fontsize=15, y=0.995 if K == 8 else 1.0)
        p = OUT / f"overview_K{K}_seed{SEED}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p.name)


def save_independence_heatmap(summary):
    P = np.full((8, 8), np.nan)
    Z = np.full((8, 8), np.nan)
    for key, v in summary["pairs"].items():
        i, j = map(int, key.split("-"))
        P[i, j] = v["ratio_mean"]
        zm = v["z_mean"]
        Z[i, j] = 12 if zm is None else min(zm, 12)  # 截断显示，数值在标注里
    logP = np.log10(P)
    div_cmap = LinearSegmentedColormap.from_list(
        "blue_surface_orange", [BLUE, SURFACE, ORANGE])
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=150)
    for ax, M, title, fmt, norm in [
        (axes[0], logP, "配对重叠比 Q_obs / (n_i n_j / M) 的 12 种子均值（log10）",
         None, TwoSlopeNorm(vmin=-0.12, vcenter=0, vmax=2.8)),
        (axes[1], Z, "环形平移零假设平均 z 分数（显示截断于 +12）",
         "{:.1f}", TwoSlopeNorm(vmin=-3, vcenter=0, vmax=12)),
    ]:
        im = ax.imshow(M, cmap=div_cmap, norm=norm)
        for i in range(8):
            for j in range(8):
                if i < j:
                    if title.startswith("配对"):
                        raw = P[i, j]
                        txt = f"{raw:.2f}" if raw < 10 else f"{raw:.0f}"
                    else:
                        zraw = summary["pairs"][f"{i}-{j}"]["z_mean"]
                        txt = "∞" if zraw is None else f"{zraw:.1f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                            color=INK)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels([f"层 {k}" for k in range(8)])
        ax.set_yticklabels([f"层 {k}" for k in range(8)])
        ax.set_title(title, fontsize=11.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=9)
    fig.suptitle("各层极值空间独立性检验：右下高密度层块 ≈ 独立（比值≈1, |z|≲2）；"
                 "左上粗层因倍频晶格锚定显著重合", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT / "fig_independence_heatmap.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.name)


def save_prediction_figure(rows):
    seeds = sorted({int(r["seed"]) for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.6), dpi=150,
                             gridspec_kw={"height_ratios": [2.1, 1]})
    x = np.arange(len(seeds))
    w = 0.38
    for col, K in enumerate((4, 8)):
        rr = [next(r for r in rows if int(r["seed"]) == s and int(r["K"]) == K)
              for s in seeds]
        pred = np.array([float(r["pred_union_plugin"]) for r in rr])
        act = np.array([int(r["actual_union"]) for r in rr])
        prior = np.array([float(r["pred_union_prior"]) for r in rr])
        rel = (pred - act) / act * 100
        relp = (prior - act) / act * 100
        mape = np.mean(np.abs(rel))
        mapep = np.mean(np.abs(relp))

        ax = axes[0, col]
        ax.bar(x - w / 2, pred, w, label="独立公式预判 Û", color=BLUE)
        ax.bar(x + w / 2, act, w, label="实测并集 U", color=INK2)
        ax.axhline(prior[0], color=ORANGE_D, ls="--", lw=1.6,
                   label="校准密度先验预判")
        ax.set_xticks(x); ax.set_xticklabels([str(s) for s in seeds])
        ax.set_ylabel("极值点数（内部像素）")
        ax.set_ylim(0, max(pred.max(), act.max(), prior[0]) * 1.15)
        ax.set_title(f"K={K}：预判 vs 实测（插件 MAPE {mape:.2f}%，"
                     f"先验 MAPE {mapep:.2f}%）", fontsize=11.5)
        ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)

        ax2 = axes[1, col]
        ax2.bar(x - w / 2, rel, w, color=BLUE, label="插件相对偏差")
        ax2.scatter(x + w / 2, relp, marker="D", s=34, color=ORANGE_D,
                    zorder=3, label="先验相对偏差")
        ax2.axhline(0, color=INK, lw=1)
        ax2.set_xticks(x); ax2.set_xticklabels([str(s) for s in seeds])
        ax2.set_ylabel("相对偏差 %")
        ax2.set_xlabel("种子 seed")
        ax2.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
        ax2.set_axisbelow(True)
        if col == 0:
            ax2.legend(fontsize=9, loc="lower right")
    fig.suptitle("总极值数预判与实测逐种子比对（N=512，M=260100）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = OUT / "fig_pred_vs_actual.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.name)


def save_correction_figure(rows, summary):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150,
                             gridspec_kw={"width_ratios": [1.05, 1, 1]})
    # layer densities
    ax = axes[0]
    dp = load_csv("layer_densities.csv")
    means = []
    for k in range(8):
        vals = [float(r["density"]) for r in dp if int(r["layer"]) == k]
        means.append(np.mean(vals))
    means = np.array(means)
    ax.bar(range(8), means * 100, color=BLUE)
    for k, v in enumerate(means):
        ax.text(k, v * 100 * 1.12, f"{v * 100:.2f}%", ha="center",
                fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xticks(range(8)); ax.set_xlabel("层")
    ax.set_ylabel("平均极值密度 %（log）")
    ax.set_title("各层极值密度（12 种子均值）", fontsize=11.5)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)

    for col, K in enumerate((4, 8), start=1):
        ax = axes[col]
        rr = [r for r in rows if int(r["K"]) == K]
        obs = np.array([int(r["corr_obs"]) for r in rr], dtype=float)
        ind = np.array([float(r["corr_ind"]) for r in rr])
        xs = np.random.default_rng(0).normal(0, 0.05, len(obs))
        ax.scatter(xs, obs, s=26, color=INK2, alpha=0.7, zorder=3,
                   label="各种子实测 C_obs")
        ax.bar([1], [obs.mean()], width=0.52, color=INK2, alpha=0.85)
        ax.bar([2], [ind.mean()], width=0.52, color=BLUE)
        ax.scatter(np.full(len(ind), 2) + xs * 0.4, ind, s=18,
                   color="white", edgecolors=BLUE_D, zorder=4)
        for xpos, val in ((1, obs.mean()), (2, ind.mean())):
            ax.text(xpos, val, f" {val:,.0f} ", ha="center",
                    va="bottom", fontsize=10.5, fontweight="bold", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=1.2, alpha=0.85))
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["实测重叠修正", "独立公式修正"])
        ax.set_xlim(0.4, 2.6)
        ax.set_ylabel("点数 / 512² 图")
        ax.set_title(f"K={K}：修正量级 实测 {obs.mean():,.0f}"
                     f"（{obs.min():.0f}–{obs.max():.0f}）", fontsize=11.5)
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    fig.suptitle("重叠修正：Σn_k − |∪E_k| 的实测值与独立容斥估计", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "fig_correction_magnitude.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.name)


def n128_check():
    """仓库默认 Size=128 口径的快速复核。"""
    n = 128
    m2 = (n - 2) ** 2
    out_rows = []
    for seed in range(1, 13):
        perm = make_perm(seed)
        masks, ns = [], []
        for k in range(8):
            e = combined_mask(value_layer(perm, n, 0.02 * 2.0 ** k))
            masks.append(e); ns.append(int(e.sum()))
        for K in (4, 8):
            U = np.zeros_like(masks[0]); S = 0
            for k in range(K):
                U |= masks[k]; S += ns[k]
            u = int(U.sum())
            p = np.array(ns[:K]) / m2
            pred = m2 * union_prediction(p)
            out_rows.append(dict(seed=seed, K=K, sum_layer=S,
                                 corr_obs=S - u, pred_union=f"{pred:.1f}",
                                 actual_union=u,
                                 relerr_pct=f"{(pred-u)/u*100:+.2f}"))
    p = OUT / "comparison_N128.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    agg = {}
    for K in (4, 8):
        rr = [r for r in out_rows if r["K"] == K]
        agg[K] = dict(
            corr_mean=float(np.mean([r["corr_obs"] for r in rr])),
            union_mean=float(np.mean([r["actual_union"] for r in rr])),
            mape=float(np.mean([abs(float(r["relerr_pct"])) for r in rr])))
    print("wrote", p.name)
    return agg, out_rows


def build_report(summary, rows, n128_agg):
    s4, s8 = summary["K4"], summary["K8"]

    def table_rows(K):
        trs = []
        for r in [r for r in rows if int(r["K"]) == K]:
            trs.append(
                f"| {r['seed']} | {r['sum_layer']} | {r['corr_obs']} | "
                f"{float(r['corr_ind']):.1f} | {float(r['pred_union_plugin']):.0f} | "
                f"{r['actual_union']} | {float(r['residual_plugin']):+.0f} | "
                f"{float(r['relerr_plugin_pct']):+.2f}% | "
                f"{float(r['relerr_prior_pct']):+.2f}% |")
        return "\n".join(trs)

    dp = load_csv("layer_densities.csv")
    layer_tbl = ["| 层 k | 频率 f | 平均点数 n_k | 平均密度 p_k |",
                 "|---:|---:|---:|---:|"]
    for k in range(8):
        rs = [r for r in dp if int(r["layer"]) == k]
        nk = np.mean([int(r["n_total"]) for r in rs])
        pk = np.mean([float(r["density"]) for r in rs])
        layer_tbl.append(f"| {k} | {0.02*2**k:.3f} | {nk:,.0f} | {pk:.5f} |")
    layer_tbl = "\n".join(layer_tbl)

    # pair table selected
    pair_lines = ["| 层对 | 重叠比均值 | 标准差 | 平移 z 均值 |",
                  "|:--|---:|---:|---:|"]
    for key, v in summary["pairs"].items():
        zm = v["z_mean"]
        zs = "∞（稀疏掩膜平移零分布标准差为 0）" if zm is None else f"{zm:.1f}"
        pair_lines.append(
            f"| {key.replace('-', '–')} | {v['ratio_mean']:.2f} | "
            f"{v['ratio_sd']:.2f} | {zs} |")
    pair_tbl = "\n".join(pair_lines)

    md = f"""# 多层极值点：空间独立性、重叠修正与总数预判

生成时间：2026-11-27 ｜ 主口径 N={N}（内部有效像素 M={summary['M']:,}=(N−2)²），
频率 Frequency=0.02、lacunarity=2、gain=0.5，与 `api.DefaultOptions` 一致（仅放大尺寸）。
评估种子 1–12；先验密度校准种子 1001–1008。基准种子 7（README 示例）。

## 0. 结论先行

- **密度公式成立**：把同一像素被多层标记视为重复计数，总极值（并集）密度恒有
  `ρ_total = Σ p_k − Δ`，其中 Δ 是重叠修正密度；独立假设下
  `ρ_total = 1 − Π_k(1−p_k)`，`Δ = Σp_k − 1 + Π(1−p_k)`，
  领头项为二阶容斥 `C₂/M = Σ_{{i<j}} p_i p_j`。
- **“近似独立”要分层看（实测，不回避反例）**：主导总计数的高密度层（k=5,6,7，
  密度 12.8%/21.7%/23.6%）配对重叠比 0.94–1.03、环形平移 |z|≲2，与空间独立完全一致；
  间隔 ≥3 个倍频的层对也基本独立（比值 0.88–1.10）。但前 4 个稀疏粗层的**逐像素精确重合**
  显著高于均匀 Bernoulli 零假设（相邻层对比值 630、157、22、7.2）。已查明这不是随机相关，
  而是 f₀=0.02=1/50 与倍频形成的**整数对齐嵌套晶格 + quintic fade 在晶格点零导数**的锚定：
  seed 7 上第 0 层 100% 极值像素位于 50 px 晶格顶点、第 1 层 100% 位于 25 px 顶点，
  二者重合点 100% 落在共享的 50 px 顶点。加亚像素相位扰动后这类重合立即跌到随机水平。
- **预判可用，且层数越多相对偏差越小**：独立公式对并集总数的插件预判，
  K=4 平均相对偏差 **{s4['relerr_plugin_mean_pct']:+.2f}%**（最大绝对 {s4['relerr_plugin_absmax_pct']:.2f}%，
  系统性偏高约 {s4['residual_plugin_mean_abs']:.0f} 点）；
  K=8 平均 **{s8['relerr_plugin_mean_pct']:+.2f}%**（最大绝对 {s8['relerr_plugin_absmax_pct']:.2f}%，
  平均绝对残差 {s8['residual_plugin_mean_abs']:.0f} 点 / 约 {s8['actual_union_mean']:,.0f} 点）。
  用独立校准种子的层密度做**真正先验预判**，K=8 的 MAPE 为
  {np.mean([abs(float(r['relerr_prior_pct'])) for r in rows if int(r['K']) == 8]):.2f}%、
  K=4 为 {np.mean([abs(float(r['relerr_prior_pct'])) for r in rows if int(r['K']) == 4]):.2f}%。
  **相对偏差不随层数扩大，反而从约 8% 收敛到不足 1%**；绝对残差从约 270 点增至约 652 点
  （量级仍小，且总数扩大了约 40 倍）。
- **重叠修正量级（给数）**：
  - **K=4：实测 Δ 计数 ≈ {s4['corr_obs_mean']:.0f} 点/图（12 种子均值，区间 {s4['corr_obs_min']}–{s4['corr_obs_max']}），
    密度 ≈ {s4['corr_obs_mean']/summary['M']:.2e}，量级 10² 个点（面积占比约 1.1×10⁻³）**；
    独立乘积公式只给出 {s4['corr_ind_mean']:.1f} 点，粗层晶格锚定使真实修正大约 30 倍。
  - **K=8：实测 Δ 计数 ≈ {s8['corr_obs_mean']:,.0f} 点/图（区间 {s8['corr_obs_min']:,}–{s8['corr_obs_max']:,}），
    密度 ≈ {s8['corr_obs_mean']/summary['M']:.3f}，量级 10⁴ 个点（约占全图 12.9%）**；
    独立公式给出 {s8['corr_ind_mean']:,.0f} 点，为实测的 {s8['corr_ratio_ind_over_obs']*100:.1f}%。
- 默认尺寸 N=128 的同口径复核（`comparison_N128.csv`）：K=4 修正 ≈ {n128_agg[4]['corr_mean']:.1f} 点、
  并集 ≈ {n128_agg[4]['union_mean']:.0f}、MAPE {n128_agg[4]['mape']:.2f}%；
  K=8 修正 ≈ {n128_agg[8]['corr_mean']:.0f} 点、并集 ≈ {n128_agg[8]['union_mean']:.0f}、
  MAPE {n128_agg[8]['mape']:.2f}%。密度结论与尺寸无关，绝对点数按 (126/510)²≈0.061 缩放。

## 1. 对象、层与极值定义

- “层”即 `internal/fbm/fbm.go` 中同一 `noise.Source` 的各个 octave 项
  `Value(x·f_k, y·f_k)`，f_k=0.02·2^k；正振幅 0.5^k 不改变极值位置。
- 极值点：内部像素 1≤x,y≤N−2 上，场值严格大于（极大）或小于（极小）全部 8 个邻域；
  极大、极小合并为极值掩膜 E_k。该口径与 `scipy.ndimage.maximum/minimum_filter`
  非平台情形逐像素一致（已验证）。
- “总极值数”= 掩膜并集 U=|∪ E_k|，即标记图上不同像素的个数；Σn_k−U 即重叠修正
  （一个像素被 r 层标记会在 Σn_k 中被数 r 次、在 U 中数 1 次，贡献 r−1）。
  注意：**fBm 叠加场自身的极值不是这个量**。seed 7 上叠加场极值 K=4 为 1,370、
  K=8 为 8,138，而标记并集为 3,230 / 130,908；“密度相加−重叠修正”只对层标记并集成立。

## 2. 保真与可复现

- `analysis/extrema_analysis.py` 逐位复现 splitmix64→Fisher-Yates→perm[512]、
  quintic fade 与双线性插值；`go run ./cmd/noisedump` 导出 Go 侧 24²×8 层基准，
  Python 对拍**最大绝对误差 = {summary['go_max_abs_diff']}**（精确为 0）。
- 复现：`python3 analysis/extrema_analysis.py && python3 analysis/make_figures.py`。

## 3. 空间独立性检验

三种统计互相印证（原始数据 `independence_pairs.csv`、`shift_test.csv`）：

1. **配对重叠比** R_ij = |E_i∩E_j| / (n_i n_j/M)：独立均匀 Bernoulli 下期望 1。
2. **φ 相关系数**：二值掩膜 Pearson 相关，独立下期望 0（见 CSV）。
3. **环形平移零假设**：把 E_i 随机环形平移 {128} 次再与 E_j 求交，保持每个掩膜自身的
   空间聚簇结构；z=(Q_obs−mean Q_shift)/sd Q_shift。

### 3.1 12 种子配对统计

{pair_tbl}

解读：k≥5 的高密度块 R≈0.94–1.03、|z|≤2.9（多数 ≤1.9），不能拒绝独立；
粗邻接层对的偏离在第 3.2 节解释。**从密度权重看，K=8 时 k≥5 三层贡献了
Σn_k 的 77%，且几乎全部二阶重叠量由密度乘积项解释（修正估计达实测 98%），
所以“总量层面的近似独立”成立。**

### 3.2 粗层反例与晶格锚定（不粉饰）

R_01≈630 不是抽样误差（z=∞，12 个种子方向一致，标准差仅 82）。机制是确定性的：

- f₀=0.02 使 0 层晶格间隔恰为 50 px，各层间隔 50/2^k px，粗层晶格顶点全部对齐到
  更细层的晶格顶点；
- quintic fade 的一阶导数在晶格点为 0，晶格点天然是驻点，1 px 邻域内场几乎不动，
  于是离散极值像素被“钉”在晶格顶点；
- seed 7 实测：层 0 极值 100% 在 50-顶点、层 1 极值 100% 在 25-顶点，
  二者的 23 个重合点 100% 在共享 50-顶点。分类本身（该点是否为各自置换表的局部极值）
  仍近似独立抛币，但像素空间支撑集重合，使 n_i n_j/M 基线失效。
- 诊断：给细层加 7.3 个噪声单位的坐标相位后，同一置换表口径下 0–1 重叠由 23 降到 0
  （随机期望 0.03）；
  统一改半像素中心采样后相邻比降为 227/47/23/7.0/1.95/0.90/0.95，K=8 预判误差降到 +0.07%。

结论措辞：**各层极值并非在严格逐像素意义下全部独立；“近似空间独立”对高密度层、
大间隔层对以及总量密度预判成立，对稀疏粗邻接层不成立，偏差量级有界且已量化。**

![独立性热力图](fig_independence_heatmap.png)

## 4. 密度期望公式

设 X_k(x) 为层 k 的极值指示变量，p_k=E[X_k]=ρ_k。由容斥：

```
ρ_total = E[1−Π_k(1−X_k)]
        = Σ p_k − Σ_(i<j) E[X_i X_j] + Σ_(i<j<l) E[X_i X_j X_l] − …
```

记 Δ = Σp_k − ρ_total 为**重叠修正密度**。若各层空间近似独立，
E[X_i X_j]≈p_i p_j、高阶联合≈乘积，则：

```
ρ_total ≈ 1 − Π_(k<K)(1−p_k)
Δ       ≈ Σp_k − 1 + Π(1−p_k)
        = Σ_(i<j) p_i p_j − Σ_(i<j<l) p_i p_j p_l + …
```

点计数预判 `Û = M·ρ_total`。当 p_k 很小时 Δ≈C₂=Σ_(i<j) p_i p_j；
K=4 正是此情形（C₂≈9.2 点），但晶格锚定使实测 Δ≈279 点；
K=8 密度不小，必须保留完整乘积（C₂={float([r for r in rows if int(r['K'])==8][0]['pair_term_C2']):,.0f} 点，
净 Δ={s8['corr_ind_mean']:,.0f} 点，高阶容斥约 −4 千点）。

### 各层平均密度

{layer_tbl}

## 5. 预判与实测逐行比对（N={N}，M={summary['M']:,}）

列含义：S=Σn_k；C_obs=S−U 实测重叠修正；C_ind=S−Û_plugin 独立公式修正；
Û_plugin=M[1−Π(1−n_k/M)]；偏差=Û−U；末列为用校准种子平均密度的先验预判相对偏差。
CSV 另含 `pred_union_cal/relerr_cal_pct`（配对膨胀校准预判，见第 6 节）。

### K=4（四档层）

| seed | S | C_obs | C_ind | Û 预判 | U 实测 | 偏差 | 相对 | 先验相对 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows(4)}

### K=8（八档层）

| seed | S | C_obs | C_ind | Û 预判 | U 实测 | 偏差 | 相对 | 先验相对 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows(8)}

![预判vs实测](fig_pred_vs_actual.png)

![修正量级](fig_correction_magnitude.png)

## 6. 层数增加，偏差是否扩大？

- **相对偏差：不扩大，显著收敛。** 插件平均相对偏差 K=4 {s4['relerr_plugin_mean_pct']:+.2f}%
  → K=8 {s8['relerr_plugin_mean_pct']:+.2f}%；12 种子最大绝对偏差
  {s4['relerr_plugin_absmax_pct']:.2f}% → {s8['relerr_plugin_absmax_pct']:.2f}%。
  原因：新增的 k=4…7 层密度大、彼此独立（R≈1），它们贡献的重叠严格服从乘积模型；
  粗层晶格相关造成的数百点额外重合，在 K=8 约 {s8['corr_obs_mean']:,.0f} 次重叠修正中
  只占约 {abs(s8['residual_plugin_mean_abs'])/s8['corr_obs_mean']*100:.1f}%、
  在 {s8['actual_union_mean']:,.0f} 点总数中仅约 0.5%。
- **绝对残差：有所增大但远慢于规模。** 平均绝对残差 {s4['residual_plugin_mean_abs']:.0f}
  → {s8['residual_plugin_mean_abs']:.0f} 点（约 2.4 倍），而总数均值
  {s4['actual_union_mean']:,.0f} → {s8['actual_union_mean']:,.0f}（约 40 倍），
  故相对误差缩小。残差在所有种子上同为正（预判偏高），与“真实重叠多于独立假设”方向一致。
- **若要消除 K=4 的系统偏置**：用校准种子（1001–1008）估计的配对膨胀系数 R_ij 把二阶项改成
  C₂=Σ R_ij p_i p_j M（高阶仍按独立乘积），在未参与校准的 1–12 种子上，
  K=4 MAPE 由 8.40% 降到 **{s4['relerr_cal_mape_pct']:.2f}%**
  （最大绝对 {s4['relerr_cal_absmax_pct']:.2f}%）；
  但对 K=8 反而从 0.50% 略升到 {s8['relerr_cal_mape_pct']:.2f}%
  （高密度时只膨胀二阶项会扰动容斥高阶平衡），故主表仍以纯净独立公式为准，
  校准列 `pred_union_cal/relerr_cal_pct` 已写入 CSV。

## 7. 极值标记图档（seed=7，分别存档）

- 四层：`overview_K4_seed7.png`（总览）；逐层 `markers_K4_layer0_seed7.png` …
  `markers_K4_layer3_seed7.png`；并集 `markers_K4_union_seed7.png`。
- 八层：`overview_K8_seed7.png`；逐层 `markers_K8_layer0…7_seed7.png`；
  并集 `markers_K8_union_seed7.png`。
- 统计图：`fig_independence_heatmap.png`、`fig_pred_vs_actual.png`、
  `fig_correction_magnitude.png`。
- 数据：`comparison_by_seed.csv`（主表）、`layer_densities.csv`、
  `independence_pairs.csv`、`shift_test.csv`、`comparison_N128.csv`、`summary.json`。

标记说明：▲ 极大（深橙）、▼ 极小（深蓝）；并集图蓝点为仅一层标记、橙点为 ≥2 层重叠。

## 8. 口径与局限

- 只统计内部像素；边界 1 px 不参与，M=(N−2)²。
- 严格比较，平台/等值不标记；本噪声为连续 float，实测无影响（与 scipy 严格口径一致）。
- “独立”是经验近似而非数学定理：确定性 splitmix 置换不提供概率论独立，
  上述 R、φ、平移 z 与预判残差是其可检验证据。
- 点数随 N、Frequency 变化；跨尺寸比较请用密度。N≤4096、Octaves≤12 为 api 校验上限。
"""
    p = OUT / "REPORT.md"
    p.write_text(md, encoding="utf-8")
    print("wrote", p.name)


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    rows = load_csv("comparison_by_seed.csv")
    fields, masks, maxs, mins, comp = seed7_bundle()
    save_layer_maps(fields, maxs, mins)
    save_union_maps(comp, masks)
    save_overviews(fields, maxs, mins, masks, comp)
    save_independence_heatmap(summary)
    save_prediction_figure(rows)
    save_correction_figure(rows, summary)
    n128_agg, _ = n128_check()
    build_report(summary, rows, n128_agg)


if __name__ == "__main__":
    main()
