#!/usr/bin/env python3
"""Curvature-budget prediction for blockwise terrain triangulation.

The predictor uses only second differences of the height field:

    Dxx = z[x+1] - 2 z[x] + z[x-1]
    Dyy = z[y+1] - 2 z[y] + z[y-1]
    kappa_B = mean(max(abs(Dxx), abs(Dyy))) over the interior of block B

With block edge L and n subdivisions per edge, s = L/n is the triangle scale.
The discrete curvature-scale law used here is e_B ~= kappa_B * s, hence

    n_pred = ceil(kappa_B * L / epsilon)
    T_pred = 2 * n_pred**2

The actual simplification count is found afterwards by trying regular
triangular grids n=1..L.  That search is validation only; the prediction path
never constructs or simplifies a mesh.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle

BLOCKS = 8
L = 16
NB = BLOCKS * BLOCKS
BLUE = "#2a78d6"
DARK_BLUE = "#184f95"
RED = "#d03b3b"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def read_pgm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:2] != b"P5":
        raise ValueError(f"only binary P5 PGM is supported: {path}")
    pos = 2
    tokens: list[int] = []
    while len(tokens) < 3:
        while pos < len(data) and data[pos] in b" \t\n\r":
            pos += 1
        if data[pos : pos + 1] == b"#":
            while pos < len(data) and data[pos] not in b"\n\r":
                pos += 1
            continue
        start = pos
        while pos < len(data) and data[pos] not in b" \t\n\r":
            pos += 1
        tokens.append(int(data[start:pos]))
    width, height, maxval = tokens
    while pos < len(data) and data[pos] in b" \t\n\r":
        pos += 1
    raw = np.frombuffer(data[pos : pos + width * height], dtype=np.uint8)
    if raw.size != width * height:
        raise ValueError("truncated PGM raster")
    return raw.reshape(height, width).astype(np.float64) / max(maxval, 1)


def bilinear(z: np.ndarray, x: float, y: float) -> float:
    h, w = z.shape
    x = min(max(x, 0.0), w - 1.0000001)
    y = min(max(y, 0.0), h - 1.0000001)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    u = x - x0
    v = y - y0
    return (
        (1 - u) * (1 - v) * z[y0, x0]
        + u * (1 - v) * z[y0, x1]
        + (1 - u) * v * z[y1, x0]
        + u * v * z[y1, x1]
    )


def triangle_error(block: np.ndarray, n: int) -> float:
    """Max vertical error of an n x n regular right-triangle mesh."""
    size = block.shape[0] - 1
    step = size / n
    zv = np.empty((n + 1, n + 1), dtype=np.float64)
    for j in range(n + 1):
        for i in range(n + 1):
            zv[j, i] = bilinear(block, i * step, j * step)

    worst = 0.0
    for y in range(size + 1):
        for x in range(size + 1):
            fi = x / step
            fj = y / step
            ci = min(int(math.floor(fi)), n - 1)
            cj = min(int(math.floor(fj)), n - 1)
            u = fi - ci
            v = fj - cj
            z00 = zv[cj, ci]
            z10 = zv[cj, ci + 1]
            z11 = zv[cj + 1, ci + 1]
            z01 = zv[cj + 1, ci]
            # Diagonal (0,0)-(1,1), consistent with build_mesh().
            if v <= u:
                zmesh = (1 - u) * z00 + (u - v) * z10 + v * z11
            else:
                zmesh = (1 - v) * z00 + (v - u) * z01 + u * z11
            worst = max(worst, abs(block[y, x] - zmesh))
    return float(worst)


def block_curvature(block: np.ndarray) -> float:
    dxx = np.zeros_like(block)
    dyy = np.zeros_like(block)
    dxx[:, 1:-1] = block[:, 2:] - 2.0 * block[:, 1:-1] + block[:, :-2]
    dyy[1:-1, :] = block[2:, :] - 2.0 * block[1:-1, :] + block[:-2, :]
    interior = np.zeros(block.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    directional = np.maximum(np.abs(dxx[interior]), np.abs(dyy[interior]))
    return float(np.mean(directional))


def build_mesh(z: np.ndarray, counts: dict[tuple[int, int], int]):
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    tris: list[list[int]] = []
    for by in range(BLOCKS):
        for bx in range(BLOCKS):
            n = counts[(by, bx)]
            base = len(xs)
            step = L / n
            for j in range(n + 1):
                for i in range(n + 1):
                    lx = i * step
                    ly = j * step
                    xs.append(bx * L + lx)
                    ys.append(by * L + ly)
                    zs.append(bilinear(z, bx * L + lx, by * L + ly))

            def vid(i: int, j: int) -> int:
                return base + j * (n + 1) + i

            for j in range(n):
                for i in range(n):
                    a = vid(i, j)
                    b = vid(i + 1, j)
                    c = vid(i + 1, j + 1)
                    d = vid(i, j + 1)
                    tris.append([a, b, c])
                    tris.append([a, c, d])
    return np.array(xs), np.array(ys), np.array(zs), np.array(tris, dtype=int)


def setup_font() -> str | None:
    # Some user-font installs are visible to fontconfig but absent from
    # matplotlib's cached ttflist. Register CJK files explicitly.
    for font_path in fm.findSystemFonts():
        if "CJK" in font_path or "wqy" in font_path.lower():
            try:
                fm.fontManager.addfont(font_path)
            except RuntimeError:
                pass
    candidates = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for name in candidates:
        if name in available:
            chosen = name
            break
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = SURFACE
    plt.rcParams["axes.facecolor"] = SURFACE
    return chosen


@dataclass
class Analysis:
    z: np.ndarray
    curvature: dict[tuple[int, int], float]
    error_by_n: dict[tuple[int, int], list[float]]
    rows_by_eps: dict[float, list[dict]]
    totals: list[dict]
    scaling_slope: float


def analyze(pgm: Path, epsilons: list[float]) -> Analysis:
    z = read_pgm(pgm)
    if z.shape != (BLOCKS * L + 1, BLOCKS * L + 1):
        raise ValueError(
            f"expected {BLOCKS * L + 1}x{BLOCKS * L + 1} raster for "
            f"{BLOCKS}x{BLOCKS} blocks of edge {L}, got {z.shape}"
        )

    curvature: dict[tuple[int, int], float] = {}
    errors: dict[tuple[int, int], list[float]] = {}
    for by in range(BLOCKS):
        for bx in range(BLOCKS):
            block = z[by * L : (by + 1) * L + 1, bx * L : (bx + 1) * L + 1]
            curvature[(by, bx)] = block_curvature(block)
            errors[(by, bx)] = [triangle_error(block, n) for n in range(1, L + 1)]

    rows_by_eps: dict[float, list[dict]] = {}
    totals: list[dict] = []
    for eps in epsilons:
        rows: list[dict] = []
        for by in range(BLOCKS):
            for bx in range(BLOCKS):
                key = (by, bx)
                kappa = curvature[key]
                raw_n = kappa * L / eps
                pred_n = max(1, min(L, int(math.ceil(raw_n - 1e-12))))
                actual_n = next(n for n, e in enumerate(errors[key], start=1) if e <= eps + 1e-12)
                pred_t = 2 * pred_n * pred_n
                actual_t = 2 * actual_n * actual_n
                rows.append(
                    {
                        "block": f"B{by}-{bx}",
                        "row": by,
                        "col": bx,
                        "kappa": kappa,
                        "scale": L,
                        "raw_n": raw_n,
                        "pred_n": pred_n,
                        "actual_n": actual_n,
                        "pred_tris": pred_t,
                        "actual_tris": actual_t,
                        "diff_tris": pred_t - actual_t,
                        "abs_diff_tris": abs(pred_t - actual_t),
                        "relative_abs_diff": abs(pred_t - actual_t) / actual_t,
                        "actual_max_error": errors[key][actual_n - 1],
                    }
                )
        worst_abs = max(r["abs_diff_tris"] for r in rows)
        for r in rows:
            r["is_worst"] = r["abs_diff_tris"] == worst_abs
        rows_by_eps[eps] = rows
        total_pred = sum(r["pred_tris"] for r in rows)
        total_actual = sum(r["actual_tris"] for r in rows)
        totals.append(
            {
                "epsilon": eps,
                "blocks": NB,
                "predicted_triangles": total_pred,
                "actual_triangles": total_actual,
                "total_diff": total_pred - total_actual,
                "total_relative_diff": (total_pred - total_actual) / total_actual,
                "exact_block_count": sum(r["pred_n"] == r["actual_n"] for r in rows),
                "underpredicted_blocks": sum(r["pred_tris"] < r["actual_tris"] for r in rows),
                "overpredicted_blocks": sum(r["pred_tris"] > r["actual_tris"] for r in rows),
                "block_triangle_wape": sum(r["abs_diff_tris"] for r in rows)
                / sum(r["actual_tris"] for r in rows),
                "worst_abs_triangle_diff": worst_abs,
                "worst_blocks": [r["block"] for r in rows if r["is_worst"]],
            }
        )

    # Empirical scaling check at power-of-two scales s = 16, 8, 4, 2.
    scales = np.array([L / n for n in (1, 2, 4, 8)], dtype=float)
    median_errors = []
    for n in (1, 2, 4, 8):
        vals = np.array([errors[(by, bx)][n - 1] for by in range(BLOCKS) for bx in range(BLOCKS)])
        median_errors.append(np.median(vals))
    slope = float(np.polyfit(np.log(scales), np.log(np.array(median_errors)), 1)[0])

    return Analysis(z, curvature, errors, rows_by_eps, totals, slope)


def write_tables(outdir: Path, analysis: Analysis, epsilons: list[float]) -> None:
    fieldnames = [
        "block", "row", "col", "kappa", "scale", "raw_n",
        "pred_n", "actual_n", "pred_tris", "actual_tris", "diff_tris",
        "abs_diff_tris", "relative_abs_diff", "actual_max_error", "is_worst", "epsilon",
    ]
    with (outdir / "block_comparison_long.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for eps in epsilons:
            for r in analysis.rows_by_eps[eps]:
                row = dict(r)
                row["epsilon"] = eps
                writer.writerow(row)

    # One wide table keeps both error levels on a single block row.
    wide_fields = ["block", "row", "col", "kappa", "scale"]
    for eps in epsilons:
        tag = f"e{eps:.3f}".replace(".", "p")
        wide_fields += [
            f"{tag}_pred_n", f"{tag}_actual_n", f"{tag}_pred_tris",
            f"{tag}_actual_tris", f"{tag}_diff_tris", f"{tag}_abs_diff_tris",
            f"{tag}_worst",
        ]
    with (outdir / "block_comparison_wide.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=wide_fields)
        writer.writeheader()
        base = analysis.rows_by_eps[epsilons[0]]
        for i, r0 in enumerate(base):
            row = {k: r0[k] for k in ["block", "row", "col", "kappa", "scale"]}
            for eps in epsilons:
                r = analysis.rows_by_eps[eps][i]
                tag = f"e{eps:.3f}".replace(".", "p")
                row.update(
                    {
                        f"{tag}_pred_n": r["pred_n"],
                        f"{tag}_actual_n": r["actual_n"],
                        f"{tag}_pred_tris": r["pred_tris"],
                        f"{tag}_actual_tris": r["actual_tris"],
                        f"{tag}_diff_tris": r["diff_tris"],
                        f"{tag}_abs_diff_tris": r["abs_diff_tris"],
                        f"{tag}_worst": int(r["is_worst"]),
                    }
                )
            writer.writerow(row)

    with (outdir / "totals.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(analysis.totals[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(analysis.totals)

    # Spatial matrices: useful for "row by row, column by column" inspection.
    for eps in epsilons:
        rows = analysis.rows_by_eps[eps]
        tag = f"eps_{eps:.3f}".replace(".", "p")
        for key, label in [("pred_tris", "pred"), ("actual_tris", "actual"), ("diff_tris", "diff")]:
            mat = np.array([[next(r for r in rows if r["row"] == by and r["col"] == bx)[key]
                             for bx in range(BLOCKS)] for by in range(BLOCKS)])
            with (outdir / f"matrix_{tag}_{label}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([f"col_{c}" for c in range(BLOCKS)])
                writer.writerows(mat.tolist())

    write_markdown_tables(outdir, analysis, epsilons)


def fmt_cell(r: dict) -> str:
    cell = f"{r['pred_tris']}/{r['actual_tris']} ({r['diff_tris']:+d})"
    if r["is_worst"]:
        return f"**{cell}**"
    return cell


def write_markdown_tables(outdir: Path, analysis: Analysis, epsilons: list[float]) -> None:
    lines = [
        "# 逐块预算与实际三角形数",
        "",
        "单元格为 `预判三角形数 / 实际三角形数 (预判-实际)`；粗体为该误差档绝对偏差最大的块。",
        "",
    ]
    for eps in epsilons:
        rows = analysis.rows_by_eps[eps]
        total = next(t for t in analysis.totals if abs(t["epsilon"] - eps) < 1e-12)
        lines += [
            f"## ε = {eps:.3f}",
            "",
            f"总预算 **{total['predicted_triangles']}**，实际 **{total['actual_triangles']}**，"
            f"相对差 **{total['total_relative_diff']:+.1%}**；"
            f"逐块三角形 WAPE={total['block_triangle_wape']:.1%}；"
            f"最大绝对偏差 {total['worst_abs_triangle_diff']}，"
            f"最大偏差块：{', '.join(total['worst_blocks'])}。",
            "",
            "| 行＼列 | " + " | ".join(f"c{bx}" for bx in range(BLOCKS)) + " |",
            "|" + "---|" * (BLOCKS + 1),
        ]
        for by in range(BLOCKS):
            cells = []
            for bx in range(BLOCKS):
                r = next(x for x in rows if x["row"] == by and x["col"] == bx)
                cells.append(fmt_cell(r))
            lines.append(f"| r{by} | " + " | ".join(cells) + " |")
        lines.append("")
        lines += [
            "| 块 | κ | 预判 n | 实际 n | 预算三角形 | 实际三角形 | 差值 | 相对绝对差 | 实测最大误差 | 最大偏差 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['block']} | {r['kappa']:.5f} | {r['pred_n']} | {r['actual_n']} | "
                f"{r['pred_tris']} | {r['actual_tris']} | {r['diff_tris']:+d} | "
                f"{r['relative_abs_diff']:.1%} | {r['actual_max_error']:.5f} | "
                f"{'⚠️' if r['is_worst'] else ''} |"
            )
        lines.append("")
    (outdir / "block_tables.md").write_text("\n".join(lines), encoding="utf-8")


def plot_curvature(outdir: Path, analysis: Analysis) -> None:
    kmat = np.array([[analysis.curvature[(by, bx)] for bx in range(BLOCKS)] for by in range(BLOCKS)])
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    im = ax.imshow(kmat, cmap="Blues")
    for by in range(BLOCKS):
        for bx in range(BLOCKS):
            color = "white" if kmat[by, bx] > kmat.mean() + 0.8 * kmat.std() else INK
            ax.text(bx, by, f"{kmat[by,bx]:.4f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(BLOCKS), [f"c{i}" for i in range(BLOCKS)])
    ax.set_yticks(range(BLOCKS), [f"r{i}" for i in range(BLOCKS)])
    ax.set_title("逐块二阶差分曲率 κ_B（均值最大方向二阶差分）")
    first_eps = next(iter(analysis.rows_by_eps))
    for worst in (r for r in analysis.rows_by_eps[first_eps] if r["is_worst"]):
        ax.add_patch(Rectangle((worst["col"] - 0.5, worst["row"] - 0.5), 1, 1,
                               fill=False, edgecolor=RED, linewidth=2.8))
    fig.colorbar(im, ax=ax, shrink=0.82, label="曲率（高度单位/格距）")
    fig.savefig(outdir / "curvature_map.png", dpi=220)
    plt.close(fig)


def plot_scaling(outdir: Path, analysis: Analysis) -> None:
    scales = np.array([L / n for n in (1, 2, 4, 8)], dtype=float)
    med = []
    q25 = []
    q75 = []
    for n in (1, 2, 4, 8):
        vals = np.array([analysis.error_by_n[(by, bx)][n - 1]
                         for by in range(BLOCKS) for bx in range(BLOCKS)])
        med.append(np.median(vals))
        q25.append(np.quantile(vals, 0.25))
        q75.append(np.quantile(vals, 0.75))
    med = np.array(med)
    q25 = np.array(q25)
    q75 = np.array(q75)

    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    ax.fill_between(scales, q25, q75, color=BLUE, alpha=0.16, linewidth=0, label="块间 25–75% 区间")
    ax.plot(scales, med, "o-", color=BLUE, linewidth=2.2, markersize=6, label="块误差中位数")
    ref1 = med[0] * (scales / scales[0])
    ref2 = med[0] * (scales / scales[0]) ** 2
    ax.plot(scales, ref1, "--", color=MUTED, linewidth=1.6, label="线性参考 ∝ s")
    ax.plot(scales, ref2, ":", color="#52514e", linewidth=1.5, label="二阶 Taylor 参考 ∝ s²")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(scales, [f"{s:g}" for s in scales])
    ax.set_xlabel("三角网尺度 s = L/n（格）")
    ax.set_ylabel("最大垂直误差（归一化高度）")
    ax.set_title(f"误差—尺度关系：实测斜率 {analysis.scaling_slope:.2f}（线性律≈1）")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(outdir / "scaling_law.png", dpi=220)
    plt.close(fig)


def plot_comparison(outdir: Path, analysis: Analysis, epsilons: list[float]) -> None:
    fig, axes = plt.subplots(1, len(epsilons), figsize=(6.4 * len(epsilons), 5.4), constrained_layout=True)
    if len(epsilons) == 1:
        axes = [axes]
    for ax, eps in zip(axes, epsilons):
        rows = analysis.rows_by_eps[eps]
        x = np.array([r["actual_tris"] for r in rows])
        y = np.array([r["pred_tris"] for r in rows])
        worst = np.array([r["is_worst"] for r in rows])
        lim = max(x.max(), y.max()) * 1.12
        ax.plot([0, lim], [0, lim], "--", color=MUTED, linewidth=1.5, label="预算=实际")
        ax.scatter(x[~worst], y[~worst], s=42, color=BLUE, edgecolor="white", linewidth=0.6, alpha=0.88)
        ax.scatter(x[worst], y[worst], s=95, color=RED, edgecolor="white", linewidth=0.8, zorder=3,
                   label="最大偏差块")
        label_groups: dict[tuple[int, int], list[str]] = {}
        for r in rows:
            if r["is_worst"]:
                key = (r["actual_tris"], r["pred_tris"])
                label_groups.setdefault(key, []).append(r["block"])
        for (xpos, ypos), labels in label_groups.items():
            ax.annotate(", ".join(labels), (xpos, ypos),
                        xytext=(7, 5), textcoords="offset points", fontsize=9,
                        color=RED, fontweight="bold")
        total = next(t for t in analysis.totals if abs(t["epsilon"] - eps) < 1e-12)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_xlabel("实际三角形数/块")
        ax.set_ylabel("预判预算/块")
        ax.set_title(f"ε={eps:.3f}｜总预算 {total['predicted_triangles']} vs "
                     f"实际 {total['actual_triangles']}")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(outdir / "budget_vs_actual.png", dpi=220)
    plt.close(fig)


def plot_mesh_result(outdir: Path, analysis: Analysis, z: np.ndarray, eps: float) -> Path:
    rows = analysis.rows_by_eps[eps]
    counts = {(r["row"], r["col"]): r["actual_n"] for r in rows}
    xs, ys, zs, tris = build_mesh(z, counts)
    tri = Triangulation(xs, ys, tris)
    face_z = zs[tris].mean(axis=1)
    worst_rows = [r for r in rows if r["is_worst"]]
    tag = f"epsilon_{eps:.3f}".replace(".", "p")
    path = outdir / f"simplified_mesh_{tag}.png"

    fig = plt.figure(figsize=(15.2, 7.4), constrained_layout=True)
    ax2 = fig.add_subplot(1, 2, 1)
    mesh = ax2.tripcolor(tri, facecolors=face_z, cmap="Blues", edgecolors="none")
    ax2.triplot(tri, color=DARK_BLUE, linewidth=0.16, alpha=0.58)
    for by in range(BLOCKS + 1):
        ax2.axhline(by * L, color=INK, linewidth=0.75, alpha=0.55)
    for bx in range(BLOCKS + 1):
        ax2.axvline(bx * L, color=INK, linewidth=0.75, alpha=0.55)
    for r in rows:
        ax2.text(r["col"] * L + L / 2, r["row"] * L + L / 2, r["block"],
                 ha="center", va="center", fontsize=5.8,
                 color=RED if r["is_worst"] else INK,
                 fontweight="bold" if r["is_worst"] else "normal", alpha=0.95)
    for r in worst_rows:
        ax2.add_patch(Rectangle((r["col"] * L, r["row"] * L), L, L, fill=False,
                                edgecolor=RED, linewidth=2.8))
    ax2.set_xlim(0, BLOCKS * L)
    ax2.set_ylim(BLOCKS * L, 0)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x（格）")
    ax2.set_ylabel("y（格）")
    total_actual = sum(r["actual_tris"] for r in rows)
    total_pred = sum(r["pred_tris"] for r in rows)
    ax2.set_title(f"俯视三角网与逐块细分｜实际 {total_actual} 三角形，预判 {total_pred}")
    fig.colorbar(mesh, ax=ax2, shrink=0.82, label="归一化高程")

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3.plot_trisurf(xs, ys, zs, triangles=tris, cmap="Blues", linewidth=0.08,
                     edgecolor=(0.094, 0.247, 0.459, 0.42), antialiased=True)
    for r in worst_rows:
        x0, y0 = r["col"] * L, r["row"] * L
        zz = 1.03
        ax3.plot([x0, x0 + L, x0 + L, x0, x0],
                 [y0, y0, y0 + L, y0 + L, y0],
                 [zz] * 5, color=RED, linewidth=2.4)
    ax3.set_xlim(0, BLOCKS * L)
    ax3.set_ylim(0, BLOCKS * L)
    ax3.set_zlim(0, 1.08)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_box_aspect((BLOCKS * L, BLOCKS * L, 26))
    ax3.view_init(elev=36, azim=-128)
    ax3.set_title("三维简化结果（红框=最大偏差块）")
    fig.suptitle(f"地形简化结果存档：ε = {eps:.3f}（归一化最大垂直误差）", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(outdir: Path, analysis: Analysis, epsilons: list[float], images: list[Path]) -> None:
    summary = {
        "pgm_raster_size": [BLOCKS * L + 1, BLOCKS * L + 1],
        "blocks": BLOCKS * BLOCKS,
        "block_edge_cells": L,
        "curvature": "mean over interior of max(abs(Dxx), abs(Dyy)); Dxx/Dyy are central second differences",
        "prediction_formula": "n_pred=ceil(kappa_B*L/epsilon), triangles=2*n_pred^2",
        "actual_method": "smallest n in 1..16 whose regular right-triangle mesh has max sampled vertical error <= epsilon",
        "scaling_slope_log_e_vs_log_s": analysis.scaling_slope,
        "epsilon_results": analysis.totals,
        "images": [p.name for p in images],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgm", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--eps", default="0.04,0.02", help="comma-separated normalized error levels")
    args = parser.parse_args()
    epsilons = [float(x.strip()) for x in args.eps.split(",") if x.strip()]
    if len(epsilons) != 2:
        raise ValueError("this report expects exactly two error levels")
    args.outdir.mkdir(parents=True, exist_ok=True)
    setup_font()
    analysis = analyze(args.pgm, epsilons)
    write_tables(args.outdir, analysis, epsilons)
    plot_curvature(args.outdir, analysis)
    plot_scaling(args.outdir, analysis)
    plot_comparison(args.outdir, analysis, epsilons)
    images = []
    for eps in epsilons:
        images.append(plot_mesh_result(args.outdir, analysis, analysis.z, eps))
    write_summary(args.outdir, analysis, epsilons, images)
    print(json.dumps(analysis.totals, ensure_ascii=False, indent=2))
    print(f"wrote artifacts to {args.outdir}")


if __name__ == "__main__":
    main()
