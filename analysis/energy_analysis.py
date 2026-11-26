#!/usr/bin/env python3
"""预判并核对 fBm 多尺度分量的方差能量占比。

输入由 cmd/energydump 生成：一个小端 float64 二进制文件，数组顺序为
原图 X，随后是生成器顺序（粗 -> 细）的 K 个加性分量。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# 中文字体；不存在时 matplotlib 会回退，不影响 CSV/JSON 数值。
for font_path in [
    "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
]:
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
GRID = "#d8d6d0"
INK = "#20201d"
MUTED = "#66645e"
DIV_CMAP = LinearSegmentedColormap.from_list("blue_surface_orange", [BLUE, "#f7f5f0", ORANGE])


def load_dump(prefix: Path) -> tuple[dict, np.ndarray]:
    meta = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    raw = np.fromfile(prefix.with_suffix(".f64"), dtype="<f8")
    n = int(meta["size"])
    k = int(meta["octaves"])
    expected = (k + 1) * n * n
    if raw.size != expected:
        raise ValueError(f"float64 数量 {raw.size} != 预期 {expected}")
    arrays = raw.reshape(k + 1, n, n)
    return meta, arrays


def predicted_shares(k: int, gain: float, lacunarity: float, exponent: float = 0.0) -> np.ndarray:
    """生成器顺序（粗 -> 细）的预判占比。

    w_j = (A_0 G^j)^2 * (f_0 L^j)^p；公共常数消去后为 (G^2 L^p)^j。
    本仓库的 amp 是整层空间振幅，平稳缩放使额外频率幂 p=0。
    """
    j = np.arange(k)
    weights = (gain**2 * lacunarity**exponent) ** j
    return weights / weights.sum()


def save_layer_image(
    path: Path,
    image: np.ndarray,
    title: str,
    subtitle: str,
    *,
    diverging: bool,
    flagged: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=170)
    if diverging:
        limit = float(np.max(np.abs(image)))
        if limit == 0.0:
            limit = 1.0
        im = ax.imshow(image, cmap=DIV_CMAP, vmin=-limit, vmax=limit, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("归一化后的有符号分量高度", color=INK)
    else:
        im = ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("归一化高度", color=INK)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(3.2 if flagged else 0.8)
        spine.set_edgecolor(RED if flagged else "#88857e")
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", pad=10)
    ax.text(
        0.5,
        -0.075,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color=RED if flagged else MUTED,
        fontweight="bold" if flagged else "normal",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_comparison_chart(rows: list[dict], path: Path, threshold_pp: float) -> None:
    # 图中最上方是展示第 2 层 = 最细尺度，符合题面层序。
    scale_rows = [r for r in rows if r["kind"] == "scale"]
    labels = [f"展示第{r['display_layer']}层\n{r['scale_name']}" for r in scale_rows]
    pred = [r["pred_pct"] for r in scale_rows]
    meas = [r["meas_pct"] for r in scale_rows]
    y = np.arange(len(scale_rows))
    h = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=170)
    b1 = ax.barh(y - h / 2, pred, height=h, color=BLUE, label="预判占比", zorder=3)
    b2 = ax.barh(y + h / 2, meas, height=h, color=ORANGE, label="分解后实测占比", zorder=3)

    for bars, values in ((b1, pred), (b2, meas)):
        for rect, value in zip(bars, values):
            ax.text(
                rect.get_width() + 0.8,
                rect.get_y() + rect.get_height() / 2,
                f"{value:.2f}%",
                va="center",
                ha="left",
                fontsize=9,
                color=INK,
            )

    for yi, row in zip(y, scale_rows):
        if row["flag"]:
            ax.text(
                max(row["pred_pct"], row["meas_pct"]) + 3.0,
                yi,
                f"超差 {row['diff_pp']:+.2f} pp",
                va="center",
                ha="left",
                fontsize=10,
                color=RED,
                fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("占层能量总和的百分比（能量 = 总体方差）", color=INK)
    ax.set_xlim(0, max(max(pred), max(meas)) * 1.42)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#aaa79f")
    ax.spines["bottom"].set_color("#aaa79f")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.set_title(
        f"预判与逐层实测两栏对照；红标为 |偏差| > {threshold_pp:g} 个百分点",
        fontsize=13,
        fontweight="bold",
        color=INK,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_contact_sheet(components_fine: np.ndarray, variances: np.ndarray, shares: np.ndarray, path: Path) -> None:
    k = components_fine.shape[0]
    fig, axes = plt.subplots(1, k, figsize=(4.0 * k, 4.2), dpi=170)
    if k == 1:
        axes = [axes]
    names = ["最细", "细", "中粗", "最粗"]
    for m, (ax, comp, var, share) in enumerate(zip(axes, components_fine, variances, shares), start=1):
        limit = float(np.max(np.abs(comp))) or 1.0
        im = ax.imshow(comp, cmap=DIV_CMAP, vmin=-limit, vmax=limit, interpolation="nearest")
        kind = names[m - 1] if m <= len(names) else f"第 {m} 层"
        ax.set_title(f"尺度 {m}（{kind}）\n实测 {share * 100:.2f}%", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("各尺度分量总览（各图独立对称色标；百分比按方差）", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path, default=Path("artifacts/energy/data/terrain_s7_n128_k4"))
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/energy"))
    ap.add_argument("--threshold-pp", type=float, default=5.0)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    meta, arrays = load_dump(args.prefix)
    x = arrays[0]
    coarse = arrays[1:]
    fine = coarse[::-1]  # 题面要求：尺度层从最细开始

    k = int(meta["octaves"])
    gain = float(meta["gain"])
    lac = float(meta["lacunarity"])
    pred_coarse = predicted_shares(k, gain, lac, exponent=0.0)
    pred_fine = pred_coarse[::-1]

    var_fine = fine.reshape(k, -1).var(axis=1)  # 总体方差 ddof=0
    total_component_var = float(var_fine.sum())
    meas_fine = var_fine / total_component_var

    # 校验线性分解：X = constant + sum(C_j)。
    constant = -float(meta["raw_min"]) * float(meta["normalization_scale"])
    reconstructed = constant + coarse.sum(axis=0)
    reconstruction_max_abs = float(np.max(np.abs(reconstructed - x)))

    original_var = float(x.var())
    rows: list[dict] = [
        {
            "display_layer": 1,
            "kind": "original",
            "scale_index": 0,
            "scale_name": "原图（总场，非额外频带）",
            "frequency": "",
            "variance": original_var,
            "pred_pct": 100.0,
            "meas_pct": 100.0,
            "diff_pp": 0.0,
            "flag": False,
            "note": "总量锚点；尺度占比的分母是四个分量方差之和",
        }
    ]

    for m in range(k):
        j = k - 1 - m  # 回到生成器的 coarse-first 下标
        diff_pp = (pred_fine[m] - meas_fine[m]) * 100.0
        flag = abs(diff_pp) > args.threshold_pp
        scale_name = "最细 octave" if m == 0 else ("最粗 octave" if m == k - 1 else f"第 {j + 1} octave")
        rows.append(
            {
                "display_layer": m + 2,
                "kind": "scale",
                "scale_index": m + 1,
                "generator_octave": j + 1,
                "scale_name": scale_name,
                "frequency": float(meta["noise_coordinates_coarse_first"][j]),
                "amplitude": float(meta["amplitudes_coarse_first"][j]),
                "variance": float(var_fine[m]),
                "pred_pct": float(pred_fine[m] * 100.0),
                "meas_pct": float(meas_fine[m] * 100.0),
                "diff_pp": float(diff_pp),
                "flag": bool(flag),
                "note": "⚠ 偏差超过 5 个百分点" if flag else "未超差",
            }
        )

    # 单独图片：第一张固定原图，其后每个尺度分量一张。
    save_layer_image(
        args.outdir / "layer_1_original.png",
        x,
        "第 1 层：原图（题面指定）",
        f"总体方差 σ²={original_var:.8f}；min–max 归一化灰度图",
        diverging=False,
    )
    for m, comp in enumerate(fine, start=1):
        row = rows[m]  # rows[0] original, rows[1] scale m=1
        flagged = bool(row["flag"])
        save_layer_image(
            args.outdir / f"layer_{m + 1}_scale{m}_{'finest' if m == 1 else 'coarsest' if m == k else 'scale'}.png",
            comp,
            f"展示第 {m + 1} 张 / 尺度 {m}：{row['scale_name']}",
            (
                f"方差={row['variance']:.8f}，实测={row['meas_pct']:.2f}%，"
                f"预判={row['pred_pct']:.2f}%，偏差={row['diff_pp']:+.2f} pp"
                + ("；【超差 >5 pp】" if flagged else "")
            ),
            diverging=True,
            flagged=flagged,
        )

    save_contact_sheet(fine, var_fine, meas_fine, args.outdir / "all_component_layers.png")
    save_comparison_chart(rows, args.outdir / "pred_vs_meas.png", args.threshold_pp)

    # CSV
    csv_path = args.outdir / "energy_comparison.csv"
    fieldnames = [
        "display_layer", "kind", "scale_index", "generator_octave", "scale_name",
        "frequency", "amplitude", "variance", "pred_pct", "meas_pct", "diff_pp", "flag", "note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Markdown 两栏占比表
    md = [
        "# 能量占比预判 vs 分解后实测",
        "",
        "能量定义为总体方差 `E=Var(·)=mean((z-mean(z))²)`；尺度占比的分母为各分量方差之和。",
        "展示顺序按题面：第 1 张/行为原图，其后尺度层从最细到最粗。",
        "",
        "| 展示层 | 层序 | 频率坐标 f | 预判占比 | 实测占比 | 偏差（预判−实测） | 判定 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r["kind"] == "original":
            md.append(
                f"| 1 | **原图（总场）** | — | 100.00%† | 100.00%† | 0.00 pp | 总量锚点 |"
            )
        else:
            mark = "⚠ **超差 >5 pp**" if r["flag"] else "合格（≤5 pp）"
            freq = f"{r['frequency']:.3f}"
            md.append(
                f"| {r['display_layer']} | 尺度 {r['scale_index']}，{r['scale_name']} | {freq} | "
                f"{r['pred_pct']:.2f}% | {r['meas_pct']:.2f}% | {r['diff_pp']:+.2f} pp | {mark} |"
            )
    md += [
        "",
        "†原图行的两个 100% 都以原图自身方差为分母，只用于满足“第一层是原图”的展示约定；"
        "四个尺度分量的两栏百分比共享另一个归一化分母，因此相加分别为 100%。",
        "",
    ]
    (args.outdir / "energy_table.md").write_text("\n".join(md), encoding="utf-8")

    result = {
        "parameters": {
            "size": meta["size"],
            "seed": meta["seed"],
            "octaves": k,
            "gain": gain,
            "lacunarity": lac,
            "base_frequency": meta["frequency"],
            "energy": "population variance (ddof=0)",
            "prediction_exponent_p_for_spatial_amplitude": 0,
            "threshold_pp": args.threshold_pp,
        },
        "original_variance": original_var,
        "component_variance_sum": total_component_var,
        "reconstruction_max_abs_error": reconstruction_max_abs,
        "predicted_shares_fine_first_pct": [float(x * 100) for x in pred_fine],
        "measured_shares_fine_first_pct": [float(x * 100) for x in meas_fine],
        "flagged_scale_indices": [r["scale_index"] for r in rows if r.get("flag")],
        "rows": rows,
    }
    (args.outdir / "energy_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "outdir": str(args.outdir),
        "pred_fine_pct": result["predicted_shares_fine_first_pct"],
        "meas_fine_pct": result["measured_shares_fine_first_pct"],
        "flagged": result["flagged_scale_indices"],
        "reconstruction_error": reconstruction_max_abs,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
