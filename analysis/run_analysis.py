"""主分析：两组分层数（8/16）下预判 vs 逐层扫描实测，逐带比对并出图。

输出（均在 artifacts/ 下）：
  tables/bands_8.csv、bands_16.csv     逐带全量数据
  tables/summary.json                  全局校验与超差带清单
  figures/hypso_8.png、hypso_16.png    分层设色填色图（各一张）
  figures/levelset_curves.png          L(h)、Λ(h) 诊断曲线
  figures/band_compare.png             预判/实测面积与宽度对比
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import font_manager

# 注册系统中的 Noto Sans CJK，保证中文标签可读
_CJK = Path("/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf")
if _CJK.exists():
    font_manager.fontManager.addfont(str(_CJK))
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
import terrain_bands as tb
from palette import RAMP8, RAMP16

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "data" / "height_s7_n128.bin"
OUT_T = ROOT / "artifacts" / "tables"
OUT_F = ROOT / "artifacts" / "figures"
THRESH = 0.10  # 相对偏差超一成即标出


def analyze_one(z: np.ndarray, mesh: dict, k: int, ramp: list[str]) -> dict:
    edges = np.linspace(0.0, 1.0, k + 1)
    bi = tb.analytic_band_integrals(mesh, edges)
    raster = tb.raster_scan(z, edges)
    clip_area = tb.exact_clipped_band_areas(mesh, edges)

    n = z.shape[0]
    raster_area = raster["cells"].astype(np.float64)          # 单位格点面积
    pred_area = bi["pred_area"]                                 # 三角剖分连续面积
    mean_l = bi["mean_L"]

    # 平均宽度 = 面积 / 带内等值线总长（同一套 L、Λ 积分给出两个量）
    pred_width = pred_area / mean_l
    meas_width = raster_area / mean_l

    def rel_err(pred, meas):
        out = np.full_like(pred, np.nan)
        nz = meas > 0
        out[nz] = (pred[nz] - meas[nz]) / meas[nz]
        return out

    area_err = rel_err(pred_area, raster_area)
    width_err = rel_err(pred_width, meas_width)
    # 连续真值与积分的偏差（验证积分本身，应≈机器精度）
    integ_err = rel_err(pred_area, clip_area)

    rows = []
    for b in range(k):
        rows.append({
            "band": b + 1,
            "h_lo": round(float(edges[b]), 6),
            "h_hi": round(float(edges[b + 1]), 6),
            "pred_area": round(float(pred_area[b]), 4),
            "raster_area": int(raster["cells"][b]),
            "clip_area": round(float(clip_area[b]), 4),
            "mean_contour_L": round(float(mean_l[b]), 4),
            "pred_width": round(float(pred_width[b]), 4),
            "meas_width": round(float(meas_width[b]), 4),
            "area_rel_err": None if np.isnan(area_err[b]) else round(float(area_err[b]), 5),
            "width_rel_err": None if np.isnan(width_err[b]) else round(float(width_err[b]), 5),
            "integral_self_err": None if np.isnan(integ_err[b]) else round(float(integ_err[b]), 8),
            "components4": int(raster["components"][b]),
            "largest_component": int(raster["largest_component"][b]),
            "area_flag_gt10pct": bool(abs(np.nan_to_num(area_err[b])) > THRESH),
            "width_flag_gt10pct": bool(abs(np.nan_to_num(width_err[b])) > THRESH),
        })

    return {
        "k": k,
        "edges": edges,
        "rows": rows,
        "band_id": raster["band_id"],
        "ramp": ramp,
        "pred_area": pred_area,
        "raster_area": raster_area,
        "clip_area": clip_area,
        "mean_l": mean_l,
        "pred_width": pred_width,
        "meas_width": meas_width,
        "area_err": area_err,
        "width_err": width_err,
        "bi": bi,
    }


def write_csv(res: dict, path: Path) -> None:
    cols = list(res["rows"][0].keys())
    lines = [",".join(cols)]
    for r in res["rows"]:
        lines.append(",".join("" if r[c] is None else str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_hypso(res: dict, z: np.ndarray, path: Path) -> None:
    k = res["k"]
    edges = res["edges"]
    cmap = ListedColormap(res["ramp"])
    # band_id 已是 0..k-1 的离散带索引，直接按索引取色；
    # BoundaryNorm 会把索引当原始值再次归一化，不能用。
    rgba = cmap(res["band_id"])
    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=150)
    ax.imshow(rgba, origin="upper", interpolation="nearest")
    ax.set_title(f"分层设色填色图（{k} 个等高度带，seed=7, 128×128）")
    ax.set_xlabel("x（格点）")
    ax.set_ylabel("y（格点）")
    # 离散色标：每色对应一个高度带区间
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=matplotlib.colors.BoundaryNorm(edges, k))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=edges, shrink=0.85, pad=0.02)
    cbar.ax.set_ylabel("归一化高度带区间", rotation=270, labelpad=16)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    z = tb.load_heightmap(str(DATA))
    mesh = tb.triangulate(z)

    # 全局恒等式闭合校验
    info = tb.analytic_triangle_integrals(mesh)
    res8 = analyze_one(z, mesh, 8, RAMP8)
    res16 = analyze_one(z, mesh, 16, RAMP16)

    write_csv(res8, OUT_T / "bands_8.csv")
    write_csv(res16, OUT_T / "bands_16.csv")
    plot_hypso(res8, z, OUT_F / "hypso_8.png")
    plot_hypso(res16, z, OUT_F / "hypso_16.png")

    # 诊断曲线 L(h)、Λ(h)
    levels = np.linspace(0, 1, 2001)
    L, Lam = tb.levelset_functions(mesh, levels)
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=150)
    ax.plot(levels, L, lw=1.8, label="L(h) 等值线欧氏总长")
    ax.plot(levels, Lam, lw=1.8, label="Λ(h)=∮ds/|∇u| 共面积因子")
    ax.set_xlabel("归一化高度 h")
    ax.set_ylabel("等值线总长")
    ax.set_title("等值线总长随高度变化（coarea：A′(h)=Λ(h)）")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_F / "levelset_curves.png", bbox_inches="tight")
    plt.close(fig)

    # 预判 vs 实测 面积/宽度 分组柱状图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    for col, res in enumerate((res8, res16)):
        idx = np.arange(1, res["k"] + 1)
        ax = axes[0, col]
        w = 0.4
        ax.bar(idx - w / 2, res["pred_area"], w, label="预判面积 ∫Λdh")
        ax.bar(idx + w / 2, res["raster_area"], w, label="实测面积（逐层扫描）")
        ax.set_title(f"{res['k']} 带：面积")
        ax.set_xlabel("色带编号")
        ax.set_ylabel("面积（格点单位）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis="y")
        ax = axes[1, col]
        ax.bar(idx - w / 2, res["pred_width"], w, label="预判宽度 A/L̄")
        ax.bar(idx + w / 2, res["meas_width"], w, label="实测宽度 A_测/L̄")
        ax.set_title(f"{res['k']} 带：平均宽度")
        ax.set_xlabel("色带编号")
        ax.set_ylabel("平均宽度（格距单位）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_F / "band_compare.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "grid": {"size": int(z.shape[0]), "cells": int(z.size),
                 "triangles": info["n_triangles"],
                 "tri_domain_area": info["total_area_Lambda"],
                 "integral_L_total": info["total_integral_L"]},
        "coarea_closure": {
            "sum_pred_area_8": float(res8["pred_area"].sum()),
            "sum_clip_area_8": float(res8["clip_area"].sum()),
            "sum_pred_area_16": float(res16["pred_area"].sum()),
            "sum_clip_area_16": float(res16["clip_area"].sum()),
            "max_integral_self_err_8": float(max(
                abs(r["integral_self_err"]) for r in res8["rows"])),
            "max_integral_self_err_16": float(max(
                abs(r["integral_self_err"]) for r in res16["rows"])),
            "raster_total_cells": int(res8["raster_area"].sum()),
        },
        "threshold_rel": THRESH,
        "flags_8": {
            "area_gt10pct": [r["band"] for r in res8["rows"] if r["area_flag_gt10pct"]],
            "width_gt10pct": [r["band"] for r in res8["rows"] if r["width_flag_gt10pct"]],
        },
        "flags_16": {
            "area_gt10pct": [r["band"] for r in res16["rows"] if r["area_flag_gt10pct"]],
            "width_gt10pct": [r["band"] for r in res16["rows"] if r["width_flag_gt10pct"]],
        },
    }
    (OUT_T / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
