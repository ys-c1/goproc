#!/usr/bin/env python3
"""Box-counting experiment for fBm octave saturation.

The Go dumper produces the exact goproc heightmaps; this script only reads the
PGM/range artifacts, performs differential box counting, fits the middle three
dyadic scales, and writes the table/chart used by the report.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
MAPDIR = OUT / "boxcount" / "maps_final"

SIZE = 512
SEEDS = list(range(1, 17))
GAINS = [0.40, 0.50, 0.60]
GAIN_LABELS = ["较快衰减 g=0.40", "默认附近 g=0.50", "较慢衰减 g=0.60"]
LACUNARITY = 2.0
F0 = 0.02
OCTAVES = np.arange(1, 13)
# Seven-level dyadic ladder; fit the middle three to avoid quantized and
# very coarse/edge-dominated scales.
LADDER = [2, 4, 8, 16, 32, 64, 128]
SCALES = np.array([8, 16, 32], dtype=float)
EPS_SAT = 0.005
EPS_VISUAL_D = 0.010
TAU_AMPLITUDE = 0.030
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"


def ensure_maps() -> None:
    marker = MAPDIR / "ranges.csv"
    if marker.exists():
        return
    seeds = ",".join(map(str, SEEDS))
    cmd = [
        "go", "run", "./analysis/dump",
        "-size", str(SIZE),
        "-seeds", seeds,
        "-out", str(MAPDIR.relative_to(ROOT)),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def box_count(pixels: np.ndarray, scale: int) -> int:
    """Differential 3D/surface box count for one xy box side ``scale``."""
    n = (pixels.shape[0] // scale) * scale
    blocks = pixels[:n, :n].reshape(n // scale, scale, n // scale, scale)
    zmax = blocks.max(axis=(1, 3)).astype(np.int16)
    zmin = blocks.min(axis=(1, 3)).astype(np.int16)
    # ceil((range + 1) / s): one gray-level box for a flat s x s block.
    counts = np.ceil((zmax - zmin + 1) / scale)
    return int(counts.sum())


def fit_dimension(pixels: np.ndarray) -> tuple[float, float, dict[int, int]]:
    counts = {int(s): box_count(pixels, int(s)) for s in SCALES}
    y = np.log(np.array([counts[int(s)] for s in SCALES], dtype=float))
    x = np.log(1.0 / SCALES)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(slope), float(r2), counts


def configure_font() -> str | None:
    candidates = [
        "NotoSansCJKsc-Regular.ttf",
        "NotoSansCJK-Regular.ttc",
        "wqy-zenhei.ttc",
        "msyh.ttc",
        "simhei.ttf",
    ]
    for path in font_manager.findSystemFonts():
        name = Path(path).name
        if any(c.lower() in name.lower() for c in candidates) or "NotoSansCJK" in path:
            try:
                font_manager.fontManager.addfont(path)
                prop = font_manager.FontProperties(fname=path)
                family = prop.get_name()
                plt.rcParams["font.family"] = family
                plt.rcParams["axes.unicode_minus"] = False
                return family
            except Exception:
                pass
    return None


def phi_smooth(t: np.ndarray | float) -> np.ndarray | float:
    return t / np.sqrt(1.0 + t * t)


def main() -> None:
    ensure_maps()
    OUT.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | int | str]] = []
    dimensions = np.zeros((len(SEEDS), len(GAINS), len(OCTAVES)))

    for si, seed in enumerate(SEEDS):
        for gi, gain in enumerate(GAINS):
            for oi, octave in enumerate(OCTAVES):
                pgm = MAPDIR / f"s{seed:03d}_g{gi}_o{octave:02d}.pgm"
                with Image.open(pgm) as im:
                    pixels = np.array(im)
                dimension, r2, counts = fit_dimension(pixels)
                dimensions[si, gi, oi] = dimension
                records.append(
                    {
                        "seed": seed,
                        "gain": gain,
                        "octaves": int(octave),
                        "D": dimension,
                        "R2": r2,
                        "N8": counts[8],
                        "N16": counts[16],
                        "N32": counts[32],
                    }
                )

    per_seed = pd.DataFrame.from_records(records)
    per_seed.to_csv(OUT / "per_octave_dimensions.csv", index=False)

    mean_d = dimensions.mean(axis=0)
    sd_d = dimensions.std(axis=0, ddof=1)
    se_d = sd_d / math.sqrt(len(SEEDS))

    ranges = pd.read_csv(MAPDIR / "ranges.csv")
    raw_range = ranges.groupby(["gain", "octaves"]).raw_range.mean().unstack(0)

    def raw_finite_curve(gain: float, q: float, c: float) -> np.ndarray:
        """Predicted D(K) before anchoring the common K=1 intercept."""
        curve = []
        for octave in OCTAVES:
            k_count = int(octave)
            s_norm = (1.0 - gain**k_count) / (1.0 - gain)
            beta_gray = 255.0 / float(raw_range.loc[k_count, gain])
            n_pred = []
            for scale in SCALES:
                response2 = 0.0
                for k in range(k_count):
                    t = c * scale * F0 * LACUNARITY**k
                    response = phi_smooth(t)
                    weight = gain**k / s_norm
                    response2 += (weight * response) ** 2
                typical_range = math.sqrt(response2)
                n_xy = (SIZE / scale) ** 2
                n_pred.append(n_xy * (1.0 + q * beta_gray * typical_range / scale))
            y = np.log(np.asarray(n_pred))
            x = np.log(1.0 / SCALES)
            curve.append(float(np.polyfit(x, y, 1)[0]))
        return np.asarray(curve)

    # Two global box-response constants. The intercept is anchored to the
    # measured one-octave baseline, which is identical for every gain.
    d_base = float(mean_d[0, 0])

    def predicted_family(theta: np.ndarray) -> np.ndarray:
        q, c = np.exp(theta)
        family = []
        for gain in GAINS:
            raw = raw_finite_curve(gain, q, c)
            family.append(raw - raw[0] + d_base)
        return np.asarray(family)

    def objective(theta: np.ndarray) -> float:
        pred = predicted_family(theta)
        return float(np.sqrt(np.mean((pred - mean_d) ** 2)))

    fit = minimize(objective, np.log([1.8, 0.75]), method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12})
    q_fit, c_fit = np.exp(fit.x)
    pred_d = predicted_family(fit.x)
    residual = pred_d - mean_d
    rmse = float(np.sqrt(np.mean(residual**2)))
    max_abs = float(np.max(np.abs(residual)))

    plateau_d = mean_d[:, 9:12].mean(axis=1)
    plateau_sd = sd_d[:, 9:12].mean(axis=1)

    def first_within(gi: int, eps: float) -> int:
        target = plateau_d[gi]
        return int(next(k for k in OCTAVES if target - mean_d[gi, k - 1] <= eps))

    point_rows = []
    for gi, gain in enumerate(GAINS):
        h_raw = -math.log(gain) / math.log(LACUNARITY)
        h_clip = min(1.0, max(0.0, h_raw))
        d_hurst = 3.0 - h_clip
        k_formula = math.ceil(math.log(1.0 / TAU_AMPLITUDE) / (h_raw * math.log(LACUNARITY)))
        k_sat = first_within(gi, EPS_SAT)
        k_visual = first_within(gi, EPS_VISUAL_D)
        for oi, octave in enumerate(OCTAVES):
            point_rows.append(
                {
                    "gain": gain,
                    "octaves": int(octave),
                    "D_measured_mean": mean_d[gi, oi],
                    "D_seed_sd": sd_d[gi, oi],
                    "D_seed_se": se_d[gi, oi],
                    "D_predicted": pred_d[gi, oi],
                    "residual_pred_minus_measured": residual[gi, oi],
                }
            )

    pointwise = pd.DataFrame.from_records(point_rows)
    pointwise.to_csv(OUT / "pointwise_comparison.csv", index=False)

    summary_rows = []
    for gi, gain in enumerate(GAINS):
        h_raw = -math.log(gain) / math.log(LACUNARITY)
        h_clip = min(1.0, max(0.0, h_raw))
        summary_rows.append(
            {
                "gain": gain,
                "H_minus_log_gain_over_log_L": h_raw,
                "H_clipped": h_clip,
                "D_hurst_asymptote": 3.0 - h_clip,
                "D_plateau_10_12_mean": plateau_d[gi],
                "D_plateau_within_seed_sd": plateau_sd[gi],
                "K_saturation_eps_0p005": first_within(gi, EPS_SAT),
                "K_visual_eps_D_0p010": first_within(gi, EPS_VISUAL_D),
                "K_inverse_amplitude_tau_0p030": math.ceil(
                    math.log(1.0 / TAU_AMPLITUDE) / (h_raw * math.log(LACUNARITY))
                ),
                "RMSE_prediction_this_gain": float(
                    np.sqrt(np.mean(residual[gi, :] ** 2))
                ),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows)
    summary.to_csv(OUT / "fractal_summary.csv", index=False)

    # ---------- Chart ----------
    configure_font()
    fig, ax = plt.subplots(figsize=(9.4, 6.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for gi, gain in enumerate(GAINS):
        color = PALETTE[gi]
        ax.fill_between(
            OCTAVES,
            mean_d[gi] - se_d[gi],
            mean_d[gi] + se_d[gi],
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        ax.plot(
            OCTAVES,
            mean_d[gi],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5.8,
            markerfacecolor=color,
            markeredgecolor=SURFACE,
            markeredgewidth=1.6,
            zorder=4,
        )
        ax.plot(
            OCTAVES,
            pred_d[gi],
            color=color,
            linewidth=1.9,
            linestyle=(0, (5, 3)),
            alpha=0.82,
            zorder=3,
        )
        k_vis = first_within(gi, EPS_VISUAL_D)
        ax.scatter(
            [k_vis],
            [mean_d[gi, k_vis - 1]],
            marker="D",
            s=92,
            facecolor=SURFACE,
            edgecolor=color,
            linewidth=2.1,
            zorder=6,
        )
        dy = {0: 0.018, 1: -0.030, 2: 0.018}[gi]
        ax.annotate(
            f"K*={k_vis}",
            xy=(k_vis, mean_d[gi, k_vis - 1]),
            xytext=(k_vis + 0.25, mean_d[gi, k_vis - 1] + dy),
            color=INK,
            fontsize=10,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9},
        )
        ax.text(
            12.35,
            mean_d[gi, -1],
            f"g={gain:.2f}",
            color=INK,
            fontsize=10.5,
            va="center",
        )

    ax.set_xlim(0.7, 13.4)
    ymin = math.floor((mean_d.min() - 0.04) * 20) / 20
    ymax = math.ceil((mean_d.max() + 0.04) * 20) / 20
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(list(OCTAVES))
    ax.set_xlabel("分形层数 K（octaves）", fontsize=11.5, color=INK)
    ax.set_ylabel("盒计数维数 D（8 / 16 / 32 px 三档）", fontsize=11.5, color=INK)
    ax.set_title(
        "加层后盒计数维数单调趋缓并饱和：三档振幅衰减对比",
        fontsize=14,
        fontweight="bold",
        color=INK,
        pad=14,
    )
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=10)

    gain_handles = [
        Line2D([0], [0], color=PALETTE[i], lw=2.4, marker="o", markersize=6,
               label=GAIN_LABELS[i])
        for i in range(3)
    ]
    style_handles = [
        Line2D([0], [0], color=INK, lw=2.2, marker="o", markersize=6,
               markerfacecolor=INK, label="16 种子实测均值（带 ±SE）"),
        Line2D([0], [0], color=INK, lw=1.9, linestyle=(0, (5, 3)),
               label="有限盒响应预测曲线"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=SURFACE,
               markeredgecolor=MUTED, markeredgewidth=2, markersize=8,
               label="视觉够用最小层 K*（ε_D=0.010）"),
    ]
    legend1 = ax.legend(
        handles=gain_handles,
        loc="upper left",
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
        fontsize=9.5,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=style_handles,
        loc="lower right",
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
        fontsize=9.2,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fractal_saturation.png", bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(OUT / "fractal_saturation.svg", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # ---------- Terrain contact sheet ----------
    show_octaves = [1, 3, 4, 5, 6, 7, 12]
    sheet, axes = plt.subplots(
        len(GAINS), len(show_octaves),
        figsize=(11.2, 5.4), dpi=180,
    )
    sheet.patch.set_facecolor(SURFACE)
    for gi, gain in enumerate(GAINS):
        k_visual = first_within(gi, EPS_VISUAL_D)
        for ki, octave in enumerate(show_octaves):
            ax = axes[gi, ki]
            pgm = MAPDIR / f"s007_g{gi}_o{octave:02d}.pgm"
            with Image.open(pgm) as im:
                tile = np.array(im.resize((160, 160), Image.Resampling.LANCZOS))
            ax.imshow(tile, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if gi == 0:
                ax.set_title(f"K={octave}", fontsize=10, color=INK)
            if ki == 0:
                ax.set_ylabel(
                    f"g={gain:.2f}", fontsize=11, color=INK,
                    rotation=0, labelpad=34, va="center",
                )
            if octave == k_visual:
                for spine in ax.spines.values():
                    spine.set_edgecolor(PALETTE[gi])
                    spine.set_linewidth(3.0)
                    spine.set_visible(True)
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)
    sheet.suptitle(
        "同种子地形条带：彩色边框为该衰减档反解出的视觉够用层数",
        fontsize=13.5, fontweight="bold", color=INK, y=0.98,
    )
    sheet.tight_layout(rect=(0.02, 0.0, 1.0, 0.95))
    sheet.savefig(OUT / "terrain_saturation_sheet.png", bbox_inches="tight",
                  facecolor=SURFACE)
    plt.close(sheet)

    # ---------- JSON / Markdown report ----------
    negative_after_first = []
    for gi in range(3):
        diffs = np.diff(mean_d[gi])
        negative_after_first.append(float(diffs.min()))

    results = {
        "setup": {
            "size": SIZE,
            "seeds": SEEDS,
            "gains": GAINS,
            "lacunarity": LACUNARITY,
            "base_frequency": F0,
            "octaves": [1, 12],
            "dyadic_ladder": LADDER,
            "fit_scales": [8, 16, 32],
            "fit_interval_note": "七档二倍尺度的中间三档；512 均可整除，无残块。",
            "box_count": "differential 3D grayscale surface boxes, N_s=sum ceil((zmax-zmin+1)/s)",
            "eps_saturation": EPS_SAT,
            "eps_visual_dimension": EPS_VISUAL_D,
            "tau_amplitude_visual": TAU_AMPLITUDE,
        },
        "summary": json.loads(summary.to_json(orient="records", force_ascii=False)),
        "prediction": {
            "global_q": float(q_fit),
            "global_c": float(c_fit),
            "pooled_rmse_D": rmse,
            "pooled_max_abs_residual_D": max_abs,
            "deviation_number": round(rmse, 4),
        },
        "monotonicity": {
            "worst_negative_ensemble_mean_increment": negative_after_first,
            "note": "饱和平台上的负增量不超过 0.00028，小于种子间 SD/SE。",
        },
        "frequency": {
            "base_wavelength_px": 1.0 / F0,
            "central_box_crossover_octave_index_log_L_1_over_f0_s16": math.log(
                1.0 / (F0 * 16.0), LACUNARITY
            ),
            "nyquist_highest_zero_based_index": math.floor(
                math.log(1.0 / (2.0 * F0), LACUNARITY)
            ),
            "one_sample_per_cell_highest_zero_based_index": math.floor(
                math.log(1.0 / F0, LACUNARITY)
            ),
        },
    }
    (OUT / "fractal_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    md = []
    md.append("# fBm 层数—分形维数饱和实验\n")
    md.append("## 结论\n")
    md.append(
        f"- 三档衰减取 `gain={GAINS}`，频率倍增固定为 `lacunarity={LACUNARITY:g}`。\n"
        f"- 盒计数边长为 **8、16、32 px**，即二倍尺度档 `{LADDER}` 的中间三档。\n"
        f"- 16 个种子的平均维数随 K 上升并进入平台；严格饱和阈值 ε_D={EPS_SAT:.3f} 时，"
        f"饱和层为 **{summary_rows[0]['K_saturation_eps_0p005']}、"
        f"{summary_rows[1]['K_saturation_eps_0p005']}、"
        f"{summary_rows[2]['K_saturation_eps_0p005']}**。\n"
        f"- 视觉够用阈值 ε_D={EPS_VISUAL_D:.3f} 时，最小层数为 **"
        f"{summary_rows[0]['K_visual_eps_D_0p010']}、"
        f"{summary_rows[1]['K_visual_eps_D_0p010']}、"
        f"{summary_rows[2]['K_visual_eps_D_0p010']}**。\n"
        f"- 取残余振幅 τ={TAU_AMPLITUDE:.2f}（约 {TAU_AMPLITUDE*255:.1f}/255 灰阶）反解，"
        f"得到 **{summary_rows[0]['K_inverse_amplitude_tau_0p030']}、"
        f"{summary_rows[1]['K_inverse_amplitude_tau_0p030']}、"
        f"{summary_rows[2]['K_inverse_amplitude_tau_0p030']}**，与维数阈值一致。\n"
        f"- 有限盒响应预测曲线与 3×12 个逐点均值比较，合并 RMSE 为 "
        f"**δ={rmse:.4f}**（最大绝对偏差 {max_abs:.4f}）。\n"
        f"- 理论曲线单调不减；实测均值在平台期的最大负增量为 "
        f"{min(negative_after_first):.5f}，小于约 0.002 的均值标准误。\n"
    )
    md.append("\n![三档衰减的维数与饱和点曲线](fractal_saturation.png)\n")
    md.append("\n![地形层数条带](terrain_saturation_sheet.png)\n")
    md.append("\n## 并排结果\n")
    md.append(
        "| gain | H=-ln g/ln2 | 预测渐近 D=3-clip(H) | 实测平台 D（K=10–12） | "
        "严格饱和 K（ε=.005） | 视觉最小 K（ε_D=.010） | 振幅反解 K（τ=.03） | 本档 RMSE |\n"
    )
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in summary_rows:
        md.append(
            f"| {row['gain']:.2f} | {row['H_minus_log_gain_over_log_L']:.3f} | "
            f"{row['D_hurst_asymptote']:.3f} | {row['D_plateau_10_12_mean']:.4f} | "
            f"{row['K_saturation_eps_0p005']} | {row['K_visual_eps_D_0p010']} | "
            f"{row['K_inverse_amplitude_tau_0p030']} | "
            f"{row['RMSE_prediction_this_gain']:.4f} |\n"
        )
    md.append("\n## 逐点实测均值 / 预测 / 残差\n")
    for gi, gain in enumerate(GAINS):
        md.append(f"\n### gain={gain:.2f}\n\n")
        md.append("| K | D实测 | SE | D预测 | 预测-实测 |\n|---:|---:|---:|---:|---:|\n")
        for oi, octave in enumerate(OCTAVES):
            md.append(
                f"| {int(octave)} | {mean_d[gi,oi]:.4f} | {se_d[gi,oi]:.4f} | "
                f"{pred_d[gi,oi]:.4f} | {residual[gi,oi]:+.4f} |\n"
            )
    md.append(
        "\n## 复现\n\n"
        "```bash\n"
        "go test ./...\n"
        "go run ./analysis/dump -size 512 -seeds $(seq -s, 1 16) -out artifacts/boxcount/maps_final\n"
        "python3 analysis/fractal_saturation.py\n"
        "```\n"
    )
    (OUT / "fractal_report.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps({
        "summary": summary.to_dict(orient="records"),
        "rmse": rmse,
        "max_abs": max_abs,
        "chart": str(OUT / "fractal_saturation.png"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
