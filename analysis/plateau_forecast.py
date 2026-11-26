#!/usr/bin/env python3
"""Pre-generation extrema-density forecast and post-generation plateau audit.

The script is intentionally self-contained: the Noise class is an exact Python
port of internal/noise/noise.go.  Actual plateau counts are read from raw
float64 output written by cmd/dumpfloats, i.e. straight from api.Generate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import linregress
from skimage.measure import label, regionprops

# ----------------------------- task constants -----------------------------

SIZE = 128
F0 = 0.02
LACUNARITY = 2.0
OCTAVES = 4
GAIN = 0.5

# Plateau rule, frozen before the seed-by-seed audit.
HIGH = 0.75
SLOPE_MAX = 0.012
MIN_AREA_PX2 = 64

# The mandated density window contains exactly 16 grid cells: 4 x 4.
DENSITY_WINDOW_CELLS = 16
DENSITY_WINDOW_SHAPE = (4, 4)
# A side-16 window is also reported so the alternative reading is explicit.
SIDE16_WINDOW_SHAPE = (16, 16)

GROUP_A = range(1, 31)
GROUP_B = range(31, 61)
SWEEP_SEEDS = range(1, 33)
HOLDOUT_SEEDS = range(501, 517)

PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "red": "#d03b3b",
    "green": "#0ca30c",
    "ink": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#d9d8d3",
    "light_blue": "#cde2fb",
}


# -------------------------- exact Go noise port ---------------------------


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


class Noise:
    """Exact port of the splitmix-seeded permutation value noise."""

    def __init__(self, seed: int):
        p = list(range(256))
        state = int(seed)
        for i in range(255, 0, -1):
            state = splitmix64(state)
            j = state % (i + 1)
            p[i], p[j] = p[j], p[i]
        self.perm = np.array(p + p, dtype=np.int64)

    @staticmethod
    def fade(t: np.ndarray | float) -> np.ndarray | float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def value_grid(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx = x - x0
        fy = y - y0
        u = self.fade(fx)
        v = self.fade(fy)
        perm = self.perm

        def corner(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
            return perm[(perm[ix & 255] + iy) & 511] / 127.5 - 1.0

        c00 = corner(x0, y0)
        c10 = corner(x0 + 1, y0)
        c01 = corner(x0, y0 + 1)
        c11 = corner(x0 + 1, y0 + 1)
        a = c00 + (c10 - c00) * u
        b = c01 + (c11 - c01) * u
        return a + (b - a) * v


def frequencies(octaves: int = OCTAVES, frequency: float = F0) -> np.ndarray:
    return np.array([frequency * LACUNARITY**k for k in range(octaves)], dtype=float)


def weights(octaves: int = OCTAVES, gain: float = GAIN) -> np.ndarray:
    a = np.array([gain**k for k in range(octaves)], dtype=float)
    return a / a.sum()


def fbm_from_noise(
    noise: Noise,
    coords: np.ndarray | None = None,
    octaves: int = OCTAVES,
    gain: float = GAIN,
    frequency: float = F0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coords is None:
        coords = np.arange(SIZE, dtype=float)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    fs = frequencies(octaves, frequency)
    w = weights(octaves, gain)
    layers = np.stack([noise.value_grid(xx * f, yy * f) for f in fs])
    raw = np.tensordot(w, layers, axes=(0, 0))
    return raw, layers, w


def normalized(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi == lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def port_heightmap(seed: int) -> np.ndarray:
    raw, _, _ = fbm_from_noise(Noise(seed))
    return normalized(raw)


# --------------------------- geomorphometry -------------------------------


def count_plateaus(h: np.ndarray, dx: float = 1.0) -> dict[str, Any]:
    gy, gx = np.gradient(h, dx)
    slope = np.hypot(gx, gy)
    mask = (h >= HIGH) & (slope <= SLOPE_MAX)
    lab = label(mask, connectivity=2)
    props = regionprops(lab)
    kept = [p for p in props if p.area * dx * dx >= MIN_AREA_PX2]
    keep_mask = np.isin(lab, np.array([p.label for p in kept], dtype=int)) if kept else np.zeros_like(mask)
    areas = sorted((float(p.area * dx * dx) for p in kept), reverse=True)
    centroids = [(float(p.centroid[1]), float(p.centroid[0])) for p in kept]
    boundary = 0
    for p in kept:
        minr, minc, maxr, maxc = p.bbox
        if minr == 0 or minc == 0 or maxr == h.shape[0] or maxc == h.shape[1]:
            boundary += 1
    return {
        "count": len(kept),
        "mask": keep_mask,
        "raw_mask": mask,
        "labels": lab,
        "areas": areas,
        "centroids": centroids,
        "boundary_count": boundary,
        "slope_mean": float(slope.mean()),
        "flat_fraction": float((slope <= SLOPE_MAX).mean()),
        "high_flat_fraction": float(mask.mean()),
    }


def extrema_centroids(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centroids and signs (+1 max, -1 min) of one-cell/table clusters."""
    maxf = ndi.maximum_filter(a, size=3, mode="nearest")
    minf = ndi.minimum_filter(a, size=3, mode="nearest")
    max_mask = a == maxf
    min_mask = a == minf
    # Outer frame has no complete 3x3 neighbourhood in-image.
    max_mask[0, :] = max_mask[-1, :] = max_mask[:, 0] = max_mask[:, -1] = False
    min_mask[0, :] = min_mask[-1, :] = min_mask[:, 0] = min_mask[:, -1] = False

    points: list[tuple[float, float]] = []
    signs: list[int] = []
    for mask, sign in ((max_mask, 1), (min_mask, -1)):
        # connectivity=2 in 2-D uses the full 8-neighbour neighbourhood.
        lab, n = label(mask, connectivity=2, return_num=True)
        for prop in regionprops(lab):
            cy, cx = prop.centroid
            points.append((cy, cx))
            signs.append(sign)
    if not points:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    return np.asarray(points, dtype=float), np.asarray(signs, dtype=int)


def window_audit(points: np.ndarray, shape: tuple[int, int]) -> dict[str, float | int]:
    wh, ww = shape
    nh, nw = SIZE // wh, SIZE // ww
    counts = np.zeros((nh, nw), dtype=int)
    if len(points):
        cy, cx = points[:, 0], points[:, 1]
        iy = np.clip(np.floor(cy / wh).astype(int), 0, nh - 1)
        ix = np.clip(np.floor(cx / ww).astype(int), 0, nw - 1)
        np.add.at(counts, (iy, ix), 1)
    total_cells = SIZE * SIZE
    return {
        "window_side_y": wh,
        "window_side_x": ww,
        "window_area_cells": wh * ww,
        "window_count": nh * nw,
        "extrema": int(len(points)),
        "density_per_cell": len(points) / total_cells,
        "mean_window_count": float(counts.mean()),
        "max_window_count": int(counts.max(initial=0)),
        "empty_windows": int((counts == 0).sum()),
        "single_extrema_windows": int((counts == 1).sum()),
        "multi_extrema_windows": int((counts >= 2).sum()),
    }


def overlapping_window_density(points: np.ndarray, shape: tuple[int, int] = DENSITY_WINDOW_SHAPE) -> float:
    """Density if every translated 4x4 window is used (overlapping audit)."""
    binary = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for cy, cx in points:
        iy, ix = int(round(cy)), int(round(cx))
        if 0 <= iy < SIZE and 0 <= ix < SIZE:
            binary[iy, ix] = 1
    wh, ww = shape
    windows = np.lib.stride_tricks.sliding_window_view(binary, (wh, ww))
    counts = windows.sum(axis=(-2, -1))
    return float(counts.sum() / (counts.shape[0] * counts.shape[1] * wh * ww))


def sparse_forecast(seed: int) -> tuple[int, dict[str, Any], np.ndarray]:
    """Density-guided 2-px lattice forecast; no 128x128 final render is read."""
    coords = np.arange(0, SIZE, 2, dtype=float)
    raw, _, _ = fbm_from_noise(Noise(seed), coords=coords)
    h = normalized(raw)
    audit = count_plateaus(h, dx=2.0)
    return audit["count"], audit, h


# ------------------------------ spectra -----------------------------------


def spectral_slope(field: np.ndarray, lo_lambda: float = 8.0, hi_lambda: float = 50.0) -> float:
    n = field.shape[0]
    window = np.outer(np.hanning(n), np.hanning(n))
    centered = (field - field.mean()) * window
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    spectrum /= n * n * float((window**2).mean())

    ky = np.fft.fftshift(np.fft.fftfreq(n, d=1.0)) * n
    qx, qy = np.meshgrid(ky, ky)
    q = np.hypot(qx, qy)
    qbin = np.floor(q).astype(int)
    bins: dict[int, list[float]] = {}
    for b, p in zip(qbin.ravel(), spectrum.ravel()):
        if b >= 1:
            bins.setdefault(int(b), []).append(float(p))

    xs: list[float] = []
    ys: list[float] = []
    for b, vals in bins.items():
        qmid = b + 0.5
        wavelength = n / qmid
        if lo_lambda <= wavelength <= hi_lambda:
            k_ang = 2 * math.pi * qmid / n
            xs.append(math.log(k_ang))
            ys.append(math.log(float(np.mean(vals))))
    if len(xs) < 3:
        return float("nan")
    return float(-np.polyfit(np.asarray(xs), np.asarray(ys), 1)[0])


def beta_theory(gain: float) -> float:
    h = -math.log(gain) / math.log(LACUNARITY)
    return 2.0 + 2.0 * h


def spectral_moment_q(k: int, gain: float) -> float:
    fs = frequencies(k)
    w = weights(k, gain)
    return float(np.sum(w * w * fs**4) / np.sum(w * w * fs**2))


# ------------------------------- plotting ---------------------------------


def setup_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def save_plain_thumbnail(h: np.ndarray, path: Path) -> None:
    g = np.clip(np.rint(h * 255), 0, 255).astype(np.uint8)
    Image.fromarray(g, mode="L").resize((256, 256), Image.Resampling.NEAREST).save(path)


def save_overlay(h: np.ndarray, mask: np.ndarray, seed: int, actual: int, pred: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=150)
    ax.imshow(h, cmap="gray", vmin=0, vmax=1)
    if mask.any():
        ax.contour(mask, levels=[0.5], colors=PALETTE["red"], linewidths=0.9)
    ax.set_title(f"seed {seed}  实际 {actual} / 预判 {pred}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def save_contact_sheet(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    fig, axes = plt.subplots(5, 6, figsize=(15, 12.5), dpi=140)
    for ax, row in zip(axes.ravel(), rows):
        ax.imshow(row["h"], cmap="gray", vmin=0, vmax=1)
        if row["mask"].any():
            ax.contour(row["mask"], levels=[0.5], colors=PALETTE["red"], linewidths=0.75)
        flag = " ⚠" if abs(row["error"]) > 2 else ""
        ax.set_title(
            f"{row['seed']}: 预{row['pred']} / 实{row['actual']}{flag}",
            fontsize=8.5,
            color=PALETTE["red"] if flag else PALETTE["ink"],
        )
        ax.axis("off")
    for ax in axes.ravel()[len(rows) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path)
    plt.close(fig)


def make_density_figure(sweep: pd.DataFrame, fit: dict[str, float], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)
    ax = axes[0]
    gains = sorted(sweep["gain"].unique())
    blues = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    for gain, color in zip(gains, blues):
        sub = sweep[sweep["gain"] == gain].sort_values("K")
        ax.plot(sub["K"], sub["rho"], marker="o", linewidth=2, color=color,
                label=f"β={beta_theory(gain):.2f}, g={gain:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("层数 K")
    ax.set_ylabel("极值密度 ρ（个/格²）")
    ax.set_title("极值密度随谱斜率与层数上升")
    ax.grid(True, which="major", color=PALETTE["grid"], linewidth=0.7)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    pred = np.exp(
        fit["intercept"]
        + fit["b_beta"] * sweep["beta_theory"]
        + fit["b_K"] * sweep["K"]
        + fit["b_betaK"] * sweep["beta_theory"] * sweep["K"]
        + fit["b_K2"] * sweep["K"] ** 2
    )
    ax.scatter(sweep["rho"], pred, s=42, color=PALETTE["blue"], edgecolor="white", linewidth=0.7)
    lo = float(min(sweep["rho"].min(), pred.min())) * 0.85
    hi = float(max(sweep["rho"].max(), pred.max())) * 1.15
    ax.plot([lo, hi], [lo, hi], color=PALETTE["orange"], linewidth=2, label="identity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("实测平均 ρ")
    ax.set_ylabel("响应面预测 ρ")
    ax.set_title(f"密度函数拟合：R²(log)={fit['r2_log']:.4f}")
    ax.grid(True, which="major", color=PALETTE["grid"], linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_flatness_figure(df: pd.DataFrame, reg: dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=160)
    colors = np.where(df["abs_error"] > 2, PALETTE["red"], PALETTE["blue"])
    ax.scatter(df["rho_low4x4"], df["rho_final"], s=34, c=colors, edgecolor="white", linewidth=0.6)
    xs = np.linspace(df["rho_low4x4"].min(), df["rho_low4x4"].max(), 100)
    ax.plot(xs, reg["intercept"] + reg["slope"] * xs, color=PALETTE["orange"], linewidth=2.2)
    ax.set_xlabel("前 3 层低频分量极值密度（4×4=16 格非重叠窗）")
    ax.set_ylabel("最终图极值密度")
    ax.set_title(f"低频骨架决定破碎度：r={reg['r']:.3f}, p={reg['pvalue']:.2g}")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_prediction_figure(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160, sharey=True)
    for ax, group, title in ((axes[0], "A", "甲组 seeds 1–30"), (axes[1], "B", "乙组 seeds 31–60")):
        sub = df[df["group"] == group].reset_index(drop=True)
        jitter = ((np.arange(len(sub)) % 7) - 3) * 0.07
        ax.scatter(sub["pred"] + jitter, sub["actual"] + jitter * 0.6, s=42,
                   color=PALETTE["blue"], alpha=0.82, label="逐种子（轻微抖动）",
                   edgecolor="white", linewidth=0.5)
        xs = np.arange(0, max(sub["actual"].max(), sub["pred"].max()) + 2)
        ax.plot(xs, xs, color=PALETTE["ink"], linewidth=1.6)
        ax.fill_between(xs, xs - 2, xs + 2, color=PALETTE["aqua"], alpha=0.13, label="±2 容差")
        bad = sub[sub["abs_error"] > 2]
        if len(bad):
            ax.scatter(bad["pred"], bad["actual"], s=110, facecolors="none",
                       edgecolors=PALETTE["red"], linewidths=1.8, label="偏差>2")
        ax.set_title(title)
        ax.set_xlabel("预判高原数")
        ax.grid(True, color=PALETTE["grid"], linewidth=0.7)
        ax.set_xticks(range(0, int(max(sub["pred"].max(), sub["actual"].max())) + 2))
    axes[0].set_ylabel("生成后实测高原数")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------- tables -----------------------------------


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---:" if key not in {"seed", "status", "reason"} else "---" for key, _ in columns) + "|"
    lines = [head, sep]
    for row in rows:
        vals = []
        for key, _ in columns:
            v = row[key]
            if isinstance(v, float):
                vals.append(f"{v:.6f}" if key.startswith("rho") else f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def reason_for(row: dict[str, Any]) -> str:
    if row["error"] > 2:
        return ("粗格把相邻高平顶在 2 px 采样上桥接/合并，预判偏多；这些种子的高频沟较窄，"
                "全分辨率才把平台切开。")
    if row["error"] < -2:
        small = row["actual_area_min_retained"]
        return (f"实测偏多：有 3 个以上面积靠近 64 px² 阈值（最小保留块 {small:.0f} px²）"
                "的小/窄平顶；2 px 稀疏格漏掉窄颈或亚样宽度，且粗格 min–max 归一化改变 0.75 阈值边界。")
    return "误差在两个以内：采样密度足以保留主导平顶，差异主要来自阈值边界像元。"


# -------------------------------- main ------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--float-dir", type=Path, default=root / "work" / "go_floats")
    parser.add_argument("--outdir", type=Path, default=root / "artifacts" / "plateau_forecast")
    args = parser.parse_args()

    out = args.outdir
    data_dir = out / "data"
    table_dir = out / "tables"
    fig_dir = out / "figures"
    thumb_dir = out / "thumbnails"
    overlay_dir = out / "overlays"
    for d in (data_dir, table_dir, fig_dir, thumb_dir / "A", thumb_dir / "B",
              overlay_dir / "A", overlay_dir / "B"):
        d.mkdir(parents=True, exist_ok=True)

    setup_font()

    # ---------- density / spectral-slope / octave sweep ----------
    sweep_rows: list[dict[str, Any]] = []
    for k in range(1, 7):
        for gain in (0.35, 0.425, GAIN, 0.575, 0.65):
            densities: list[float] = []
            betas: list[float] = []
            for seed in SWEEP_SEEDS:
                raw, _, _ = fbm_from_noise(Noise(seed), octaves=k, gain=gain)
                points, _ = extrema_centroids(raw)
                densities.append(len(points) / (SIZE * SIZE))
                betas.append(spectral_slope(normalized(raw)))
            rho = float(np.mean(densities))
            sweep_rows.append({
                "K": k,
                "gain": gain,
                "lacunarity": LACUNARITY,
                "beta_theory": beta_theory(gain),
                "beta_empirical_128": float(np.nanmean(betas)),
                "beta_empirical_sd": float(np.nanstd(betas)),
                "rho": rho,
                "rho_per_16cell_window": rho * DENSITY_WINDOW_CELLS,
                "Q42": spectral_moment_q(k, gain),
            })
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(data_dir / "density_sweep.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    X = np.column_stack([
        np.ones(len(sweep)),
        sweep["beta_theory"].to_numpy(),
        sweep["K"].to_numpy(),
        sweep["beta_theory"].to_numpy() * sweep["K"].to_numpy(),
        sweep["K"].to_numpy() ** 2,
    ])
    y_log = np.log(sweep["rho"].to_numpy())
    coef, *_ = np.linalg.lstsq(X, y_log, rcond=None)
    pred_log = X @ coef
    ss_res = float(np.sum((y_log - pred_log) ** 2))
    ss_tot = float(np.sum((y_log - y_log.mean()) ** 2))
    r2_log = 1 - ss_res / ss_tot
    mape = float(np.mean(np.abs(np.exp(pred_log) - sweep["rho"]) / sweep["rho"]))

    # Independent seed holdout for the fitted response surface.
    holdout_rows: list[dict[str, Any]] = []
    for k in range(1, 7):
        for gain in (0.35, 0.425, GAIN, 0.575, 0.65):
            ds = []
            for seed in HOLDOUT_SEEDS:
                raw, _, _ = fbm_from_noise(Noise(seed), octaves=k, gain=gain)
                pts, _ = extrema_centroids(raw)
                ds.append(len(pts) / SIZE**2)
            beta = beta_theory(gain)
            rho_pred = math.exp(
                coef[0] + coef[1] * beta + coef[2] * k + coef[3] * beta * k + coef[4] * k * k
            )
            holdout_rows.append({
                "K": k, "gain": gain, "beta_theory": beta,
                "rho_holdout": float(np.mean(ds)), "rho_pred": rho_pred,
                "relative_error": abs(rho_pred - float(np.mean(ds))) / float(np.mean(ds)),
            })
    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(data_dir / "density_holdout.csv", index=False)
    holdout_mape = float(holdout["relative_error"].mean())

    fit = {
        "intercept": float(coef[0]),
        "b_beta": float(coef[1]),
        "b_K": float(coef[2]),
        "b_betaK": float(coef[3]),
        "b_K2": float(coef[4]),
        "r2_log": float(r2_log),
        "fit_mape": mape,
        "holdout_mape": holdout_mape,
        "window_cells": DENSITY_WINDOW_CELLS,
        "window_shape": list(DENSITY_WINDOW_SHAPE),
        "window_overlap": False,
    }
    (data_dir / "density_fit.json").write_text(json.dumps(fit, ensure_ascii=False, indent=2) + "\n")
    make_density_figure(sweep, fit, fig_dir / "density_law.png")

    # ---------- seed-by-seed prediction / actual audit ----------
    go_checks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    overlap_densities: list[float] = []
    contact_rows: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}

    all_seeds = list(GROUP_A) + list(GROUP_B)
    for seed in all_seeds:
        f64_path = args.float_dir / f"seed_{seed:04d}.f64"
        if not f64_path.exists():
            raise FileNotFoundError(
                f"缺少 {f64_path}; 先运行 go run ./cmd/dumpfloats --seeds 1-60 --out {args.float_dir}"
            )
        h_go = np.fromfile(f64_path, dtype="<f8").reshape(SIZE, SIZE)

        h_port = port_heightmap(seed)
        max_diff = float(np.max(np.abs(h_go - h_port)))
        go_checks.append({"seed": seed, "max_abs_diff": max_diff})

        actual_audit = count_plateaus(h_go, dx=1.0)
        actual = int(actual_audit["count"])
        pred, pred_audit, h_coarse = sparse_forecast(seed)

        # Pre-generation low-frequency skeleton: first three octaves.
        raw_low, low_layers, low_w = fbm_from_noise(Noise(seed))
        low3 = np.sum(low_w[:3, None, None] * low_layers[:3], axis=0)
        low_points, low_signs = extrema_centroids(low3)
        rho_low = len(low_points) / SIZE**2
        w4 = window_audit(low_points, DENSITY_WINDOW_SHAPE)
        w16 = window_audit(low_points, SIDE16_WINDOW_SHAPE)
        overlap_rho = overlapping_window_density(low_points, DENSITY_WINDOW_SHAPE)
        overlap_densities.append(overlap_rho)

        final_points, _ = extrema_centroids(h_go)
        rho_final = len(final_points) / SIZE**2

        group = "A" if seed in GROUP_A else "B"
        row: dict[str, Any] = {
            "seed": seed,
            "group": group,
            "beta_theory": beta_theory(GAIN),
            "K": OCTAVES,
            "pred": pred,
            "actual": actual,
            "error": pred - actual,
            "abs_error": abs(pred - actual),
            "within_2": abs(pred - actual) <= 2,
            "rho_low3": rho_low,
            "rho_low4x4": float(w4["density_per_cell"]),
            "low_extrema": int(w4["extrema"]),
            "w4_empty": w4["empty_windows"],
            "w4_multi": w4["multi_extrema_windows"],
            "rho_low16x16": float(w16["density_per_cell"]),
            "w16_empty": w16["empty_windows"],
            "rho_overlap4x4": overlap_rho,
            "rho_final": rho_final,
            "actual_boundary_components": actual_audit["boundary_count"],
            "actual_area_min_retained": min(actual_audit["areas"]) if actual_audit["areas"] else 0.0,
            "actual_areas": " ".join(f"{a:.0f}" for a in actual_audit["areas"]),
            "pred_areas": " ".join(f"{a:.0f}" for a in pred_audit["areas"]),
            "go_port_max_diff": max_diff,
            "status": "合格" if abs(pred - actual) <= 2 else "超差>2",
        }
        row["reason"] = reason_for(row) if abs(pred - actual) > 2 else ""
        rows.append(row)

        save_plain_thumbnail(h_go, thumb_dir / group / f"seed_{seed:04d}.png")
        save_overlay(
            h_go, actual_audit["mask"], seed, actual, pred,
            overlay_dir / group / f"overlay_seed_{seed:04d}.png",
        )
        contact_rows[group].append({
            "seed": seed, "pred": pred, "actual": actual, "error": pred - actual,
            "h": h_go, "mask": actual_audit["mask"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "seed_forecast_all.csv", index=False)
    pd.DataFrame(go_checks).to_csv(data_dir / "go_port_check.csv", index=False)

    columns = [
        ("seed", "种子"), ("rho_low4x4", "低频极值密度"), ("low_extrema", "极值数"),
        ("pred", "预判数"), ("actual", "实际数"), ("error", "差(预-实)"),
        ("abs_error", "|差|"), ("status", "判定"),
    ]
    for group, seed_range, name in (("A", GROUP_A, "group_A"), ("B", GROUP_B, "group_B")):
        sub = [r for r in rows if r["group"] == group]
        (table_dir / f"{name}.md").write_text(
            f"# {name} 逐种子预判与实测（种子 {seed_range.start}–{seed_range.stop - 1}）\n\n"
            + markdown_table(sub, columns) + "\n",
            encoding="utf-8",
        )
        sub_df = df[df["group"] == group]
        sub_df.to_csv(table_dir / f"{name}.csv", index=False)
        save_contact_sheet(contact_rows[group], fig_dir / f"contact_sheet_{group}.png",
                           f"{'甲' if group == 'A' else '乙'}组：预判 vs 生成后实测（红轮廓=实测高原）")

    bad = df[df["abs_error"] > 2].copy()
    bad.to_csv(data_dir / "deviations_gt2.csv", index=False)
    if len(bad):
        bad_cols = [
            ("seed", "种子"), ("pred", "预判"), ("actual", "实际"), ("error", "偏差"),
            ("actual_areas", "实测块面积"), ("actual_area_min_retained", "最小保留面积"),
            ("reason", "原因"),
        ]
        (table_dir / "deviations_gt2.md").write_text(
            "# 偏差超过两个的种子\n\n" + markdown_table(bad.to_dict("records"), bad_cols) + "\n",
            encoding="utf-8",
        )

    reg = linregress(df["rho_low4x4"], df["rho_final"])
    reg_json = {
        "slope": float(reg.slope),
        "intercept": float(reg.intercept),
        "r": float(reg.rvalue),
        "r2": float(reg.rvalue**2),
        "pvalue": float(reg.pvalue),
        "stderr": float(reg.stderr),
    }
    (data_dir / "flatness_regression.json").write_text(json.dumps(reg_json, indent=2) + "\n")
    make_flatness_figure(df, reg_json, fig_dir / "flatness_link.png")
    make_prediction_figure(df, fig_dir / "pred_vs_actual.png")

    # Aggregate summary.
    summary = {
        "seeds": len(df),
        "within_2": int(df["within_2"].sum()),
        "gt2": int((~df["within_2"]).sum()),
        "mae": float(df["abs_error"].mean()),
        "bias": float(df["error"].mean()),
        "group_A_within2": int(df[df.group == "A"]["within_2"].sum()),
        "group_B_within2": int(df[df.group == "B"]["within_2"].sum()),
        "go_port_max_diff_all": float(df["go_port_max_diff"].max()),
        "overlap4x4_mean_density": float(np.mean(overlap_densities)),
        "nonoverlap4x4_mean_density": float(df["rho_low4x4"].mean()),
        "overlap_over_nonoverlap": float(np.mean(overlap_densities) / df["rho_low4x4"].mean()),
        "density_fit": fit,
        "flatness_regression": reg_json,
        "bad_seeds": bad["seed"].tolist(),
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
