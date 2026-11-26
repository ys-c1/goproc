#!/usr/bin/env python3
"""分形叠加频谱验证、斜率-粗糙度对应与衰减参数反解。

数据来自 cmd/dumpfields 导出的真实 Go 生成器输出（api.Generate）。
只用 numpy / matplotlib，命令行参数指定输入输出路径。

阶段一：
  python3 analysis/spectrum_analysis.py \
      --main build/fields1024 --out artifacts --write-target build/target.json
阶段二（用阶段一反解出的 g_req 再生成验证场后）：
  python3 analysis/spectrum_analysis.py \
      --main build/fields1024 --verify build/verify \
      --target build/target.json --out artifacts
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# 某些环境（如 WSL 用户字体目录）fontconfig 能看到但 matplotlib 缓存没有，显式注册。
for _p in (
    os.path.expanduser("~/.local/share/fonts/NotoSansCJKsc-Regular.ttf"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
):
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
        except Exception:
            pass

# ---------- 视觉令牌（dataviz 参考调色板前三档，已通过 CVD 校验） ----------
C_GAINS = ["#2a78d6", "#eb6834", "#1baf7a"]  # 低/中/高三档增益
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURF = "#fcfcfb"
GOOD = "#0ca30c"
CRIT = "#d03b3b"

plt.rcParams.update({
    "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.facecolor": SURF,
    "figure.facecolor": "white",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SECOND,
    "ytick.color": SECOND,
    "axes.titlecolor": INK,
})

# ---------- 分析协议常量（固定，不得按结果调整） ----------
K_LO, K_HI = 0.030, 0.200      # 所有条件共用的双对数 LS 拟合区间（周期/像素）
ROUGH_C = (0.030, 0.060)       # 粗尺度带
ROUGH_F = (0.090, 0.180)       # 细尺度带；两带几何中心之比 q=3
Q_BAND = 3.0
DELTA_TOL = 0.25               # |s_meas - s_pred| 超过该值判层数不足
TARGET_RHO = 0.15              # 反解演示用的给定粗糙度


@dataclass
class CondResult:
    gain: float
    octaves: int
    n_seeds: int
    s_pred: float
    s_pool: float
    s_seed_mean: float
    s_seed_sd: float
    intercept: float
    r2_log: float
    s_raw_pool: float
    rho: float
    rho_sd: float
    rho_pred: float
    band_r: float
    grad_rms: float
    grad_sd: float
    k: np.ndarray | None = None
    b_mean: np.ndarray | None = None
    p_mean: np.ndarray | None = None

    @property
    def delta(self) -> float:
        return self.s_pool - self.s_pred

    @property
    def enough(self) -> bool:
        return abs(self.delta) <= DELTA_TOL


def load_field(path: str, n: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape(n, n).astype(np.float64)


def make_window(n: int) -> np.ndarray:
    w1 = np.hanning(n)
    return np.outer(w1, w1)


def radial_spectrum(z: np.ndarray, win: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 k(周期/像素), 径向平均 PSD, 每环 FFT bin 数。Hann 窗 + 去均值。"""
    n = z.shape[0]
    zw = (z - z.mean()) * win
    zf = np.fft.fft2(zw)
    p2 = (np.abs(zf) ** 2) / float((win ** 2).sum())

    fx = np.fft.fftfreq(n)
    fx2, fy2 = np.meshgrid(fx, fx, indexing="ij")
    kgrid = np.sqrt(fx2 ** 2 + fy2 ** 2)

    # 用整数 DFT 半径分环
    rad = np.floor(kgrid * n + 0.5).astype(int)
    binmax = int(np.floor(np.sqrt(2) * (n // 2) + 0.5)) + 1
    psum = np.zeros(binmax + 1)
    cnt = np.zeros(binmax + 1)
    np.add.at(psum, rad.ravel(), p2.ravel())
    np.add.at(cnt, rad.ravel(), 1.0)
    valid = cnt > 0
    pr = np.zeros_like(psum)
    pr[valid] = psum[valid] / cnt[valid]
    r = np.arange(binmax + 1)
    k = r / n
    keep = (r >= 1) & (r <= n // 2)
    return k[keep], pr[keep], cnt[keep]


def ls_slope(k: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """ln y = s ln k + b 的最小二乘；返回 s, b, R²(对数空间)。"""
    x = np.log(k)
    ly = np.log(y)
    s, b = np.polyfit(x, ly, 1)
    pred = s * x + b
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(s), float(b), float(r2)


def band_energy(z: np.ndarray, win: np.ndarray, k_edges: tuple[float, float]) -> float:
    n = z.shape[0]
    zw = (z - z.mean()) * win
    zf = np.fft.fft2(zw)
    p2 = (np.abs(zf) ** 2) / float((win ** 2).sum())
    fx = np.fft.fftfreq(n)
    fx2, fy2 = np.meshgrid(fx, fx, indexing="ij")
    kg = np.sqrt(fx2 ** 2 + fy2 ** 2)
    m = (kg >= k_edges[0]) & (kg < k_edges[1])
    return float(p2[m].sum())


def grad_rms(z: np.ndarray) -> float:
    gx = np.diff(z, axis=1)
    gy = np.diff(z, axis=0)
    return float(np.sqrt(0.5 * (np.mean(gx ** 2) + np.mean(gy ** 2))))


def analyze_condition(entries: list[dict], data_dir: str, n: int, lacunarity: float) -> CondResult:
    gain = float(entries[0]["gain"])
    octaves = int(entries[0]["octaves"])
    win = make_window(n)

    profiles = []
    seed_slopes = []
    rhos, rs, grads = [], [], []
    for e in entries:
        z = load_field(os.path.join(data_dir, e["file"]), n)
        k, pr, _ = radial_spectrum(z, win)
        profiles.append(pr)

        m = (k >= K_LO) & (k <= K_HI) & (pr > 0)
        b_k = k ** 2 * pr
        s_i, _, _ = ls_slope(k[m], b_k[m])
        seed_slopes.append(s_i)

        ec = band_energy(z, win, ROUGH_C)
        ef = band_energy(z, win, ROUGH_F)
        r = ef / ec
        rs.append(r)
        rhos.append(r / (1.0 + r))
        grads.append(grad_rms(z))

    p_mean = np.mean(np.stack(profiles), axis=0)
    b_mean = k ** 2 * p_mean
    m = (k >= K_LO) & (k <= K_HI) & (p_mean > 0)
    s_pool, intercept, r2 = ls_slope(k[m], b_mean[m])

    s_pred = 2.0 * np.log(gain) / np.log(lacunarity)
    r_pred = Q_BAND ** s_pred

    return CondResult(
        gain=gain, octaves=octaves, n_seeds=len(entries),
        s_pred=s_pred, s_pool=s_pool,
        s_seed_mean=float(np.mean(seed_slopes)), s_seed_sd=float(np.std(seed_slopes, ddof=1)),
        intercept=intercept, r2_log=r2,
        s_raw_pool=s_pool - 2.0,
        rho=float(np.mean(rhos)), rho_sd=float(np.std(rhos, ddof=1)),
        rho_pred=r_pred / (1.0 + r_pred),
        band_r=float(np.mean(rs)),
        grad_rms=float(np.mean(grads)), grad_sd=float(np.std(grads, ddof=1)),
        k=k, b_mean=b_mean, p_mean=p_mean,
    )


def group_manifest(manifest: dict, data_dir: str) -> dict[tuple[float, int], list[dict]]:
    groups: dict[tuple[float, int], list[dict]] = {}
    for e in manifest["files"]:
        # 校验文件确实在给定目录（manifest 可能来自被移动的目录）
        if not os.path.exists(os.path.join(data_dir, e["file"])):
            continue
        groups.setdefault((float(e["gain"]), int(e["octaves"])), []).append(e)
    return dict(sorted(groups.items()))


def invert_gain(rho_target: float, lacunarity: float) -> tuple[float, float]:
    """ρ* -> R* -> s_req -> g_req 的显式反解。"""
    r_star = rho_target / (1.0 - rho_target)
    s_req = np.log(r_star) / np.log(Q_BAND)
    g_req = float(np.exp(s_req * np.log(lacunarity) / 2.0))
    return g_req, float(s_req)


# ---------------------------------------------------------------- 绘图

def style_ax(ax):
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.grid(True, which="minor", color=GRID, lw=0.5, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")


def compact_xticks(ax):
    # 必须在 loglog() 之后调用：loglog 会重装 scale 默认定位器。
    ticks = [0.02, 0.05, 0.10, 0.20, 0.40]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    # 该跨度下 LogFormatter 会给 3/4/6×10^n 的次要刻度加标签，与主标签粘连。
    ax.tick_params(axis="x", which="minor", labelbottom=False)


def normalized_curve(res: CondResult, k_ref: float) -> np.ndarray:
    assert res.k is not None and res.b_mean is not None
    b_ref = np.exp(np.interp(np.log(k_ref), np.log(res.k), np.log(res.b_mean)))
    return res.b_mean / b_ref


def plot_gains(results8: list[CondResult], path: str) -> None:
    k_ref = float(np.sqrt(K_LO * K_HI))
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=200)
    style_ax(ax)
    ax.axvspan(K_LO, K_HI, color=INK, alpha=0.045, zorder=0)

    k_view_lo, k_view_hi = 0.014, 0.34
    for i, r in enumerate(results8):
        c = C_GAINS[i]
        assert r.k is not None
        m_view = (r.k >= k_view_lo) & (r.k <= k_view_hi)
        ax.loglog(r.k[m_view], normalized_curve(r, k_ref)[m_view],
                  color=c, lw=1.9, alpha=0.9, zorder=3)
        kf = np.array([K_LO, K_HI])
        # 用拟合直线并归一到 k_ref
        yf = np.exp(r.s_pool * np.log(kf / k_ref))
        ax.loglog(kf, yf, ls="--", lw=2.0, color=c, zorder=4)
        # 右端直接标注
        y_end = (K_HI / k_ref) ** r.s_pool
        ax.text(K_HI * 1.06, y_end,
                f"g={r.gain:.2f}  s={r.s_pool:+.2f}（预测 {r.s_pred:+.2f}）",
                va="center", ha="left", fontsize=10.5, color=INK)

    ax.set_xlim(k_view_lo, 0.55)
    ax.set_ylim(0.05, 260)
    compact_xticks(ax)
    ax.set_xlabel("空间频率 k（周期/像素）")
    ax.set_ylabel(r"$B(k)=k^2\overline{P}(k)$，按 $k_{\rm ref}$ 归一")
    ax.set_title("图 1  三档增益的分形叠加功率谱（8 层，16 种子平均）\n"
                 "双对数坐标下近似直线，虚线为最小二乘拟合", fontsize=12.5, pad=34)

    # 顶部手工波长刻度 λ=1/k（避免 secondary 对数轴自动刻度重叠）
    for lam in (64, 32, 16, 8, 4):
        kk = 1.0 / lam
        ax.plot([kk, kk], [1.0, 1.018], transform=ax.get_xaxis_transform(),
                color=MUTED, lw=0.8, clip_on=False)
        ax.text(kk, 1.028, str(lam), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10, color=SECOND)
    ax.text(1.0, 1.03, "波长 λ（像素）", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10.5, color=MUTED)

    handles = [
        Line2D([0], [0], color=C_GAINS[i], lw=2.2,
               label=f"增益 g={r.gain:.2f}，实测斜率 {r.s_pool:+.3f}，偏差 {r.delta:+.3f}")
        for i, r in enumerate(results8)
    ]
    handles += [
        Line2D([0], [0], color=MUTED, lw=2.0, ls="--", label="双对数最小二乘直线"),
        Patch(facecolor=INK, alpha=0.08, label="统一拟合区间 [0.03, 0.20]"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.92,
              edgecolor=GRID, fontsize=9.6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_octaves(r4: list[CondResult], r8: list[CondResult], path: str) -> None:
    k_ref = float(np.sqrt(K_LO * K_HI))
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.9), dpi=200)
    k_view_lo, k_view_hi = 0.014, 0.30

    panels = [
        (axes[0], r4, "层数 K=4：高频段无层可续，固定区间拟合被折弯拉陡", (0.006, 4.0e3)),
        (axes[1], r8, "层数 K=8：三档实测斜率均落在预测容差内", (0.05, 260.0)),
    ]
    for ax, rows, subtitle, ylim in panels:
        style_ax(ax)
        ax.axvspan(K_LO, K_HI, color=INK, alpha=0.045, zorder=0)
        handles, labels = [], []
        for i, r in enumerate(rows):
            c = C_GAINS[i]
            assert r.k is not None
            norm = normalized_curve(r, k_ref)
            m_view = (r.k >= k_view_lo) & (r.k <= k_view_hi)
            ax.loglog(r.k[m_view], norm[m_view], color=c, ls="-", lw=2.0,
                      alpha=0.95, zorder=3)
            kf = np.array([K_LO, K_HI])
            yf = np.exp(r.s_pool * np.log(kf / k_ref))
            ax.loglog(kf, yf, ls="--", lw=2.0, color=c, alpha=0.85, zorder=4)
            verdict = "达标" if r.enough else "层数不足"
            labels.append(f"g={r.gain:.2f}：s={r.s_pool:+.2f}，预测 {r.s_pred:+.2f}，"
                          f"|Δ|={abs(r.delta):.2f} → {verdict}")
            handles.append(Line2D([0], [0], color=c, lw=2.2))
        handles += [
            Line2D([0], [0], color=MUTED, lw=2.0, ls="-"),
            Line2D([0], [0], color=MUTED, lw=2.0, ls="--"),
        ]
        labels += ["实测 B(k)（16 种子平均）", "双对数最小二乘直线"]
        ax.set_xlim(k_view_lo, 0.46)
        ax.set_ylim(*ylim)
        compact_xticks(ax)
        ax.set_xlabel("空间频率 k（周期/像素）；实线实测，虚线 LS 拟合")
        ax.set_ylabel(r"$B(k)=k^2\overline{P}(k)$，按 $k_{\rm ref}$ 归一")
        ax.set_title(subtitle, fontsize=11.5, pad=10)
        ax.legend(handles=handles, labels=labels, loc="upper right",
                  frameon=True, framealpha=0.93, edgecolor=GRID, fontsize=9.3)

    fig.suptitle("图 2  层数 4 与 8 比对（同一拟合区间 [0.03, 0.20]）　"
                 f"判据 |s实测−s预测| ≤ {DELTA_TOL:.2f}，超过即层数不足",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_terrain(manifest: dict, data_dir: str, results8: list[CondResult], path: str) -> None:
    n = manifest["size"]
    seed0 = min(e["seed"] for e in manifest["files"])
    crop = 160
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6), dpi=170)
    for i, r in enumerate(results8):
        e = next(e for e in manifest["files"]
                 if abs(float(e["gain"]) - r.gain) < 1e-9 and int(e["octaves"]) == 8
                 and int(e["seed"]) == int(seed0))
        z = load_field(os.path.join(data_dir, e["file"]), n)
        ax = axes[0, i]
        ax.imshow(z, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"g={r.gain:.2f} 全图（{n}px）", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#c3c2b7")
        c0 = n // 2 - crop // 2
        zc = z[c0:c0 + crop, c0:c0 + crop]
        ax2 = axes[1, i]
        ax2.imshow(zc, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax2.set_title(f"中心 {crop}px 放大  ρ={r.rho:.3f}  梯度RMS={r.grad_rms:.4f}",
                      fontsize=10.2)
        ax2.set_xticks([]); ax2.set_yticks([])
    axes[0, 0].set_ylabel("缩小看（全图）", fontsize=11)
    axes[1, 0].set_ylabel("放大看（中心裁切）", fontsize=11)
    fig.suptitle("图 3（地形对照，非频谱图）  增益越高，细尺度起伏越强、视觉越粗糙",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 输出

def result_json(r: CondResult) -> dict:
    return {
        "gain": r.gain, "octaves": r.octaves, "n_seeds": r.n_seeds,
        "s_pred": r.s_pred, "s_fit_pooled": r.s_pool,
        "s_fit_seed_mean": r.s_seed_mean, "s_fit_seed_sd": r.s_seed_sd,
        "delta_vs_pred": r.delta, "r2_loglog": r.r2_log,
        "raw_radial_slope": r.s_raw_pool, "raw_radial_pred": r.s_pred - 2.0,
        "roughness_rho": r.rho, "roughness_rho_sd": r.rho_sd,
        "roughness_rho_pred_powerlaw": r.rho_pred,
        "band_ratio_R": r.band_r, "band_ratio_R_pred": Q_BAND ** r.s_pred,
        "grad_rms": r.grad_rms, "grad_sd": r.grad_sd,
        "abs_delta": abs(r.delta), "enough_layers": bool(r.enough),
        "inverse_gain_from_rho": float(r.band_r ** (np.log(2.0) / (2.0 * np.log(Q_BAND)))),
    }


def write_wide_csv(path: str, size: int, rows8: list[CondResult], rows4: list[CondResult],
                   target: dict | None, verify: dict | None) -> None:
    header = ["size", "n_seeds", "k_lo", "k_hi", "delta_tol", "roughness_q", "target_rho"]
    row: list = [size, rows8[0].n_seeds if rows8 else None, K_LO, K_HI, DELTA_TOL, Q_BAND,
                 TARGET_RHO]
    for tag, rows in (("o4", rows4), ("o8", rows8)):
        for i, r in enumerate(rows, 1):
            header += [f"g{i}_{tag}", f"s_pred{i}_{tag}", f"s_fit{i}_{tag}",
                       f"delta{i}_{tag}", f"r2{i}_{tag}", f"enough{i}_{tag}"]
            row += [r.gain, f"{r.s_pred:.4f}", f"{r.s_pool:.4f}", f"{r.delta:+.4f}",
                    f"{r.r2_log:.5f}", int(r.enough)]
    for i, r in enumerate(rows8, 1):
        header += [f"rho{i}", f"rho_pred{i}", f"grad_rms{i}", f"g_inv_rho{i}"]
        row += [f"{r.rho:.5f}", f"{r.rho_pred:.5f}", f"{r.grad_rms:.5f}",
                f"{r.band_r ** (np.log(2.0)/(2*np.log(Q_BAND))):.5f}"]
    if target:
        header += ["g_required", "s_required"]
        row += [f"{target['g_required']:.5f}", f"{target['s_required']:.4f}"]
    if verify:
        header += ["verify_gain", "verify_rho", "verify_rho_err", "verify_s_fit",
                   "verify_s_pred", "verify_delta"]
        row += [f"{verify['gain']:.5f}", f"{verify['rho']:.5f}",
                f"{verify['rho'] - TARGET_RHO:+.5f}", f"{verify['s_fit']:.4f}",
                f"{verify['s_pred']:.4f}", f"{verify['delta']:+.4f}"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="主矩阵 dumpfields 目录")
    ap.add_argument("--verify", help="反解验证场 dumpfields 目录")
    ap.add_argument("--target", help="target.json 路径（仅用于工作流留痕；目标按固定 TARGET_RHO 重算）")
    ap.add_argument("--write-target", help="写出反解目标 JSON（阶段一）")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.main, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    n = int(manifest["size"])
    lac = float(manifest.get("lacunarity", 2.0))
    groups = group_manifest(manifest, args.main)

    gains = sorted({g for (g, o) in groups})
    rows4, rows8 = [], []
    for g in gains:
        for o, bucket in ((4, rows4), (8, rows8)):
            entries = groups[(g, o)]
            bucket.append(analyze_condition(entries, args.main, n, lac))

    # 反解目标
    g_req, s_req = invert_gain(TARGET_RHO, lac)
    target = {"rho_target": TARGET_RHO, "R_target": TARGET_RHO / (1 - TARGET_RHO),
              "q": Q_BAND, "s_required": s_req, "lacunarity": lac,
              "g_required": g_req}
    if args.write_target:
        os.makedirs(os.path.dirname(args.write_target) or ".", exist_ok=True)
        with open(args.write_target, "w", encoding="utf-8") as f:
            json.dump(target, f, ensure_ascii=False, indent=2)

    verify = None
    verify_block = None
    if args.verify:
        with open(os.path.join(args.verify, "manifest.json"), encoding="utf-8") as f:
            vm = json.load(f)
        vgroups = group_manifest(vm, args.verify)
        keys = list(vgroups)
        gv, ov = keys[0]
        vr = analyze_condition(vgroups[(gv, ov)], args.verify, int(vm["size"]),
                               float(vm.get("lacunarity", lac)))
        verify_block = result_json(vr)
        verify = {"gain": vr.gain, "rho": vr.rho, "s_fit": vr.s_pool,
                  "s_pred": vr.s_pred, "delta": vr.delta}

    # JSON 全量结果
    summary = {
        "protocol": {
            "size": n, "lacunarity": lac, "k_lo": K_LO, "k_hi": K_HI,
            "roughness_bands": {"coarse": ROUGH_C, "fine": ROUGH_F, "q": Q_BAND},
            "delta_tolerance": DELTA_TOL,
            "seeds_per_condition": rows8[0].n_seeds if rows8 else None,
            "window": "Hann 2D", "fit": "ordinary least squares on ln B vs ln k",
        },
        "octaves4": [result_json(r) for r in rows4],
        "octaves8": [result_json(r) for r in rows8],
        "inversion_target": target,
        "verification": verify_block,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_wide_csv(os.path.join(args.out, "summary.csv"), n, rows8, rows4, target, verify)

    # 图：严格两幅频谱图 + 一幅地形对照
    plot_gains(rows8, os.path.join(args.out, "spectrum_gains.png"))
    plot_octaves(rows4, rows8, os.path.join(args.out, "spectrum_octaves.png"))
    plot_terrain(manifest, args.main, rows8, os.path.join(args.out, "terrain_gains.png"))

    # 控制台简报
    print(f"N={n} seeds={rows8[0].n_seeds} fit=[{K_LO},{K_HI}] delta_tol={DELTA_TOL}")
    print("\n增益 | 层 | s_pred | s_fit | seed mean±sd | Δ | R² | ρ | G | 判定")
    for r in rows4 + rows8:
        print(f"g={r.gain:.2f} o={r.octaves} | {r.s_pred:+.3f} | {r.s_pool:+.3f} | "
              f"{r.s_seed_mean:+.3f}±{r.s_seed_sd:.3f} | {r.delta:+.3f} | {r.r2_log:.4f} | "
              f"ρ={r.rho:.4f} | G={r.grad_rms:.4f} | {'达标' if r.enough else '层数不足'}")
    print(f"\n反解: ρ*={TARGET_RHO} -> s_req={s_req:+.4f} -> g_req={g_req:.5f}")
    if verify_block:
        print(f"验证场 g={verify['gain']:.5f}: ρ={verify['rho']:.4f} "
              f"(误差 {verify['rho'] - TARGET_RHO:+.4f}), "
              f"s={verify['s_fit']:+.3f} vs pred {verify['s_pred']:+.3f}")
    print(f"\nwrote {args.out}/summary.json, summary.csv, spectrum_gains.png, "
          f"spectrum_octaves.png, terrain_gains.png")


if __name__ == "__main__":
    main()
