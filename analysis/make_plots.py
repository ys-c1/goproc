"""出图：曲率分布、两套档的 N(E) 曲线与拐点、以及曲率二阶差分诊断图。

每个种子 × 每套阈值档各自存档：
  seed{S}_{sched}_curve.png       N 对 RMSE / Emax 双曲线（共享 x 子图，各自独立轴）
  seed{S}_{sched}_knee_diag.png   RMSE 曲线的离散曲率与二阶差分（变号诊断）
另出：
  seed{S}_curvature_dist.png      曲率直方图 + KDE + 众数线 + CDF
  prediction_vs_measured.png      全部 12 组的档位预判 vs 实测对照

运行：cd /tmp && python3 -P <repo>/analysis/make_plots.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 注册系统里的 Noto Sans CJK，保证中文标注不缺字。
_CJK = "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK)
    CJK_FAMILY = fm.FontProperties(fname=_CJK).get_name()
else:
    CJK_FAMILY = "DejaVu Sans"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrainlib as tl  # noqa: E402
from run_experiment import SEEDS, SMOOTH, TIERS, make_schedules, run_seed  # noqa: E402

REPO = "/home/ys/feedback/uidreczz28eCcwNSxAr"
OUT = f"{REPO}/analysis/out"

# dataviz 参考调色板（已通过 validate_palette.js，light 模式）
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"
VIOLET, RED = "#6250d6", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": CJK_FAMILY, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot_curvature(res: dict) -> None:
    samp = res["samp"]
    lo, hi = float(np.quantile(samp, 0.001)), float(np.quantile(samp, 0.995))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    from scipy.stats import gaussian_kde
    dens = gaussian_kde(samp)(np.linspace(lo, hi, 600))

    ax = axes[0]
    ax.hist(samp, bins=80, range=(lo, hi), density=True, color=BLUE, alpha=0.25,
            edgecolor="none", label="曲率直方图（内部顶点）")
    ax.plot(np.linspace(lo, hi, 600), dens, color=BLUE, lw=2, label="KDE 密度")
    ax.axvline(res["kstar"], color=ORANGE, lw=2,
               label=f"众数 κ*={res['kstar']:.5f}")
    ax.set_xlabel("顶点曲率 κ（离散拉普拉斯幅值）")
    ax.set_ylabel("密度")
    ax.set_title(f"seed {res['seed']}：曲率分布")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # 独立面板画 CDF，避免双轴（单轴原则：两种量纲分面展示）。
    axc = axes[1]
    xsorted = np.sort(samp)
    cdf = np.arange(1, xsorted.size + 1) / xsorted.size
    axc.plot(xsorted, cdf, color=AQUA, lw=2, label="经验累积 F(κ)")
    axc.axvline(res["kstar"], color=ORANGE, lw=2, label=f"κ*={res['kstar']:.5f}")
    axc.axhline(res["qstar"], color=VIOLET, lw=1.4, ls="--",
                label=f"拐点分位 q*=F(κ*)={res['qstar']:.3f}")
    axc.scatter([res["kstar"]], [res["qstar"]], s=70, color=RED, zorder=5)
    axc.set_xlabel("顶点曲率 κ")
    axc.set_ylabel("累积分位 F(κ)")
    axc.set_xlim(0, hi)
    axc.set_ylim(0, 1.02)
    axc.set_title("曲率累积与拐点分位")
    axc.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/seed{res['seed']}_curvature_dist.png")
    plt.close(fig)


def plot_curve(res: dict, sched_name: str) -> None:
    sched = res["schedules"][sched_name]
    sw = sched["sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    for ax, metric, color, title in (
        (axes[0], "rmse", BLUE, "N 对 RMSE"),
        (axes[1], "emax", ORANGE, "N 对最大绝对误差 Emax"),
    ):
        x, y = sw[metric], sw["nvert"].astype(float)
        ax.plot(x, y, color=color, lw=2, marker="o", ms=3.5,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=0.6,
                label="完整简化扫描")
        pt = sched["pred_tier"]
        k = sched["metrics"][metric]["knee"]
        ax.scatter([x[pt]], [y[pt]], s=130, facecolor="none", edgecolor=VIOLET,
                   lw=2, zorder=5, label=f"预判档 #{pt}")
        ax.scatter([x[k]], [y[k]], s=90, marker="D", facecolor=RED,
                   edgecolor=SURFACE, lw=0.8, zorder=6,
                   label=f"实测拐点 #{k}（Δ={k - pt:+d} 档）")
        ax.set_xlabel(f"{metric}（高度归一化单位）")
        ax.set_ylabel("保留顶点数 N")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(f"seed {res['seed']} · 阈值档 {sched_name}（{TIERS} 档）", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/seed{res['seed']}_{sched_name}_curve.png")
    plt.close(fig)


def plot_knee_diag(res: dict, sched_name: str) -> None:
    """离散曲率及其二阶差分，直接展示“变号取最近一处”的判定过程（RMSE）。"""
    sched = res["schedules"][sched_name]
    sw = sched["sweep"]
    x = sw["rmse"]
    y = sw["nvert"].astype(float)
    kg = tl.discrete_curvature(x, y, smooth=SMOOTH)
    d2 = np.full_like(kg, np.nan)
    d2[1:-1] = kg[2:] - 2 * kg[1:-1] + kg[:-2]
    tiers = np.arange(TIERS)
    pt = sched["pred_tier"]
    k = sched["metrics"]["rmse"]["knee"]
    flips = sched["metrics"]["rmse"]["flips"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), dpi=150, sharex=True)
    axes[0].plot(tiers[1:-1], kg[1:-1], color=BLUE, lw=2, marker="o", ms=3)
    axes[0].axvline(pt, color=VIOLET, lw=1.6, ls="--", label=f"预判档 #{pt}")
    axes[0].axvline(k, color=RED, lw=1.6, ls=":", label=f"选中变号点 #{k}")
    axes[0].set_ylabel("离散曲率 κ_g")
    axes[0].set_title(f"seed {res['seed']} · {sched_name} · RMSE 曲线拐点诊断")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].plot(tiers[2:-1], d2[2:-1], color=ORANGE, lw=2, marker="o", ms=3)
    for f in flips:
        axes[1].axvline(f, color=AQUA, lw=0.9, alpha=0.7)
    axes[1].axvline(pt, color=VIOLET, lw=1.6, ls="--")
    axes[1].axvline(k, color=RED, lw=1.6, ls=":")
    axes[1].set_ylabel("曲率二阶差分 Δ²κ_g")
    axes[1].set_xlabel("档位 tier（青色竖线 = 全部变号点；取距预判档最近者）")
    fig.tight_layout()
    fig.savefig(f"{OUT}/seed{res['seed']}_{sched_name}_knee_diag.png")
    plt.close(fig)


def plot_prediction_vs_measured(all_results: list[dict]) -> None:
    labels, pred, meas, delta = [], [], [], []
    for res in all_results:
        for sched_name, sched in res["schedules"].items():
            for metric in ("rmse", "emax"):
                md = sched["metrics"][metric]
                labels.append(f"s{res['seed']}\n{sched_name[0]}\n{metric[0].upper()}")
                pred.append(sched["pred_tier"])
                meas.append(md["knee"])
                delta.append(md["delta_tier"])
    idx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    ax.plot(idx, pred, color=BLUE, lw=0, marker="o", ms=8, label="预判档位")
    ax.plot(idx, meas, color=ORANGE, lw=0, marker="D", ms=7, label="实测拐点档位")
    for i, d in enumerate(delta):
        ax.plot([i, i], [pred[i], meas[i]], color=MUTED, lw=1, zorder=0)
        if d:
            ax.annotate(f"{d:+d}", (i, max(pred[i], meas[i]) + 0.35),
                        ha="center", fontsize=8.5, color=RED)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("档位")
    ax.set_ylim(-1, TIERS + 2)
    ax.set_title("12 组（3 种子 × 2 阈值档 × 2 误差指标）拐点档位：预判 vs 实测")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/prediction_vs_measured.png")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    all_results = [run_seed(s) for s in SEEDS]
    for res in all_results:
        plot_curvature(res)
        for sched_name in make_schedules(res["samp"]):
            plot_curve(res, sched_name)
            plot_knee_diag(res, sched_name)
    plot_prediction_vs_measured(all_results)
    print(f"图表已写入 {OUT}")


if __name__ == "__main__":
    main()
