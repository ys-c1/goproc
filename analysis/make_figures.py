#!/usr/bin/env python3
"""生成全部图件：两张各组极值点分布图（必需）+ 规律证明/预判/乘积/方差诊断图。"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "artifacts" / "extrema" / "data"
OUT = ROOT / "artifacts" / "extrema"

# 中文字体
FONT = "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

# dataviz 色板（light）
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; YELLOW = "#eda100"
VIOLET = "#6250d6"; RED = "#e34948"; INK = "#0b0b0b"; INK2 = "#52514e"
GRID = "#e2e2df"; SURF = "#fcfcfb"

R = json.load(open(OUT / "results.json"))
ENTRIES = json.load(open(DATA / "manifest.json"))
F0, LAC, C = 0.02, 2.0, R["C_weighted"]


def load_map(role, n, seed=1, g=None):
    for e in ENTRIES:
        if e["role"] == role and e["octaves"] == n and e["seed"] == seed:
            if g is None or abs(e["gain"] - g) < 1e-9:
                z = np.fromfile(DATA / e["file"], dtype="<f8")
                return z.reshape(e["size"], e["size"]), e
    raise KeyError(role, n, seed)


def extrema_points(z):
    n = z.shape[0]
    zc = z[1:-1, 1:-1]
    mx = np.ones(zc.shape, bool); mn = np.ones(zc.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            w = z[1 + dy:n - 1 + dy, 1 + dx:n - 1 + dx]
            mx &= zc > w; mn &= zc < w
    ys, xs = np.nonzero(mx)
    maxp = np.column_stack([xs + 1, ys + 1])
    ys, xs = np.nonzero(mn)
    minp = np.column_stack([xs + 1, ys + 1])
    return maxp, minp


# ---------------------------------------------------------------- 两张分布图
_rng = np.random.default_rng(0)


def _thin(p, cap_per_class=2600):
    """极密集层做空间公平随机抽样以便看清分布，标题注明显示比例。"""
    if len(p) > cap_per_class:
        idx = _rng.choice(len(p), cap_per_class, replace=False)
        return p[idx], len(p)
    return p, len(p)


def group_map_figure(role, g, fname, subtitle):
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 8.1), dpi=150)
    fig.patch.set_facecolor("white")
    for n, ax in enumerate(axes.ravel(), start=1):
        z, e = load_map(role, n, seed=1)
        size = e["size"]
        margin = 50
        maxp, minp = extrema_points(z)
        ax.imshow(z, cmap="gray", origin="upper", interpolation="nearest",
                  vmin=max(0, np.percentile(z, 0.5)), vmax=min(1, np.percentile(z, 99.5)))
        mp, nmax_full = _thin(maxp)
        np_, nmin_full = _thin(minp)
        thinned = len(mp) < nmax_full or len(np_) < nmin_full
        if thinned:
            psize = 5.0
        else:
            psize = {1: 15, 2: 13, 3: 9, 4: 6}.get(n, 2.8)
        ax.scatter(mp[:, 0], mp[:, 1], marker="^", s=psize, c=RED,
                   linewidths=0, alpha=0.9 if thinned else 0.85,
                   rasterized=True, label="极大值")
        ax.scatter(np_[:, 0], np_[:, 1], marker="v", s=psize, c=BLUE,
                   linewidths=0, alpha=0.9 if thinned else 0.85,
                   rasterized=True, label="极小值")
        ax.add_patch(Rectangle((margin, margin), size - 2 * margin, size - 2 * margin,
                               fill=False, edgecolor=YELLOW, lw=0.9, ls="--", alpha=0.95))
        row = next(x for x in R[role] if x["n"] == n)
        show_note = "；密集层为随机抽样显示" if thinned else ""
        ax.set_title(f"N={n}　本图极值 {nmax_full + nmin_full} 个\n"
                     f"六种子平均最近邻 {row['meas']:.2f} px{show_note}",
                     fontsize=9.8, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#bdbdb8"); sp.set_linewidth(0.8)
    # 图例只放一次
    handles = [plt.Line2D([], [], marker="^", ls="", color=RED, ms=8, label="局部极大"),
               plt.Line2D([], [], marker="v", ls="", color=BLUE, ms=8, label="局部极小"),
               plt.Line2D([], [], ls="--", color=YELLOW, lw=1.4, label="边界剔除带（50 px）")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.004))
    fig.suptitle(subtitle, fontsize=14.5, color=INK, y=0.985, fontweight="bold")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.075,
                        wspace=0.06, hspace=0.42)
    fig.savefig(OUT / fname, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", fname)


group_map_figure("A", 0.5, "extrema_distribution_group_A.png",
                 "第一组 A：默认增益 g=0.50，层数 N=1–8（768²，种子 1；统计用 6 个种子）")
group_map_figure("B", 0.8, "extrema_distribution_group_B.png",
                 "第二组 B：高增益 g=0.80，层数 N=1–8（768²，种子 1；统计用 6 个种子）")

# ---------------------------------------------------------------- 反比律证明图
cal = R["calibration"]
fs = np.array([x["f"] for x in cal])
ms = np.array([x["mean"] for x in cal])
Cs = np.array([x["C"] for x in cal])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=150)
fig.patch.set_facecolor("white")
ells = 1 / fs
ax1.scatter(ells, ms, s=46, color=BLUE, zorder=3, edgecolor="white", linewidth=0.6)
xx = np.linspace(0, 82, 50)
ax1.plot(xx, C * xx, color=ORANGE, lw=2, label=f"拟合 d̄ = {C:.3f}·ℓ")
ax1.set_xlabel("单层晶格间距 ℓ = 1/f（px）"); ax1.set_ylabel("实测平均最近邻间距 d̄（px）")
ax1.set_title("平均间距正比于晶格间距（反比于频率）", fontsize=11.5)
ax1.grid(color=GRID, lw=0.8); ax1.set_axisbelow(True)
ax1.legend(frameon=False, fontsize=10)
ax1.set_xlim(0, 84); ax1.set_ylim(0, 100)

ax2.axhline(C, color=ORANGE, lw=2, label=f"加权平均 C={C:.3f}")
ax2.scatter(fs, Cs, s=46, color=AQUA, zorder=3, edgecolor="white", linewidth=0.6)
ax2.set_xscale("log"); ax2.set_xlabel("基础频率 f（每像素噪声坐标，对数轴）")
ax2.set_ylabel("d̄ · f"); ax2.set_ylim(1.05, 1.16)
ax2.set_title(f"d̄·f ≈ 常数；log-log 斜率 = {R['loglog_slope']:.4f}", fontsize=11.5)
ax2.grid(color=GRID, lw=0.8, which="both"); ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "proof_inverse_frequency.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote proof_inverse_frequency.png")

# ---------------------------------------------------------------- 预判对比图
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), dpi=150, sharey=True)
fig.patch.set_facecolor("white")
for ax, role, g, col in ((axes[0], "A", 0.5, BLUE), (axes[1], "B", 0.8, ORANGE)):
    rows = R[role]
    n = np.array([x["n"] for x in rows])
    meas = np.array([x["meas"] for x in rows])
    se = np.array([x["se_mean"] for x in rows])
    dtop = np.array([x["d_top_pred"] for x in rows])
    dsat = np.array([x["d_sat_pred"] for x in rows])
    dprod = np.array([x["d_product_pred"] for x in rows])
    ax.errorbar(n, meas, yerr=1.96 * se, fmt="o", ms=7, color=col, zorder=4,
                markeredgecolor="white", markeredgewidth=0.7, label="实测均值（95% 区间）")
    ax.plot(n, dtop, "-", color=INK, lw=1.8, label="预判 C/f_top（顶层频率）")
    ax.plot(n, dsat, "--", color=AQUA, lw=2.0, label="加像素地板修正")
    ax.plot(n, dprod, ":", color=VIOLET, lw=2.0, label="产品频率 C/($f_0$gN)")
    # 标出 |err_top|>2 的层
    for x in rows:
        if abs(x["err_top"]) > 2:
            ax.scatter([x["n"]], [x["meas"]], s=190, facecolor="none",
                       edgecolor=RED, lw=1.8, zorder=5)
        if abs(x["meas"] - x["d_sat_pred"]) > 2:
            ax.annotate("二阶层偏强，修正模型仍超 2px", (x["n"], x["meas"]),
                        textcoords="offset points", xytext=(26, 22), fontsize=8.8,
                        color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.set_yscale("log"); ax.set_xticks(n)
    ax.set_xlabel("层数 N")
    ax.set_title(f"{role} 组：增益 g={g}", fontsize=11.5)
    ax.grid(color=GRID, lw=0.8, which="both"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.6, loc="upper right")
axes[0].set_ylabel("平均最近邻间距（px，对数轴）")
fig.suptitle("预判 vs 实测：红圈为纯顶层频率模型偏差 >2 px 的层", fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(OUT / "prediction_vs_measurement.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote prediction_vs_measurement.png")

# ---------------------------------------------------------------- 乘积检验图
pf = R["product_factorial"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5), dpi=150)
fig.patch.set_facecolor("white")
# 左：等 gN 配对（显式标注文本与错位，避免重叠）
pairs = [(2.0, BLUE, [((2 / 3), 3, "g=2/3", (0, 11)),
                      (0.5, 4, "g=0.50", (-2, 13)),
                      (0.4, 5, "g=0.40", (0, 11))]),
         (3.0, ORANGE, [(0.75, 4, "g=0.75", (8, -16)),
                        (0.5, 6, "g=0.50", (0, 11))]),
         (4.0, AQUA, [(0.8, 5, "g=0.80", (20, -12)),
                      (0.5, 8, "g=0.50", (0, 11))])]
for P, col, cfgs in pairs:
    xs, ys = [], []
    for gg, nn, lab, off in cfgs:
        rec = next(x for x in pf if abs(x["g"] - gg) < 1e-3 and x["n"] == nn)
        xs.append(nn); ys.append(rec["meas"])
    ax1.plot(xs, ys, "-o", color=col, lw=1.8, ms=8,
             label=f"gN={P:g}（产品相同）", markeredgecolor="white")
    for gg, nn, lab, off in cfgs:
        rec = next(x for x in pf if abs(x["g"] - gg) < 1e-3 and x["n"] == nn)
        ax1.annotate(lab, (nn, rec["meas"]), textcoords="offset points",
                     xytext=off, ha="center", fontsize=8.5, color=INK2)
nn_axis = np.arange(1, 9)
ax1.plot(nn_axis, [C / (F0 * LAC ** (n - 1)) for n in nn_axis], "--",
         color=INK2, lw=1.4, label="C/f_top 参考线")
ax1.set_yscale("log"); ax1.set_xticks(range(1, 9))
ax1.set_xlabel("层数 N"); ax1.set_ylabel("实测平均最近邻间距（px，对数轴）")
ax1.set_title("相同 g×N 产品，间距并不相同", fontsize=11.5)
ax1.grid(color=GRID, lw=0.8, which="both"); ax1.set_axisbelow(True)
ax1.legend(frameon=False, fontsize=9)

# 右：固定 N=5 变增益
g5 = sorted([x for x in pf if x["n"] == 5], key=lambda x: x["g"])
gx = [x["g"] for x in g5]; gy = [x["meas"] for x in g5]
ax2.plot(gx, gy, "-o", color=VIOLET, lw=1.8, ms=8, markeredgecolor="white")
for x, y in zip(gx, gy):
    ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=8.5, color=INK2)
ax2.axhline(C / (F0 * LAC ** 4), color=INK2, ls="--", lw=1.4,
            label=f"C/f_top={C/(F0*LAC**4):.2f} px")
ax2.set_xlabel("增益 g（固定 N=5）"); ax2.set_ylabel("实测平均间距（px）")
ax2.set_title("增益只做次比例修正并趋饱和", fontsize=11.5)
ax2.grid(color=GRID, lw=0.8); ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "product_claim_test.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote product_claim_test.png")

# ---------------------------------------------------------------- 方差量级图
fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
fig.patch.set_facecolor("white")
for role, col, lab in (("A", BLUE, "A 组 g=0.5"), ("B", ORANGE, "B 组 g=0.8")):
    rows = R[role]
    m = np.array([x["meas"] for x in rows])
    v = np.array([x["var"] for x in rows])
    ax.scatter(m ** 2, v, s=52, color=col, label=lab, zorder=4,
               edgecolor="white", linewidth=0.6)
cm2 = np.array([x["mean"] ** 2 for x in cal])
cv = np.array([x["var"] for x in cal])
ax.scatter(cm2, cv, s=34, color=AQUA, marker="s", label="单层标定", zorder=3,
           edgecolor="white", linewidth=0.5)
xx = np.logspace(math.log10(0.5), math.log10(4000), 100)
for k, ls, lab in ((0.035, ":", "单层晶格 κ≈0.035"),
                   (0.10, "--", "κ=0.10"),
                   (0.20, "-", "κ=0.20（fBm 估计）"),
                   (0.273, "-.", "泊松上限 0.273")):
    ax.plot(xx, k * xx, ls, color=INK2 if k != 0.2 else RED,
            lw=1.8 if k == 0.2 else 1.2, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("平均间距平方 d̄²（px²，对数轴）")
ax.set_ylabel("最近邻距离方差 s²（px²，对数轴）")
ax.set_title("方差量级：Var ≈ κ·d̄²，多层 fBm  κ≈0.1–0.2", fontsize=12)
ax.grid(color=GRID, lw=0.8, which="both"); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, ncol=2)
ax.set_xlim(0.4, 6000); ax.set_ylim(0.2, 600)
fig.tight_layout()
fig.savefig(OUT / "variance_order.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote variance_order.png")
print("ALL FIGURES DONE")
