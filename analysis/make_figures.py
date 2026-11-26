"""生成留档图与分析图表（matplotlib Agg，Noto CJK 标签）。

输出到 artifacts/png/：
  render_<case>.png            九个双光源案例的单独留档图（转自 Go 生产 PGM）
  fig_renders_grid.png         3 地形 × 3 方位差效果图矩阵
  fig_beat_envelope.png        N_beat 随 Δ 的 sin 包络与三档标注
  fig_flip_vs_delta.png        实测逐行翻转数随 Δ 变化（三地形，证明与 Δ 解耦）
  fig_row_compare.png          Δ=90 下逐行：角拍频 vs 线性预测 vs Go 实测
  fig_angular_sweep.png        固定像素方位扫描的余弦周期（角拍频直接证据）
"""
import csv
import os
import site
import sys

# -I 隔离模式不加载 user site；matplotlib 装在固定的受信任用户目录，显式加入。
sys.path.insert(0, site.getusersitepackages())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

_CJK_TTF = "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf"
font_manager.fontManager.addfont(_CJK_TTF)
_CJK_NAME = font_manager.FontProperties(fname=_CJK_TTF).get_name()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgmio import read_pgm
from render_beat import (normals, strike_angles, beat_periods_row,
                         render, render_linear, quantize8, count_flips)

plt.rcParams["font.family"] = _CJK_NAME
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
ROOT = ".."
PNG = f"{ROOT}/artifacts/png"
os.makedirs(PNG, exist_ok=True)

TERRAINS = [("fbm", "fBM 随机山脊", 15.0), ("b1", "B1 平行正弦山脊", 20.0), ("b2", "B2 弯曲山脊", 30.0)]
DELTAS = [15.0, 90.0, 165.0]
PHI_M = 45.0


def style_ax(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def fig_renders():
    """效果图矩阵 + 单独 PNG 留档。"""
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    for i, (t, tlabel, elev) in enumerate(TERRAINS):
        for j, d in enumerate(DELTAS):
            tag = f"{t}_d{int(d)}"
            _, img = read_pgm(f"{ROOT}/artifacts/renders/go_{tag}.pgm")
            plt.imsave(f"{PNG}/render_{tag}.png", img, cmap="gray", vmin=0, vmax=1)
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"Δ={int(d)}°", fontsize=13)
            if j == 0:
                ax.set_ylabel(tlabel, fontsize=12)
    fig.suptitle("双光源 Lambert 效果图（Go 生产渲染，仰角按地形无钳制选取）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{PNG}/fig_renders_grid.png", dpi=130)
    plt.close(fig)


def fig_envelope():
    ds = np.linspace(0, 180, 721)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, (t, tlabel, _) in zip(axes, TERRAINS):
        stem = {"fbm": "fbm_s7", "b1": "b1_sinusoid", "b2": "b2_meander"}[t]
        _, h = read_pgm(f"{ROOT}/artifacts/heightmaps/{stem}.pgm")
        _, _, _, gx, gy = normals(h)
        beta = strike_angles(gx, gy)
        gf = np.nansum(np.abs(np.sin(np.deg2rad(beta - PHI_M))))
        nb = gf * np.sin(np.deg2rad(ds))
        ax.plot(ds, nb, color=BLUE, linewidth=2, zorder=3)
        for d, c in zip(DELTAS, (ORANGE, AQUA, "#6250d6")):
            v = gf * np.sin(np.deg2rad(d))
            ax.scatter([d], [v], color=c, s=55, zorder=4)
            ax.annotate(f"Δ={int(d)}°\n{v:.0f}", (d, v), textcoords="offset points",
                        xytext=(0, 10 if d != 90 else -34), ha="center", fontsize=9, color=c)
        ax.set_title(tlabel, fontsize=12)
        ax.set_xlabel("两灯方位差 Δ（度）")
        style_ax(ax)
    axes[0].set_ylabel("角拍频周期数 N_beat")
    fig.suptitle("N_beat(Δ) = sinΔ · Σ|cosγ|：严格正弦包络，峰值在 90°，15°与165°对称", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{PNG}/fig_beat_envelope.png", dpi=130)
    plt.close(fig)


def fig_flip_vs_delta():
    rows = list(csv.DictReader(open(f"{ROOT}/artifacts/csv/summary.csv")))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {"fbm": BLUE, "b1": ORANGE, "b2": AQUA}
    for t, tlabel, _ in TERRAINS:
        xs, ys = [], []
        for r in rows:
            if r["terrain"] == t:
                xs.append(float(r["delta"]))
                ys.append(float(r["obs8_mean"]))
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        ax.plot(xs, ys, marker="o", markersize=8, linewidth=2, color=colors[t], label=tlabel, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9, color=colors[t])
    ax.set_xticks(DELTAS)
    ax.set_xlabel("两灯方位差 Δ（度）")
    ax.set_ylabel("逐行亮暗翻转数（256 行均值，Go 留档图统计）")
    ax.set_title("空间翻转数被地形频率锁定，不随 Δ 拍频变化（Δ 只改变对比度）")
    ax.legend(frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{PNG}/fig_flip_vs_delta.png", dpi=130)
    plt.close(fig)


def fig_row_compare():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=False)
    for ax, (t, tlabel, elev) in zip(axes, TERRAINS):
        stem = {"fbm": "fbm_s7", "b1": "b1_sinusoid", "b2": "b2_meander"}[t]
        _, h = read_pgm(f"{ROOT}/artifacts/heightmaps/{stem}.pgm")
        _, _, _, gx, gy = normals(h)
        beta = strike_angles(gx, gy)
        d = 90.0
        _, go_img = read_pgm(f"{ROOT}/artifacts/renders/go_{t}_d90.pgm")
        size = h.shape[0]
        beat = np.array([beat_periods_row(beta[y], PHI_M, d) for y in range(size)])
        sp8 = np.array([count_flips(quantize8(render_linear(h, PHI_M, d, elev))[y]) for y in range(size)])
        obs = np.array([count_flips(go_img[y]) for y in range(size)])
        ys = np.arange(size)
        ax.plot(beat, ys, color=BLUE, linewidth=1.6, label="角拍频 N_beat（仅走向预测）")
        ax.plot(sp8, ys, color=AQUA, linewidth=1.6, linestyle="--", label="线性解析预测 N_sp（8-bit）")
        ax.plot(obs, ys, color=ORANGE, linewidth=1.2, alpha=0.85, label="Go 渲染实测翻转")
        ax.invert_yaxis()
        ax.set_title(f"{tlabel}（Δ=90°）", fontsize=12)
        ax.set_xlabel("单行翻转数 / 周期数")
        if ax is axes[0]:
            ax.set_ylabel("行号 y")
        style_ax(ax)
    axes[-1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("逐行对照：线性解析预测与实测完全重合；角拍频给出的是角空间量，不等于空间翻转数", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{PNG}/fig_row_compare.png", dpi=130)
    plt.close(fig)


def fig_angular_sweep():
    """B1 一个像素的方位扫描：I_var(phi_m) 为纯余弦，Δ 只缩放幅度。"""
    _, h = read_pgm(f"{ROOT}/artifacts/heightmaps/b1_sinusoid.pgm")
    nx, ny, nz, gx, gy = normals(h)
    y, x = 60, 128
    elev = 20.0
    e = np.deg2rad(elev)
    ce, se = np.cos(e), np.sin(e)
    sweep = np.linspace(0, 360, 1441)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for d, c in zip(DELTAS, (ORANGE, AQUA, "#6250d6")):
        vals = []
        for pm in sweep:
            p1, p2 = np.deg2rad(pm - d / 2), np.deg2rad(pm + d / 2)
            d1 = max(0.0, nx[y, x] * ce * np.sin(p1) + ny[y, x] * ce * np.cos(p1) + nz[y, x] * se)
            d2 = max(0.0, nx[y, x] * ce * np.sin(p2) + ny[y, x] * ce * np.cos(p2) + nz[y, x] * se)
            vals.append((d1 + d2) / 2)
        vals = np.array(vals)
        ax.plot(sweep, vals - vals.mean(), color=c, linewidth=2,
                label=f"Δ={int(d)}°（幅度∝cos(Δ/2)={np.cos(np.deg2rad(d/2)):.3f}）")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("双灯中心方位 φ_m（度，扫描 0→360°）")
    ax.set_ylabel("去直流后的双灯响应")
    ax.set_title("固定像素方位扫描：每个完整余弦周期恰 2 次亮暗翻转，与 Δ 无关（B1 正弦山脊像素）")
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{PNG}/fig_angular_sweep.png", dpi=130)
    plt.close(fig)


def main():
    fig_renders()
    fig_envelope()
    fig_flip_vs_delta()
    fig_row_compare()
    fig_angular_sweep()
    print("图表与留档 PNG ->", os.path.abspath(PNG))


if __name__ == "__main__":
    main()
