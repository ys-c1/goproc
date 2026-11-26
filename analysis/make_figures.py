"""Generate the three marked maps (octave tiers 2/4/6) and correlation summary.

Reads ONLY authoritative Go outputs:
  artifacts/repetition/raw/layer_{shared,layered}_{0..5}.f64  (cmd/repeataudit)
  artifacts/repetition/repeat_coordinates.csv
  artifacts/repetition/repeat_metrics.json
and writes marked_K{2,4,6}.png + correlation_tiers.png + noise_floor.json.
"""
from __future__ import annotations
import csv, json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

_noto = None
_fp = "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf"
if pathlib.Path(_fp).exists():
    font_manager.fontManager.addfont(_fp)
    _noto = font_manager.FontProperties(fname=_fp).get_name()
    plt.rcParams["font.sans-serif"] = [_noto, "DejaVu Sans"]
    plt.rcParams["font.monospace"] = [_noto, "DejaVu Sans Mono"]
    plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

ROOT = pathlib.Path("/home/ys/feedback/uidreczz28eB3n7piyx")
ART = ROOT / "artifacts/repetition"
RAW = ART / "raw"
S = 1024
GAIN = 0.5

# CVD-aware categorical level colors (dataviz reference categorical slots).
LEVEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#c98500", "#e87ba4", "#6250d6"]
C_BUG = "#d64545"
C_FIX = "#1b8a4b"
INK = "#1b1b1a"
MUTED = "#52514e"


def load_layers():
    out = {"shared": [], "layered": []}
    for mode in out:
        for i in range(6):
            g = np.fromfile(RAW / f"layer_{mode}_{i}.f64", dtype="<f8").reshape(S, S)
            out[mode].append(g)
    return out


def composite(layers, K):
    acc = np.zeros((S, S))
    amp, norm = 1.0, 0.0
    for i in range(K):
        acc += layers[i] * amp
        norm += amp
        amp *= GAIN
    z = acc / norm
    lo, hi = float(z.min()), float(z.max())
    return (z - lo) / (hi - lo)


def load_coords():
    rows = []
    with open(ART / "repeat_coordinates.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            for k in ("tier", "cx0", "cy0", "level", "layer", "center_x",
                      "center_y", "x0", "y0", "x1", "y1", "side"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def boxes_for(rows, K, level=None):
    for r in rows:
        if r["tier"] != K:
            continue
        if level is not None and r["level"] != level:
            continue
        yield r


def halo_rect(ax, r, color, lw=1.4, ls="-", alpha=1.0, halo=True):
    x, y = r["x0"], r["y0"]
    w, h = r["x1"] - r["x0"], r["y1"] - r["y0"]
    if halo:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="black",
                               lw=lw + 1.8, ls=ls, alpha=0.45, zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color,
                           lw=lw, ls=ls, alpha=alpha, zorder=6))


def draw_all_boxes(ax, rows, K, fixed=False):
    for r in boxes_for(rows, K):
        col = "#9fe0b6" if fixed else LEVEL_COLORS[r["level"]]
        if fixed:
            halo_rect(ax, r, col, lw=1.0, ls=(0, (4, 2)), alpha=0.95)
        else:
            lw = 2.2 if r["level"] == 0 else max(0.7, 1.7 - 0.22 * r["level"])
            halo_rect(ax, r, col, lw=lw)
        if r["level"] == 0:
            ax.text(r["x0"] + 6, r["y0"] + 22, r["anchor_id"], color="white",
                    fontsize=10, weight="bold", zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])


def tier_figure(layers, rows, metrics_by_k, K):
    m = metrics_by_k[K]
    fig = plt.figure(figsize=(22, 13.2), dpi=110)
    outer = GridSpec(2, 1, height_ratios=[1.32, 1.0], hspace=0.16,
                     top=0.90, bottom=0.045, left=0.025, right=0.985)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.05, 1.05, 0.95], wspace=0.06)

    zs = composite(layers["shared"], K)
    zl = composite(layers["layered"], K)

    ax0 = fig.add_subplot(top[0, 0])
    ax0.imshow(zs, cmap="terrain", origin="upper")
    draw_all_boxes(ax0, rows, K)
    ax0.set_title(f"共用置换表（缺陷） composite, K={K} 层 — 全部 {9*K} 个标记盒 / "
                  f"{9*(K-1)} 个重复片", fontsize=13, weight="bold", color=C_BUG)
    ax0.set_xlim(0, S); ax0.set_ylim(S, 0); ax0.set_xticks([]); ax0.set_yticks([])

    ax1 = fig.add_subplot(top[0, 1])
    ax1.imshow(zl, cmap="terrain", origin="upper")
    draw_all_boxes(ax1, rows, K, fixed=True)
    ax1.set_title("按层派生子表（修复） 同一批坐标 — 同源重复消失",
                  fontsize=13, weight="bold", color=C_FIX)
    ax1.set_xlim(0, S); ax1.set_ylim(S, 0); ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(top[0, 2]); ax2.axis("off")
    _stat_panel(ax2, m, K)

    bot = outer[1].subgridspec(2, 6, wspace=0.06, hspace=0.28)
    for i in range(K):
        axt = fig.add_subplot(bot[0, i])
        axt.imshow(layers["shared"][i], cmap="gray", origin="upper",
                   vmin=-1, vmax=1)
        for r in boxes_for(rows, K, level=i):
            halo_rect(axt, r, LEVEL_COLORS[i], lw=1.6)
        axt.set_title(f"shared L{i}  f={0.02*2**i:g}/px", fontsize=10.5, color=C_BUG)
        axt.set_xticks([]); axt.set_yticks([])
        axb = fig.add_subplot(bot[1, i])
        axb.imshow(layers["layered"][i], cmap="gray", origin="upper",
                   vmin=-1, vmax=1)
        for r in boxes_for(rows, K, level=i):
            halo_rect(axb, r, "#9fe0b6", lw=1.4, ls=(0, (4, 2)))
        axb.set_title(f"layered sub-table L{i}", fontsize=10.5, color=C_FIX)
        axb.set_xticks([]); axb.set_yticks([])

    fig.suptitle(
        f"相邻层共用置换表的尺度同源重复  |  种子=7  size=1024  f0=0.02  ρ=lacunarity=2  "
        f"P=256 cells  Π=P/gcd(P,ρ−1)=256  本档顶层周期 T{K-1}={m['top_period_px']:.0f}px",
        fontsize=15, weight="bold", y=0.975)
    p = ART / f"marked_K{K}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


def _stat_panel(ax, m, K):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.02, 0.97, "周期 / 数量 / 相关", fontsize=14, weight="bold",
            color=INK, va="top")
    lines = [
        f"频率比 ρ = {m['frequency_ratio']}",
        f"置换表周期 P = {m['perm_period_cells']} 格点",
        f"Vernier 重合周期 Π = P/gcd(P,ρ−1) = {m['coincidence_period_cells']} 格点",
        f"基础像素周期 T0 = P/f0 = {m['base_period_px']:.0f} px",
        f"本档顶层 L{K-1} 周期 T{K-1} = P/(f0·ρ^{K-1}) = {m['top_period_px']:.0f} px",
        f"3×3 锚点 = {m['anchors']}，配对 = {m['pairs']}",
        f"重复片(孪生盒) = {m['repeat_patches']}，标记盒总数 = {m['marked_boxes']}",
    ]
    y = 0.90
    for ln in lines:
        ax.text(0.02, y, ln, fontsize=11.5, color=INK, va="top", family="monospace")
        y -= 0.062
    # correlation bars
    yb = 0.30
    ax.text(0.02, yb + 0.10, "比例对齐山脊相关 r（同一批坐标）", fontsize=12,
            weight="bold", color=INK, va="top")
    _bar(ax, 0.02, yb, 0.92, m["shared_corr_mean"], C_BUG,
         f"shared  {m['shared_corr_mean']:.6f} (min {m['shared_corr_min']:.3f}, 全部=1)")
    lm, ls = m["layered_corr_mean"], m["layered_corr_sd"]
    _bar(ax, 0.02, yb - 0.13, 0.92, max(0.0, lm), C_FIX,
         f"layered {lm:+.4f} ± {ls:.3f}  → 噪声水平")
    ax.text(0.02, 0.04,
            "位置公式:  c_k = c0 / ρ^k,  side_k = side0 / ρ^k   （与种子无关）",
            fontsize=11, color=MUTED, style="italic", va="bottom", family="monospace")


def _bar(ax, x, y, w, val, color, label):
    ax.add_patch(Rectangle((x, y - 0.045), w, 0.07, fill=False, edgecolor="#c9c8c2", lw=1))
    fw = w * min(1.0, abs(val))
    ax.add_patch(Rectangle((x, y - 0.045), fw, 0.07, facecolor=color, edgecolor="none", alpha=.9))
    ax.text(x, y + 0.045, label, fontsize=11, color=INK, va="bottom")


def correlation_summary(metrics_by_k, noise):
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=130)
    Ks = [2, 4, 6]
    x = np.arange(len(Ks)); w = 0.36
    shared = [metrics_by_k[k]["shared_corr_mean"] for k in Ks]
    lm = [metrics_by_k[k]["layered_corr_mean"] for k in Ks]
    ls = [metrics_by_k[k]["layered_corr_sd"] for k in Ks]
    nm = [noise[k]["null_mean"] for k in Ks]
    ns = [noise[k]["null_sd"] for k in Ks]
    ax.bar(x - w/2, shared, w, label="shared perm（缺陷）", color=C_BUG)
    ax.bar(x + w/2, lm, w, yerr=ls, capsize=4, label="per-layer sub-table（修复）",
           color=C_FIX, alpha=.9)
    ax.errorbar(x + w/2, nm, yerr=ns, fmt="D", color="#444", ms=7,
                label="无关片零分布 mean±sd", capsize=4)
    ax.axhline(0, color="#888", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([f"K={k}\n(n={9*(k-1)} 对)" for k in Ks])
    ax.set_ylabel("比例对齐 Pearson r")
    ax.set_ylim(-0.7, 1.15)
    ax.set_title("三档层数：同一坐标上的山脊相关，修复后落入噪声零分布", fontsize=13, weight="bold")
    for xi, v in zip(x - w/2, shared):
        ax.text(xi, v + .03, f"{v:.3f}", ha="center", fontsize=11, weight="bold")
    for xi, v, s in zip(x + w/2, lm, ls):
        ax.text(xi, v + s + .05, f"{v:+.3f}", ha="center", fontsize=10.5)
    ax.legend(loc="lower left", framealpha=.95)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    p = ART / "correlation_tiers.png"
    fig.savefig(p, facecolor="white"); plt.close(fig)
    print("wrote", p)


def noise_floor(layers, rows, metrics_by_k):
    """Unrelated-pair null: correlate matched-shape but coordinate-scrambled
    boxes across the two independent tables, many draws, per tier."""
    rng = np.random.default_rng(20261127)
    out = {}
    for K in (2, 4, 6):
        vals = []
        coords = [r for r in rows if r["tier"] == K]
        # sample many level edges
        edges = [(r, next(q for q in coords if q["anchor_id"] == r["anchor_id"]
                          and q["level"] == r["level"] + 1))
                 for r in coords if r["level"] < K - 1]
        for r, q in edges:
            wf = q["side"]
            for _ in range(40):
                # random non-corresponding locations, same side, layered tables
                lvl = q["level"]
                g1 = layers["layered"][lvl]
                g0 = layers["layered"][r["level"]]
                xA = rng.integers(0, S - wf); yA = rng.integers(0, S - wf)
                A = g1[yA:yA+wf, xA:xA+wf]
                wk = r["side"]; half = wk // 2
                # random coarse box of wk then decimate rho to wf
                xB = rng.integers(0, S - wk); yB = rng.integers(0, S - wk)
                B = g0[yB:yB+wk, xB:xB+wk][::2, ::2]
                if B.shape != (wf, wf):
                    continue
                a = A.ravel() - A.mean(); b = B.ravel() - B.mean()
                den = np.sqrt((a*a).sum() * (b*b).sum())
                if den > 0:
                    vals.append(float((a*b).sum()/den))
        vals = np.array(vals)
        out[K] = {"null_mean": float(vals.mean()), "null_sd": float(vals.std()),
                  "null_p2.5": float(np.quantile(vals, .025)),
                  "null_p97.5": float(np.quantile(vals, .975)), "n": int(vals.size)}
        # also carry layered measured for convenience
        out[K]["layered_mean"] = metrics_by_k[K]["layered_corr_mean"]
        out[K]["layered_sd"] = metrics_by_k[K]["layered_corr_sd"]
    (ART / "noise_floor.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote noise_floor.json")
    return out


def main():
    layers = load_layers()
    rows = load_coords()
    metrics = json.loads((ART / "repeat_metrics.json").read_text())
    mbk = {m["octaves"]: m for m in metrics["tiers"]}
    nf = noise_floor(layers, rows, mbk)
    for K in (2, 4, 6):
        tier_figure(layers, rows, mbk, K)
    correlation_summary(mbk, nf)


if __name__ == "__main__":
    main()
