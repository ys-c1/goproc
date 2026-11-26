#!/usr/bin/env python3
"""光照分析与验证脚本。

只读取 goproc shade 写出的存档（float64 bin + stats json），用 numpy 独立重算：
  1. 三档光向的点积亮度分布、偏度，以及与一阶预测偏度的差；
  2. 分辨率等价性（f·dx 不变时高度场、粗胞元差分、点积光照逐点相等）；
  3. 直方图与梯度/光照面板图。
所有图内文字用英文，避免环境缺中文字体产生豆腐块；解释见 REPORT.md。
"""
import json
import math
import os
import site
import sys

# 以 python3 -I 运行时用户 site-packages 不在路径上；显式补入固定路径，
# 不读取 cwd，保持与 -I 同等的注入防护。
_user_site = os.path.join(site.getuserbase(), "lib",
                          f"python{sys.version_info.major}.{sys.version_info.minor}",
                          "site-packages")
if os.path.isdir(_user_site) and _user_site not in sys.path:
    sys.path.append(_user_site)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz 校验通过的三槽位分类色（蓝/橙/青），浅色表面。
COLORS = {"nw": "#2a78d6", "top": "#eb6834", "ne": "#1baf7a"}
LABELS = {"nw": "NW upper-left (az 45, alt 45)",
          "top": "Top (alt 90)",
          "ne": "NE upper-right (az 135, alt 45)"}
ANGLES = {
    "nw": (45.0, 45.0),
    "top": (0.0, 90.0),
    "ne": (135.0, 45.0),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "system-ui"],
    "font.size": 10,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "#fcfcfb",
})


def load_bin(path, n):
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def skew(v):
    """总体偏度 E[(X-mu)^3]/sigma^3，与 Go hmap.Stats 同一定义。"""
    x = np.asarray(v, dtype=np.float64)
    d = x - x.mean()
    s = x.std()  # 总体标准差，除以 N
    if s == 0:
        return 0.0
    return float((d**3).mean() / s**3)


def light_vector(az_deg, alt_deg):
    az, alt = math.radians(az_deg), math.radians(alt_deg)
    ca = math.cos(alt)
    return np.array([ca * math.cos(az), -ca * math.sin(az), math.sin(alt)])


def dot_shade(gx, gy, l):
    """与 Go ShadeLinear 同一公式：L = l.n，n=(-gx,-gy,1)/sqrt(1+g^2)。"""
    den = np.sqrt(1.0 + gx**2 + gy**2)
    return (l[0] * (-gx) + l[1] * (-gy) + l[2]) / den


def main(root):
    shade_dir = os.path.join(root, "artifacts", "shade")
    meta = json.load(open(os.path.join(shade_dir, "metadata.json")))
    n = meta["size"]
    go_stats = {s["name"]: s for s in
                json.load(open(os.path.join(shade_dir, "shading_stats.json")))}

    gx = load_bin(os.path.join(shade_dir, "gradient_x.bin"), n)
    gy = load_bin(os.path.join(shade_dir, "gradient_y.bin"), n)
    slopes2 = gx**2 + gy**2

    rows = []
    linears = {}
    gps = {}
    for name, (az, alt) in ANGLES.items():
        l = light_vector(az, alt)
        assert abs(np.linalg.norm(l) - 1.0) < 1e-12
        L = load_bin(os.path.join(shade_dir, name, "linear.bin"), n).ravel()
        linears[name] = L

        # 独立重算点积，与存档 linear.bin 逐点核对。
        L_recalc = dot_shade(gx, gy, l).ravel()
        max_err = float(np.abs(L - L_recalc).max())

        # 一阶预测：L ≈ l_z - (l_x gx + l_y gy)，预测分布形状 = 水平投影 gp = g.e_xy。
        # 正上光（alt=90°）水平分量严格为零：一阶亮度是常数，预测偏度为零。
        hnorm = math.hypot(l[0], l[1])
        if alt < 89.9 and hnorm > 0:
            ex, ey = l[0] / hnorm, l[1] / hnorm
            gp = (gx * ex + gy * ey).ravel()
            gp_stored = load_bin(os.path.join(shade_dir, name, "gradient_parallel.bin"), n).ravel()
            gp_err = float(np.abs(gp - gp_stored).max())
            pred_skew = skew(gp)
        else:
            gp = np.zeros_like(L)
            gp_err = 0.0
            pred_skew = 0.0
        gps[name] = gp

        measured = skew(L)
        go_linear = go_stats[name]["linear"]["skew"]
        rows.append({
            "angle": name,
            "azimuth_deg": az,
            "altitude_deg": alt,
            "measured_skew_python": measured,
            "measured_skew_go": go_linear,
            "delta_python_minus_go": measured - go_linear,
            "predicted_first_order_skew": pred_skew,
            "delta_measured_minus_predicted": measured - pred_skew,
            "mean": float(L.mean()),
            "std": float(L.std()),
            "min": float(L.min()),
            "max": float(L.max()),
            "neg_frac": float((L < 0).mean()),
            "dot_recalc_max_abs_err": max_err,
            "gp_archive_max_abs_err": gp_err,
        })

    # 仿射不变性数值复核：aL+b（a>0）偏度不变。
    aff = skew(2.5 * linears["nw"] + 3.7) - skew(linears["nw"])

    # 正上光二阶预测：L=(1+|g|^2)^-1/2 ≈ 1 - |g|^2/2，主导偏度 = -skew(|g|^2)。
    s2 = slopes2.ravel()
    top_second_order = -skew(s2)
    top_meas = skew(linears["top"])

    # 斜光一阶近似质量：L 与 lz - (l_x gx + l_y gy) 的相关系数。
    first_order_corr = {}
    for name in ("nw", "ne"):
        az, alt = ANGLES[name]
        l = light_vector(az, alt)
        lin = l[2] - (l[0] * gx + l[1] * gy)
        first_order_corr[name] = float(np.corrcoef(linears[name], lin.ravel())[0, 1])

    out = {
        "size": n,
        "skew_definition": "population skew E[(X-mu)^3]/sigma^3",
        "prediction_rule": "oblique: skew(g.e_xy); top: 0 (first-order constant)",
        "angles": rows,
        "affine_invariance_max_abs_skew_change": float(abs(aff)),
        "top_light_second_order_check": {
            "measured_skew": top_meas,
            "predicted_minus_skew_of_slope_squared": top_second_order,
            "gap": top_meas - top_second_order,
        },
        "oblique_first_order_correlation": first_order_corr,
        "terrain": {
            "mean_slope_squared": float(slopes2.mean()),
            "rms_slope": float(math.sqrt(slopes2.mean())),
            "max_slope": float(math.sqrt(slopes2.max())),
        },
    }
    with open(os.path.join(shade_dir, "skew_comparison.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---------- 分辨率等价性 ----------
    eq = equivalence(root)
    with open(os.path.join(shade_dir, "equivalence.json"), "w") as f:
        json.dump(eq, f, indent=2, ensure_ascii=False)

    # ---------- 图 ----------
    fig_dir = os.path.join(shade_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_histograms(linears, rows, fig_dir)
    plot_panels(gps, shade_dir, n, fig_dir)
    plot_gradients(root, shade_dir, n, fig_dir)
    plot_equivalence(root, eq, fig_dir)

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(eq, indent=2, ensure_ascii=False))


def equivalence(root):
    a_dir = os.path.join(root, "artifacts", "equiv", "a_n128_f0.02")
    b_dir = os.path.join(root, "artifacts", "equiv", "b_n256_f0.01")
    ha = load_bin(os.path.join(a_dir, "height.bin"), 128)
    hb = load_bin(os.path.join(b_dir, "height.bin"), 256)
    na, nb = 128, 256
    fa, fb, dxa, dxb = 0.02, 0.01, 1.0, 2.0
    assert fa * dxa == fb * dxb

    # raw 存档是同一 fBm 函数在匹配索引上的直接采样，未做各自的 min-max 归一化，
    # 因此高度场应逐点相等（机器精度），等价性证明不被仿射常数污染。
    h_match = np.abs(ha[:na - 1, :na - 1] - hb[::2, ::2][:na - 1, :na - 1]).max()

    # 前向差分（单位像素）。
    dxa_gx = np.diff(ha, axis=1)[:-1, :]
    dxa_gy = np.diff(ha, axis=0)[:, :-1]
    # B 的粗胞元差分：跨 2 个像素，对应同一物理格距 dx'=2。
    coarse_x = hb[:-2:2, 2::2] - hb[:-2:2, :-2:2]
    coarse_y = hb[2::2, :-2:2] - hb[:-2:2, :-2:2]

    h_match = np.abs(ha[:na - 1, :na - 1] - hb[::2, ::2][:na - 1, :na - 1]).max()
    cx_err = np.abs(dxa_gx - coarse_x).max()
    cy_err = np.abs(dxa_gy - coarse_y).max()

    # D/f 不变量。
    dof_x = np.abs(dxa_gx / dxa / fa - coarse_x / dxb / fb).max()
    dof_y = np.abs(dxa_gy / dxa / fa - coarse_y / dxb / fb).max()

    # 点积光照：A 单位梯度 vs B 粗胞元梯度，逐角度。
    per_angle = {}
    for name, (az, alt) in ANGLES.items():
        l = light_vector(az, alt)
        la = dot_shade(dxa_gx, dxa_gy, l)
        lb = dot_shade(coarse_x, coarse_y, l)
        diff = np.abs(la - lb)
        per_angle[name] = {"max_abs_shading_err": float(diff.max()),
                           "mean_abs_shading_err": float(diff.mean())}

    # 反例：B 若用朴素单位像素（细）差分，算子不等价，用偏度差展示。
    fine_gx = np.diff(hb, axis=1)[:-1, :]
    fine_gy = np.diff(hb, axis=0)[:, :-1]
    l = light_vector(45, 45)
    fine_skew = skew(dot_shade(fine_gx, fine_gy, l).ravel())
    coarse_skew = skew(dot_shade(coarse_x, coarse_y, l).ravel())
    a_skew = skew(dot_shade(dxa_gx, dxa_gy, l).ravel())

    return {
        "config_A": {"N": na, "f": fa, "dx": dxa, "q_f_times_dx": fa * dxa},
        "config_B": {"N": nb, "f": fb, "dx": dxb, "q_f_times_dx": fb * dxb},
        "height_match_max_abs_err": float(h_match),
        "coarse_cell_gradient_max_abs_err": {"x": float(cx_err), "y": float(cy_err)},
        "D_over_f_invariant_max_abs_err": {"x": float(dof_x), "y": float(dof_y)},
        "shading_pointwise": per_angle,
        "naive_fine_operator_counterexample": {
            "note": "B 用单位像素细差分时步长只有 dx'/2，不与 A 等价",
            "A_unit_skew_nw": a_skew,
            "B_coarse_skew_nw": coarse_skew,
            "B_fine_skew_nw": fine_skew,
            "coarse_vs_A_skew_gap": coarse_skew - a_skew,
            "fine_vs_A_skew_gap": fine_skew - a_skew,
        },
        "_shade_diff_maps_note": "差值图在 plot_equivalence 中重算",
    }


def plot_histograms(linears, rows, fig_dir):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5.2),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    sk = {r["angle"]: r for r in rows}

    # 左面板：两档斜光，同箱同尺度，偏度方向直接可见。
    oblique = [v for v in linears.values()]
    lo = min(linears[a].min() for a in ("nw", "ne"))
    hi = max(linears[a].max() for a in ("nw", "ne"))
    bins = np.linspace(lo, hi, 60)
    for name in ("nw", "ne"):
        L = linears[name]
        axl.hist(L, bins=bins, density=True, histtype="step", lw=2,
                 color=COLORS[name], label=LABELS[name])
        axl.axvline(L.mean(), color=COLORS[name], lw=1, ls=":", alpha=0.8)
    axl.set_xlabel("Lambertian dot-product brightness L = l·n")
    axl.set_ylabel("probability density")
    axl.set_title("Oblique light: shape governed by projected gradient")
    axl.legend(frameon=False, loc="upper right")

    # 右面板：正上光，值域极窄（亮度集中在 1 附近），独立尺度。
    t = linears["top"]
    axr.hist(t, bins=40, density=True, histtype="step", lw=2,
             color=COLORS["top"], label=LABELS["top"])
    axr.axvline(t.mean(), color=COLORS["top"], lw=1, ls=":", alpha=0.8)
    axr.set_xlabel("L (note narrow range)")
    axr.set_title("Top light: peak pinned at 1")
    axr.legend(frameon=False, loc="upper right")

    txt = "\n".join(
        f"{a}: skew={sk[a]['measured_skew_python']:+.4f}, "
        f"pred={sk[a]['predicted_first_order_skew']:+.4f}, "
        f"delta(meas-pred)={sk[a]['delta_measured_minus_predicted']:+.4f}"
        for a in ("nw", "top", "ne"))
    fig.text(0.01, 0.01, txt, fontsize=9, color="#0b0b0b", family="monospace",
             bbox=dict(facecolor="white", edgecolor="#c3c2b7", alpha=0.9))
    fig.suptitle("Brightness histograms under three light directions", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(fig_dir, "fig_hist.png"), dpi=160)
    plt.close(fig)


def plot_panels(gps, shade_dir, n, fig_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    for col, name in enumerate(("nw", "top", "ne")):
        gp = gps[name].reshape(n, n)
        v = np.abs(gp).max()
        im0 = axes[0, col].imshow(gp, cmap="gray", vmin=-v, vmax=v)
        axes[0, col].set_title(f"{LABELS[name]}\nprojected gradient g·e_xy")
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)
        L = load_bin(os.path.join(shade_dir, name, "linear.bin"), n)
        im1 = axes[1, col].imshow(L, cmap="gray")
        axes[1, col].set_title("dot-product shading L = l·n")
        fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_panels.png"), dpi=160)
    plt.close(fig)


def plot_gradients(root, shade_dir, n, fig_dir):
    h = load_bin(os.path.join(shade_dir, "height.bin"), n)
    gx = load_bin(os.path.join(shade_dir, "gradient_x.bin"), n)
    gy = load_bin(os.path.join(shade_dir, "gradient_y.bin"), n)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (img, title, sym) in zip(axes, [
        (h, "height h (normalized 0..1)", False),
        (gx, "forward-diff gradient gx", True),
        (gy, "forward-diff gradient gy", True),
    ]):
        if sym:
            v = np.abs(img).max()
            im = ax.imshow(img, cmap="gray", vmin=-v, vmax=v)
        else:
            im = ax.imshow(img, cmap="gray")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_gradients.png"), dpi=160)
    plt.close(fig)


def plot_equivalence(root, eq, fig_dir):
    a_dir = os.path.join(root, "artifacts", "equiv", "a_n128_f0.02")
    b_dir = os.path.join(root, "artifacts", "equiv", "b_n256_f0.01")
    ha = load_bin(os.path.join(a_dir, "height.bin"), 128)
    hb = load_bin(os.path.join(b_dir, "height.bin"), 256)
    l = light_vector(45, 45)
    gax = np.diff(ha, axis=1)[:-1, :]
    gay = np.diff(ha, axis=0)[:, :-1]
    la = dot_shade(gax, gay, l)
    cx = hb[:-2:2, 2::2] - hb[:-2:2, :-2:2]
    cy = hb[2::2, :-2:2] - hb[:-2:2, :-2:2]
    lb_coarse = dot_shade(cx, cy, l)
    diff = np.abs(la - lb_coarse)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    im0 = axes[0].imshow(la, cmap="gray")
    axes[0].set_title("A: N=128, f=0.02, dx=1 (unit forward diff)")
    im1 = axes[1].imshow(lb_coarse, cmap="gray")
    axes[1].set_title("B: N=256, f=0.01, dx=2 (coarse 2-px cell diff)")
    im2 = axes[2].imshow(diff, cmap="gray", vmin=0,
                         vmax=max(diff.max(), 1e-12))
    axes[2].set_title(f"pointwise |LA-LB|, max={diff.max():.2e}")
    for im, ax in zip((im0, im1, im2), axes):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_equivalence.png"), dpi=160)
    plt.close(fig)
    eq.pop("_diff_maps", None)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    main(root)
