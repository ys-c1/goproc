#!/usr/bin/env python3
"""曲率尾部指数预判简化曲线的对数斜率。

实验模型（非重叠局部重建）
--------------------------
1. 曲率取离散 Laplacian 幅值 κ=|Dxx+Dyy|。
2. 只允许删除内部棋盘格 x+y 为偶数的顶点；任一被删顶点的四邻点均保留。
3. 被删点用四邻点均值（旋转方格的双线性中心值）重建，因此
   e_i = |h_i - \\hat h_i| = κ_i/4 是逐点精确恒等式。
4. 曲线使用累计 L1 误差 A=Σ e_i；MAE=A/n，RMSE 另列。固定除以 n 不改变对数斜率。

脚本只依赖 numpy + matplotlib；SciPy 不是必需项。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

for _font_path in (
    Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.ttf",
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
):
    if _font_path.exists():
        font_manager.fontManager.addfont(_font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_font_path).get_name()
        break

BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#d03b3b"
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "font.family": ["Noto Sans CJK SC", "system-ui", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    }
)


def load_goraw(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    if blob[:8] != b"GORAW001":
        raise ValueError(f"expected GORAW001, got {blob[:8]!r}")
    n = int(np.frombuffer(blob, dtype="<i8", offset=8, count=1)[0])
    data = np.frombuffer(blob, dtype="<f8", offset=24, count=n * n)
    return data.reshape(n, n).copy()


def curvature_and_eligible(h: np.ndarray):
    n = h.shape[0]
    lap = np.zeros_like(h)
    lap[1:-1, 1:-1] = (
        h[2:, 1:-1]
        + h[:-2, 1:-1]
        + h[1:-1, 2:]
        + h[1:-1, :-2]
        - 4.0 * h[1:-1, 1:-1]
    )
    kappa = np.abs(lap)
    err = kappa / 4.0
    y, x = np.indices((n, n))
    eligible = (x > 0) & (x < n - 1) & (y > 0) & (y < n - 1) & ((x + y) % 2 == 0)
    return kappa, err, eligible


def fit_lower_tail_alpha(kappa: np.ndarray, eligible: np.ndarray) -> dict:
    """在预先指定的 F∈[0.01,0.20] 尾部窗内拟合 ln F = ln C + α ln κ。"""
    q = np.linspace(0.01, 0.20, 60)
    tau = np.quantile(kappa[eligible], q)
    lx = np.log(tau)
    ly = np.log(q)
    alpha, log_c = np.polyfit(lx, ly, 1)
    pred = alpha * lx + log_c
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return {
        "alpha": float(alpha),
        "C": float(np.exp(log_c)),
        "r2": float(r2),
        "quantile_window": [0.01, 0.20],
        "n_fit_points": len(q),
    }


def curve_at_cutoffs(sorted_err: np.ndarray, eps: np.ndarray, n_pixels: int):
    """给定局部误差阈值 ε，返回 R、累计 L1、MAE、RMSE。"""
    rows = []
    for e in eps:
        r = int(np.searchsorted(sorted_err, e, side="right"))
        cut = sorted_err[:r]
        a = float(cut.sum())
        rmse = float(np.sqrt(np.sum(cut * cut) / n_pixels))
        rows.append((e, r, a, a / n_pixels, rmse))
    return np.asarray(rows, dtype=float)


def ols_log_slope(r: np.ndarray, a: np.ndarray) -> float:
    return float(np.polyfit(np.log(r), np.log(a), 1)[0])


def write_group(
    group: str,
    schedule_eps: np.ndarray,
    center_idx: list[int],
    sorted_err: np.ndarray,
    n_pixels: int,
    alpha: float,
    outdir: Path,
) -> list[dict]:
    full = curve_at_cutoffs(sorted_err, schedule_eps, n_pixels)
    q_emp = np.array([np.mean(sorted_err <= e) for e in full[:, 0]])

    curve_csv = outdir / f"curve_{group}.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["j", "epsilon", "curvature_tau=4eps", "empirical_quantile", "R", "A_L1", "MAE", "RMSE"])
        for j, row in enumerate(full):
            eps, r, a, mae, rmse = row
            w.writerow([j, f"{eps:.12g}", f"{4*eps:.12g}", f"{q_emp[j]:.9f}", int(r), f"{a:.12g}", f"{mae:.12g}", f"{rmse:.12g}"])

    pred_a = 1.0 + 1.0 / alpha
    pred_rmse = 0.5 + 1.0 / alpha
    records = []
    for k, j in enumerate(center_idx, start=1):
        ix = [j - 1, j, j + 1]
        trip = full[ix]
        # 三点对数坐标 OLS。
        s_fit = ols_log_slope(trip[:, 1], trip[:, 2])
        x = np.log(trip[:, 1])
        y = np.log(trip[:, 2])
        s_diff = float((y[-1] - y[0]) / (x[-1] - x[0]))
        linear_slope = float((trip[-1, 2] - trip[0, 2]) / (trip[-1, 1] - trip[0, 1]))
        eps_center = float(full[j, 0])
        rec = {
            "group": group,
            "k": k,
            "center_j": j,
            "triplet_j": f"{j-1},{j},{j+1}",
            "epsilon_center": eps_center,
            "tau_center": 4.0 * eps_center,
            "quantile_center": float(q_emp[j]),
            "eps_minus": float(trip[0, 0]),
            "eps_plus": float(trip[2, 0]),
            "R_minus": int(trip[0, 1]),
            "R_center": int(trip[1, 1]),
            "R_plus": int(trip[2, 1]),
            "A_minus": float(trip[0, 2]),
            "A_center": float(trip[1, 2]),
            "A_plus": float(trip[2, 2]),
            "s_fit": s_fit,
            "s_log_endpoint_diff": s_diff,
            "linear_marginal_dA_dR": linear_slope,
            "marginal_over_epsilon": linear_slope / eps_center,
            "alpha_tail": alpha,
            "pred_slope_L1": pred_a,
            "pred_slope_RMSE": pred_rmse,
            "rel_dev": s_fit / pred_a - 1.0,
            "over20": abs(s_fit / pred_a - 1.0) > 0.20,
        }
        records.append(rec)

    detail_csv = outdir / f"slopes_{group}.csv"
    fields = list(records[0].keys())
    with detail_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rec in records:
            w.writerow(rec)

    plot_group(group, full, q_emp, records, pred_a, outdir)
    return records


def plot_group(group: str, full: np.ndarray, q_emp: np.ndarray, records: list[dict], pred: float, outdir: Path):
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=170)
    ax.plot(
        full[:, 1],
        full[:, 2],
        "-",
        color=BLUE,
        lw=1.8,
        alpha=0.72,
        marker="o",
        ms=2.8,
        label="完整取档曲线（仅用于事后核对）",
        zorder=2,
    )

    ok = [r for r in records if not r["over20"]]
    bad = [r for r in records if r["over20"]]
    if ok:
        ax.scatter(
            [r["R_center"] for r in ok],
            [r["A_center"] for r in ok],
            s=58,
            marker="o",
            facecolor=SURFACE,
            edgecolor=BLUE,
            linewidth=1.8,
            label="三点斜率偏差 ≤20%",
            zorder=4,
        )
    if bad:
        ax.scatter(
            [r["R_center"] for r in bad],
            [r["A_center"] for r in bad],
            s=82,
            marker="X",
            color=RED,
            edgecolor=INK,
            linewidth=0.5,
            label="偏差 >20%",
            zorder=5,
        )
        for bi, r in enumerate(bad):
            if group == "B" and len(bad) > 1 and bi == 0:
                offset = (-86, -24)
                ha = "left"
            elif group == "B" and bi == len(bad) - 1:
                offset = (8, 10)
                ha = "left"
            else:
                offset = (7, -4)
                ha = "left"
            ax.annotate(
                f"#{r['k']} {100*r['rel_dev']:+.0f}%",
                (r["R_center"], r["A_center"]),
                xytext=offset,
                textcoords="offset points",
                ha=ha,
                fontsize=9,
                color=RED,
                weight="bold",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8, alpha=0.8),
            )

    anchor = records[len(records) // 2]
    rr = np.array([records[0]["R_minus"], records[-1]["R_plus"]], dtype=float)
    aa = anchor["A_center"] * (rr / anchor["R_center"]) ** pred
    ax.plot(rr, aa, "--", color=ORANGE, lw=2.0, label=rf"尾部指数预判斜率 $1+1/\hat\alpha={pred:.3f}$", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="both", alpha=0.9)
    ax.grid(True, which="minor", axis="both", alpha=0.35)
    ax.set_xlabel("单位顶点减少数 R（个，对数坐标）")
    ax.set_ylabel("累计 L1 误差 A（高度单位，对数坐标）")
    title = "A 组：等误差阈值取档" if group == "A" else "B 组：等分位（等删除点数）取档"
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor="#c3c2b7")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0, x1 * 1.28)
    ax.set_ylim(y0, y1 * 1.32)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / f"curve_{group}.png", bbox_inches="tight")
    fig.savefig(outdir / f"curve_{group}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_tail(kappa: np.ndarray, eligible: np.ndarray, tail: dict, outdir: Path):
    q = np.linspace(0.002, 0.50, 120)
    tau = np.quantile(kappa[eligible], q)
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=170)
    ax.loglog(tau, q, "-", color=BLUE, lw=2.0, label="经验 CDF $F(\\kappa)$")
    fit_q = np.linspace(tail["quantile_window"][0], tail["quantile_window"][1], 60)
    fit_tau = np.quantile(kappa[eligible], fit_q)
    xx = np.array([fit_tau[0], fit_tau[-1]])
    yy = tail["C"] * xx ** tail["alpha"]
    ax.loglog(xx, yy, "--", color=ORANGE, lw=2.2, label=rf"尾部拟合 $\alpha={tail['alpha']:.3f}$, $R^2={tail['r2']:.4f}$")
    ax.axvspan(fit_tau[0], fit_tau[-1], color=ORANGE, alpha=0.08, label="拟合分位窗 1%–20%")
    ax.set_xlabel("曲率阈值 κ（对数坐标）")
    ax.set_ylabel("累计分位 F(κ)（对数坐标）")
    ax.set_title("只使用曲率数组的低曲率尾部指数拟合")
    ax.grid(True, which="both", alpha=0.45)
    ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor="#c3c2b7")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "curvature_tail.png", bbox_inches="tight")
    fig.savefig(outdir / "curvature_tail.svg", bbox_inches="tight")
    plt.close(fig)


def write_markdown_columns(all_records: dict[str, list[dict]], tail: dict, outdir: Path):
    a = all_records["A"]
    b = all_records["B"]
    lines = [
        "| k | A: ε 阈值 | A: 三点 j−1,j,j+1 | A: R−/R0/R+ | A: s₃ | A: 偏差 | B: ε 阈值 | B: 三点 j−1,j,j+1 | B: R−/R0/R+ | B: s₃ | B: 偏差 |",
        "|---:|---:|:---:|:---|---:|---:|---:|:---:|:---|---:|---:|",
    ]
    pred = 1.0 + 1.0 / tail["alpha"]
    for ra, rb in zip(a, b):
        lines.append(
            "| {k} | {ea:.3e} | {ta} | {ra0}/{rac}/{rap} | {sa:.3f} | {da:+.1%}{fa} | {eb:.3e} | {tb} | {rb0}/{rbc}/{rbp} | {sb:.3f} | {db:+.1%}{fb} |".format(
                k=ra["k"],
                ea=ra["epsilon_center"],
                ta=ra["triplet_j"],
                ra0=ra["R_minus"],
                rac=ra["R_center"],
                rap=ra["R_plus"],
                sa=ra["s_fit"],
                da=ra["rel_dev"],
                fa=" ⚠" if ra["over20"] else "",
                eb=rb["epsilon_center"],
                tb=rb["triplet_j"],
                rb0=rb["R_minus"],
                rbc=rb["R_center"],
                rbp=rb["R_plus"],
                sb=rb["s_fit"],
                db=rb["rel_dev"],
                fb=" ⚠" if rb["over20"] else "",
            )
        )
    (outdir / "slope_columns.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    detail_lines = ["# 超过两成偏差的阈值\n"]
    for group, recs in all_records.items():
        bad = [r for r in recs if r["over20"]]
        detail_lines.append(f"## {group} 组：{len(bad)} 个\n")
        if not bad:
            detail_lines.append("无。\n")
        for r in bad:
            detail_lines.append(
                f"- #{r['k']}，中心档 j={r['center_j']}，ε={r['epsilon_center']:.6e}，"
                f"F={r['quantile_center']:.3f}，s₃={r['s_fit']:.4f}，"
                f"预判={pred:.4f}，偏差={100*r['rel_dev']:+.1f}%；"
                f"三点 ε−/ε0/ε+={r['eps_minus']:.3e}/{r['epsilon_center']:.3e}/{r['eps_plus']:.3e}，"
                f"R={r['R_minus']}/{r['R_center']}/{r['R_plus']}。"
            )
        detail_lines.append("")
    (outdir / "over20_explanations.md").write_text("\n".join(detail_lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    h = load_goraw(args.raw)
    n = h.shape[0]
    n_pixels = n * n
    kappa, err, eligible = curvature_and_eligible(h)
    eligible_err = err[eligible]
    sorted_err = np.sort(eligible_err)

    # 精确恒等式数值核对：|h - four-neighbor mean| = |L|/4 = κ/4。
    neigh_mean = np.zeros_like(h)
    neigh_mean[1:-1, 1:-1] = 0.25 * (
        h[2:, 1:-1] + h[:-2, 1:-1] + h[1:-1, 2:] + h[1:-1, :-2]
    )
    residual = h - neigh_mean
    identity_abs_err = float(np.max(np.abs(np.abs(residual[eligible]) - err[eligible])))

    tail = fit_lower_tail_alpha(kappa, eligible)
    plot_tail(kappa, eligible, tail, args.outdir)

    # A 组：0..ε90 作 30 个等间隔区间；中心档位置显式列出。
    eps_a_max = float(np.quantile(eligible_err, 0.90))
    eps_a = np.linspace(0.0, eps_a_max, 31)
    a_centers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 21, 24, 27, 29]
    records_a = write_group("A", eps_a, a_centers, sorted_err, n_pixels, tail["alpha"], args.outdir)

    # B 组：0..98% 分位作 40 个等间距分位区间；j=3,6,...,39。
    q_b = np.linspace(0.0, 0.98, 41)
    eps_b = np.quantile(eligible_err, q_b)
    b_centers = list(range(3, 40, 3))
    records_b = write_group("B", eps_b, b_centers, sorted_err, n_pixels, tail["alpha"], args.outdir)

    write_markdown_columns({"A": records_a, "B": records_b}, tail, args.outdir)

    summary = {
        "raw": str(args.raw),
        "n": n,
        "n_pixels": n_pixels,
        "eligible_vertices_M": int(eligible.sum()),
        "other_parity_interior_retained": int(eligible.sum()),
        "boundary_vertices": 4 * (n - 1),
        "identity_abs_max_error": identity_abs_err,
        "tail": tail,
        "predicted_log_slope_L1": 1 + 1 / tail["alpha"],
        "predicted_log_slope_RMSE": 0.5 + 1 / tail["alpha"],
        "predicted_log_slope_threshold_vs_R": 1 / tail["alpha"],
        "A_over20": [r["k"] for r in records_a if r["over20"]],
        "B_over20": [r["k"] for r in records_b if r["over20"]],
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
