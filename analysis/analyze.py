"""拉伸噪声主脊方向：理论预判 vs 梯度加权直方图实测。

坐标与角度约定（全流程统一）：
  x 轴正东，y 轴向北；方向角以正东为 0°、逆时针为正，方向折叠到 [0,180°)。
  PGM 行号向南增大，故第 row 行对应北向坐标 (SIZE-1-row)。

每个 case：
1. 读 Go 生成的 PGM（行序翻转到北向上）；
2. 中心差分梯度，梯度模长加权方向直方图（180 个 1° 桶，折叠 mod π），
   峰值桶做三点抛物线亚桶插值 -> 实测主方向角（题目指定估计器）；
3. 结构张量主特征向量方向 + 相干性（连续加权的对照估计器）；
4. Python 独立复刻管线逐像素交叉核对；
5. 理论预判 = θ + atan(r)（折回 [0,180)）；逐拉伸比出表，偏差 >5° 标注并解释；
6. 每组一张方向叠加图（灰度底图 + 三类方向线 + 直方图）；
7. 8 种子集成直方图与 r=0 基线用于区分随机失稳与系统性格点偏差。

运行：.venv/bin/python -I analyze.py
"""
import csv
import json
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgm_noise import ValueNoise, fbm_eval, warp_xy, read_pgm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PGM_DIR = os.path.join(ROOT, "artifacts", "pgm")
FIG_DIR = os.path.join(ROOT, "artifacts", "figures")
RES_DIR = os.path.join(ROOT, "artifacts", "results")

SIZE, SEED, OCTAVES, LACUNARITY, GAIN, FREQ = 512, 20261127, 4, 2.0, 0.5, 0.02
NBINS = 180
DEVIATION_LIMIT_DEG = 5.0
ENSEMBLE_SEEDS = [SEED + i * 101 for i in range(8)]
SMOOTH_SIGMA_DEG = 4.0

BLUE, ORANGE = "#006BA4", "#D94822"  # 已通过 CVD 校验；线型/文字标签做二次编码


# ---------- 估计器 ----------

def gradient_field(z_native):
    """对 PGM 原生行序数组（第 0 行为北边缘）求数学坐标梯度。

    数组行号向南增大，而北向坐标 y_north = SIZE-1-row，故
    ∂H/∂y_north = -∂H/∂row；列号即向东 x，gx 直接取列轴导数。
    返回 (gx, gy_north, 模长)，角度 arctan2(gy_north, gx) 即东零逆时针。
    """
    d_row = np.gradient(z_native, axis=0)  # 沿向南行号
    gx = np.gradient(z_native, axis=1)     # 沿向东列号
    gy = -d_row                            # 转成向北坐标导数
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def weighted_histogram(gx, gy, mag):
    """梯度模长加权方向直方图，方向折叠 mod π。"""
    inner = (slice(1, -1), slice(1, -1))
    angle = np.arctan2(gy[inner], gx[inner]) % math.pi
    counts, _ = np.histogram(angle, bins=np.linspace(0, math.pi, NBINS + 1),
                             weights=mag[inner])
    centers = (np.arange(NBINS) + 0.5) * (180.0 / NBINS)
    return centers, counts


def smooth_circular(counts, sigma_deg):
    half = 30
    k = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (k / sigma_deg) ** 2)
    kernel /= kernel.sum()
    padded = np.r_[counts[-half:], counts, counts[:half]]
    return np.convolve(padded, kernel, mode="valid")


def parabolic_peak(centers, counts):
    k = int(np.argmax(counts))
    left = counts[(k - 1) % NBINS]
    center = counts[k]
    right = counts[(k + 1) % NBINS]
    denom = left - 2 * center + right
    shift = 0.5 * (left - right) / denom if denom != 0 else 0.0
    return (centers[k] + shift * (180.0 / NBINS)) % 180.0


def structure_tensor(gx, gy):
    """梯度外积均值的主特征向量方向（度）与相干性。"""
    jxx, jyy, jxy = float(np.mean(gx * gx)), float(np.mean(gy * gy)), float(np.mean(gx * gy))
    tr, disc = jxx + jyy, math.hypot(jxx - jyy, 2 * jxy)
    lam1, lam2 = (tr + disc) / 2, (tr - disc) / 2
    angle = math.degrees(0.5 * math.atan2(2 * jxy, jxx - jyy)) % 180.0
    coherence = (lam1 - lam2) / (lam1 + lam2)
    return angle, coherence


def predicted_angle_deg(theta_deg, ratio):
    return (theta_deg + math.degrees(math.atan(ratio))) % 180.0


def angular_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


# ---------- 生成与交叉核对 ----------

def render_python(seed, theta_deg, ratio, size=SIZE):
    """独立 Python 复刻，返回 PGM 原生行序数组（第 0 行为北边缘）。

    第 row 行的北向采样坐标为 (size-1-row)，与 api.Generate 完全一致。
    """
    theta = math.radians(theta_deg)
    rows, cols = np.mgrid[0:size, 0:size]
    north = size - 1 - rows
    wx, wy = warp_xy(cols * FREQ, north * FREQ, theta, ratio)
    return fbm_eval(ValueNoise(seed), wx, wy, OCTAVES, LACUNARITY, GAIN)


def cross_validate(spec):
    # PGM 第 row 行与 Python h[row] 使用同一映射（北向坐标 size-1-row），直接逐行比对。
    h = render_python(SEED, spec["thetaDeg"], spec["ratio"])
    lo, hi = float(h.min()), float(h.max())
    expected = np.clip(np.floor((h - lo) / (hi - lo) * 255 + 0.5), 0, 255).astype(np.uint8)
    actual = read_pgm(os.path.join(PGM_DIR, spec["file"])).astype(np.uint8)
    return int(np.count_nonzero(expected != actual)), float(np.abs(
        expected.astype(np.float64) - actual.astype(np.float64)).max())


def analyze_case(spec):
    h = read_pgm(os.path.join(PGM_DIR, spec["file"]))  # 原生行序，第 0 行=北
    gx, gy, mag = gradient_field(h)
    centers, counts = weighted_histogram(gx, gy, mag)
    measured = parabolic_peak(centers, counts)
    st_angle, coherence = structure_tensor(gx, gy)
    pred = predicted_angle_deg(spec["thetaDeg"], spec["ratio"])
    mismatch, max_abs = cross_validate(spec)
    return {
        "spec": spec, "h": h, "centers": centers, "counts": counts,
        "measured": measured, "predicted": pred,
        "deviation": angular_diff(measured, pred),
        "st_angle": st_angle, "st_diff": angular_diff(st_angle, pred),
        "coherence": coherence, "mismatch": mismatch, "max_abs": max_abs,
        "peak_to_median": float(counts.max() / np.median(counts)),
    }


def ensemble_analysis(theta_deg, ratio):
    """8 种子：逐种子直方图平均 -> 集成原始峰与 σ=4° 平滑峰；集成结构张量。"""
    acc = np.zeros(NBINS)
    jxx = jyy = jxy = 0.0
    for sd in ENSEMBLE_SEEDS:
        h = render_python(sd, theta_deg, ratio)
        gx, gy, _ = gradient_field(h)
        _, counts = weighted_histogram(gx, gy, mag=np.hypot(gx, gy))
        acc += counts
        jxx += np.mean(gx * gx)
        jyy += np.mean(gy * gy)
        jxy += np.mean(gx * gy)
    centers = (np.arange(NBINS) + 0.5) * (180.0 / NBINS)
    raw_peak = parabolic_peak(centers, acc)
    smooth_peak = parabolic_peak(centers, smooth_circular(acc, SMOOTH_SIGMA_DEG))
    n = len(ENSEMBLE_SEEDS)
    tr, disc = (jxx + jyy) / n, math.hypot((jxx - jyy) / n, 2 * jxy / n)
    st = math.degrees(0.5 * math.atan2(2 * jxy / n, (jxx - jyy) / n)) % 180.0
    return raw_peak, smooth_peak, st


# ---------- 出图 ----------

def plot_group(group_name, cases, results):
    n = len(cases)
    fig = plt.figure(figsize=(4.0 * n, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, n, height_ratios=[1.25, 1.0])
    by_name = {r["spec"]["name"]: r for r in results}
    for j, spec in enumerate(cases):
        r = by_name[spec["name"]]
        ax0 = fig.add_subplot(gs[0, j])
        # PGM 原生行序即北在顶（第 0 行），用 upper 原点；叠加角为东零逆时针的数学角。
        ax0.imshow(r["h"], cmap="gray", origin="upper",
                   extent=(0, SIZE, SIZE, 0), interpolation="nearest")
        cx = cy = SIZE / 2.0
        L = SIZE * 0.42
        for angle, color, ls, lw, label in [
            (spec["thetaDeg"], BLUE, "-", 1.8, "stretch dir θ"),
            (r["predicted"], ORANGE, "--", 1.8, "predicted θ+atan(r)"),
            (r["measured"], "white", "-.", 1.7, "measured hist peak"),
        ]:
            a = math.radians(angle)
            # 屏幕行坐标向南，故数学上的向北分量 sin(a) 映射为 -sin(a) 的行位移。
            ax0.plot([cx - L * math.cos(a), cx + L * math.cos(a)],
                     [cy + L * math.sin(a), cy - L * math.sin(a)],
                     color=color, ls=ls, lw=lw, label=label,
                     solid_capstyle="round", zorder=5)
        # 罗盘：正东=0°，逆时针；extent 下列像素即东向坐标，行像素南向增大。
        ax0.annotate("", xy=(SIZE * 0.93, SIZE * 0.12), xytext=(SIZE * 0.80, SIZE * 0.12),
                     arrowprops=dict(arrowstyle="->", color="lime", lw=1.4), zorder=6)
        ax0.text(SIZE * 0.94, SIZE * 0.12, "E 0°", color="lime", fontsize=8,
                 va="center", ha="left", zorder=6)
        ax0.text(SIZE * 0.80, SIZE * 0.06, "CCW", color="lime", fontsize=7,
                 va="bottom", ha="left", zorder=6)
        ax0.set_title(f"r={spec['ratio']:g}\npred={r['predicted']:.2f}°  "
                      f"meas={r['measured']:.2f}°\n|Δ|={r['deviation']:.2f}°  "
                      f"coh={r['coherence']:.2f}", fontsize=9)
        ax0.set_xticks([])
        ax0.set_yticks([])
        if j == 0:
            ax0.legend(fontsize=7, loc="upper left", framealpha=0.85)

        ax1 = fig.add_subplot(gs[1, j])
        ax1.bar(r["centers"], r["counts"] / r["counts"].max(),
                width=180.0 / NBINS, color="#6f6f6f", align="center")
        for angle, color, ls in [
            (spec["thetaDeg"], BLUE, "-"),
            (r["predicted"], ORANGE, "--"),
            (r["measured"], "black", "-."),
        ]:
            ax1.axvline(angle, color=color, ls=ls, lw=1.4)
        ax1.set_xlim(0, 180)
        ax1.set_xlabel("gradient direction (deg, east=0, CCW)", fontsize=8)
        ax1.set_ylabel("weighted count (norm.)", fontsize=8)
        ax1.tick_params(labelsize=8)
        ax1.grid(alpha=0.25, linewidth=0.6)
    fig.suptitle(f"Group {group_name}: stretch θ={cases[0]['thetaDeg']:g}° — "
                 "blue=stretch dir, orange=predicted ridge, black=measured histogram peak",
                 fontsize=11)
    out = os.path.join(FIG_DIR, f"overlay_{group_name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------- 主流程 ----------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(PGM_DIR, "manifest.json"), encoding="utf-8") as fh:
        specs = json.load(fh)
    groups = {}
    for spec in specs:
        groups.setdefault(spec["group"], []).append(spec)

    all_results, over_limit = [], []
    ensemble_rows = []
    for group_name in sorted(groups):
        cases = sorted(groups[group_name], key=lambda s: s["ratio"])
        results = [analyze_case(s) for s in cases]
        all_results.extend(results)

        rows = []
        for r in results:
            s = r["spec"]
            ens_raw, ens_smooth, ens_st = ensemble_analysis(s["thetaDeg"], s["ratio"])
            row = {
                "ratio": f"{s['ratio']:g}",
                "theta_deg": f"{s['thetaDeg']:.2f}",
                "atan_r_deg": f"{math.degrees(math.atan(s['ratio'])):.3f}",
                "predicted_deg": f"{r['predicted']:.3f}",
                "hist_peak_measured_deg": f"{r['measured']:.3f}",
                "hist_abs_diff_deg": f"{r['deviation']:.3f}",
                "structure_tensor_deg": f"{r['st_angle']:.3f}",
                "tensor_abs_diff_deg": f"{r['st_diff']:.3f}",
                "coherence": f"{r['coherence']:.4f}",
                "ensemble8_hist_peak_deg": f"{ens_raw:.3f}",
                "ensemble8_smooth4_peak_deg": f"{ens_smooth:.3f}",
                "ensemble8_tensor_deg": f"{ens_st:.3f}",
                "over_5deg": "YES" if r["deviation"] > DEVIATION_LIMIT_DEG else "no",
                "px_mismatch": r["mismatch"],
            }
            rows.append(row)
            ensemble_rows.append({
                "group": group_name, "ratio": s["ratio"], "theta_deg": s["thetaDeg"],
                "predicted_deg": round(r["predicted"], 4),
                "single_hist_peak_deg": round(r["measured"], 4),
                "single_hist_diff_deg": round(r["deviation"], 4),
                "ensemble8_hist_peak_deg": round(ens_raw, 4),
                "ensemble8_hist_diff_deg": round(angular_diff(ens_raw, r["predicted"]), 4),
                "ensemble8_smooth4_peak_deg": round(ens_smooth, 4),
                "ensemble8_smooth4_diff_deg": round(angular_diff(ens_smooth, r["predicted"]), 4),
                "ensemble8_tensor_deg": round(ens_st, 4),
                "ensemble8_tensor_diff_deg": round(angular_diff(ens_st, r["predicted"]), 4),
                "coherence": round(r["coherence"], 4),
                "over_5deg_single": bool(r["deviation"] > DEVIATION_LIMIT_DEG),
            })
            if r["deviation"] > DEVIATION_LIMIT_DEG:
                over_limit.append(r)

        csv_path = os.path.join(RES_DIR, f"angles_{group_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        fig_path = plot_group(group_name, cases, results)
        print(f"[{group_name}] {len(cases)} cases -> {csv_path} ; {fig_path}")

    # 集成稳健性表
    ens_path = os.path.join(RES_DIR, "ensemble_robustness.csv")
    with open(ens_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ensemble_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ensemble_rows)

    # r=0 各向同性基线：量化引擎自身的轴对齐偏好
    h0 = render_python(SEED, 0.0, 0.0)
    gx0, gy0, mag0 = gradient_field(h0)
    c0, counts0 = weighted_histogram(gx0, gy0, mag0)
    _, coh0 = structure_tensor(gx0, gy0)
    baseline = {
        "note": "r=0 各向同性基线：无拉伸时直方图仍偏好水平/竖直桶，即值噪声引擎的固有格点各向异性",
        "top_bins_deg": [float(c0[k]) for k in np.argsort(counts0)[::-1][:6]],
        "top_bin_count_over_median": [float(counts0[k] / np.median(counts0))
                                      for k in np.argsort(counts0)[::-1][:6]],
        "coherence": coh0,
    }
    with open(os.path.join(RES_DIR, "baseline_r0.json"), "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2, ensure_ascii=False)

    with open(os.path.join(RES_DIR, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump([{
            "group": r["spec"]["group"], "name": r["spec"]["name"],
            "theta_deg": r["spec"]["thetaDeg"], "ratio": r["spec"]["ratio"],
            "predicted_deg": round(r["predicted"], 4),
            "measured_deg": round(r["measured"], 4),
            "abs_diff_deg": round(r["deviation"], 4),
            "structure_tensor_deg": round(r["st_angle"], 4),
            "tensor_diff_deg": round(r["st_diff"], 4),
            "coherence": round(r["coherence"], 4),
            "over_5deg": bool(r["deviation"] > DEVIATION_LIMIT_DEG),
            "px_mismatch": r["mismatch"],
        } for r in all_results], fh, indent=2, ensure_ascii=False)

    print("\n=== 逐拉伸比角度比对（度，正东=0 逆时针；直方图加权峰值）===")
    for r in all_results:
        s = r["spec"]
        flag = "  <<< 超5°" if r["deviation"] > DEVIATION_LIMIT_DEG else ""
        print(f"{s['group']} r={s['ratio']:>3g}: 预测={r['predicted']:7.3f}  "
              f"实测={r['measured']:7.3f}  偏差={r['deviation']:5.3f}  "
              f"张量={r['st_angle']:7.3f}(Δ{r['st_diff']:5.2f})  "
              f"相干={r['coherence']:.3f}  像素不符={r['mismatch']}{flag}")
    worst = max(all_results, key=lambda r: r["deviation"])
    print(f"\n最大直方图偏差: {worst['deviation']:.3f}° ({worst['spec']['name']})")
    print(f"超 5° case 数: {len(over_limit)}/{len(all_results)}")
    print(f"交叉验证像素不符: {sum(r['mismatch'] for r in all_results)} / "
          f"{len(all_results) * SIZE * SIZE}")
    print(f"集成稳健性表: {ens_path}")
    print(f"r=0 基线: {os.path.join(RES_DIR, 'baseline_r0.json')}")


if __name__ == "__main__":
    main()
