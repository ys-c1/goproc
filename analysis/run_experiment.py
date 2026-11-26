"""正式实验：两套阈值档下的拐点预判 vs 完整简化扫描实测。

两套阈值档（每组 41 档，含 tau=0 完整网格基线）：
  A. 等曲率间距档  tau_a = linspace(0, kappa 的 98% 分位, 41)
  B. 等分位间距档  tau_b = F_n^{-1}(linspace(0, 0.98, 41))，每档曲率样本量近似相等

预判路径（不做完整扫描）：
  1) 由内部顶点曲率样本做 KDE，取众数 κ*，分位 q* = F_n(κ*)；
  2) 保留顶点数 N* 直接数边界 + κ>=κ* 的内部顶点（无三角剖分）；
  3) 误差阈值函数 g 用 κ* 邻域内 7 个局部探针标定仿射形式 E = a*tau + b，
     探针只做少量 Delaunay 重建，不触碰两套完整扫描档；
  4) E*_pred = g(κ*)。

对照路径：两套档各自做完整 Delaunay 简化扫描，离散曲率二阶差分变号，
  取距预判档位最近的一处作为实测拐点。

运行：cd /tmp && python3 -P <repo>/analysis/run_experiment.py
输出：analysis/out/ 下 CSV / PNG；汇总打印到 stdout。
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrainlib as tl

REPO = "/home/ys/feedback/uidreczz28eCcwNSxAr"
OUT = f"{REPO}/analysis/out"
SEEDS = (7, 1, 42)
TIERS = 41
PROBE_HALF_WIDTH = 0.35   # 局部探针取 [0.65κ*, 1.35κ*]
PROBES = 7
SMOOTH = 5


def make_schedules(samp: np.ndarray) -> dict[str, np.ndarray]:
    hi = float(np.quantile(samp, 0.98))
    a = np.linspace(0.0, hi, TIERS)
    b = np.quantile(samp, np.linspace(0.0, 0.98, TIERS))
    b[0] = 0.0
    return {"A_uniform_kappa": a, "B_uniform_quantile": b}


def local_probe_g(h: np.ndarray, kappa: np.ndarray, kstar: float) -> dict[str, tuple[float, float]]:
    """κ* 邻域内少量探针，标定 E = a*tau + b（RMSE 与 Emax 各一组）。"""
    taus = np.linspace(kstar * (1.0 - PROBE_HALF_WIDTH), kstar * (1.0 + PROBE_HALF_WIDTH), PROBES)
    rows = [tl.simplify_and_error(h, kappa, float(t)) for t in taus]
    out: dict[str, tuple[float, float]] = {}
    for j, metric in enumerate(("rmse", "emax"), start=1):
        e = np.array([r[j] for r in rows])
        a, b = np.polyfit(taus, e, 1)
        out[metric] = (float(a), float(b))
    out["_taus"] = taus  # type: ignore[assignment]
    out["_rows"] = rows  # type: ignore[assignment]
    return out


def nearest_tier(taus: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(taus - value)))


def run_seed(seed: int) -> dict:
    h = tl.load_raw(f"{REPO}/artifacts/raw/seed{seed}.raw")
    n = h.shape[0]
    kappa = tl.curvature(h)
    samp = tl.interior_curvature(kappa)
    n_boundary = 4 * (n - 1)

    # ---- 预判（无完整扫描）----
    kstar, qstar = tl.curvature_mode(samp)
    n_pred = tl.retained_count(kappa, kstar)
    probe = local_probe_g(h, kappa, kstar)
    g_coef = {m: probe[m] for m in ("rmse", "emax")}
    e_pred = {m: g_coef[m][0] * kstar + g_coef[m][1] for m in ("rmse", "emax")}

    schedules = make_schedules(samp)
    result = {
        "seed": seed, "n": n, "n_boundary": n_boundary,
        "kstar": kstar, "qstar": qstar, "n_pred": n_pred,
        "g_coef": g_coef, "e_pred": e_pred, "schedules": {}, "samp": samp,
    }

    for sched_name, taus in schedules.items():
        sw = tl.sweep(h, kappa, taus)
        pred_tier = nearest_tier(taus, kstar)
        sched = {"taus": taus, "sweep": sw, "pred_tier": pred_tier, "metrics": {}}
        for metric in ("rmse", "emax"):
            x = sw[metric]
            y = sw["nvert"].astype(float)
            kg = tl.discrete_curvature(x, y, smooth=SMOOTH)
            d2 = np.full_like(kg, np.nan)
            d2[1:-1] = kg[2:] - 2.0 * kg[1:-1] + kg[:-2]
            flips = [i for i in range(2, len(kg) - 1)
                     if np.isfinite(d2[i - 1]) and np.isfinite(d2[i]) and d2[i - 1] * d2[i] < 0]
            knee = min(flips, key=lambda i: abs(i - pred_tier))
            sched["metrics"][metric] = {
                "kg": kg, "d2": d2, "flips": flips, "knee": knee,
                "delta_tier": knee - pred_tier,
            }
        result["schedules"][sched_name] = sched
    return result


def write_csvs(all_results: list[dict]) -> None:
    # 每种子 × 每档组：逐行表
    for res in all_results:
        for sched_name, sched in res["schedules"].items():
            path = f"{OUT}/seed{res['seed']}_{sched_name}_tiers.csv"
            sw = sched["sweep"]
            with open(path, "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow([
                    "tier", "tau", "quantile_F(tau)", "nvert_measured", "rmse_measured",
                    "emax_measured", "ntri", "predicted_knee_tier", "rmse_knee_tier",
                    "emax_knee_tier",
                ])
                for i in range(TIERS):
                    # 分位列用经验 CDF 现算，保持与曲率累积口径一致。
                    f_tau = float(np.mean(res["samp"] <= sw["tau"][i]))
                    wr.writerow([
                        i, f"{sw['tau'][i]:.8f}", f"{f_tau:.6f}", int(sw["nvert"][i]),
                        f"{sw['rmse'][i]:.8f}", f"{sw['emax'][i]:.8f}", int(sw["ntri"][i]),
                        sched["pred_tier"],
                        sched["metrics"]["rmse"]["knee"],
                        sched["metrics"]["emax"]["knee"],
                    ])
    # 汇总预判 vs 实测
    with open(f"{OUT}/knee_prediction_summary.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "seed", "schedule", "metric", "kstar_mode", "qstar_quantile",
            "pred_tier", "pred_tau", "pred_N", "pred_E",
            "measured_knee_tier", "measured_tau", "measured_N", "measured_E",
            "delta_tier", "abs_delta_tier",
        ])
        for res in all_results:
            for sched_name, sched in res["schedules"].items():
                sw = sched["sweep"]
                for metric in ("rmse", "emax"):
                    k = sched["metrics"][metric]["knee"]
                    wr.writerow([
                        res["seed"], sched_name, metric,
                        f"{res['kstar']:.8f}", f"{res['qstar']:.6f}",
                        sched["pred_tier"], f"{sw['tau'][sched['pred_tier']]:.8f}",
                        res["n_pred"], f"{res['e_pred'][metric]:.8f}",
                        k, f"{sw['tau'][k]:.8f}", int(sw["nvert"][k]),
                        f"{sw[metric][k]:.8f}",
                        sched["metrics"][metric]["delta_tier"],
                        abs(sched["metrics"][metric]["delta_tier"]),
                    ])


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    all_results = [run_seed(s) for s in SEEDS]
    write_csvs(all_results)
    for res in all_results:
        print(f"\n===== seed {res['seed']}  κ*={res['kstar']:.6f} q*={res['qstar']:.4f} "
              f"N_pred={res['n_pred']} =====")
        for m in ("rmse", "emax"):
            a, b = res["g_coef"][m]
            print(f"  g_{m}(τ) = {a:.3f}·τ + {b:.5f}  →  E*_pred={res['e_pred'][m]:.6f}")
        for sched_name, sched in res["schedules"].items():
            for metric in ("rmse", "emax"):
                md = sched["metrics"][metric]
                sw = sched["sweep"]
                k = md["knee"]
                print(f"  [{sched_name}/{metric}] 预测档={sched['pred_tier']} "
                      f"实测拐点档={k} Δ={md['delta_tier']:+d}  "
                      f"E={sw[metric][k]:.6f} N={int(sw['nvert'][k])} 变号点数={len(md['flips'])}")
    print(f"\nCSV 已写入 {OUT}")


if __name__ == "__main__":
    main()
