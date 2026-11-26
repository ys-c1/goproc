#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层值噪声（goproc fbm octaves）极值点的空间独立性、重叠修正与总数预判。

设计：
- 噪声由本脚本按 internal/noise/noise.go 逐位复现（splitmix64 + Fisher-Yates
  + quintic fade），并通过 `go run ./cmd/noisedump` 与 Go 实现逐值对拍。
- 极值点 = 内部像素上的严格 8 邻域局部极大/极小（与 scipy maximum/minimum_filter
  的严格口径一致，已交叉验证）。
- “总极值数”= 各层极值掩膜的并集（同像素被多层标记只计一次）；这是
  “密度相加 - 重叠修正”公式唯一直接对应的量。叠加场本身的极值另作补充对照。

输出（默认 artifacts/extrema/）：
- comparison_by_seed.csv   每种子 × K=4/8 的预判与实测逐行数据
- layer_densities.csv      每层密度
- independence_pairs.csv   多种子配对重叠比 / phi
- shift_test.csv           环形平移零假设 z 分数
- summary.json             头条数字
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------- noise exact
def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def make_perm(seed: int) -> np.ndarray:
    """与 noise.New 完全相同的置换表，返回长度 512 的 int64 数组。"""
    p = list(range(256))
    state = int(seed)
    for i in range(255, 0, -1):
        state = splitmix64(state)
        j = state % (i + 1)
        p[i], p[j] = p[j], p[i]
    return np.array([p[i & 255] for i in range(512)], dtype=np.int64)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def value_layer(perm: np.ndarray, n: int, freq: float,
                ox: float = 0.0, oy: float = 0.0) -> np.ndarray:
    """生成一层未加权 Value 噪声场，坐标 x*freq (+phase)，与 Go 同序运算。"""
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    X = xs * freq + ox
    Y = ys * freq + oy
    x0 = np.floor(X).astype(np.int64)
    y0 = np.floor(Y).astype(np.int64)
    u = _fade(X - x0)
    v = _fade(Y - y0)

    def grad(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        h = perm[(perm[ix & 255] + iy) & 511]
        return h / 127.5 - 1.0

    c00 = grad(x0, y0)
    c10 = grad(x0 + 1, y0)
    c01 = grad(x0, y0 + 1)
    c11 = grad(x0 + 1, y0 + 1)
    a = c00 + (c10 - c00) * u
    b = c01 + (c11 - c01) * u
    return a + (b - a) * v


def go_crosscheck(n: int = 24, seed: int = 7, octaves: int = 8) -> float:
    """与 cmd/noisedump 的 Go 输出对拍，返回最大绝对误差。"""
    out = subprocess.run(
        ["go", "run", "./cmd/noisedump",
         "-size", str(n), "-seed", str(seed), "-octaves", str(octaves)],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    ref = json.loads(out.stdout)
    perm = make_perm(seed)
    f0 = ref["frequency"]
    worst = 0.0
    for k, row in enumerate(ref["layers"]):
        a = value_layer(perm, n, f0 * ref["lacunarity"] ** k)
        worst = max(worst, float(np.max(np.abs(a.ravel() - np.array(row)))))
    return worst


# ------------------------------------------------------------- extrema masks
def extrema_masks(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回内部网格 (n-2)^2 上的 (极大掩膜, 极小掩膜)，严格 8 邻域。"""
    c = h[1:-1, 1:-1]
    nbr = [
        h[:-2, 1:-1], h[2:, 1:-1], h[1:-1, :-2], h[1:-1, 2:],
        h[:-2, :-2], h[:-2, 2:], h[2:, :-2], h[2:, 2:],
    ]
    mx = np.ones(c.shape, dtype=bool)
    mn = np.ones(c.shape, dtype=bool)
    for z in nbr:
        mx &= c > z
        mn &= c < z
    return mx, mn


def combined_mask(h: np.ndarray) -> np.ndarray:
    mx, mn = extrema_masks(h)
    return mx | mn


# ------------------------------------------------------------------- metrics
def union_prediction(ps: np.ndarray) -> float:
    """独立 Bernoulli 掩膜下的并集概率 1 - prod(1-p_k)。"""
    return 1.0 - float(np.prod(1.0 - ps))


def pair_correction(ps: np.ndarray) -> float:
    """容斥二阶（配对）项之和 sum_{i<j} p_i p_j，重叠修正的领头项。"""
    k = len(ps)
    return sum(ps[i] * ps[j] for i in range(k) for j in range(i + 1, k))


def phi_coefficient(ei: np.ndarray, ej: np.ndarray, m2: int) -> float:
    pi = ei.sum() / m2
    pj = ej.sum() / m2
    pij = (ei & ej).sum() / m2
    denom = np.sqrt(pi * (1 - pi) * pj * (1 - pj))
    return 0.0 if denom == 0 else (pij - pi * pj) / denom


def shift_z(ei: np.ndarray, ej: np.ndarray, rng: np.random.Generator,
            n_shift: int = 128) -> tuple[float, float, float]:
    """环形平移零假设：保持两掩膜各自空间结构，返回 (观测重合, 均值, z)。"""
    m = ei.shape[0]
    obs = int((ei & ej).sum())
    qs = np.empty(n_shift, dtype=np.int64)
    for t in range(n_shift):
        dy = int(rng.integers(0, m))
        dx = int(rng.integers(0, m))
        rolled = np.roll(np.roll(ei, dy, axis=0), dx, axis=1)
        qs[t] = (rolled & ej).sum()
    mu = float(qs.mean())
    sd = float(qs.std(ddof=1)) if n_shift > 1 else 0.0
    z = (obs - mu) / sd if sd > 1e-12 else (0.0 if obs == mu else np.inf)
    return obs, mu, z


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--frequency", type=float, default=0.02)
    ap.add_argument("--eval-seeds", type=int, nargs="+",
                    default=list(range(1, 13)))
    ap.add_argument("--cal-seeds", type=int, nargs="+",
                    default=[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008])
    ap.add_argument("--shift-seeds", type=int, nargs="+", default=[1, 7, 42])
    ap.add_argument("--shifts", type=int, default=128)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "extrema")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Go 对拍中 ...")
    diff = go_crosscheck()
    print(f"  最大逐值误差 = {diff}")
    assert diff < 1e-10, "Python 噪声复现与 Go 不一致"

    n = args.n
    m2 = (n - 2) ** 2
    f0 = args.frequency

    def seed_layers(seed: int, k_max: int = 8):
        perm = make_perm(seed)
        fields, masks, nmax, nmin = [], [], [], []
        for k in range(k_max):
            h = value_layer(perm, n, f0 * 2.0 ** k)
            mx, mn = extrema_masks(h)
            fields.append(h)
            masks.append(mx | mn)
            nmax.append(int(mx.sum()))
            nmin.append(int(mn.sum()))
        return fields, masks, nmax, nmin

    # ---- calibration: a-priori layer densities + pair inflation R_ij --
    cal_p = np.zeros(8)
    r_sum = np.zeros((8, 8))
    r_cnt = np.zeros((8, 8))
    for s in args.cal_seeds:
        _, masks, _, _ = seed_layers(s)
        cal_p += np.array([e.sum() / m2 for e in masks])
        nsv = np.array([e.sum() for e in masks], dtype=float)
        for i in range(8):
            for j in range(i + 1, 8):
                expect = nsv[i] * nsv[j] / m2
                if expect > 0:
                    r_sum[i, j] += int((masks[i] & masks[j]).sum()) / expect
                    r_cnt[i, j] += 1
    cal_p /= len(args.cal_seeds)
    r_bar = np.where(r_cnt > 0, r_sum / np.maximum(r_cnt, 1), 1.0)

    density_rows = []
    comp_rows = []
    pair_rows = []

    for seed in args.eval_seeds:
        fields, masks, nmax, nmin = seed_layers(seed)
        ns = np.array([e.sum() for e in masks], dtype=float)
        ps = ns / m2
        for k in range(8):
            density_rows.append(dict(
                seed=seed, layer=k, freq=f0 * 2.0 ** k,
                n_total=int(ns[k]), n_max=nmax[k], n_min=nmin[k],
                density=f"{ps[k]:.8f}"))

        # pair independence stats
        for i in range(8):
            for j in range(i + 1, 8):
                q = int((masks[i] & masks[j]).sum())
                expect = ns[i] * ns[j] / m2
                pair_rows.append(dict(
                    seed=seed, i=i, j=j, obs=q,
                    expect=f"{expect:.4f}",
                    ratio=f"{q / expect:.4f}" if expect > 0 else "nan",
                    phi=f"{phi_coefficient(masks[i], masks[j], m2):.6f}"))

        for K in (4, 8):
            pK = ps[:K]
            nK = ns[:K]
            S = int(nK.sum())
            union = np.zeros_like(masks[0])
            for k in range(K):
                union |= masks[k]
            U = int(union.sum())
            c_obs = S - U
            u_ind = m2 * union_prediction(pK)
            c_ind = S - u_ind
            c2 = m2 * pair_correction(pK)
            u_prior = m2 * union_prediction(cal_p[:K])
            # 校准配对膨胀：只修正二阶项，高阶仍按独立乘积
            extra_pair = sum(
                (r_bar[i, j] - 1.0) * nK[i] * nK[j] / m2
                for i in range(K) for j in range(i + 1, K))
            u_cal = u_ind - extra_pair
            comp_rows.append(dict(
                n=n, seed=seed, K=K, M=m2,
                sum_layer=S,
                corr_obs=c_obs,
                corr_ind=f"{c_ind:.1f}",
                pair_term_C2=f"{c2:.1f}",
                pred_union_plugin=f"{u_ind:.1f}",
                actual_union=U,
                residual_plugin=f"{u_ind - U:+.1f}",
                relerr_plugin_pct=f"{(u_ind - U) / U * 100:+.3f}",
                pred_union_prior=f"{u_prior:.1f}",
                relerr_prior_pct=f"{(u_prior - U) / U * 100:+.3f}",
                pred_union_cal=f"{u_cal:.1f}",
                relerr_cal_pct=f"{(u_cal - U) / U * 100:+.3f}",
                density_actual=f"{U / m2:.6f}",
                density_pred=f"{u_ind / m2:.6f}",
            ))

    # ---- toroidal shift z for selected seeds --------------------------
    shift_rows = []
    for seed in args.shift_seeds:
        if seed in args.eval_seeds:
            idx = args.eval_seeds.index(seed)
            # regenerate (cheap enough, keeps code simple)
        _, masks, _, _ = seed_layers(seed)
        rng = np.random.default_rng(20261127 + seed)
        for i in range(8):
            for j in range(i + 1, 8):
                obs, mu, z = shift_z(masks[i], masks[j], rng, args.shifts)
                shift_rows.append(dict(
                    seed=seed, i=i, j=j, obs=obs,
                    shift_mean=f"{mu:.3f}", z=f"{z:.2f}"))

    # ---- write CSVs ----------------------------------------------------
    def write_csv(name: str, rows: list[dict]):
        path = args.out / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", path.relative_to(REPO))

    write_csv("comparison_by_seed.csv", comp_rows)
    write_csv("layer_densities.csv", density_rows)
    write_csv("independence_pairs.csv", pair_rows)
    write_csv("shift_test.csv", shift_rows)

    # ---- aggregate summary --------------------------------------------
    summary: dict = {"n": n, "M": m2, "frequency": f0,
                     "eval_seeds": args.eval_seeds,
                     "cal_seeds": args.cal_seeds,
                     "go_max_abs_diff": diff,
                     "cal_density": cal_p.tolist()}
    for K in (4, 8):
        rows = [r for r in comp_rows if r["K"] == K]
        rel = np.array([float(r["relerr_plugin_pct"]) for r in rows])
        relp = np.array([float(r["relerr_prior_pct"]) for r in rows])
        relc = np.array([float(r["relerr_cal_pct"]) for r in rows])
        res = np.array([float(r["residual_plugin"]) for r in rows])
        co = np.array([r["corr_obs"] for r in rows], dtype=float)
        ci = np.array([float(r["corr_ind"]) for r in rows])
        U = np.array([r["actual_union"] for r in rows], dtype=float)
        summary[f"K{K}"] = dict(
            actual_union_mean=float(U.mean()),
            actual_union_min=int(U.min()), actual_union_max=int(U.max()),
            relerr_plugin_mean_pct=float(rel.mean()),
            relerr_plugin_absmax_pct=float(np.abs(rel).max()),
            residual_plugin_mean_abs=float(np.abs(res).mean()),
            relerr_prior_mean_pct=float(relp.mean()),
            relerr_prior_absmax_pct=float(np.abs(relp).max()),
            relerr_cal_mean_pct=float(relc.mean()),
            relerr_cal_mape_pct=float(np.abs(relc).mean()),
            relerr_cal_absmax_pct=float(np.abs(relc).max()),
            corr_obs_mean=float(co.mean()),
            corr_obs_min=int(co.min()), corr_obs_max=int(co.max()),
            corr_ind_mean=float(ci.mean()),
            corr_ratio_ind_over_obs=float(ci.mean() / co.mean()),
        )
    # pair ratio aggregates by pair
    pair_summary = {}
    for i in range(8):
        for j in range(i + 1, 8):
            rs = [float(r["ratio"]) for r in pair_rows if r["i"] == i and r["j"] == j]
            zs = [float(r["z"]) for r in shift_rows if r["i"] == i and r["j"] == j]
            zm = float(np.mean(zs))
            za = float(np.max(np.abs(zs)))
            pair_summary[f"{i}-{j}"] = dict(
                ratio_mean=float(np.mean(rs)), ratio_sd=float(np.std(rs)),
                z_mean=None if not np.isfinite(zm) else zm,
                z_absmax=None if not np.isfinite(za) else za)
    summary["pairs"] = pair_summary
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", (args.out / "summary.json").relative_to(REPO))
    print(json.dumps({k: summary[k] for k in ("K4", "K8")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
