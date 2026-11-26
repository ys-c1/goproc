#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层填色锯齿分析：叠加模型 + 同号条件反解 + 逐像素统计对账。

记号
  z ∈ [0,1] 归一化高度；L 个色阶，阶宽 h=1/L，色阶边界在 k/L，k=1..L-1。
  连续色阶坐标 w = z/h（单位是“个色阶”），边界即整数 k。
  离散斜率（格边 i->j）: s_ij = w_j - w_i。
  量化两档:
      floor 档 qf = floor(w)，量化误差 ef = floor(w)-w = -r,  r={w}∈[0,1)，ef∈(-1,0]
      ceil  档 qc = ceil(w)， 量化误差 ec = ceil(w)-w  = 1-r （r>0 时）, ec∈[0,1)
  叠加式（表观阶跃 = 斜率效应 + 量化误差效应）:
      q_j - q_i = s_ij + (e_j - e_i)

同号条件（反解位置用）
  把每条格边按高度定向 a->b（w_a<=w_b，故 s>=0）。令 ra={w_a}, rb={w_b}。
  floor 档: a 侧误差为负 <=> ra>0；b 侧须落在边界正侧（上一档）<=> floor(w_b)=k>floor(w_a)。
            锯齿 ⇔ ra>0 且 floor(w_a)<floor(w_b)，即“负误差端 + 正侧端”。
  ceil  档: a 侧误差为正 <=> ra>0；b 侧越过整数 <=> ceil(w_b)>ceil(w_a)。
            锯齿 ⇔ ra>0 且 ceil(w_a)<ceil(w_b)，即“正误差端 + 负侧端”。
  统一读法：边两端的量化误差符号与其沿坡方向（斜率符号）一致时出锯齿；
  条件只含端点 w，可直接反解出格边，无需先生成填色图。

统计口径（独立第二条代码路径）
  锯齿 = 跨越同一色阶边界 k 的相邻两像素（4-邻域，无向边只计一次）。
  直接在量化图 q 上比较：边对每个满足 min(q)<k<=max(q) 的边界 k 贡献一次。
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


def load_raw(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    n = int(round(math.sqrt(raw.size)))
    if n * n != raw.size:
        raise ValueError(f"数据长度 {raw.size} 不是完全平方")
    return raw.reshape(n, n).astype(np.float64)


def all_edges(n: int):
    """全部无向格边，每边一次。返回 (端点索引 i, 端点索引 j)。"""
    g = np.arange(n * n).reshape(n, n)
    h_i, h_j = g[:, :-1].ravel(), g[:, 1:].ravel()
    v_i, v_j = g[:-1, :].ravel(), g[1:, :].ravel()
    return np.concatenate([h_i, v_i]), np.concatenate([h_j, v_j])


def model_predict(w: np.ndarray, mode: str):
    """同号条件反解。返回每边信息。

    边界归属统一按几何阈值：边跨过整数 k ⇔ min(w)<k<=max(w)。
      floor 档: ef=-r 恒非正。低端 a 带负误差 <=> r_a>0；高端 b 在 k 正侧。
                锯齿 ⇔ r_a>0 且 floor(w_a)<floor(w_b)。
      ceil  档: ec=1-r（r>0 时）恒正。低端 a 带正误差 <=> r_a>0；高端 b 越过整数。
                锯齿 ⇔ r_a>0 且 ceil(w_a)<ceil(w_b)。
    两种写法对 r>0 的边给出同一边集与同一 k=floor(max(w)) 归属。
    """
    n = w.shape[0]
    wi = w.ravel()
    e_i, e_j = all_edges(n)
    zi, zj = wi[e_i], wi[e_j]
    lo = np.minimum(zi, zj)
    hi = np.maximum(zi, zj)
    r_lo = lo - np.floor(lo)
    if mode == "floor":
        hit = (r_lo > 0.0) & (np.floor(lo) < np.floor(hi))
    elif mode == "ceil":
        hit = (r_lo > 0.0) & (np.ceil(lo) < np.ceil(hi))
    else:
        raise ValueError(mode)
    f_lo = np.floor(lo).astype(np.int64)
    f_hi = np.floor(hi).astype(np.int64)
    multi = (f_hi - f_lo) > 1
    return e_i, e_j, hit, f_lo, f_hi, multi


def pixel_stat(w: np.ndarray, mode: str):
    """逐像素统计（独立路径）：先量化成档位图 q，再逐边按几何阈值 k 计数。

    边 (i,j) 对每个满足 min(w)<k<=max(w) 的整数 k 贡献一次锯齿；
    另返回由量化图直接判定的逐边布尔集 hit_q，供与模型同号判据逐边对账。
    """
    n = w.shape[0]
    wi = w.ravel()
    q = np.floor(wi).astype(np.int64) if mode == "floor" else np.ceil(wi).astype(np.int64)
    e_i, e_j = all_edges(n)
    zi, zj = wi[e_i], wi[e_j]
    lo = np.minimum(zi, zj)
    hi = np.maximum(zi, zj)
    f_lo = np.floor(lo).astype(np.int64)
    f_hi = np.floor(hi).astype(np.int64)
    per_k: dict[int, int] = {}
    for a, b in zip(f_lo, f_hi):
        for k in range(a + 1, b + 1):
            per_k[k] = per_k.get(k, 0) + 1
    exact = int(np.count_nonzero(wi == np.floor(wi)))
    qgap = np.maximum(q[e_i], q[e_j]) - np.minimum(q[e_i], q[e_j])
    hit_q = qgap > 0
    return per_k, q.reshape(n, n), exact, int(np.count_nonzero(qgap > 1)), hit_q


def analyze(z: np.ndarray, levels: int, mode: str):
    n = z.shape[0]
    w = np.clip(z, 0.0, 1.0) * levels
    w = np.minimum(w, np.nextafter(levels, np.float64(-np.inf)))  # 顶点归入最高档

    e_i, e_j, hit, f_lo, f_hi, multi = model_predict(w, mode)
    stat_per_k, qimg, exact_int, multi_stat, hit_q = pixel_stat(w, mode)

    # 逐边集合对账：模型同号判据 vs 量化图直判
    set_mismatch = int(np.count_nonzero(hit != hit_q))

    # 模型侧按边界汇总，口径与统计侧完全一致（多边界边逐 k 计）
    model_per_k = {k: 0 for k in range(1, levels)}
    for a, b in zip(f_lo, f_hi):
        for k in range(a + 1, b + 1):
            if 1 <= k < levels:
                model_per_k[k] += 1

    jag_pixels = np.unique(np.concatenate([e_i[hit], e_j[hit]]))
    marked = np.zeros(n * n, dtype=bool)
    marked[jag_pixels] = True

    rows = []
    for k in range(1, levels):
        m, s = model_per_k[k], stat_per_k.get(k, 0)
        dev = abs(m - s) / s if s else (0.0 if m == 0 else math.inf)
        rows.append({"boundary": k, "model": int(m), "pixel_stat": int(s),
                     "rel_dev": dev, "flag_gt_20pct": dev > 0.20})

    info = {
        "n": n, "levels": levels, "mode": mode,
        "total_edges": int(e_i.size),
        "model_total": int(sum(model_per_k.values())),
        "stat_total": int(sum(stat_per_k.values())),
        "jagged_edges": int(hit.sum()),
        "jagged_pixels": int(marked.sum()),
        "multi_boundary_edges": int(multi.sum()),
        "exact_integer_samples": exact_int,
        "edge_set_mismatch": set_mismatch,
        "per_boundary": rows,
    }
    return info, marked.reshape(n, n), qimg.reshape(n, n), w.reshape(n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    z = load_raw(args.raw)
    report = {"source": os.path.basename(args.raw), "shape": list(z.shape),
              "z_range": [float(z.min()), float(z.max())], "runs": []}
    for L in args.levels:
        for mode in ("floor", "ceil"):
            info, *_ = analyze(z, L, mode)
            report["runs"].append(info)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"wrote": args.out, "runs": len(report["runs"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
