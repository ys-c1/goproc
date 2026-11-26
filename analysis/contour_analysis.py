"""等高线总长与坡度分布关系分析。

理论主线（连续情形，coarea 公式）：
    对高度场 h: Ω→[0,1]，高度 α 的等高线长度为 L(α)，坡度 s=|∇h|，则
        ∫_0^1 L(α) dα = ∫_Ω |∇h| dA = A · E[s]。
    等高距 dh 下层级数约为 1/dh（归一化高度跨度 H=1），故
        L_total(dh) ≈ A · E[s] / dh。
    即等高线总长与坡度期望成正比、与等高距成反比——任务书所述"成反比"
    以及"高度跨度除以坡度期望"的写法在量纲与单调性上均不成立，本脚本以
    推导与实验给出纠正。

计长口径（任务书规定）：marching-squares 交点吸附到网格边中点，每条连接
要么是横竖像素边（长度 1），要么跨格对角（长度 √2）。该口径量的是数字
阶梯长度，与连续 coarea 的欧氏长度不同。为此同时给出两种坡度期望：
  * slope_euclid  : 连续 coarea 对应的欧氏坡度 E[|∇h|]（中心差分）；
  * slope_digital : 与数字阶梯计长严格匹配的坡度，即沿水平集直接积分的
                    全变差 TV_digital / A（只数边界边，不生成等高线折线）。
实测表明后者在该计长口径下把预判偏差压到一成以内。

仅依赖 numpy / matplotlib，配合 python3 -I 运行。
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

SQRT2 = math.sqrt(2.0)

# marching-squares 16 情形 -> 连接的网格边。
# 角点：左下=1 右下=2 右上=4 左上=8；边：0=底 1=右 2=顶 3=左。
# 5、10 为鞍部歧义情形，按"保持左下组分连通"解。
MS_LINKS = {
    0: (),
    1: ((0, 3),),
    2: ((0, 1),),
    3: ((1, 3),),
    4: ((1, 2),),
    5: ((0, 3), (1, 2)),
    6: ((0, 2),),
    7: ((2, 3),),
    8: ((2, 3),),
    9: ((0, 2),),
    10: ((0, 1), (2, 3)),
    11: ((1, 2),),
    12: ((1, 3),),
    13: ((0, 1),),
    14: ((0, 3),),
    15: (),
}

# 每种 marching-squares 情形在中点吸附口径下的单元内连接长度：
# 相邻边（共享一个角点）连接是对角跨格，长 √2；对边连接是横竖像素边，长 1；
# 鞍部情形 5、10 含两条对角连接，共 2√2。
_OPPOSITE = {(0, 2), (1, 3)}
MS_WEIGHT = np.zeros(16, dtype=np.float64)
for _code, _links in MS_LINKS.items():
    MS_WEIGHT[_code] = sum(
        1.0 if tuple(sorted(_e)) in _OPPOSITE else SQRT2 for _e in _links)


# ---------------------------------------------------------------- 数据读取
def load_heightmap(path: str | Path) -> np.ndarray:
    """读取 dumphm 写出的 GOPROC-HM 全精度二进制高度场。"""
    raw = Path(path).read_bytes()
    if raw[:9] != b"GOPROC-HM":
        raise ValueError(f"bad magic in {path}")
    (n,) = struct.unpack_from("<I", raw, 9)
    data = np.frombuffer(raw, dtype="<f8", offset=13, count=n * n)
    return data.reshape(n, n).astype(np.float64, copy=True)


# ---------------------------------------------------------------- 坡度统计
def slope_euclid(h: np.ndarray) -> dict:
    """连续 coarea 口径：欧氏坡度 |∇h| 的分布统计（间距 1 像素）。"""
    gy, gx = np.gradient(h)
    s = np.hypot(gx, gy)
    return {
        "mean": float(s.mean()),
        "mean_interior": float(s[1:-1, 1:-1].mean()),
        "median": float(np.median(s)),
        "p90": float(np.percentile(s, 90)),
        "max": float(s.max()),
        "tv_euclid": float(s.sum()),
        "tv_l1": float(np.abs(gx).sum() + np.abs(gy).sum()),
    }


def digital_tv(h: np.ndarray) -> dict:
    """数字阶梯口径的全变差，与 marching-squares 中点计长严格同源。

    离散 coarea：TV = ∫ L(α) dα。中点连接完全落在单个网格单元内
    （相邻边连接长 √2、对边连接长 1、鞍部两条对角共 2√2），因此边界
    长度是单元局部量，全局积分等于各单元积分之和。对一个四角高度为
    c1..c4（对应位 1,2,4,8）的单元，让阈值 α 从最低角扫到最高角：
    每越过一个角点，marching-squares 情形 code 改变一次，单元在该
    高度区间贡献 W[code]×高度间隔，W 即中点连接长度表。把所有单元
    求和即得 TV，全程只读取高度场本身，不生成任何等高线折线。
    """
    # 四个角点按 bit1(左下)、bit2(右下)、bit4(右上)、bit8(左上) 排列
    cells = np.stack([
        h[1:, :-1].ravel(), h[1:, 1:].ravel(),
        h[:-1, 1:].ravel(), h[:-1, :-1].ravel(),
    ], axis=1)
    bits = np.array([1, 2, 4, 8], dtype=np.int64)
    order = np.argsort(cells, axis=1)
    sorted_vals = np.sort(cells, axis=1)
    sorted_bits = np.take_along_axis(
        np.broadcast_to(bits, cells.shape), order, axis=1)
    tv = 0.0
    for k in range(3):
        # 区间 (sorted_vals[k], sorted_vals[k+1]) 内高于阈值的角点为 k+1..3
        code = sorted_bits[:, k + 1:].sum(axis=1)
        gaps = sorted_vals[:, k + 1] - sorted_vals[:, k]
        tv += float((MS_WEIGHT[code] * gaps).sum())
    return {"tv_digital": tv}


# ---------------------------------------------------------------- 等高线
def _edge_node(e: int, i: int, j: int, n: int) -> int:
    """网格边 -> 半像素整数格 (u=2x, v=2y) 上的边中点节点编号。"""
    m = 2 * n + 1
    if e == 0:
        u, v = 2 * i + 1, 2 * j
    elif e == 1:
        u, v = 2 * (i + 1), 2 * j + 1
    elif e == 2:
        u, v = 2 * i + 1, 2 * (j + 1)
    else:
        u, v = 2 * i, 2 * j + 1
    return v * m + u


def contour_level(h: np.ndarray, level: float) -> dict:
    """单高度层级的 marching-squares 连接集合与数字阶梯长度。"""
    n = h.shape[0]
    above = h >= level
    edges: set[tuple[int, int]] = set()
    straight = 0
    diagonal = 0
    m = 2 * n + 1
    for j in range(n - 1):
        r0 = above[j]
        r1 = above[j + 1]
        for i in range(n - 1):
            code = ((1 if r1[i] else 0) | (2 if r1[i + 1] else 0)
                    | (4 if r0[i + 1] else 0) | (8 if r0[i] else 0))
            for ea, eb in MS_LINKS[code]:
                na = _edge_node(ea, i, j, n)
                nb = _edge_node(eb, i, j, n)
                key = (na, nb) if na < nb else (nb, na)
                if key in edges:
                    continue
                edges.add(key)
                ua, va = na % m, na // m
                ub, vb = nb % m, nb // m
                if ua == ub or va == vb:
                    straight += 1
                else:
                    diagonal += 1
    return {
        "level": level,
        "edges": edges,
        "length": straight + diagonal * SQRT2,
        "n_straight": straight,
        "n_diagonal": diagonal,
    }


def trace_chains(edges: set[tuple[int, int]], n: int) -> list[list[tuple[float, float]]]:
    """把边集合组织成闭合环或开口折线，返回物理坐标序列（像素单位）。

    节点编号位于宽 2n+1 的半像素整数格上，物理坐标 = 编号坐标 / 2。
    """
    width = 2 * n + 1
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    remaining = set(edges)
    raw_chains: list[list[int]] = []
    for s in (p for p, q in adj.items() if len(q) == 1):
        if not any(s in e for e in remaining):
            continue
        chain = [s]
        prev, cur = -1, s
        while True:
            nxts = [q for q in adj[cur] if q != prev]
            if not nxts:
                break
            nxt = nxts[0]
            e = (cur, nxt) if cur < nxt else (nxt, cur)
            if e not in remaining:
                break
            remaining.discard(e)
            chain.append(nxt)
            prev, cur = cur, nxt
        raw_chains.append(chain)
    while remaining:
        s = next(iter(remaining))[0]
        chain = [s]
        prev, cur = -1, s
        while True:
            nxts = [q for q in adj[cur] if q != prev]
            if not nxts:
                break
            nxt = nxts[0]
            e = (cur, nxt) if cur < nxt else (nxt, cur)
            if e not in remaining:
                break
            remaining.discard(e)
            chain.append(nxt)
            prev, cur = cur, nxt
            if cur == s:
                break
        raw_chains.append(chain)
    out = []
    for ch in raw_chains:
        out.append([((p % width) / 2.0, (p // width) / 2.0) for p in ch])
    return out


def levels_for(dh: float, eps: float = 1e-12) -> np.ndarray:
    """开区间 (0,1) 内 k*dh 的层级，避开归一化边界 0 与 1。"""
    out = []
    k = 1
    while k * dh < 1.0 - eps:
        out.append(k * dh)
        k += 1
    return np.array(out)


def measure_contours(h: np.ndarray, dh: float) -> dict:
    """某等高距下逐层提取等高线、逐条累加总长。"""
    n = h.shape[0]
    levels = levels_for(dh)
    total = 0.0
    n_straight = 0
    n_diagonal = 0
    n_chains = 0
    per_level = []
    chains_by_level = {}
    for lv in levels:
        r = contour_level(h, float(lv))
        total += r["length"]
        n_straight += r["n_straight"]
        n_diagonal += r["n_diagonal"]
        chains = trace_chains(r["edges"], n)
        n_chains += len(chains)
        chains_by_level[round(float(lv), 6)] = chains
        per_level.append({
            "level": round(float(lv), 6),
            "length": round(r["length"], 4),
            "n_chains": len(chains),
            "n_straight": r["n_straight"],
            "n_diagonal": r["n_diagonal"],
        })
    return {
        "dh": dh,
        "n_levels": len(levels),
        "total_length": total,
        "n_straight": n_straight,
        "n_diagonal": n_diagonal,
        "n_chains_total": n_chains,
        "per_level": per_level,
        "_chains": chains_by_level,
    }


# ---------------------------------------------------------------- 解析自检
def analytic_self_test() -> dict:
    """斜面与圆锥：coarea 给出可手算的期望值。"""
    n = 129
    yy, xx = np.mgrid[0:n, 0:n]
    plane = xx / (n - 1)
    sp = slope_euclid(plane)
    mp = measure_contours(plane, 0.1)
    rmax = (n - 1) / 2
    r = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
    cone = np.clip(1.0 - r / rmax, 0.0, 1.0)
    sc = slope_euclid(cone)
    mc = measure_contours(cone, 0.1)
    tvp = digital_tv(plane)
    tvc = digital_tv(cone)
    return {
        "plane": {
            "mean_slope": sp["mean"],
            "expected_slope": 1.0 / (n - 1),
            "mean_level_len": mp["total_length"] / mp["n_levels"],
            "expected_level_len": n - 1,
            "digital_tv": tvp["tv_digital"],
            "expected_tv": float(n - 1),
        },
        "cone": {
            "mean_slope": sc["mean"],
            "expected_slope_continuum": 1.0 / rmax,
            "measured_total_dh0.1": mc["total_length"],
            "digital_tv": tvc["tv_digital"],
            "note": "锥顶与裁剪边界存在离散化误差，连续期望仅作量级参照",
        },
    }


# ---------------------------------------------------------------- 主流程
def analyze(h: np.ndarray, tag: str, dhs: list[float]) -> dict:
    n = h.shape[0]
    area = n * n
    se = slope_euclid(h)
    tvd = digital_tv(h)
    slope_digital = tvd["tv_digital"] / area
    entry = {
        "tag": tag,
        "n": n,
        "area_px2": area,
        "h_min": float(h.min()),
        "h_max": float(h.max()),
        "slope_euclid": se,
        "slope_digital_mean": slope_digital,
        "tv_digital": tvd,
        "contours": [],
    }
    for dh in dhs:
        m = measure_contours(h, dh)
        m.pop("_chains", None)
        pred_digital = area * slope_digital / dh
        pred_euclid = area * se["mean"] / dh
        pred_l1 = area * (se["tv_l1"] / area) / dh
        m["pred_digital"] = pred_digital
        m["pred_euclid"] = pred_euclid
        m["pred_l1"] = pred_l1
        m["dev_digital"] = (m["total_length"] - pred_digital) / pred_digital
        m["dev_euclid"] = (m["total_length"] - pred_euclid) / pred_euclid
        m["dev_l1"] = (m["total_length"] - pred_l1) / pred_l1
        entry["contours"].append(m)
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="artifacts/data")
    ap.add_argument("--out", default="artifacts/data/results.json")
    ap.add_argument("--dh", type=float, nargs="+", default=[0.10, 0.05])
    args = ap.parse_args()

    results = {"self_test": analytic_self_test(), "terrains": []}
    for tag, fname in [
        ("A_oct2", "terrain_A_oct2.bin"),
        ("B_oct8", "terrain_B_oct8.bin"),
    ]:
        h = load_heightmap(Path(args.data_dir) / fname)
        results["terrains"].append(analyze(h, tag, args.dh))

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results["self_test"], ensure_ascii=False, indent=2))
    for t in results["terrains"]:
        print(f"\n== {t['tag']}  E[slope]_euclid={t['slope_euclid']['mean']:.6f} "
              f"E[slope]_digital={t['slope_digital_mean']:.6f} ==")
        for c in t["contours"]:
            print(f"dh={c['dh']:.2f} levels={c['n_levels']:2d} "
                  f"L={c['total_length']:10.2f} pred_dig={c['pred_digital']:10.2f} "
                  f"dev_dig={c['dev_digital']*100:+5.2f}% "
                  f"dev_euclid={c['dev_euclid']*100:+6.2f}% "
                  f"dev_l1={c['dev_l1']*100:+6.2f}%")


if __name__ == "__main__":
    main()
