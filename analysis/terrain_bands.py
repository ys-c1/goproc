"""分层设色预判与逐层扫描实测的核心库。

数学记号（详见 analysis/REPORT.md 的证明）：
  u(x,y)          归一化高度场，定义域 D
  C_h = {u = h}   高度 h 的等值线
  L(h) = ∮_{C_h} ds                 等值线欧氏总长
  Λ(h) = ∮_{C_h} ds/|∇u|            斜率加权等值线总长（共面积因子）
  A(a,b) = area{a ≤ u < b}          色带面积
coarea 恒等式： A'(h) = Λ(h)，故 A(a,b)=∫_a^b Λ(h) dh。
仅当 |∇u|≡1 时 Λ=L，题目“导数=等值线总长”的朴素形式才逐字成立。
离散实现把网格三角剖分为线性三角面；每块面上等值线长度关于 h 分段
线性，段积分有闭式解（analytic_band_integrals），机器精度闭合。
"""

from __future__ import annotations

import numpy as np


def load_heightmap(path: str) -> np.ndarray:
    """读取 cmd/dumpheight 的二进制：int64 N 后接 N*N 个小端 float64。"""
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(8), dtype="<i8")[0])
        data = np.frombuffer(f.read(), dtype="<f8")
    if data.size != n * n:
        raise ValueError(f"尺寸 {n} 与数据量 {data.size} 不符")
    return data.reshape(n, n).astype(np.float64, copy=True)


def triangulate(z: np.ndarray) -> dict:
    """把 N×N 格点三角剖分为 2(N-1)^2 个直角等腰三角面。

    每块三角面记录三个顶点高度 h0,h1,h2、面积 w、梯度模 g、
    以及用于等值线求交的预算系数。顶点 (x,y) 位于整数格点，格距=1。
    """
    n = z.shape[0]
    # 每个单元两个三角面：
    #   T0: (i,j),(i+1,j),(i,j+1)      直角在 (i,j)
    #   T1: (i+1,j+1),(i,j+1),(i+1,j)  直角在 (i+1,j+1)
    jj, ii = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="xy")
    i = ii.ravel()
    j = jj.ravel()
    v00 = z[j, i]
    v10 = z[j, i + 1]
    v01 = z[j + 1, i]
    v11 = z[j + 1, i + 1]

    # T0、T1 只需顶点高度；坐标为整数格点、格距 1，面积统一 1/2。
    tri_area = 0.5

    # T0 顶点高度
    a0 = np.stack([v00, v10, v01], axis=1)
    # T1 顶点高度
    a1 = np.stack([v11, v01, v10], axis=1)
    hv = np.concatenate([a0, a1], axis=0)  # (2M,3)

    # 梯度模：线性插值在同一单元内唯一，T0、T1 共享同一梯度。
    gx = v10 - v00
    gy = v01 - v00
    g_cell = np.hypot(gx, gy)
    g = np.concatenate([g_cell, g_cell])

    m = hv.shape[0]
    hmin = hv.min(axis=1)
    hmax = hv.max(axis=1)
    hmid = np.median(hv, axis=1)
    out = {
        "n": n,
        "tri_area": np.full(m, tri_area),
        "grad": g,
        "h": hv,
        "hmin": hmin,
        "hmax": hmax,
        "hmid": hmid,
    }
    # 等值线段长通用公式（对线性三角面严格成立，见 REPORT.md 附录）：
    #   孤立顶点高度 h_iso，另两顶点高度 d1、d2（均从 h_iso 起算），
    #   seg(h) = 2·g·w·|h-h_iso| / |d1·d2|
    # 低端段孤立点=最低顶点 hmin（另两点为 hmid,hmax），
    # 高端段孤立点=最高顶点 hmax（另两点为 hmin,hmid）。
    # 系数 c_lo、c_hi 即 seg(h)=c·(h-h_iso) 中的 c（段内 h-h_iso 保号）。
    denom_lo = np.abs((hmid - hmin) * (hmax - hmin))
    denom_hi = np.abs((hmin - hmax) * (hmid - hmax))
    gw2 = 2.0 * g * tri_area
    with np.errstate(divide="ignore", invalid="ignore"):
        c_lo = np.where(denom_lo > 0, gw2 / denom_lo, 0.0)
        c_hi = np.where(denom_hi > 0, gw2 / denom_hi, 0.0)
    # 高端段 seg=c_hi·(hmax-h)（从 hmax 处的 0 向下增长），
    # 在 (hmid,hmax) 内 hmax-h>0，levelset 与积分据此写符号。
    out["c_lo"] = c_lo
    out["c_hi"] = c_hi
    out["flat"] = g <= 1e-15
    return out


def levelset_functions(mesh: dict, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算各高度 h 处的 L(h) 与 Λ(h)，全向量化 marching triangles。

    每块三角面在 (hmin,hmid) 与 (hmid,hmax) 上各有一段线性等值线长度
    seg(h)=c|h-h_端点|。把所有三角面的两类“线性段”收集成区间事件，
    对查询高度用 searchsorted 统计覆盖区间并累加端点贡献。
    """
    hmin = mesh["hmin"]
    hmid = mesh["hmid"]
    hmax = mesh["hmax"]
    c_lo = mesh["c_lo"]
    c_hi = mesh["c_hi"]
    g = mesh["grad"]
    inv_g = np.zeros_like(g)
    nz = ~mesh["flat"]
    inv_g[nz] = 1.0 / g[nz]

    # 低端段 seg=c_lo·(h-hmin)，h∈(hmin,hmid]，零值点 hmin
    # 高端段 seg=c_hi·(hmax-h)，h∈[hmid,hmax)，零值点 hmax
    # 公共折点 hmid 处两段值相等，统一按闭区间 (a,b] 计数，点值正确。
    seg_lo = {
        "a": hmin, "b": hmid,
        "ch": c_lo, "const": -c_lo * hmin,
        "wl": c_lo * inv_g, "cl": -c_lo * hmin * inv_g,
    }
    seg_hi = {
        "a": hmid, "b": hmax,
        "ch": -c_hi, "const": c_hi * hmax,
        "wl": -c_hi * inv_g, "cl": c_hi * hmax * inv_g,
    }

    lvl = np.asarray(levels, dtype=np.float64)
    l_total = np.zeros_like(lvl)
    lam_total = np.zeros_like(lvl)
    for seg in (seg_lo, seg_hi):
        a0, b0 = seg["a"], seg["b"]
        # 零宽段（a==b）两端点段长均为 0，排除不影响另一段
        keep = b0 > a0
        a, b = a0[keep], b0[keep]
        ch, const = seg["ch"][keep], seg["const"][keep]
        wl, cl = seg["wl"][keep], seg["cl"][keep]
        order_a, order_b = np.argsort(a), np.argsort(b)
        sa, sb = np.sort(a), np.sort(b)
        cnt_lo = np.searchsorted(sa, lvl, side="right")
        cnt_hi = np.searchsorted(sb, lvl, side="left")
        sumch_a = np.concatenate([[0.0], np.cumsum(ch[order_a])])
        sumch_b = np.concatenate([[0.0], np.cumsum(ch[order_b])])
        sumc_a = np.concatenate([[0.0], np.cumsum(const[order_a])])
        sumc_b = np.concatenate([[0.0], np.cumsum(const[order_b])])
        sumwl_a = np.concatenate([[0.0], np.cumsum(wl[order_a])])
        sumwl_b = np.concatenate([[0.0], np.cumsum(wl[order_b])])
        sumcl_a = np.concatenate([[0.0], np.cumsum(cl[order_a])])
        sumcl_b = np.concatenate([[0.0], np.cumsum(cl[order_b])])
        if (cnt_lo - cnt_hi < 0).any():
            raise RuntimeError("区间计数出现负值")
        l_total += lvl * (sumch_a[cnt_lo] - sumch_b[cnt_hi])
        l_total += sumc_a[cnt_lo] - sumc_b[cnt_hi]
        lam_total += lvl * (sumwl_a[cnt_lo] - sumwl_b[cnt_hi])
        lam_total += sumcl_a[cnt_lo] - sumcl_b[cnt_hi]
    l_total = np.maximum(l_total, 0.0)
    lam_total = np.maximum(lam_total, 0.0)
    return l_total, lam_total


def analytic_triangle_integrals(mesh: dict) -> dict:
    """解析真值：每块线性三角面上 ∫Λ dh=面积、∫L dh=g·面积（全局求和）。

    单三角面 Λ 积分恒等于其面积，故全域 ∫_0^1 Λ dh 必须等于三角剖分
    总面积 2(N-1)^2·(1/2)=(N-1)^2，与采样积分互相校验。
    """
    area = mesh["tri_area"]
    return {
        "total_area_Lambda": float(area.sum()),
        "total_integral_L": float((mesh["grad"] * area).sum()),
        "n_triangles": int(area.size),
    }


def _linear_segment_integral(c, lo, hi, zero_at, sign):
    """线性段在每个 [lo,hi] 上的积分，全向量化（lo、hi、c 形状相同）。

    sign=+1：seg=c·(h-zero_at)；sign=-1：seg=c·(zero_at-h)。
    输入已保证 hi>lo；不满足的位置由调用方置零。
    """
    if sign > 0:
        return c * ((hi * hi - lo * lo) / 2 - zero_at * (hi - lo))
    return c * (zero_at * (hi - lo) - (hi * hi - lo * lo) / 2)


def analytic_band_integrals(mesh: dict, edges: np.ndarray) -> dict:
    """逐三角面分段解析积分（向量化），无重复计数。

    每块三角面的等值线长度在 (hmin,hmid)、(hmid,hmax) 上各为线性段，
    Λ 段为 seg/g。对被色带边切出的子区间直接积分线性函数再求和；
    线性段积分对区间端点不敏感（单点测度为零），公共折点不会重复计数。
    """
    hmin = mesh["hmin"][:, None]
    hmid = mesh["hmid"][:, None]
    hmax = mesh["hmax"][:, None]
    c_lo = mesh["c_lo"][:, None]
    c_hi = mesh["c_hi"][:, None]
    g = mesh["grad"][:, None]
    flat = mesh["flat"][:, None]

    a = edges[:-1][None, :]
    b = edges[1:][None, :]

    # 低端段 [hmin,hmid]，seg=c_lo·(h-hmin)
    lo_lo = np.maximum(a, hmin)
    hi_lo = np.minimum(b, hmid)
    valid_lo = (~flat) & (hi_lo > lo_lo)
    il = np.where(
        valid_lo,
        _linear_segment_integral(c_lo, lo_lo, hi_lo, hmin, +1),
        0.0,
    )
    # 高端段 [hmid,hmax]，seg=c_hi·(hmax-h)
    lo_hi = np.maximum(a, hmid)
    hi_hi = np.minimum(b, hmax)
    valid_hi = (~flat) & (hi_hi > lo_hi)
    ih = np.where(
        valid_hi,
        _linear_segment_integral(c_hi, lo_hi, hi_hi, hmax, -1),
        0.0,
    )

    lint_band = (il + ih).sum(axis=0)             # ∫L
    area_band = ((il + ih) / np.where(flat, 1.0, g)).sum(axis=0)  # ∫Λ
    mean_l = lint_band / (edges[1:] - edges[:-1])
    edge_l, edge_lam = levelset_functions(mesh, edges)
    return {
        "pred_area": area_band,
        "integral_L": lint_band,
        "mean_L": mean_l,
        "edge_L": edge_l,
        "edge_Lambda": edge_lam,
    }


def _fraction_belane(mesh: dict, h: float) -> np.ndarray:
    """每块三角面中 {u < h} 的面积占比（全向量化）。

    d_i = h - u_i，d_i>0 表示顶点在阈值之下（位于 {u<h} 一侧）。
    恰一个顶点在内部时，截得小三角形面积比为两条边截比之积：
      F = d_i^2 / ((d_i-d_j)(d_i-d_k))
    恰两个顶点在内部时取补集。
    """
    hv = mesh["h"]
    d = h - hv  # (n_tri, 3)
    # 恰在阈值上的顶点（d=0）计入 {u<h} 一侧；占比对边界测度不敏感，
    # 但计数归类必须与参考实现一致，否则 d=0 的面会被误判。
    inside = d >= 0
    n_in = inside.sum(axis=1)
    frac = np.zeros(hv.shape[0])
    frac[n_in == 3] = 1.0

    def one_inside(row_index):
        # 内部顶点 i，另两点 j、k
        dd = d[row_index]
        idx = np.argmax(inside[row_index], axis=1)
        rows = np.arange(dd.shape[0])
        di = dd[rows, idx]
        dj = dd[rows, (idx + 1) % 3]
        dk = dd[rows, (idx + 2) % 3]
        return di ** 2 / ((di - dj) * (di - dk))

    one = n_in == 1
    if one.any():
        frac[one] = one_inside(np.where(one)[0])
    two = n_in == 2
    if two.any():
        # 外部顶点 k，对 -d 套用同一公式得到补集占比
        row_index = np.where(two)[0]
        dneg = -d[row_index]
        outside = dneg > 0  # 原 d<0 的唯一顶点（d=0 已归内部侧）
        idx = np.argmax(outside, axis=1)
        rows = np.arange(dneg.shape[0])
        dk = dneg[rows, idx]
        di = dneg[rows, (idx + 1) % 3]
        dj = dneg[rows, (idx + 2) % 3]
        complement = dk ** 2 / ((dk - di) * (dk - dj))
        frac[two] = 1.0 - complement

    return np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0)


def exact_clipped_band_areas(mesh: dict, edges: np.ndarray) -> np.ndarray:
    """三角面解析裁切的色带面积（连续真值，非格点计数）。"""
    area = mesh["tri_area"]
    k = edges.size - 1
    out = np.zeros(k)
    for b in range(k):
        f_hi = _fraction_belane(mesh, edges[b + 1])
        f_lo = _fraction_belane(mesh, edges[b])
        out[b] = float(np.sum(area * (f_hi - f_lo)))
    return out


def _label_connected4(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """纯 NumPy 四邻域连通标记（并查集），返回 (标签图, 连通域数)。

    对角接触不算连通。两遍扫描：第一遍赋临时标签并合并，第二拍压平。
    """
    parent: list[int] = []

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    labels = np.zeros(mask.shape, dtype=np.int64)
    next_label = 1
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            up = labels[r - 1, c] if r > 0 else 0
            left = labels[r, c - 1] if c > 0 else 0
            neigh = [x for x in (up, left) if x > 0]
            if not neigh:
                parent.append(len(parent))
                labels[r, c] = next_label
                next_label += 1
                continue
            roots = sorted({find(int(x) - 1) for x in neigh})
            m_lab = roots[0] + 1
            labels[r, c] = m_lab
            for x in roots[1:]:
                union(roots[0], x)

    # 压平并重新编号为 1..k
    remap: dict[int, int] = {}
    out = np.zeros_like(labels)
    for r in range(rows):
        for c in range(cols):
            if labels[r, c] == 0:
                continue
            root = find(int(labels[r, c]) - 1)
            if root not in remap:
                remap[root] = len(remap) + 1
            out[r, c] = remap[root]
    return out, len(remap)


def raster_scan(z: np.ndarray, edges: np.ndarray) -> dict:
    """逐层扫描实测：单位格点按高度归入色带，面积按格点数计。

    连通域按四邻域统计（对角接触不算连通），纯 NumPy 实现无外部依赖。
    """
    n = z.shape[0]
    band_id = np.searchsorted(edges, z, side="right") - 1
    band_id = np.clip(band_id, 0, edges.size - 2)

    k_bands = edges.size - 1
    cells = np.zeros(k_bands, dtype=np.int64)
    n_comp = np.zeros(k_bands, dtype=np.int64)
    largest = np.zeros(k_bands, dtype=np.int64)
    for b in range(k_bands):
        mask = band_id == b
        cells[b] = int(mask.sum())
        if cells[b] > 0:
            lab, cnt = _label_connected4(mask)
            n_comp[b] = int(cnt)
            if cnt > 0:
                sizes = np.bincount(lab.ravel())
                largest[b] = int(sizes[1:].max())
    return {
        "band_id": band_id,
        "cells": cells,
        "components": n_comp,
        "largest_component": largest,
    }
