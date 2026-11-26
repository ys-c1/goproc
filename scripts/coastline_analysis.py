#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海岸线幂律发散分析。

流程:
  1. 读取 Go 生成器导出的 float64 裸高度图;
  2. 二维径向功率谱估计谱斜率 beta2D, 并以行(列)一维谱直接估 beta1D 交叉验证;
  3. marching squares (线性亚格点插值, 鞍点按角点均值判定) 提取等高线;
  4. 规尺法在多档步长 epsilon 下丈量海岸线总长;
  5. 四档步长 (32,16,8,2 px, 跨度 16 倍 >= 一个数量级) 拟合 L = C * eps^(-mu);
  6. 用拟合指数外推/内插其余各档长度, 与实测逐档比对;
  7. 盒维数独立交叉验证;
  8. 存档 CSV/JSON/PNG。

仅依赖 numpy 与 PIL。运行: python3 -I coastline_analysis.py <data_dir> <out_dir>
"""
import json
import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 常量
LEVEL = 0.5                 # 海岸线等高线水位 (高度图归一化到 [0,1])
# 全部丈量步长 (px); 非整数步长沿弧长参数化, 与整数等价
RULER_ALL = [64.0, 32.0, 27.95, 16.0, 12.21, 8.0, 5.33, 4.0, 2.0, 1.0]
# 幂律拟合四档: 公比 12^(1/3)=2.289, 跨度 12x (1.08 个数量级),
# 且全部 >= 5.33px, 落在谱标度区 (波长 >=4px 的 fBm 有效倍频程) 内
FIT_STEPS = [64.0, 27.95, 12.21, 5.33]
PRED_STEPS = RULER_ALL                     # 逐档比对 (含拟合档回归残差与截止区外推检验)
ANCHOR_EPS = 8.0                           # 预判幅度常数的单一锚点档 (不在四档拟合点内)
MIN_CHAIN_LEN = 64.0                            # 丢弃短于该弧长的碎链
K_FIT = (3, 16)                                 # 谱拟合波数区间 (cycle/图幅): fBm 干净标度区
BOX_FIT = (4.0, 64.0)                          # 盒维数拟合步长区间 (避开 <4px 截止区)
DEV_FLAG = 0.10                                 # 偏离一成即标记

# marching squares 边: 0=底(BL->BR) 1=右(BR->TR) 2=顶(TR->TL) 3=左(TL->BL)
# 位序: bit0 BL, bit1 BR, bit2 TR, bit3 TL
SEG_TABLE = {
    0: [], 15: [],
    1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
    6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)], 9: [(0, 2)],
    11: [(1, 2)], 12: [(1, 3)], 13: [(0, 1)], 14: [(3, 0)],
    # 5 与 10 为鞍点歧义, 运行时按角点均值裁决
}


def load_f64(path, n=1024):
    f = np.fromfile(path, dtype="<f8")
    assert f.size == n * n, (path, f.size)
    return f.reshape(n, n)


# ---------------------------------------------------------------- 谱分析
def radial_psd(h):
    """加 Hann 窗的二维 FFT, 返回径向平均 PSD 与波数 k(cycle/图幅)。"""
    n = h.shape[0]
    win = np.hanning(n)
    hw = (h - h.mean()) * np.outer(win, win)
    F = np.fft.fftshift(np.fft.fft2(hw))
    psd = (F.real ** 2 + F.imag ** 2) / (n * n)
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int64)
    kmax = n // 2 - 1
    k = np.arange(1, kmax + 1)
    rmax = int(r.max()) + 1
    acc = np.zeros(rmax)
    cnt = np.zeros(rmax)
    np.add.at(acc, r.ravel(), psd.ravel())
    np.add.at(cnt, r.ravel(), 1.0)
    return k, acc[1:kmax + 1] / cnt[1:kmax + 1]


def line1d_psd(h):
    """行/列一维谱平均, 直接估计 beta1D。"""
    n = h.shape[0]
    win = np.hanning(n)

    def avg1d(arr2d):
        x = (arr2d - arr2d.mean(axis=1, keepdims=True)) * win
        F = np.fft.rfft(x, axis=1)
        p = (F.real ** 2 + F.imag ** 2) / n
        return p.mean(axis=0)

    prow = avg1d(h)
    pcol = avg1d(h.T)
    p = 0.5 * (prow + pcol)
    k = np.arange(p.size)
    return k, p


def loglog_slope(k, p, k_lo, k_hi):
    m = (k >= k_lo) & (k <= k_hi) & (p > 0)
    x = np.log10(k[m].astype(float))
    y = np.log10(p[m])
    b, a = np.polyfit(x, y, 1)          # y = b*x + a
    yhat = b * x + a
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, a, r2


# ---------------------------------------------------------------- marching squares
def extract_contours(f, level=LEVEL):
    n = f.shape[0]
    above = f > level
    # 有跨号的格子掩码
    diff = (above[:-1, :-1] != above[:-1, 1:]) | (above[:-1, :-1] != above[1:, :-1]) | \
           (above[:-1, 1:] != above[1:, 1:]) | (above[1:, :-1] != above[1:, 1:])
    cells = np.argwhere(diff)                  # (y, x)
    segs = []

    def edge_point(ix, iy, edge):
        # 角点: BL(ix,iy) BR(ix+1,iy) TR(ix+1,iy+1) TL(ix,iy+1)
        bl = f[iy, ix]; br = f[iy, ix + 1]
        tr = f[iy + 1, ix + 1]; tl = f[iy + 1, ix]
        if edge == 0:
            t = (level - bl) / (br - bl); return (ix + t, float(iy))
        if edge == 1:
            t = (level - br) / (tr - br); return (float(ix + 1), iy + t)
        if edge == 2:
            t = (level - tl) / (tr - tl); return (ix + t, float(iy + 1))
        t = (level - bl) / (tl - bl); return (float(ix), iy + t)

    for iy, ix in cells:
        bl = above[iy, ix]; br = above[iy, ix + 1]
        tr = above[iy + 1, ix + 1]; tl = above[iy + 1, ix]
        code = int(bl) | (int(br) << 1) | (int(tr) << 2) | (int(tl) << 3)
        edges = SEG_TABLE.get(code)
        if edges is None:
            meanv = (f[iy, ix] + f[iy, ix + 1] + f[iy + 1, ix + 1] + f[iy + 1, ix]) / 4
            if code == 5:        # BL, TR 在上
                edges = [(0, 1), (3, 2)] if meanv > level else [(0, 3), (1, 2)]
            else:                # code == 10: BR, TL 在上
                edges = [(0, 3), (1, 2)] if meanv > level else [(0, 1), (3, 2)]
        for e0, e1 in edges:
            segs.append((edge_point(ix, iy, e0), edge_point(ix, iy, e1)))
    return segs, n


def build_chains(segs, n):
    """把线段按端点拼成折线, 区分闭合环与开口链。"""
    key = lambda p: (round(p[0], 6), round(p[1], 6))
    adj = {}
    for i, (p0, p1) in enumerate(segs):
        adj.setdefault(key(p0), []).append(i)
        adj.setdefault(key(p1), []).append(i)
    used = [False] * len(segs)
    chains = []

    def other(sid, k):
        p0, p1 = segs[sid]
        return p1 if key(p0) == k else p0

    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        p0, p1 = segs[i]
        chain = [p0, p1]
        # 正向
        while True:
            cands = [s for s in adj[key(chain[-1])] if not used[s]]
            if not cands:
                break
            sid = cands[0]
            used[sid] = True
            chain.append(other(sid, key(chain[-1])))
        # 反向
        while True:
            cands = [s for s in adj[key(chain[0])] if not used[s]]
            if not cands:
                break
            sid = cands[0]
            used[sid] = True
            chain.insert(0, other(sid, key(chain[0])))
        closed = key(chain[0]) == key(chain[-1])
        if closed:
            chain = chain[:-1]
        chains.append((chain, closed))

    # 过滤碎链
    kept = []
    for chain, closed in chains:
        pts = np.array(chain)
        seg_len = np.sqrt(np.diff(pts, axis=0) ** 2).sum(axis=1)
        S = seg_len.sum() + (0.0 if closed else 0.0)
        if closed:
            S = np.sqrt(((pts - np.roll(pts, -1, axis=0)) ** 2).sum(axis=1)).sum()
        if S >= MIN_CHAIN_LEN and len(chain) >= 8:
            kept.append((chain, closed))
    return merge_open(kept, n)


def merge_open(chains, n):
    """把端点邻近且都在图内的开口链接起来 (域边界上的端点不跨接)。"""
    open_chains = [list(c) for c, cl in chains if not cl]
    closed = [c for c, cl in chains if cl]
    interior = lambda p: 0.5 < p[0] < n - 0.5 and 0.5 < p[1] < n - 0.5
    changed = True
    while changed:
        changed = False
        for i in range(len(open_chains)):
            for j in range(i + 1, len(open_chains)):
                a, b = open_chains[i], open_chains[j]
                pairs = [
                    (a[-1], b[0], a + b[1:]),
                    (a[-1], b[-1], a + b[::-1][1:]),
                    (a[0], b[0], b + a[1:]),
                    (a[0], b[-1], b[::-1] + a[1:]),
                ]
                for pa, pb, merged in pairs:
                    if interior(pa) and interior(pb) and math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= 1.6:
                        open_chains[i] = merged
                        del open_chains[j]
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return [(c, True) for c in closed] + [(c, False) for c in open_chains]


# ---------------------------------------------------------------- 规尺法
def _point_at(pts, cum, s):
    j = int(np.searchsorted(cum, s, side="right"))
    j = min(max(j, 1), len(pts) - 1)
    span = cum[j] - cum[j - 1]
    t = 0.0 if span == 0 else (s - cum[j - 1]) / span
    return pts[j - 1] + t * (pts[j] - pts[j - 1])


def ruler_walk(chain, closed, eps):
    """经典圆规/分度规尺法 (compass/divider):
    从端点起, 每步沿折线走到与当前量点欧氏距离恰为 eps 的点。
    实现: 弧长参数化 + 在 [s_prev+eps, S] 上二分; 下界 s_prev+eps 保证沿弧长前进
    (弦距 <= 弧长, 故真正交点的弧长必 >= s_prev+eps), 天然避开 u=0 与折返。
    L = N*eps + 末端残余弦。返回 (L, 量点序列)。"""
    pts = np.array(chain, dtype=float)
    if closed:
        pts = np.vstack([pts, pts[0]])
    seg = np.sqrt(np.diff(pts, axis=0) ** 2).sum(axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    S = float(cum[-1])
    if S <= eps:
        return float(np.linalg.norm(pts[-1] - pts[0])), pts[[0, -1]]

    q = [pts[0].copy()]
    s_prev = 0.0
    pivot = pts[0].copy()
    count = 0
    max_steps = int(S / max(eps, 1e-9)) + 4
    while count < max_steps:
        lo = min(s_prev + eps, S)          # 弦距<=弧长 -> 交点弧长不小于此
        if lo >= S:
            break                            # 剩余弧长不足一个步长
        # 开口链尾段: 若到终点直线距已不足 eps, 作为尾弦收尾
        if not closed and float(np.linalg.norm(_point_at(pts, cum, S) - pivot)) < eps:
            break
        hi = S
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            d = float(np.linalg.norm(_point_at(pts, cum, mid) - pivot))
            if d < eps:
                lo = mid
            else:
                hi = mid
        s_new = 0.5 * (lo + hi)
        pivot = _point_at(pts, cum, s_new)
        q.append(pivot)
        count += 1
        s_prev = s_new
    q = np.array(q)
    if closed:
        tail = float(np.linalg.norm(pts[0] - q[-1]))
    else:
        tail = float(np.linalg.norm(pts[-1] - q[-1]))
    return float(count * eps + tail), q


def ruler_length(chain, closed, eps):
    return ruler_walk(chain, closed, eps)


def measure(chains, eps):
    total = 0.0
    walks = []
    for chain, closed in chains:
        L, q = ruler_length(chain, closed, eps)
        total += L
        walks.append(q)
    return total, walks


# ---------------------------------------------------------------- 盒维数
def box_counting(chains, sizes=(1, 2, 4, 8, 16, 32, 64)):
    pts = np.vstack([np.array(c, dtype=float) for c, _ in chains])
    out = []
    for eps in sizes:
        boxes = set()
        qx = (pts[:, 0] / eps).astype(int)
        qy = (pts[:, 1] / eps).astype(int)
        nbox = len(set(zip(qx.tolist(), qy.tolist())))
        out.append((float(eps), nbox))
    return out


# ---------------------------------------------------------------- 出图
FONT = ImageFont.load_default()


def plot_coastline(f, chains, walks32, path, tag):
    n = f.shape[0]
    base = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(base, "L").convert("RGB")
    dr = ImageDraw.Draw(img)
    for chain, _ in chains:
        p = [(x, y) for x, y in chain]
        if len(p) > 1:
            dr.line(p, fill=(255, 0, 0), width=1)
    # 右图: eps=32 规尺折线
    img2 = Image.fromarray(base, "L").convert("RGB")
    dr2 = ImageDraw.Draw(img2)
    for q in walks32:
        if len(q) > 1:
            dr2.line([(x, y) for x, y in q], fill=(0, 170, 255), width=1)
            dr2.ellipse([q[0, 0] - 2, q[0, 1] - 2, q[0, 0] + 2, q[0, 1] + 2], fill=(255, 255, 0))
    canvas = Image.new("RGB", (n * 2 + 20, n + 30), (255, 255, 255))
    canvas.paste(img, (0, 30)); canvas.paste(img2, (n + 20, 30))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), f"{tag}  left: coastline (level={LEVEL}, red)   right: ruler walk eps=32px (blue)", fill=(0, 0, 0), font=FONT)
    canvas.save(path)


def plot_fit(ruler_rows, mu, intercept, beta2d, mu_spec, path, tag):
    W, H = 900, 620
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    xs = np.log10(np.array([r[0] for r in ruler_rows]))
    ys = np.log10(np.array([r[1] for r in ruler_rows]))
    x0, x1 = xs.min() - 0.15, xs.max() + 0.15
    y0, y1 = ys.min() - 0.08, ys.max() + 0.08

    def PX(x): return 90 + (x - x0) / (x1 - x0) * (W - 160)
    def PY(y): return H - 80 - (y - y0) / (y1 - y0) * (H - 160)

    dr.rectangle([PX(x0), PY(y1), PX(x1), PY(y0)], outline=(0, 0, 0))
    for gx in np.arange(math.ceil(x0 * 2) / 2, x1, 0.5):
        dr.line([PX(gx), PY(y0), PX(gx), PY(y1)], fill=(230, 230, 230))
        dr.text((PX(gx) - 14, PY(y0) + 6), f"{gx:.1f}", fill=(0, 0, 0), font=FONT)
    for gy in np.arange(math.ceil(y0 * 4) / 4, y1, 0.25):
        dr.line([PX(x0), PY(gy), PX(x1), PY(gy)], fill=(230, 230, 230))
        dr.text((PX(x0) - 52, PY(gy) - 5), f"{gy:.2f}", fill=(0, 0, 0), font=FONT)
    # 规尺拟合线 (仅在拟合区间); 负斜率须按数组顺序取端点
    lx = np.log10(np.array(FIT_STEPS))
    ly = intercept - mu * lx
    dr.line([(PX(lx[0]), PY(ly[0])), (PX(lx[-1]), PY(ly[-1]))], fill=(255, 0, 0), width=2)
    # 谱指数预判线 (过锚点, 绿色虚线效果用短段模拟)
    L_anchor = dict(ruler_rows)[ANCHOR_EPS]
    C = L_anchor * ANCHOR_EPS ** mu_spec
    px = np.log10(np.array([r[0] for r in ruler_rows]))
    py = np.log10(C) - mu_spec * px
    dr.line([(PX(px.min()), PY(py[0])), (PX(px.max()), PY(py[-1]))], fill=(0, 150, 0), width=2)
    fit_set = set(FIT_STEPS)
    for r, x, y in zip(ruler_rows, xs, ys):
        if r[0] == ANCHOR_EPS:
            col = (255, 140, 0)      # 锚点: 橙
        elif r[0] in fit_set:
            col = (200, 0, 0)       # 拟合档: 红
        else:
            col = (0, 0, 220)      # 检验档: 蓝
        dr.ellipse([PX(x) - 5, PY(y) - 5, PX(x) + 5, PY(y) + 5], fill=col)
        dr.text((PX(x) + 8, PY(y) - 16), f"eps={r[0]:g}", fill=col, font=FONT)
    dr.text((90, 12), f"{tag}  log10 L vs log10 eps   red line ruler-fit mu_ruler={mu:.4f} (D={1+mu:.4f}); "
                      f"green line spectrum-pred mu_spec={mu_spec:.4f}", fill=(0, 0, 0), font=FONT)
    dr.text((90, 30), f"red=4 fit steps (64/27.95/12.21/5.33, span 12x), orange=anchor eps={ANCHOR_EPS:g}, "
                      f"blue=check steps; beta2D={beta2d:.3f}", fill=(80, 80, 80), font=FONT)
    dr.text((W // 2 - 40, H - 30), "log10(eps / px)", fill=(0, 0, 0), font=FONT)
    dr.text((12, H // 2), "log10(L / px)", fill=(0, 0, 0), font=FONT)
    img.save(path)


def plot_spectra(specs, path):
    W, H = 980, 640
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    allx = np.concatenate([np.log10(k[1:]) for k, _ in specs])
    ally = np.concatenate([np.log10(p[1:]) for _, p in specs])
    x0, x1 = allx.min(), allx.max()
    y0, y1 = ally.min(), ally.max() + 0.3

    def PX(x): return 90 + (x - x0) / (x1 - x0) * (W - 160)
    def PY(y): return H - 80 - (y - y0) / (y1 - y0) * (H - 170)

    dr.rectangle([PX(x0), PY(y1), PX(x1), PY(y0)], outline=(0, 0, 0))
    colors = [(200, 0, 0), (0, 110, 0), (0, 0, 220)]
    for (k, p), (tag, b2d), col in zip(specs, specs_meta, colors):
        m = (k >= K_FIT[0]) & (k <= K_FIT[1])
        pts = [(PX(x), PY(y)) for x, y in zip(np.log10(k[1:]), np.log10(p[1:]))]
        dr.line(pts, fill=col, width=1)
        lx = np.log10(k[m].astype(float))
        ly = np.polyval(np.polyfit(lx, np.log10(p[m]), 1), lx)
        # 负斜率: 必须按数组顺序取两端, 不能用 min/max 配对 (否则画出上升线)
        dr.line([(PX(lx[0]), PY(ly[0])), (PX(lx[-1]), PY(ly[-1]))], fill=col, width=3)
        dr.text((PX(x1) - 250, PY(y1) + 18 + colors.index(col) * 16),
                f"{tag}: beta2D={b2d:.3f}", fill=col, font=FONT)
    dr.text((90, 12), f"radial-mean 2D power spectral density (Hann-windowed FFT), thick = fit k in {list(K_FIT)}",
            fill=(0, 0, 0), font=FONT)
    dr.text((W // 2 - 60, H - 30), "log10(k / cycles per map)", fill=(0, 0, 0), font=FONT)
    dr.text((12, H // 2), "log10 PSD", fill=(0, 0, 0), font=FONT)
    img.save(path)


# ---------------------------------------------------------------- 主流程
def analyze(tag, f64_path, out_dir, gain):
    f = load_f64(f64_path)
    n = f.shape[0]
    res = {"tag": tag, "n": n, "level": LEVEL, "sea_fraction": float((f < LEVEL).mean()),
           "gain": gain, "H_theory": -math.log2(gain),
           "mu_theory": 1.0 + math.log2(gain), "D_theory": 2.0 + math.log2(gain)}

    # 谱 (b2d, b1d 为带符号对数斜率, 约定 PSD ~ k^b)
    k2, p2 = radial_psd(f)
    b2d, a2d, r2b = loglog_slope(k2, p2, K_FIT[0], K_FIT[1])
    k1, p1 = line1d_psd(f)
    b1d_fit, _, r2_1 = loglog_slope(k1, p1, K_FIT[0], K_FIT[1])
    b1d_from2d = b2d + 1.0                       # 切片定理: b1D = b2D + 1
    H_from_spec = -0.5 * b2d - 1.0               # b2D = -(2H+2)
    # mu = 1-H = 2 + b2D/2 = (3+b1D)/2 ;  D = 2-H = 3 + b2D/2 = (5+b1D)/2
    res["spectrum"] = {
        "k_fit": list(K_FIT), "beta2D": float(b2d), "beta2D_r2": float(r2b),
        "beta1D_direct": float(b1d_fit), "beta1D_from2D": float(b1d_from2d), "beta1D_r2": float(r2_1),
        "H_from_spectrum": float(H_from_spec),
        "D_spec": float(3.0 + b2d / 2.0),
        "mu_spec": float(2.0 + b2d / 2.0),
        "D_spec_from_b1D": float((5.0 + b1d_fit) / 2.0),
        "mu_spec_from_b1D": float((3.0 + b1d_fit) / 2.0),
    }

    # 海岸线
    segs, _ = extract_contours(f)
    chains = build_chains(segs, n)
    n_closed = sum(1 for _, cl in chains if cl)
    n_open = sum(1 for _, cl in chains if not cl)
    total_nodes = sum(len(c) for c, _ in chains)
    res["contour"] = {"segments": len(segs), "chains_closed": n_closed,
                      "chains_open": n_open, "nodes": total_nodes}

    # 规尺丈量
    rows = []
    walks_keep = None
    for eps in RULER_ALL:
        L, walks = measure(chains, eps)
        rows.append((eps, L))
        if eps == 32.0:
            walks_keep = walks
    res["ruler_raw"] = [{"eps": e, "L": L} for e, L in rows]

    # 四档幂律拟合 (仅用于独立估计规尺指数, 与谱指数对照)
    fit = [(e, L) for e, L in rows if e in FIT_STEPS]
    X = np.log10(np.array([e for e, _ in fit]))
    Y = np.log10(np.array([L for _, L in fit]))
    slope, intercept = np.polyfit(X, Y, 1)
    mu_ruler = -slope
    Yhat = slope * X + intercept
    r2 = 1 - ((Y - Yhat) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    res["power_law"] = {"fit_steps": FIT_STEPS, "span_decade": math.log10(max(FIT_STEPS) / min(FIT_STEPS)),
                        "mu_ruler": float(mu_ruler), "D_ruler": float(1 + mu_ruler),
                        "intercept_log10C": float(intercept), "r2": float(r2)}

    # 预判: 指数完全取自谱 (mu_spec), 不使用规尺拟合指数, 也不生成新分辨率地形。
    # 幅度常数只需一个锚点丈量值标定 (取 ANCHOR_EPS, 不与四档拟合点重合)。
    mu_pred = float(res["spectrum"]["mu_spec"])
    L_anchor = dict(rows)[ANCHOR_EPS]
    C_anchor = L_anchor * ANCHOR_EPS ** mu_pred          # L(eps)=C*eps^-mu
    res["prediction_meta"] = {"mu_used": mu_pred, "mu_source": "spectrum: mu=2+beta2D/2",
                              "anchor_eps": ANCHOR_EPS, "anchor_L": float(L_anchor),
                              "C_anchor": float(C_anchor)}
    pred_rows = []
    for eps, Lm in rows:
        Lp = C_anchor * eps ** (-mu_pred)
        dev = (Lp - Lm) / Lm
        pred_rows.append({"eps": float(eps), "L_meas": float(Lm), "L_pred": float(Lp),
                          "dev": float(dev), "in_fit": bool(eps in FIT_STEPS),
                          "is_anchor": bool(eps == ANCHOR_EPS),
                          "flag": bool(abs(dev) > DEV_FLAG)})
    res["prediction"] = pred_rows

    # 盒维数 (只在标度区内拟合, 避开 <4px 的网格截止)
    bc = box_counting(chains)
    bx_all = np.array([e for e, _ in bc])
    by_all = np.array([c for _, c in bc], dtype=float)
    m = (bx_all >= BOX_FIT[0]) & (bx_all <= BOX_FIT[1])
    dbox, _ = np.polyfit(np.log10(bx_all[m]), np.log10(by_all[m]), 1)
    res["box_counting"] = {"eps_N": [{"eps": float(e), "N": int(c)} for e, c in bc],
                           "fit_eps": list(BOX_FIT), "D_box": float(-dbox)}

    # 存档
    with open(os.path.join(out_dir, "data", f"result_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    import csv
    with open(os.path.join(out_dir, "data", f"ruler_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["eps_px", "L_meas_px", "L_pred_px(spectrum mu)", "rel_dev",
                    "in_fit_steps", "is_anchor", "flag_gt10pct"])
        for r in pred_rows:
            w.writerow([r["eps"], f"{r['L_meas']:.6f}", f"{r['L_pred']:.6f}",
                        f"{r['dev']:.6f}", int(r["in_fit"]), int(r["is_anchor"]), int(r["flag"])])
    plot_coastline(f, chains, walks_keep, os.path.join(out_dir, "figures", f"coastline_{tag}.png"), tag)
    plot_fit(rows, mu_ruler, intercept, b2d, mu_pred,
             os.path.join(out_dir, "figures", f"fit_{tag}.png"), tag)
    return res, (k2, p2)


specs_meta = []

def main():
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    terrains = [("T1_g057", "t1_g057.f64", 0.57),
                ("T2_g060", "t2_g060.f64", 0.60),
                ("T3_g062", "t3_g062.f64", 0.62)]
    all_res, specs = [], []
    for tag, fn, gain in terrains:
        res, spec = analyze(tag, os.path.join(data_dir, fn), out_dir, gain)
        all_res.append(res)
        specs.append(spec)
        specs_meta.append((tag, res["spectrum"]["beta2D"]))
    plot_spectra(specs, os.path.join(out_dir, "figures", "spectra_summary.png"))
    with open(os.path.join(out_dir, "data", "results_all.json"), "w") as fh:
        json.dump(all_res, fh, indent=2, ensure_ascii=False)

    # 控制台摘要
    print("=" * 88)
    for r in all_res:
        s, pl, ct = r["spectrum"], r["power_law"], r["contour"]
        print(f"[{r['tag']}] gain={r['gain']} H_theory={r['H_theory']:.4f} "
              f"sea={r['sea_fraction']:.3f} chains(closed/open)={ct['chains_closed']}/{ct['chains_open']} "
              f"nodes={ct['nodes']}")
        print(f"  spectrum(k {s['k_fit']}): b2D={s['beta2D']:.4f}(r2={s['beta2D_r2']:.4f}) "
              f"b1D_direct={s['beta1D_direct']:.4f} b1D=b2D+1={s['beta1D_from2D']:.4f}  H_spec={s['H_from_spectrum']:.4f}")
        pm = r["prediction_meta"]
        print(f"  theory: mu={r['mu_theory']:.4f} D={r['D_theory']:.4f} | "
              f"from-spectrum(PREDICTION mu): mu={s['mu_spec']:.4f} D={s['D_spec']:.4f} | "
              f"ruler-fit: mu={pl['mu_ruler']:.4f} D={pl['D_ruler']:.4f}(r2={pl['r2']:.5f}) | "
              f"D_box={r['box_counting']['D_box']:.4f}")
        print(f"  prediction: mu_source={pm['mu_source']} anchor eps={pm['anchor_eps']:g} "
              f"L_anchor={pm['anchor_L']:.2f}")
        print(f"  {'eps':>6} {'L_meas':>12} {'L_pred':>12} {'dev%':>8} {'fit?':>5} {'anchor?':>7} flag")
        for row in r["prediction"]:
            print(f"  {row['eps']:>6g} {row['L_meas']:>12.3f} {row['L_pred']:>12.3f} "
                  f"{row['dev'] * 100:>7.2f}% {('Y' if row['in_fit'] else '-'):>5} "
                  f"{('Y' if row['is_anchor'] else '-'):>7} {('>10%' if row['flag'] else '')}")
    print("=" * 88)


if __name__ == "__main__":
    main()
