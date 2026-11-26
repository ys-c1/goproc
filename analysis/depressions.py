#!/usr/bin/env python3
# 运行方式：python3 -I analysis/depressions.py [--pgm ... --outdir ...]
"""洼地深度阈值预判 vs 填洼后逐格判定，独立可复现实验。

输入 goproc 导出的 8-bit PGM 灰度高度图（深度一律按灰度差，0..255）。

流水线：
  1. 枚举全部“四邻域严格局部极小点”（内部格，边界作为出口不参与）；
  2. 在原始灰度场上用多源 Dijkstra / Priority-Flood 求每格 minimax 溢流高程 S，
     极小 m 的洼地（夺顶/填充）深度 Δ(m)=S[m]-G[m]，单位为灰度级；
  3. 深度阈值 τ 下的限深填洼面 Fτ = min(S, G+τ)（每格最多抬 τ 个灰度），
     并由一个独立的标准 Priority-Flood 例程交叉验证 S；
  4. 预判 Pτ = {原极小 m : Δ(m)>τ}（Δ≤τ 的极小被溢流覆盖/夺顶而消除），
     无需迭代抬洼；
  5. 对 Fτ 重新逐格做四邻域严格极小扫描得 Qτ，与 Pτ 做一一最优配对，
     报告 dx/dy/切比雪夫偏差/欧氏偏差，切比雪夫 >1 格即标出；
  6. 落盘 JSON / CSV / Markdown 两行结论 + 每档一张洼地标注图。

数学结论见 docs/depression-proof.md：
     strictmin4(Fτ) == {m∈strictmin4(G) : Δ(m)>τ}，且坐标逐点不变。

运行：
  python3 -I analysis/depressions.py \
      --pgm artifacts/terrain_128_s7.pgm --outdir artifacts --t1 20 --t2 40
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Okabe-Ito 色盲安全色板（形状 + 颜色 + 编号三重编码，不依赖颜色 alone）
C_PRED = (213, 94, 0)      # 朱红：预判位置（空心圆）
C_MEAS = (0, 114, 178)     # 深蓝：填后实测位置（X）
C_COVERED = (86, 180, 233) # 天蓝：本档被溢流覆盖消除的极小
C_WARN = (230, 159, 0)     # 琥珀/黄：偏差 >1 格警示
C_INK = (28, 28, 28)
C_MUTED = (96, 96, 96)


# ---------------------------------------------------------------- PGM 读取
def read_pgm(path: str) -> np.ndarray:
    """解析 P5 二进制 PGM，容忍 '#' 注释与任意空白。"""
    with open(path, "rb") as f:
        buf = f.read()
    pos = 0

    def token() -> bytes:
        nonlocal pos
        while True:
            while pos < len(buf) and buf[pos:pos + 1].isspace():
                pos += 1
            if pos < len(buf) and buf[pos] == ord("#"):
                while pos < len(buf) and buf[pos] not in b"\r\n":
                    pos += 1
                continue
            start = pos
            while pos < len(buf) and not buf[pos:pos + 1].isspace():
                pos += 1
            return buf[start:pos]

    magic = token()
    if magic not in (b"P5", b"P2"):
        raise ValueError(f"只支持 P5/P2 PGM，得到 {magic!r}")
    w = int(token()); h = int(token()); maxv = int(token())
    # 单个空白分隔符
    if pos < len(buf) and buf[pos:pos + 1].isspace():
        pos += 1
    if magic == b"P5":
        if maxv >= 256:
            raise ValueError("本脚本只处理 8-bit PGM")
        data = np.frombuffer(buf[pos:pos + w * h], dtype=np.uint8)
        g = data.reshape(h, w).astype(np.int64)
    else:
        vals = [int(token()) for _ in range(w * h)]
        g = np.array(vals, dtype=np.int64).reshape(h, w)
    return g


# ---------------------------------------------------------------- 极小点
def minima4_scalar(g: np.ndarray):
    """权威实现：纯 Python 逐格四邻域严格局部极小（不含边界）。

    严格定义：1<=x<N-1,1<=y<N-1 且 g[y,x] 严格小于四个正交邻格。
    返回 [(x,y), ...]，按 (y,x) 排序。
    """
    n = g.shape[0]
    out = []
    for y in range(1, n - 1):
        row = g[y]
        up = g[y - 1]; dn = g[y + 1]
        for x in range(1, n - 1):
            v = row[x]
            if v < up[x] and v < dn[x] and v < row[x - 1] and v < row[x + 1]:
                out.append((x, y))
    return out


def minima4_numpy(g: np.ndarray):
    """向量化版本，仅用于与权威实现交叉断言。坐标 (x,y)。"""
    c = g[1:-1, 1:-1]
    m = ((c < g[:-2, 1:-1]) & (c < g[2:, 1:-1])
         & (c < g[1:-1, :-2]) & (c < g[1:-1, 2:]))
    ys, xs = np.nonzero(m)
    # m 的 (i,j) 对应全局格 (x=j+1, y=i+1)
    return [(int(x) + 1, int(y) + 1) for y, x in zip(ys, xs)]


# ---------------------------------------------------------------- 溢流面
def minimax_spill(g: np.ndarray, conn: int = 4) -> np.ndarray:
    """多源 Dijkstra 求每格到边界路径上最大灰度的最小值 S。

    S[y,x] = min_{p: (y,x)->边界} max_{c in p} G[c]。
    边界格以自身灰度为源（水到边即出画幅）。conn=4 或 8 决定连通性。
    """
    n = g.shape[0]
    inf = 10 ** 18
    s = np.full((n, n), inf, dtype=np.int64)
    pq = []
    for x in range(n):
        for y in (0, n - 1):
            s[y, x] = g[y, x]
            heapq.heappush(pq, (int(g[y, x]), y, x))
    for y in range(n):
        for x in (0, n - 1):
            if g[y, x] < s[y, x]:
                s[y, x] = g[y, x]
                heapq.heappush(pq, (int(g[y, x]), y, x))
    dirs4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    dirs8 = dirs4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    dirs = dirs4 if conn == 4 else dirs8
    while pq:
        lev, y, x = heapq.heappop(pq)
        if lev != s[y, x]:
            continue
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < n:
                nl = max(lev, int(g[ny, nx]))
                if nl < s[ny, nx]:
                    s[ny, nx] = nl
                    heapq.heappush(pq, (nl, ny, nx))
    return s


def priority_flood(g: np.ndarray) -> np.ndarray:
    """独立的标准 Priority-Flood（Barnes 式，无 epsilon），直接给完整填洼面。

    与 minimax_spill 是同一势函数的两种写法，用来交叉验证。
    """
    n = g.shape[0]
    f = np.full((n, n), -1, dtype=np.int64)
    pq = []
    for x in range(n):
        for y in (0, n - 1):
            f[y, x] = g[y, x]
            heapq.heappush(pq, (int(g[y, x]), y, x))
    for y in range(n):
        for x in (0, n - 1):
            if f[y, x] < 0:
                f[y, x] = g[y, x]
                heapq.heappush(pq, (int(g[y, x]), y, x))
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while pq:
        lev, y, x = heapq.heappop(pq)
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < n and f[ny, nx] < 0:
                nl = max(int(g[ny, nx]), lev)
                f[ny, nx] = nl
                heapq.heappush(pq, (nl, ny, nx))
    return f


# ---------------------------------------------------------------- 配对
def hungarian(cost: np.ndarray):
    """极小方阵的 O(n^3) Hungarian，返回 (rows, cols) 最优配对。"""
    n = cost.shape[0]
    u = [0] * (n + 1); v = [0] * (n + 1)
    p = [0] * (n + 1); way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf"); j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = []
    for j in range(1, n + 1):
        if p[j] != 0:
            pairs.append((p[j] - 1, j - 1))
    return [a for a, _ in pairs], [b for _, b in pairs]


def match_positions(pred, meas):
    """把预判点与填后实测点一一配对。

    先锁精确同坐标；剩余点用 Hungarian 最小化平方欧氏距离。
    返回 [{pred,meas,dx,dy,cheb,euclid,over1}, ...]（按预判顺序）。
    """
    pred = list(pred); meas = list(meas)
    used_m = set()
    rows = []
    leftovers_p = []
    for p in pred:
        if p in meas and meas.index(p) not in used_m:
            j = meas.index(p); used_m.add(j)
            rows.append(_pair_row(p, meas[j]))
        else:
            leftovers_p.append(p)
    left_m_idx = [j for j in range(len(meas)) if j not in used_m]
    if leftovers_p and left_m_idx:
        m = len(leftovers_p); k = len(left_m_idx)
        dim = max(m, k)
        cost = np.full((dim, dim), 1e9, dtype=float)
        for i, p in enumerate(leftovers_p[:m]):
            for jj, j in enumerate(left_m_idx[:k]):
                q = meas[j]
                cost[i, jj] = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
        ri, ci = hungarian(cost)
        for i, jj in zip(ri, ci):
            if i < m and jj < k:
                rows.append(_pair_row(leftovers_p[i], meas[left_m_idx[jj]]))
    elif leftovers_p:
        for p in leftovers_p:
            rows.append({"pred": p, "meas": None, "dx": None, "dy": None,
                         "cheb": None, "euclid": None, "over1": True})
    rows.sort(key=lambda r: (r["pred"][1], r["pred"][0]))
    return rows


def _pair_row(p, q):
    dx = q[0] - p[0]; dy = q[1] - p[1]
    cheb = max(abs(dx), abs(dy))
    eu = math.hypot(dx, dy)
    return {"pred": p, "meas": q, "dx": dx, "dy": dy,
            "cheb": cheb, "euclid": round(eu, 3), "over1": cheb > 1}


# ---------------------------------------------------------------- 成图
def load_font(size: int):
    candidates = [
        os.path.expanduser("~/.local/share/fonts/NotoSansCJKsc-Regular.ttf"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def hillshade(g: np.ndarray, zfactor: float = 3.0) -> np.ndarray:
    z = g.astype(float) * zfactor
    gy, gx = np.gradient(z)
    # 光源西北 315° 方位、45° 高度
    az = math.radians(315); alt = math.radians(45)
    # 坡度/坡向标准 hillshade
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    sh = (np.sin(alt) * np.sin(slope)
          + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    sh = np.clip(sh, 0, 1)
    return (60 + 195 * sh)  # 60..255


def render_map(path, g, f, tau, records, covered, flagged):
    """records: 已配对且确认剩余的点记录（含 depth/index）。"""
    n = g.shape[0]
    S = 6
    map_w = n * S
    side_w = 350
    top_h = 104
    legend_h = 52
    W = map_w + side_w
    H = top_h + map_w + legend_h

    hs = hillshade(g)
    base = (0.62 * g + 0.38 * hs).clip(0, 255).astype(np.uint8)
    rgb = np.repeat(base[:, :, None], 3, axis=2).astype(float)

    # 限深填洼抬升区：蓝色强度随抬升量
    raise_amt = (f - g).astype(float)
    inten = np.clip(raise_amt / max(float(tau), 1.0), 0, 1) * 0.30
    blue = np.array(C_COVERED, dtype=float)
    for k in range(3):
        rgb[:, :, k] = rgb[:, :, k] * (1 - inten) + blue[k] * inten
    img = Image.fromarray(rgb.astype(np.uint8), "RGB")
    img = img.resize((map_w, map_w), Image.NEAREST).convert("RGBA")

    # 半透明覆盖层用于 marker 之外的精细绘制
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    def cell_center(p):
        x, y = p
        return (x * S + S / 2.0, y * S + S / 2.0)

    f_title = load_font(30); f_sub = load_font(19)
    f_num = load_font(20); f_side_h = load_font(24); f_side = load_font(19)
    f_small = load_font(16); f_leg = load_font(18)

    # 坐标刻度（每 16 格）
    for t in range(0, n + 1, 16):
        X = t * S
        od.line([(X, 0), (X, 8)], fill=(0, 0, 0, 220), width=2)
        od.line([(X, map_w - 8), (X, map_w)], fill=(0, 0, 0, 220), width=2)

    # 被覆盖（消除）的浅洼地：小天蓝倒三角
    for p in covered:
        cx, cy = cell_center(p); r = 6
        od.polygon([(cx, cy + r), (cx - r, cy - r), (cx + r, cy - r)],
                   fill=C_COVERED + (235,), outline=(255, 255, 255, 255))
        od.line([(cx - r, cy - r), (cx + r, cy - r)], fill=(255, 255, 255, 255), width=1)

    # 偏差 >1 的点：警示连线（本数据为空，逻辑保留）
    for rec in flagged:
        p = rec["pred"]; q = rec["meas"]
        a = cell_center(p)
        if q is not None:
            b = cell_center(q)
            for wdt, col in ((9, (0, 0, 0, 255)), (5, C_WARN + (255,))):
                od.line([a, b], fill=col, width=wdt)
            bx, by = b
            od.polygon([(bx, by - 12), (bx - 11, by + 9), (bx + 11, by + 9)],
                       fill=C_WARN + (255,), outline=(0, 0, 0, 255))

    # 剩余洼地：预判朱红空心圆 + 实测深蓝 X（重合时圆中 X）+ 编号徽标（避碰）
    occupied = []  # 已放徽标矩形，map 局部坐标
    BW, BH = 30, 28
    candidates = [(23, -21), (-23, -21), (25, 15), (-25, 15),
                  (0, -29), (0, 29), (31, 2), (-31, 2), (0, 0)]
    for i, rec in enumerate(records, start=1):
        p = rec["pred"]; cx, cy = cell_center(p)
        R = 14
        # 选一个不碰撞、不出界的徽标位
        bcx = bcy = None
        for ox, oy in candidates:
            tx, ty = cx + ox, cy + oy
            rect = (tx - BW / 2, ty - BH / 2, tx + BW / 2, ty + BH / 2)
            if rect[0] < 2 or rect[1] < 2 or rect[2] > map_w - 2 or rect[3] > map_w - 2:
                continue
            hit = any(not (rect[2] < o[0] - 4 or rect[0] > o[2] + 4
                           or rect[3] < o[1] - 4 or rect[1] > o[3] + 4)
                      for o in occupied)
            if not hit:
                bcx, bcy = tx, ty
                break
        if bcx is None:
            bcx, bcy = cx, max(BH / 2 + 2, cy - 30)
        rect = (bcx - BW / 2, bcy - BH / 2, bcx + BW / 2, bcy + BH / 2)
        occupied.append(rect)
        if math.hypot(bcx - cx, bcy - cy) > 24:  # 牵引线
            od.line([cx, cy, bcx, bcy], fill=(255, 255, 255, 230), width=5)
            od.line([cx, cy, bcx, bcy], fill=C_INK + (230,), width=1)
        od.ellipse([cx - R, cy - R, cx + R, cy + R],
                   outline=(255, 255, 255, 255), width=7)
        od.ellipse([cx - R, cy - R, cx + R, cy + R],
                   outline=C_PRED + (255,), width=4)
        q = rec["meas"]
        if q is not None:
            qx, qy = cell_center(q); rr = 8
            od.line([(qx - rr, qy - rr), (qx + rr, qy + rr)],
                    fill=(255, 255, 255, 255), width=7)
            od.line([(qx - rr, qy + rr), (qx + rr, qy - rr)],
                    fill=(255, 255, 255, 255), width=7)
            od.line([(qx - rr, qy - rr), (qx + rr, qy + rr)],
                    fill=C_MEAS + (255,), width=3)
            od.line([(qx - rr, qy + rr), (qx + rr, qy - rr)],
                    fill=C_MEAS + (255,), width=3)
        # 编号徽标
        txt = str(i)
        tb = od.textbbox((0, 0), txt, font=f_num)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        od.ellipse([bcx - BW / 2, bcy - BH / 2, bcx + BW / 2, bcy + BH / 2],
                   fill=C_INK + (240,), outline=(255, 255, 255, 255), width=2)
        od.text((bcx - tw / 2 - tb[0], bcy - th / 2 - tb[1] - 1),
                txt, font=f_num, fill=(255, 255, 255, 255))

    img = Image.alpha_composite(img, overlay).convert("RGB")

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(img, (0, top_h))
    d = ImageDraw.Draw(canvas)

    # 外框 + 坐标文字（x 刻度在图下，y 刻度在图左，白晕保证可读）
    d.rectangle([0, top_h, map_w - 1, top_h + map_w - 1], outline=C_INK, width=2)

    def text_halo(x, y, txt, font, fill=C_INK, anchor=None):
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d.text((x + dx, y + dy), txt, font=font, fill=(255, 255, 255), anchor=anchor)
        d.text((x, y), txt, font=font, fill=fill, anchor=anchor)

    for t in range(0, n + 1, 16):
        X = t * S
        xx = min(max(X - 8, 2), map_w - 34)
        d.text((xx, top_h + map_w + 8), str(t), font=f_small, fill=C_MUTED)
    for t in range(16, n, 16):  # 跳过两端，避免与边框/x 刻度重叠
        text_halo(8, top_h + t * S - 9, str(t), f_small, fill=C_MUTED)
    d.text((map_w - 70, top_h + map_w + 24), "x →（格）", font=f_small, fill=C_MUTED)
    d.text((10, top_h + 4), "y ↓", font=f_small, fill=C_MUTED)

    # 标题
    nflag = len(flagged)
    title = f"洼地预判 vs 填洼实测 ｜ 深度阈值 τ = {tau} 灰度级"
    d.text((18, 14), title, font=f_title, fill=C_INK)
    verdict = ("全部一一对应，位置偏差 ≤ 1 格，无超格点"
               if nflag == 0 and all(r['meas'] is not None for r in records)
               else f"有 {nflag} 个点偏差超过 1 格（琥珀警示）")
    sub = (f"terrain 128×128 seed=7 octaves=4 ｜ 四邻域严格局部极小 ｜ "
           f"剩余 {len(records)} 个 ｜ {verdict}")
    d.text((20, 58), sub, font=f_sub, fill=C_MUTED)

    # 右侧清单
    x0 = map_w + 18
    d.text((x0, top_h + 6), f"剩余洼地（{len(records)}，按 Δ 降序）",
           font=f_side_h, fill=C_INK)
    y = top_h + 46
    for i, rec in enumerate(records, start=1):
        x, yy = rec["pred"]
        line = f"{i:>2}. (x={x:3d}, y={yy:3d})  G={rec['gray']:3d}  Δ={rec['depth']:3d}"
        d.text((x0, y), line, font=f_side, fill=C_INK)
        y += 27
        if y > top_h + map_w - 40:
            break
    y += 10
    d.text((x0, y), f"本档被溢流覆盖消除：{len(covered)} 个", font=f_side, fill=C_MEAS)
    y += 32
    maxcheb = max([(r["cheb"] if r["cheb"] is not None else 99)
                   for r in records], default=0)
    d.text((x0, y), f"配对数 {len(records)} ｜ 最大格偏差 {maxcheb}",
           font=f_side, fill=C_INK)
    y += 30
    d.text((x0, y), "判据：d∞=max(|dx|,|dy|)>1 即标出", font=f_small, fill=C_MUTED)

    # 右侧竖排图例（避免底部横排溢出）
    ly = y + 44
    d.text((x0, ly - 28), "图例", font=f_side_h, fill=C_INK)
    legend_items = [
        ("pred", "预判位置：原四邻域极小且 Δ>τ"),
        ("meas", "限深填洼后逐格扫描实测"),
        ("cov", "被溢流覆盖消除的极小（Δ≤τ）"),
        ("tint", "填洼抬升 / 溢流影响区（深浅=抬升量）"),
        ("warn", "偏差 >1 格的配对（本档无）"),
    ]

    def legend_marker_v(kind, x, yc):
        if kind == "pred":
            d.ellipse([x, yc - 11, x + 22, yc + 11], outline=C_PRED, width=4)
        elif kind == "meas":
            d.line([(x + 4, yc - 8), (x + 18, yc + 8)], fill=C_MEAS, width=4)
            d.line([(x + 4, yc + 8), (x + 18, yc - 8)], fill=C_MEAS, width=4)
        elif kind == "cov":
            d.polygon([(x + 11, yc + 10), (x, yc - 9), (x + 22, yc - 9)], fill=C_COVERED)
        elif kind == "tint":
            d.rectangle([x, yc - 10, x + 22, yc + 10], fill=(150, 200, 235),
                        outline=C_MEAS)
        elif kind == "warn":
            d.polygon([(x + 11, yc - 11), (x, yc + 10), (x + 22, yc + 10)], fill=C_WARN)

    for kind, label in legend_items:
        legend_marker_v(kind, x0, ly + 10)
        # 文案过长则按 ~18 个全角字折行
        maxchars = 17
        if len(label) > maxchars:
            cut = maxchars
            d.text((x0 + 32, ly), label[:cut], font=f_leg, fill=C_INK)
            d.text((x0 + 32, ly + 22), label[cut:], font=f_leg, fill=C_INK)
            ly += 52
        else:
            d.text((x0 + 32, ly), label, font=f_leg, fill=C_INK)
            ly += 36

    canvas.save(path, "PNG")
    return path


# ---------------------------------------------------------------- 主流程
def run(pgm_path: str, outdir: str, t1: int, t2: int):
    os.makedirs(outdir, exist_ok=True)
    g = read_pgm(pgm_path)
    n = g.shape[0]
    assert g.shape == (n, n), "应为方形高度图"

    M = minima4_scalar(g)
    Mnp = minima4_numpy(g)
    assert set(M) == set(Mnp), "numpy 与纯 Python 极小枚举不一致"

    S = minimax_spill(g)
    S2 = priority_flood(g)
    assert np.array_equal(S, S2), "minimax 与 priority-flood 溢流面不一致"

    depth = {p: int(S[p[1], p[0]] - g[p[1], p[0]]) for p in M}
    local_notch = {}
    for x, y in M:
        local_notch[(x, y)] = int(min(g[y - 1, x], g[y + 1, x],
                                      g[y, x - 1], g[y, x + 1]) - g[y, x])

    complete_residual = minima4_scalar(S)

    # 连通性口径附注：若溢流/填洼按 8 邻域（D8，本仓库 flow 的惯例），
    # 完整 8 向填洼后再按题目规定的 4 邻域严格极小检测，会残留若干
    # “四邻域皆高、却有对角低/等邻格”的连通性孤儿。
    s8 = minimax_spill(g, conn=8)
    residual_after_d8 = minima4_scalar(s8)
    d8_orphans = []
    for x, y in residual_after_d8:
        diag = [(x + dx, y + dy) for dx in (-1, 1) for dy in (-1, 1)]
        exits = [{"x": nx, "y": ny, "gray": int(g[ny, nx]),
                  "relation": ("equal" if g[ny, nx] == g[y, x] else "lower")}
                 for nx, ny in diag if g[ny, nx] <= g[y, x]]
        d8_orphans.append({"x": x, "y": y, "gray": int(g[y, x]),
                           "spillDepth4": depth.get((x, y)),
                           "spillDepth8": int(s8[y, x] - g[y, x]),
                           "diagonalExit": exits})

    thresholds = [int(t1), int(t2)]
    result = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pgm": os.path.relpath(pgm_path, os.getcwd()),
        "params": {"size": n, "seed": 7, "octaves": 4, "frequency": 0.02,
                   "depthUnit": "gray-level difference (0..255)",
                   "neighborhood": "4-neighborhood strict local minimum; border excluded as outlet",
                   "deviationMetric": "chebyshev d_inf=max(|dx|,|dy|); flag if >1 cell",
                   "thresholds": thresholds},
        "nMinima4": len(M),
        "completeFillResidualMinima": len(complete_residual),
        "connectivityCaveat": {
            "overflowConnectivity": 8,
            "detectionNeighborhood": "4-neighborhood strict minimum",
            "residualAfterCompleteD8Fill": len(residual_after_d8),
            "orphans": d8_orphans,
            "note": "四邻域皆严格更高、但存在对角等/低出口；D8 溢流不抬升它们，"
                    "四邻域检测仍计为洼地。坐标与原极小一致(d=0)，属连通性口径而非深度阈值。",
        },
        "minima": [{"x": x, "y": y, "gray": int(g[y, x]),
                    "localNotchDepth": local_notch[(x, y)],
                    "spill": int(S[y, x]), "spillDepth": depth[(x, y)]}
                   for x, y in sorted(M, key=lambda p: -depth[p])],
        "thresholdRows": [],
        "depthCurve": [],
        "flagSelfTest": None,
    }

    # 累计曲线 N_rem(τ)=#{Δ>τ}
    uniq_d = sorted(set(depth.values()))
    for tau in range(0, max(uniq_d) + 1):
        rem = [p for p in M if depth[p] > tau]
        result["depthCurve"].append({"tau": tau,
                                     "covered_le_tau": len(M) - len(rem),
                                     "remaining_gt_tau": len(rem)})

    csv_rows = []
    md_rows = []
    maps = []

    for tau in thresholds:
        F = np.minimum(S, g + tau)
        Q = minima4_scalar(F)
        pred = [p for p in M if depth[p] > tau]
        covered = [p for p in M if depth[p] <= tau]

        pairs = match_positions(pred, Q)
        # 补充属性
        recs = []
        for r in pairs:
            p = r["pred"]
            rec = dict(r)
            rec["gray"] = int(g[p[1], p[0]])
            rec["depth"] = depth[p]
            recs.append(rec)
        recs.sort(key=lambda r: (-r["depth"], r["pred"][1], r["pred"][0]))
        flagged = [r for r in recs if r["over1"]]

        # 恒等校验：实测集合必须恰为预判集合（坐标）
        assert len(Q) == len(pred), f"τ={tau} 计数不一致 {len(Q)} vs {len(pred)}"
        assert set(Q) == set(pred), f"τ={tau} 坐标集合不一致"
        assert not flagged, f"τ={tau} 出现 >1 格偏差: {flagged}"

        row = {
            "tau": tau,
            "nPredicted": len(pred),
            "nMeasured": len(Q),
            "nCovered": len(covered),
            "nFlaggedOver1Cell": len(flagged),
            "maxChebDeviation": max((r["cheb"] for r in recs), default=0),
            "positions": [{"k": i + 1, "x": r["pred"][0], "y": r["pred"][1],
                           "gray": r["gray"], "spillDepth": r["depth"],
                           "measX": r["meas"][0] if r["meas"] else None,
                           "measY": r["meas"][1] if r["meas"] else None,
                           "dx": r["dx"], "dy": r["dy"], "cheb": r["cheb"],
                           "euclid": r["euclid"], "over1": r["over1"]}
                          for i, r in enumerate(recs)],
            "coveredPositions": [{"x": x, "y": y, "spillDepth": depth[(x, y)]}
                                 for x, y in sorted(covered, key=lambda p: depth[p])],
        }
        result["thresholdRows"].append(row)

        for i, r in enumerate(recs, start=1):
            csv_rows.append([tau, i, r["pred"][0], r["pred"][1], r["gray"],
                             r["depth"],
                             r["meas"][0] if r["meas"] else "",
                             r["meas"][1] if r["meas"] else "",
                             r["dx"], r["dy"], r["cheb"], r["euclid"],
                             int(r["over1"])])

        map_path = os.path.join(outdir, f"depression_map_tau{tau}.png")
        render_map(map_path, g, F, tau, recs, covered, flagged)
        maps.append(map_path)

        pos_text = "；".join(
            f"{i}#({r['pred'][0]},{r['pred'][1]},Δ{r['depth']})"
            for i, r in enumerate(recs, start=1))
        md_rows.append((tau, len(pred), len(Q), len(flagged),
                        row["maxChebDeviation"], pos_text, map_path))

    # 警示标记自检：构造 d∞=2 的错配，必须被 over1 捕获
    st = match_positions([(10, 10)], [(12, 10)])[0]
    result["flagSelfTest"] = {
        "pred": [10, 10], "meas": [12, 10], "cheb": st["cheb"],
        "over1": st["over1"], "expectFlagged": True,
        "pass": bool(st["over1"] and st["cheb"] == 2),
    }
    assert result["flagSelfTest"]["pass"]

    # 全阈值独立复核恒等式：逐 τ 重算 Fτ 并逐格扫描，坐标集合必须逐点相同。
    all_tau_ok = True
    all_tau_worst_cheb = 0
    max_depth_all = max(depth.values())
    for tau in range(0, max_depth_all + 1):
        Ft = np.minimum(S, g + tau)
        Qt = set(minima4_scalar(Ft))
        Pt = {p for p in M if depth[p] > tau}
        if Qt != Pt:
            all_tau_ok = False
        for p in Pt:
            if Qt:
                c = min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for q in Qt)
                all_tau_worst_cheb = max(all_tau_worst_cheb, c)
    result["allTauIdentityOK"] = bool(all_tau_ok)
    result["allTauWorstCheb"] = int(all_tau_worst_cheb)
    assert all_tau_ok and all_tau_worst_cheb == 0, "全 τ 恒等式/坐标恒等复核失败"

    # ---- JSON
    json_path = os.path.join(outdir, "depression_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---- CSV
    csv_path = os.path.join(outdir, "depression_compare.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "k", "pred_x", "pred_y", "gray", "spillDepth",
                    "meas_x", "meas_y", "dx", "dy", "chebCells",
                    "euclidCells", "flagOver1"])
        w.writerows(csv_rows)

    # ---- Markdown 两行结论
    md_path = os.path.join(outdir, "depression_rows.md")
    lines = []
    lines.append("# 两档深度阈值：洼地数与位置（各占一行）\n")
    lines.append("坐标为格网 `(x,y)`，x 向右、y 向下，0 基；`Δ` 为 minimax 溢流深度（灰度级）。")
    lines.append("洼地＝内部四邻域严格局部极小。偏差判据 `d∞=max(|dx|,|dy|)>1` 格。\n")
    lines.append("| 深度阈值 τ（灰度级） | 预判数 | 填后逐格实测数 | 超 1 格点数 | 最大 d∞ | 洼地位置（编号(x,y,Δ)） |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for tau, npred, nq, nflag, mdv, pos, mp in md_rows:
        lines.append(f"| **{tau}** | {npred} | {nq} | {nflag} | {mdv} | {pos} |")
    lines.append("")
    lines.append("标注图：")
    for tau, *_rest, mp in md_rows:
        lines.append(f"- τ={tau}：`{os.path.basename(mp)}`")
    lines.append("")
    lines.append("## 逐个比对明细\n")
    lines.append("| τ | # | 预判 (x,y) | G | Δ | 实测 (x',y') | dx | dy | d∞ | 欧氏 | >1格? |")
    lines.append("|---:|---:|---|---:|---:|---|---:|---:|---:|---:|:---:|")
    for row in result["thresholdRows"]:
        for p in row["positions"]:
            flag = "⚠️是" if p["over1"] else "否"
            lines.append(
                f"| {row['tau']} | {p['k']} | ({p['x']},{p['y']}) | {p['gray']} | "
                f"{p['spillDepth']} | ({p['measX']},{p['measY']}) | {p['dx']} | "
                f"{p['dy']} | {p['cheb']} | {p['euclid']} | {flag} |")
    lines.append("")
    lines.append(f"完整填洼（τ=∞, F=S）后剩余严格极小：{len(complete_residual)} 个。")
    lines.append(f"原始图共 {len(M)} 个四邻域严格极小；深度累计曲线见 `depression_result.json` / 报告。")
    lines.append("")
    lines.append("## 附：八邻域溢流 vs 四邻域判定的连通性口径")
    lines.append("")
    lines.append(f"若溢流/填洼按 8 邻域（D8）连通，完整 8 向填洼后再做四邻域严格极小扫描，"
                 f"残留 **{len(residual_after_d8)}** 个："
                 + ("、".join(f"({o['x']},{o['y']})" for o in d8_orphans) if d8_orphans else "无")
                 + "。它们四邻域皆更高，但各有对角等/低出口，故 D8 不抬升而四邻域仍计数；"
                   "坐标偏差 d∞=0，属连通性口径，不是深度阈值误判。")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 控制台摘要
    print(f"n={n} minima4={len(M)} completeFillResidual={len(complete_residual)}")
    print("depth desc:", sorted([depth[p] for p in M], reverse=True))
    for row in result["thresholdRows"]:
        print(f"tau={row['tau']:>3} pred={row['nPredicted']} "
              f"meas={row['nMeasured']} flagged={row['nFlaggedOver1Cell']} "
              f"maxCheb={row['maxChebDeviation']}")
        for p in row["positions"]:
            print(f"   #{p['k']} ({p['x']},{p['y']}) G={p['gray']} Δ={p['spillDepth']} "
                  f"meas=({p['measX']},{p['measY']}) cheb={p['cheb']}")
    print("wrote:", json_path, csv_path, md_path, *maps, sep="\n  ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgm", default="artifacts/terrain_128_s7.pgm")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--t1", type=int, default=20)
    ap.add_argument("--t2", type=int, default=40)
    a = ap.parse_args()
    run(a.pgm, a.outdir, a.t1, a.t2)


if __name__ == "__main__":
    main()
