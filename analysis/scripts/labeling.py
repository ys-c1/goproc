#!/usr/bin/env python3
"""
等高线数字标签放置：整数分配建模 + 贪心/穷举对照。

输入: Go 端 api.Generate 导出的 128x128 float64 高度图（raw little-endian）。
输出:
  work/results.json          候选、贪心与穷举结果
  work/candidates.csv        候选明细
  figures/*.png              可视化

代价定义（题目）:
  c_i = b_i * kappa_i
  b_i     : 标签框内被覆盖的等值线像素数（所有等值线的并掩码，框落在网格内）
  kappa_i : 标签锚点处水平集曲率绝对值 |κ| = |(fy² fxx - 2 fx fy fxy + fx² fyy) / (fx²+fy²)^(3/2)|
整数分配:
  min  Σ c_i x_i   s.t.  Σ x_i = k, x_i ∈ {0,1}
贪心: 按 (c_i, y0_i, x0_i) 升序取前 k（代价并列按标签框左上角坐标排序，保证唯一）。
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ----------------------------- 参数 -----------------------------
N = 128
LEVELS = [0.30, 0.40, 0.50, 0.60, 0.70]
BOX_W, BOX_H = 9, 5          # 标签框宽 × 高（像素）
PER_LEVEL = 4                # 每级候选数 → 共 20 个候选
MIN_SPACING = 12.0           # 候选锚点最小欧氏间距（保证标签框两两不相交）
KS = [6, 12]                 # 两组标签数
GRAD_EPS = 1e-6


# ----------------------------- 工具 -----------------------------
def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """可分离高斯滤波，边界反射延拓。"""
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()

    def conv1d(v, axis):
        pad = [(0, 0)] * v.ndim
        pad[axis] = (r, r)
        p = np.pad(v, pad, mode="reflect")
        out = np.zeros_like(v)
        sl = [slice(None)] * v.ndim
        for i, w in enumerate(k):
            sl[axis] = slice(i, i + v.shape[axis])
            out += w * p[tuple(sl)]
        return out

    return conv1d(conv1d(a, 0), 1)


def contour_mask(h: np.ndarray, level: float) -> np.ndarray:
    """边缘穿越法：网格边两端跨越 level，标记该边两侧的栅格像素。"""
    above = h >= level
    m = np.zeros_like(above)
    cx = above[:, :-1] != above[:, 1:]
    m[:, :-1] |= cx
    m[:, 1:] |= cx
    cy = above[:-1, :] != above[1:, :]
    m[:-1, :] |= cy
    m[1:, :] |= cy
    return m


def derivatives(h: np.ndarray):
    """中心差分（边界前向/后向），作用于平滑后的高度图。"""
    fx = np.zeros_like(h)
    fy = np.zeros_like(h)
    fx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / 2.0
    fx[:, 0] = h[:, 1] - h[:, 0]
    fx[:, -1] = h[:, -1] - h[:, -2]
    fy[1:-1, :] = (h[2:, :] - h[:-2, :]) / 2.0
    fy[0, :] = h[1, :] - h[0, :]
    fy[-1, :] = h[-1, :] - h[-2, :]
    fxx = np.zeros_like(h)
    fxx[:, 1:-1] = h[:, 2:] - 2 * h[:, 1:-1] + h[:, :-2]
    fyy = np.zeros_like(h)
    fyy[1:-1, :] = h[2:, :] - 2 * h[1:-1, :] + h[:-2, :]
    fxy = np.zeros_like(h)
    fxy[1:-1, 1:-1] = (h[2:, 2:] - h[2:, :-2] - h[:-2, 2:] + h[:-2, :-2]) / 4.0
    return fx, fy, fxx, fyy, fxy


def levelset_curvature(h: np.ndarray) -> np.ndarray:
    """等高线（水平集）曲率绝对值。"""
    fx, fy, fxx, fyy, fxy = derivatives(h)
    g2 = fx ** 2 + fy ** 2
    num = fy ** 2 * fxx - 2 * fx * fy * fxy + fx ** 2 * fyy
    kappa = np.zeros_like(h)
    ok = g2 > GRAD_EPS ** 2
    kappa[ok] = np.abs(num[ok]) / g2[ok] ** 1.5
    return kappa


@dataclass
class Candidate:
    idx: int
    level: float
    cx: int
    cy: int
    x0: int
    y0: int
    b: int          # 框内等值线像素数（并掩码）
    kappa: float    # 锚点曲率绝对值
    cost: float     # b * kappa


def box_pixels(c: Candidate) -> tuple[slice, slice]:
    return (slice(c.y0, c.y0 + BOX_H), slice(c.x0, c.x0 + BOX_W))


def occlusion_union(choices: list[Candidate], union_mask: np.ndarray) -> int:
    m = np.zeros_like(union_mask)
    for c in choices:
        sy, sx = box_pixels(c)
        m[sy, sx] = True
    return int(np.count_nonzero(m & union_mask))


def boxes_disjoint(cands: list[Candidate]) -> bool:
    """断言候选标签框两两不相交。"""
    occ = np.zeros((N, N), dtype=np.int32)
    for c in cands:
        sy, sx = box_pixels(c)
        if occ[sy, sx].any():
            return False
        occ[sy, sx] = 1
    return True


def pick_candidates(level: float, cm: np.ndarray, kappa: np.ndarray,
                    union_mask: np.ndarray, rng: np.random.Generator) -> list[Candidate]:
    """在一条等值线上空间均匀地抽 4 个候选，满足框在网格内且与已选锚点间距足够。"""
    ys, xs = np.nonzero(cm)
    order = np.argsort(ys * N + xs)
    ys, xs = ys[order], xs[order]
    chosen: list[tuple[int, int]] = []
    # 沿等值线像素序列等距取探针，扰动起点保证确定性
    probes = np.linspace(0, len(ys) - 1, PER_LEVEL * 8).astype(int)
    perm = rng.permutation(len(probes))
    out: list[tuple[int, int]] = []
    for p in perm:
        y, x = int(ys[probes[p]]), int(xs[probes[p]])
        x0, y0 = x - BOX_W // 2, y - BOX_H // 2
        if x0 < 0 or y0 < 0 or x0 + BOX_W > N or y0 + BOX_H > N:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < MIN_SPACING ** 2 for px, py in chosen):
            continue
        chosen.append((x, y))
        out.append((x, y, x0, y0))
        if len(out) == PER_LEVEL:
            break
    if len(out) < PER_LEVEL:
        raise RuntimeError(f"level {level}: 仅得到 {len(out)} 个候选，调小 MIN_SPACING")
    res = []
    for x, y, x0, y0 in out:
        sy, sx = slice(y0, y0 + BOX_H), slice(x0, x0 + BOX_W)
        b = int(np.count_nonzero(union_mask[sy, sx]))
        res.append((x, y, x0, y0, b, float(kappa[y, x])))
    return res


def main() -> None:
    root = Path(sys.argv[1])
    work = root / "work"
    figs = root / "figures"
    work.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    h = np.fromfile(work / "heightmap.bin", dtype="<f8").reshape(N, N)
    hs = gaussian_blur(h, sigma=1.2)
    kappa = levelset_curvature(hs)

    masks = {lv: contour_mask(h, lv) for lv in LEVELS}
    union_mask = np.zeros((N, N), dtype=bool)
    for m in masks.values():
        union_mask |= m

    rng = np.random.default_rng(20261128)
    raw: list[tuple] = []
    for lv in LEVELS:
        raw.extend([(lv, *t) for t in pick_candidates(lv, masks[lv], kappa, union_mask, rng)])

    cands = []
    for i, (lv, x, y, x0, y0, b, kap) in enumerate(raw):
        cands.append(Candidate(i, lv, x, y, x0, y0, b, kap, b * kap))
    # 题目规定的全局次序：代价升序，并列按标签坐标（框左上角 y0 再 x0）
    cands.sort(key=lambda c: (c.cost, c.y0, c.x0, c.idx))
    for rank, c in enumerate(cands):
        c.idx = rank

    assert boxes_disjoint(cands), "候选标签框存在相交，遮挡不具可加性"

    # ---------------- 贪心 ----------------
    greedy_by_k = {k: cands[:k] for k in KS}

    # ---------------- 穷举 ----------------
    def evaluate(k: int):
        combos = list(itertools.combinations(range(len(cands)), k))
        rows = []
        for comb in combos:
            sel = [cands[i] for i in comb]
            cost = sum(c.cost for c in sel)
            px = occlusion_union(sel, union_mask)  # 框不相交 ⇒ 等于 Σ b_i
            rows.append((cost, px, comb))
        # 穷举最优 A：代价最小；并列按坐标次序组合（与贪心同序，结果唯一）
        rows.sort(key=lambda t: (t[0], tuple((cands[i].y0, cands[i].x0) for i in t[2])))
        opt_cost = rows[0]
        # 穷举最优 B（对照）：遮挡像素最少；并列再按代价
        rows.sort(key=lambda t: (t[1], t[0], tuple((cands[i].y0, cands[i].x0) for i in t[2])))
        opt_px = rows[0]
        return opt_cost, opt_px, len(combos)

    exhaustive = {}
    for k in KS:
        opt_cost, opt_px, ncomb = evaluate(k)
        exhaustive[k] = {"opt_cost": opt_cost, "opt_px": opt_px, "n_combos": ncomb}

    # ---------------- 汇总 ----------------
    summary = []
    for k in KS:
        g = greedy_by_k[k]
        g_cost = sum(c.cost for c in g)
        g_px = occlusion_union(g, union_mask)
        oc = exhaustive[k]["opt_cost"]
        opx = exhaustive[k]["opt_px"]
        gap_cost = abs(g_cost - oc[0])
        gap_px_vs_pxopt = (g_px - opx[1]) / opx[1] if opx[1] else 0.0
        summary.append({
            "k": k,
            "n_combos": exhaustive[k]["n_combos"],
            "greedy": {"cost": g_cost, "pixels": g_px, "idx": [c.idx for c in g]},
            "exhaust_cost_opt": {"cost": oc[0], "pixels": oc[1], "idx": list(oc[2])},
            "exhaust_pixel_opt": {"cost": opx[0], "pixels": opx[1], "idx": list(opx[2])},
            "cost_gap": gap_cost,
            "pixel_gap_vs_pixel_opt": gap_px_vs_pxopt,
        })

    with open(work / "candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "level", "cx", "cy", "x0", "y0", "b_pixels", "kappa", "cost"])
        for c in cands:
            w.writerow([c.idx, c.level, c.cx, c.cy, c.x0, c.y0, c.b, f"{c.kappa:.6f}", f"{c.cost:.6f}"])

    payload = {
        "params": {"N": N, "levels": LEVELS, "box_w": BOX_W, "box_h": BOX_H,
                   "per_level": PER_LEVEL, "min_spacing": MIN_SPACING, "ks": KS,
                   "blur_sigma": 1.2},
        "candidates": [asdict(c) for c in cands],
        "summary": summary,
    }
    with open(work / "results.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 控制台简报
    print(f"{'rank':>4} {'level':>5} {'(cx,cy)':>9} {'(x0,y0)':>9} {'b':>3} {'kappa':>9} {'cost':>10}")
    for c in cands:
        print(f"{c.idx:>4} {c.level:>5.2f} {f'({c.cx},{c.cy})':>9} "
              f"{f'({c.x0},{c.y0})':>9} {c.b:>3} {c.kappa:>9.5f} {c.cost:>10.5f}")
    for s in summary:
        print("=" * 70)
        print(f"k={s['k']}  组合数={s['n_combos']}")
        print(f"  贪心:        代价={s['greedy']['cost']:.5f}  遮挡像素={s['greedy']['pixels']}")
        print(f"  穷举(代价优): 代价={s['exhaust_cost_opt']['cost']:.5f}  遮挡像素={s['exhaust_cost_opt']['pixels']}")
        print(f"  穷举(像素优): 代价={s['exhaust_pixel_opt']['cost']:.5f}  遮挡像素={s['exhaust_pixel_opt']['pixels']}")
        print(f"  代价差={s['cost_gap']:.3e}  对像素最优的像素差距={s['pixel_gap_vs_pixel_opt']*100:.2f}%")


if __name__ == "__main__":
    main()
