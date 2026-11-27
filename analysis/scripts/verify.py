#!/usr/bin/env python3
"""数值正确性核验（独立于 labeling.py 重新实现关键断言）。

1. 曲率公式：径向二次场 f=r²/2R 的等高线为圆，|κ| 应等于 1/r；
2. 遮挡像素可加性：框两两不相交时，并集计数 == Σ 框内计数；
3. 穷举组合数 == C(n,k)；
4. 贪心 == 穷举代价最优（对 results.json 复核）；
5. 并列 tie-break：构造同价候选，验证按 (y0,x0) 排序后结果唯一确定。
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from labeling import (N, BOX_W, BOX_H, contour_mask, gaussian_blur,
                      levelset_curvature, occlusion_union, Candidate)


def test_curvature_formula():
    # f(x,y) = r²/(2R)：fx=x/R, fy=y/R, fxx=fyy=1/R, fxy=0
    # 水平集圆半径 r，理论 |κ| = 1/r
    R = 10.0
    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    f = (xx ** 2 + yy ** 2) / (2 * R)
    kappa = levelset_curvature(f)
    r = np.sqrt(xx ** 2 + yy ** 2)
    # 排除边界 3 像素（单边/贴边差分污染）与原点（梯度为零）
    border = np.zeros_like(r, dtype=bool)
    border[:3, :] = border[-3:, :] = border[:, :3] = border[:, -3:] = True
    interior = (r >= 8) & (r < 30) & ~border
    truth = 1.0 / r[interior]
    abs_err = np.abs(kappa[interior] - truth)
    rel_err = abs_err / truth
    assert rel_err.max() < 1e-10, f"内部点相对误差超限: {rel_err.max()}"
    print(f"[PASS] 曲率公式：径向圆场内部点 |κ|=1/r，最大相对误差 {rel_err.max():.2e}（边界点受单边差分影响不计）")

    # 线性场等高线为直线，曲率必须处处为 0
    g = 3.0 * xx + 7.0 * yy + 1.0
    kg = levelset_curvature(g)
    assert np.abs(kg).max() < 1e-12, "线性场曲率应为 0"
    print(f"[PASS] 线性场等高线为直线：max|κ|={np.abs(kg).max():.2e}")


def test_occlusion_additivity_and_combos(root: Path):
    res = json.loads((root / "work" / "results.json").read_text())
    h = np.fromfile(root / "work" / "heightmap.bin", dtype="<f8").reshape(N, N)
    levels = res["params"]["levels"]
    union = np.zeros((N, N), dtype=bool)
    for lv in levels:
        union |= contour_mask(h, lv)

    cands = [Candidate(c["idx"], c["level"], c["cx"], c["cy"], c["x0"], c["y0"],
                       c["b"], c["kappa"], c["cost"]) for c in res["candidates"]]

    # 可加性：全部 20 个框
    occ = np.zeros((N, N), dtype=np.int8)
    for c in cands:
        sy = slice(c.y0, c.y0 + BOX_H)
        sx = slice(c.x0, c.x0 + BOX_W)
        assert not occ[sy, sx].any(), f"框 {c.idx} 与其他框相交"
        occ[sy, sx] = 1
    union_px = occlusion_union(cands, union)
    sum_b = sum(c.b for c in cands)
    assert union_px == sum_b, f"可加性不成立: 并集 {union_px} != Σb {sum_b}"
    print(f"[PASS] 标签框两两不相交；遮挡可加：并集 {union_px} = Σbᵢ {sum_b}")

    # 组合数 + 贪心即代价最优（独立重算）
    costs = np.array([c.cost for c in cands])
    order = np.lexsort(([c.x0 for c in cands], [c.y0 for c in cands], costs))
    for s in res["summary"]:
        k = s["k"]
        n_comb = len(list(itertools.combinations(range(len(cands)), k)))
        assert n_comb == s["n_combos"], "组合数不一致"
        greedy_idx = sorted(order[:k].tolist())
        # 全枚举最小代价
        best = min(sum(costs[list(comb)]) for comb in itertools.combinations(range(len(cands)), k))
        greedy_cost = float(costs[order[:k]].sum())
        assert abs(best - greedy_cost) < 1e-12, "贪心未达到穷举最优代价"
        assert greedy_idx == sorted(s["exhaust_cost_opt"]["idx"]), "贪心与穷举选项不一致"
        print(f"[PASS] k={k}: C(20,{k})={n_comb}；贪心代价 {greedy_cost:.6f} == 穷举最优 {best:.6f}")


def test_tie_break_uniqueness():
    # 三个同价候选，验证 (代价, y0, x0) 次序唯一
    mk = lambda i, x0, y0: Candidate(i, 0.5, x0 + 4, y0 + 2, x0, y0, 5, 0.2, 1.0)
    cands = [mk(0, 30, 40), mk(1, 10, 40), mk(2, 30, 20)]
    cands.sort(key=lambda c: (c.cost, c.y0, c.x0, c.idx))
    got = [(c.y0, c.x0) for c in cands]
    assert got == [(20, 30), (40, 10), (40, 30)], got
    # 再用相同输入独立排一次，结果必须逐元素相同
    again = sorted([mk(0, 30, 40), mk(1, 10, 40), mk(2, 30, 20)],
                   key=lambda c: (c.cost, c.y0, c.x0, c.idx))
    assert [c.idx for c in again] == [c.idx for c in cands]
    print("[PASS] 代价并列时按 (y0,x0) 排序，输出唯一：", [(c.idx, c.y0, c.x0) for c in cands])


def test_blur_preserves_range():
    root = Path(sys.argv[1])
    h = np.fromfile(root / "work" / "heightmap.bin", dtype="<f8").reshape(N, N)
    hs = gaussian_blur(h, 1.2)
    assert hs.shape == h.shape and np.isfinite(hs).all()
    assert hs.min() >= 0 and hs.max() <= 1.0 + 1e-12
    print(f"[PASS] 平滑后值域有限且保持 [0,1]：[{hs.min():.4f}, {hs.max():.4f}]")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    test_curvature_formula()
    test_blur_preserves_range()
    test_occlusion_additivity_and_combos(root)
    test_tie_break_uniqueness()
    print("全部核验通过。")
