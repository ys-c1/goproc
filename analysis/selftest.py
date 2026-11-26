"""核心库数值自检：不依赖地形生成器，用解析构造验证恒等式。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import terrain_bands as tb


def check(name, got, want, tol=1e-10):
    rel = abs(got - want) / max(1.0, abs(want))
    status = "OK " if rel < tol else "FAIL"
    print(f"[{status}] {name}: got={got:.10f} want={want:.10f} rel={rel:.2e}")
    return rel < tol


def main():
    ok = True

    # 1) 斜面 u=(x+y)/... 取 u=x/N，|∇u|=1/N，等值线为竖直线，L(h)=N
    n = 40
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    z = xx / (n - 1)
    mesh = tb.triangulate(z)
    info = tb.analytic_triangle_integrals(mesh)
    levels = np.linspace(0, 1, 2001)
    L, Lam = tb.levelset_functions(mesh, levels)
    int_lam = np.trapezoid(Lam, levels)
    int_l = np.trapezoid(L, levels)
    # 均匀采样积分只作诊断，允许采样截断误差（精确版见下方折点积分）
    ok &= check("斜面 采样∫Λ≈(N-1)^2", int_lam, (n - 1) ** 2, 1e-2)
    ok &= check("斜面解析 ∫Λ", info["total_area_Lambda"], (n - 1) ** 2, 1e-12)
    # 内部高度：L(h)=N（沿 y 方向跨 N 个格点的线段长 N-1，三角剖分边界为 (N-1)）
    interior = (levels > 0.05) & (levels < 0.95)
    ok &= check("斜面 L(h)≈N-1（等值线总长）", np.median(L[interior]), n - 1, 1e-12)
    # Λ = L/g, g=1/(N-1) → 中位数应为 (N-1)^2
    ok &= check("斜面 Λ(h)≈(N-1)^2", np.median(Lam[interior]), (n - 1) ** 2, 1e-12)
    ok &= check("斜面 采样∫L≈g·面积", int_l, info["total_integral_L"], 1e-2)

    # 2) 抛物面 u=(x^2+y^2) 归一化，验证 ∫Λ=面积 且 解析裁切=积分
    r2 = (xx - n / 2) ** 2 + (yy - n / 2) ** 2
    z2 = r2 / r2.max()
    mesh2 = tb.triangulate(z2)
    edges = np.linspace(0, 1, 9)
    bi = tb.analytic_band_integrals(mesh2, edges)
    ok &= check("抛物面 ∫Λ=面积（逐面解析）", bi["pred_area"].sum(), (n - 1) ** 2, 1e-12)
    exact = tb.exact_clipped_band_areas(mesh2, edges)
    ok &= check("解析裁切 vs Λ 积分（最大带偏差）",
                np.abs(bi["pred_area"] - exact).max(), 0.0, 1e-12)
    ok &= check("解析裁切面积求和=总面积", exact.sum(), (n - 1) ** 2, 1e-12)

    # 3) 随机光滑场：恒等式闭合
    rng = np.random.default_rng(0)
    z3 = rng.random((30, 30))
    mesh3 = tb.triangulate(z3)
    bi3 = tb.analytic_band_integrals(mesh3, np.linspace(0, 1, 9))
    ok &= check("随机场 ∫Λ=面积", bi3["pred_area"].sum(), 29 ** 2, 1e-12)

    # 4) 斜面逐面解析积分精确闭合
    bi1 = tb.analytic_band_integrals(mesh, np.linspace(0, 1, 9))
    ok &= check("斜面逐面 ∫Λ", bi1["pred_area"].sum(), (n - 1) ** 2, 1e-12)
    ok &= check("斜面逐面 ∫L=g·面积",
                sum(bi1["mean_L"][b] * (1 / 8) for b in range(8)),
                info["total_integral_L"], 1e-12)

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
