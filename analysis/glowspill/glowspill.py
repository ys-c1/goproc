#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高度分层边界光晕外溢宽度上界的数值验证。

几何约定（局部坐标系）：
  边界为平面曲线 y=f(x)，分层区域内部在 y<=f(x) 一侧，光晕加在外部
  y>=f(x) 一侧；光晕 = 到边界距离 <=r 的外侧点集，r 为光晕半径；
  光晕外轮廓 = 边界沿外法向(+y)平移 r 的等距曲线（offset curve）。
  曲率带符号：凸向光晕侧(+y)为正，凹向光晕侧为负，直线为 0。

外溢宽度 w（对直线边界严格为 0）：
  取与光晕同宽的窗口 |x|<=r，外轮廓在窗口两端相对切线参照线 y=r 的
  总抬升，以二阶差分度量：
      w_delta = 2 * ( y_env(r) - r )
  单侧最大外溢（矢高 sagitta，对照量）：
      w_sag = max_{|x|<=r} y_env(x) - r

曲率估计（曲率按二阶差分估）：
  顶点 f'(0)=0，固定物理步长 h=1 像素（与栅格格距解耦）：
      k_est = ( f(h) - 2 f(0) + f(-h) ) / h^3      （带符号）
      R_est = 1 / |k_est|
  无量纲上界 U = r/R_est；长度量纲上界 r*U = r^2/R_est。

紧界构造：
  抛物线 f(x)=x^2/(2R)，顶点曲率 1/R；R 增大时局部曲率 -> 0，边界局部
  趋于直线。验证 w_delta/r -> r/R，即 eta=(w/r)/(r/R) -> 1，上界紧。
  对照：同顶点曲率的圆弧（常曲率）外偏移等距线在窗口端反而下沉，
  w_delta 为负，给出与抛物线相反的极端。

栅格化误差：逐格心判定距离 <=r，列顶是最高光晕格心，单列系统偏低至多一个
格距 cell；当真实外溢小于一个格距时粗网格无法分辨，需用细分网格
（cell=0.5/0.25/0.1）观察向解析值的收敛。强曲率下栅格的竖直列度量与法向
等距线会分叉，见正文“度量分叉对照”。

仅依赖标准库，以 python3 -I 运行。
"""
import math
import sys

INF = math.inf


# ---------------------------------------------------------------- 曲线定义
def f_para(x, R):
    """抛物线，顶点 (0,0)，曲率 +1/R，凸向光晕侧 +y。"""
    return x * x / (2.0 * R)


def f_circle(x, R):
    """圆心 (0,-R)、半径 R 的圆弧，顶点 (0,0)，曲率 +1/R（常曲率）。"""
    return -R + math.sqrt(max(0.0, R * R - x * x))


def f_line(x, R):
    """严格直线（曲率 0 的零对照）。"""
    return 0.0


CURVES = {"抛物线": f_para, "圆弧": f_circle, "直线": f_line}


def curvature_2nd_diff(f, R, h):
    """按二阶差分估顶点带符号曲率（局部坐标 f'(0)=0）。"""
    return (f(h, R) - 2.0 * f(0.0, R) + f(-h, R)) / h ** 3


# ------------------------------------------------------ 解析外轮廓（等距曲线）
def envelope_para(x, r, R):
    """抛物线沿 +y 法向平移 r 的外轮廓。"""
    slope = x / R
    return f_para(x, R) + r / math.sqrt(1.0 + slope * slope)


def envelope_circle(x, r, R):
    """圆弧凸侧(+y)距离 r 的外偏移等距线：同心、半径 R+r 的圆弧。"""
    return -R + math.sqrt(max(0.0, (R + r) ** 2 - x * x))


def envelope_line(x, r, R):
    return r


ENVELOPES = {"抛物线": envelope_para, "圆弧": envelope_circle, "直线": envelope_line}


# -------------------------------------------------------------- 栅格实测
def _dist_circle(R, px, py):
    """点到圆弧（圆心 (0,-R)、半径 R、上半圆）的最短距离，解析解。

    最近点在查询点与圆心的连线上、落在圆弧上的那个交点；
    连线竖直退化（px=0）时最近点为顶点 (0,0)。
    """
    vx, vy = px, py + R                      # 查询点相对圆心的向量
    norm = math.hypot(vx, vy)
    if norm < 1e-15:
        return R
    tx, ty = R * vx / norm, -R + R * vy / norm
    return math.hypot(tx - px, ty - py)


def _dist_parabola(R, px, py):
    """点到抛物线 f(t)=t²/(2R) 的最短距离，解驻点三次方程，取全局最小。

    驻点条件 (t-px)+(t²/(2R)-py)·(t/R)=0，即
        t³ + 2R(R-py) t - 2R² px = 0 .
    三次项系数恒正，至多三个实根；对全部实根与（兜底的）t=px 取最小距离。
    用判别式分类的卡尔达诺公式求实根，无第三方依赖。
    """
    p_coef = 2.0 * R * (R - py)
    q_coef = -2.0 * R * R * px

    def obj(t):
        return math.hypot(t - px, t * t / (2.0 * R) - py)

    cands = [px]
    disc = (q_coef / 2.0) ** 2 + (p_coef / 3.0) ** 3
    if disc >= 0.0:
        sd = math.sqrt(disc)
        u = math.copysign(abs(-q_coef / 2.0 + sd) ** (1.0 / 3.0), -q_coef / 2.0 + sd)
        v = math.copysign(abs(-q_coef / 2.0 - sd) ** (1.0 / 3.0), -q_coef / 2.0 - sd)
        cands.append(u + v)
    else:
        rho = math.sqrt((-p_coef / 3.0) ** 3)
        ang = math.acos(max(-1.0, min(1.0, (-q_coef / 2.0) / rho)))
        m = 2.0 * math.sqrt(-p_coef / 3.0)
        for k in (0, 1, 2):
            cands.append(m * math.cos((ang + 2.0 * math.pi * k) / 3.0))
    return min(obj(t) for t in cands)


def dist_to_curve(f, R, px, py):
    """点 (px,py) 到曲线 y=f(x) 的最短距离（解析求解）。"""
    if f is f_circle:
        return _dist_circle(R, px, py)
    if f is f_para:
        return _dist_parabola(R, px, py)
    return abs(py)                                    # 直线 f=0


def raster_envelope(f, R, r, cell):
    """在与光晕同宽的窗口 |x|<=r 内逐列栅格化光晕。

    逐格心判断到曲线距离 <=r，取每列最高光晕格心为该列外轮廓高度。
    这就是像素栅格化的真实结果：列顶系统偏低至多一个格距 cell，
    强曲率或窄光晕时整列可能为空（measured_spill 对端列空值如实标注 None）。
    """
    n = int(round(2.0 * r / cell))
    xs = [(-r + (i + 0.5) * cell) for i in range(n)]
    y_top = 1.6 * r + 2.0
    ny = int(math.ceil(y_top / cell))
    tops = []
    for x in xs:
        col_top = None
        for j in range(ny):
            y = (j + 0.5) * cell
            if dist_to_curve(f, R, x, y) <= r:
                col_top = y
        tops.append(col_top)
    return xs, tops


def measured_spill(xs, tops, r, cell):
    """从栅格外轮廓估计外溢，窗口严格限制在 |x|<=r。

    w_delta：最靠近 ±r 的两个非空端列相对切线参照 y=r 的双侧抬升之和；
             端列缺失（该列无任何光晕格心）时返回 None，由上层如实标注，
             不得用内侧列冒充窗口端点。
    w_sag ：窗口内非空列外轮廓相对 y=r 的最大单侧超出（矢高）。
    列顶为最高光晕格心，系统偏低至多一个格距 cell。
    """
    pos = [i for i, v in enumerate(tops) if v is not None]
    if not pos:
        return None, None
    ir = min(pos, key=lambda i: abs(xs[i] - r))
    il = min(pos, key=lambda i: abs(xs[i] + r))
    if abs(abs(xs[ir]) - r) > 0.5 * cell + 1e-12:
        w_delta = None
    else:
        w_delta = (tops[ir] - r) + (tops[il] - r)
    window = [tops[i] for i in pos if abs(xs[i]) <= r + 1e-12]
    return w_delta, max(window) - r


def column_height_errors(f, env_f, R, r, cell):
    """逐列比较栅格列顶与解析外轮廓在同一列 x 处的高度。

    返回 (最大绝对误差, 平均带符号偏差, 有效列数)。
    近直线工况下列顶应系统偏低至多一个格距，即偏差落在 [-cell, 0]；
    强曲率下竖直列与法向等距线分叉，误差会超出该带，作为对照保留。
    """
    xs, tops = raster_envelope(f, R, r, cell)
    errs = []
    for x, t in zip(xs, tops):
        if t is None:
            continue
        errs.append(t - env_f(x, r, R))
    if not errs:
        return None, None, 0
    return max(abs(e) for e in errs), sum(errs) / len(errs), len(errs)


# -------------------------------------------------------------------- 单例
def run_case(name, R, r, cell):
    f = CURVES[name]
    env_f = ENVELOPES[name]
    h_curv = 1.0          # 曲率二阶差分固定用 1 像素物理步长，与栅格解耦

    if name == "直线":
        k_est, R_est, U, U_len = 0.0, INF, 0.0, 0.0
        w_delta_a = w_sag_a = 0.0
    else:
        k_est = curvature_2nd_diff(f, R, h_curv)
        R_est = 1.0 / abs(k_est)
        U = r / R_est
        U_len = r * U
        y_end_a = env_f(r, r, R)
        w_delta_a = 2.0 * (y_end_a - r)
        dense = (-r + 2.0 * r * i / 4000.0 for i in range(4001))
        w_sag_a = max(env_f(x, r, R) for x in dense) - r

    xs, tops = raster_envelope(f, R if name != "直线" else 1.0, r, cell)
    w_delta_m, w_sag_m = measured_spill(xs, tops, r, cell)
    if name == "直线":
        col_max, col_bias, ncol = column_height_errors(f, env_f, 1.0, r, cell)
    else:
        col_max, col_bias, ncol = column_height_errors(f, env_f, R, r, cell)

    return dict(name=name, R=R, r=r, cell=cell, k_est=k_est, R_est=R_est,
                U=U, U_len=U_len, w_delta_a=w_delta_a, w_sag_a=w_sag_a,
                w_delta_m=w_delta_m, w_sag_m=w_sag_m,
                col_max=col_max, col_bias=col_bias, ncol=ncol,
                err_band=2.0 * cell)


def fmt(v, nd=5):
    if v is None:
        return " -- "
    if math.isinf(v):
        return " inf"
    return f"{v:.{nd}f}"


def print_table(rows, title):
    print("=" * 138)
    print(title)
    print("=" * 138)
    print(f"{'曲线':<6}{'R':>7}{'r':>4}{'格距':>6}{'曲率估计':>9}{'上界r/R':>8}"
          f"{'长度上界r²/R':>11}{'解析wΔ':>9}{'wΔ/Ulen':>9}{'实测wΔ':>9}"
          f"{'列高最大误差':>12}{'列高均偏差':>11}{'有效列':>7}")
    print("-" * 138)
    for d in rows:
        ratio = d["w_delta_a"] / d["U_len"] if d["U_len"] > 0 else 0.0
        m_str = "端列缺" if d["w_delta_m"] is None else f"{d['w_delta_m']:.3f}"
        print(f"{d['name']:<6}{fmt(d['R'],1):>7}{d['r']:>4}{d['cell']:>6.2f}"
              f"{fmt(d['k_est'],5):>9}{fmt(d['U'],5):>8}"
              f"{fmt(d['U_len'],4):>11}{fmt(d['w_delta_a'],4):>9}{fmt(ratio,4):>9}"
              f"{m_str:>9}{fmt(d['col_max'],4):>12}{fmt(d['col_bias'],4):>11}"
              f"{d['ncol']:>7}")
    print("  列高误差=栅格列顶−同列解析外轮廓；近直线工况下应落在 [-格距, 0] 量化带内。")
    print()


def self_check():
    """把四条核心结论变成可执行断言，全部通过才返回。"""
    # 1) 紧性：解析比值 eta 随 R 增大单调趋于 1，且全程 eta<1（不越上界）
    etas = []
    for R in (16, 32, 64, 128, 256, 512, 1024):
        d = run_case("抛物线", R, 4, 0.1)
        etas.append(d["w_delta_a"] * R / 16.0)
    assert all(e < 1.0 for e in etas), "存在越上界的外溢比值"
    assert all(etas[i] < etas[i + 1] for i in range(len(etas) - 1)), "eta 未单调趋紧"
    assert abs(etas[-1] - 1.0) < 0.01, "最大 R 处未贴近上界"

    # 2) 量化带：近直线工况（r/R<=0.125）逐列高度误差落在 [-cell, 0] 附近
    for cell in (1.0, 0.5, 0.25, 0.1):
        for R in (32, 64, 128):
            d = run_case("抛物线", R, 4, cell)
            assert d["col_max"] <= cell + 1e-9, f"列高误差超格距: {d}"
            assert -cell - 1e-9 <= d["col_bias"] <= 1e-9, f"列高均偏差越带: {d}"

    # 3) 零对照：直线外溢严格为 0，栅格仅残留量化偏差
    for cell in (1.0, 0.1):
        d = run_case("直线", INF, 4, cell)
        assert d["w_delta_a"] == 0.0 and abs(d["col_bias"] + cell / 2) < 1e-9

    # 4) 圆弧（负曲率）外溢不出现正向：解析 w_delta 恒非正
    for R in (16, 32, 64):
        assert run_case("圆弧", R, 4, 1.0)["w_delta_a"] < 0.0
    print("自检通过：① eta 单调趋 1 且不越界 ② 近直线列高误差在量化带内 "
          "③ 直线零外溢 ④ 圆弧无正向外溢\n")


def main():
    self_check()
    # 两组光晕半径。主表取 r/R<=0.25 的近直线工况：此时边界法向接近竖直，
    # 栅格列顶与法向等距线的 y 坐标在量化带内一致。
    # R=8 的强曲率案例单列为“度量分叉对照”，展示竖直列与法向偏移的差异。
    radii = (4, 8)
    R_main = (16, 32, 64, 128)
    cells = (1.0, 0.5, 0.25, 0.1)
    R_tight = (8, 16, 32, 64, 128, 256, 512, 1024)   # 紧性序列（解析）
    tables = {c: [] for c in cells}

    for r in radii:
        for R in R_main:
            for c in cells:
                tables[c].append(run_case("抛物线", R, r, c))
                tables[c].append(run_case("圆弧", R, r, c))
        for c in cells:
            tables[c].append(run_case("直线", INF, r, c))

    for c in cells:
        label = "粗网格（真实像素栅格化）" if c == 1.0 else f"细分网格（格距 {c}，验证向解析值收敛）"
        print_table(tables[c], label)

    # 度量分叉对照：强曲率下竖直列顶 vs 法向等距线 y
    print("=" * 104)
    print("度量分叉对照（强曲率，粗网格）：栅格按竖直列取列顶，与法向等距线 y 坐标是两个对象")
    print("=" * 104)
    for r in radii:
        R = 8
        xs, tops = raster_envelope(f_para, float(R), r, 1.0)
        wm, _ = measured_spill(xs, tops, r, 1.0)
        d = run_case("抛物线", R, r, 1.0)
        print(f"  r={r} R={R}（r/R={r/R:.2f}）：栅格实测 wΔ={wm}  解析法向等距线 wΔ={d['w_delta_a']:.4f}"
              f"  上界 r²/R={r*r/R:.3f}  —— 近直线工况(r/R→0)下二者合一")
    print()

    print("=" * 104)
    print("紧性检验（抛物线，解析值）：eta=(wΔ/r)/(r/R)=wΔ·R/r²，R→∞（局部曲率→0）时 eta→1")
    print("=" * 104)
    for r in radii:
        for R in R_tight:
            d = run_case("抛物线", R, r, 0.1)
            eta = d["w_delta_a"] * R / (r * r)
            print(f"  r={r:>2}  R={R:>5}  r/R={r/R:9.6f}  长度上界r²/R={r*r/R:9.5f}"
                  f"  wΔ={d['w_delta_a']:10.6f}  wΔ/r={d['w_delta_a']/r:10.6f}  eta={eta:9.6f}")
        print()

    # 紧性构造的显式坐标样例
    print("=" * 104)
    print("紧界构造的坐标样例（抛物线 f(x)=x²/(2R)，光晕外轮廓点 (x, y_env)，单位：像素）")
    print("=" * 104)
    for r in radii:
        R = 512
        print(f"  r={r}, R={R}（顶点曲率 1/R={1/R:.5f}，r/R={r/R:.4f}）：")
        for x in (-r, -r / 2, 0.0, r / 2, r):
            yb = f_para(x, R)
            ye = envelope_para(x, r, R)
            print(f"    x={x:6.2f}  边界 y={yb:9.5f}  外轮廓 y_env={ye:10.6f}"
                  f"  相对 y=r 抬升={ye-r:+.6f}")
        print()


if __name__ == "__main__":
    sys.exit(main())
