#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标定: Koch 曲线 (严格自相似, D=log4/log3, mu=D-1), 验证圆规规尺实现。"""
import math
import sys
import numpy as np
sys.path.insert(0, "/home/ys/feedback/uidreczz28eCcwNSdbd/scripts")
from coastline_analysis import ruler_walk


def koch(level):
    """返回单位长度 Koch 折线顶点, 自底向上迭代。方向角 0°。"""
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
    for _ in range(level):
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            v = b - a
            p1 = a + v / 3
            p3 = a + 2 * v / 3
            ang = math.atan2(v[1], v[0]) + math.pi / 3
            p2 = p1 + np.linalg.norm(v) / 3 * np.array([math.cos(ang), math.sin(ang)])
            out += [p1, p2, p3, b]
        pts = out
    return pts


D_true = math.log(4) / math.log(3)
mu_true = D_true - 1
print(f"Koch D={D_true:.5f} mu={mu_true:.5f}")
for level in (6, 7, 8):
    pts = np.array(koch(level))
    # 放大到像素尺度, 使最小特征 > 1px
    scale = 3 ** level
    chain = (pts * scale).tolist()
    L0 = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) * scale
    # 规尺步长取 3 的幂 (与 Koch 自相似尺度对齐) 及若干中间值
    eps_list = [3 ** k for k in range(level, 0, -1)]
    rows = []
    for e in eps_list:
        L, q = ruler_walk(chain, False, float(e))
        rows.append((e, L, len(q)))
    le = np.log(np.array([r[0] for r in rows], float))
    lL = np.log(np.array([r[1] for r in rows], float))
    b, a = np.polyfit(le, lL, 1)
    print(f"\nlevel={level} npts={len(pts)} arc={L0:.1f}  ruler mu={-b:.5f} D={1-b:.5f}")
    for e, L, nq in rows:
        print(f"  eps={e:8d} L={L:12.2f} N={nq:6d}  L/eps={L/e:8.2f}")
