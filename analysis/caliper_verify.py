#!/usr/bin/env python3
"""独立复算：凸包、旋转卡壳、暴力最远点对，与 Go 产物逐项对拍。

只读 artifacts/caliper 下的地形 PGM 与 summary.json，不调用 Go 代码。
候选点口径：地形 PGM 的 8 位灰度 >= 128。该条件与原始 float64 高度 >= 0.5
严格等价——Gray8 取整为 floor(255x+0.5)，floor(...)>=128 当且仅当 x>=0.5。
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np


def read_pgm(path):
    data = Path(path).read_bytes()
    m = re.match(rb"P5\n(\d+) (\d+)\n255\n", data)
    if not m:
        raise SystemExit(f"{path}: 非预期 PGM 文件头")
    w, h = int(m.group(1)), int(m.group(2))
    body = data[m.end():m.end() + w * h]
    if len(body) != w * h:
        raise SystemExit(f"{path}: 像素字节数不符")
    return np.frombuffer(body, dtype=np.uint8).reshape(h, w)


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 1:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def antipodal_by_definition(h):
    """逐边取最大面积顶点，枚举全部对踵点对（与卡壳无关的真值定义）。"""
    n = len(h)
    out = set()
    if n < 2:
        return out
    if n == 2:
        return {(0, 1)}
    for i in range(n):
        ni = (i + 1) % n
        areas = [cross(h[i], h[ni], h[j]) for j in range(n)]
        mx = max(areas)
        for j, v in enumerate(areas):
            if v == mx:
                out.add((min(i, j), max(i, j)))
                out.add((min(ni, j), max(ni, j)))
    out.discard((0, 0))
    return out


def calipers(h):
    """旋转卡壳：返回 (对踵点对集合, 轮数, 游标步数)。"""
    n = len(h)
    if n < 2:
        return set(), 0, 0
    if n == 2:
        return {(0, 1)}, 1, 0
    cand, j, steps = set(), 1, 0
    rounds = 0
    for i in range(n):
        ni = (i + 1) % n
        rounds += 1
        guard = 0
        while True:
            nj = (j + 1) % n
            if cross(h[i], h[ni], h[nj]) > cross(h[i], h[ni], h[j]):
                j = nj
                steps += 1
                guard += 1
                if guard > 4 * n:
                    raise SystemExit("卡壳游标未收敛")
                continue
            break
        cand.add((min(i, j), max(i, j)))
        cand.add((min(ni, j), max(ni, j)))
        nj = (j + 1) % n
        if cross(h[i], h[ni], h[nj]) == cross(h[i], h[ni], h[j]):
            cand.add((min(i, nj), max(i, nj)))
            cand.add((min(ni, nj), max(ni, nj)))
    return cand, rounds, steps


def brute(points):
    p = np.asarray(points, dtype=np.int64)
    best = -1
    pair = None
    for i in range(len(p)):
        d = ((p - p[i]) ** 2).sum(axis=1)
        m = int(d.max())
        if m > best:
            best = m
            j = int(d.argmax())
            pair = tuple(sorted((tuple(p[i]), tuple(p[j]))))
    return best, pair


def round_px(d2):
    # 与 Go 相同：整数平方根后比较余量，避免浮点。
    n = math.isqrt(d2)
    if d2 - n * n > (n + 1) * (n + 1) - d2:
        n += 1
    return n


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/caliper")
    summary = json.loads((out / "summary.json").read_text())
    for st in summary:
        if abs(st["threshold_height"] - 0.5) > 1e-12:
            raise SystemExit(
                f"画幅 {st['frame']} 的阈值是 {st['threshold_height']}，"
                "本脚本只按 8 位灰度 >= 128 复算；该等价仅对阈值 0.5 成立。"
            )
    failures = 0
    print("画幅  候选   凸包  卡壳轮  步数  对踵  预判d  实测d  一致")
    for st in summary:
        img = read_pgm(out / f"terrain_{st['frame']}.pgm")
        ys, xs = np.where(img >= 128)
        pts = list(zip(xs.tolist(), ys.tolist()))
        h = hull(pts)
        cand, rounds, steps = calipers(h)
        definition = antipodal_by_definition(h)
        best2, best_pair = brute(pts)
        pred2 = max(dist2(h[a], h[b]) for a, b in cand)

        checks = {
            "候选点数": (len(pts), st["candidates"]),
            "凸包顶点数": (len(h), st["hull_vertices"]),
            "卡壳轮数": (rounds, st["caliper_rounds"]),
            "游标步数": (steps, st["caliper_cursor_steps"]),
            "对踵点对数": (len(cand), st["antipodal_count"]),
            "卡壳=定义集": (cand == definition, True),
            "预判d2": (pred2, st["predicted"]["dist2"]),
            "实测d2": (best2, st["measured"]["dist2"]),
            "预判取整": (round_px(pred2), st["predicted"]["dist_px"]),
            "实测取整": (round_px(best2), st["measured"]["dist_px"]),
            "凸包顶点集": (set(h), {tuple(p) for p in
                                   [(v["X"], v["Y"]) for v in st["hull"]]}),
        }
        bad = [k for k, (got, want) in checks.items() if got != want]
        failures += len(bad)
        print(f"{st['frame']:>4} {len(pts):>6} {len(h):>5} {rounds:>7} {steps:>5} "
              f"{len(cand):>5} {round_px(pred2):>6} {round_px(best2):>6} "
              f"{'一致' if not bad else '不一致 ' + ','.join(bad)}")
        if best_pair and st["measured"]["dist2"] == best2:
            got = tuple(sorted(((st["measured"]["a"]["X"], st["measured"]["a"]["Y"]),
                                (st["measured"]["b"]["X"], st["measured"]["b"]["Y"]))))
            if got != best_pair:
                print(f"  ! 暴力枚举最远点对坐标不符：Go {got} / 复算 {best_pair}")
                failures += 1
    if failures:
        raise SystemExit(f"独立复算发现 {failures} 处不一致")
    print("独立复算全部一致。")


if __name__ == "__main__":
    main()
