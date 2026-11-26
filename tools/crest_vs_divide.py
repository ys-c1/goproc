#!/usr/bin/env python3
"""逐像素比对：逐格分界线（格边集合）与独立脊线（脊像素集合）。

输入
  --heights F   高度图，四字节小端 uint32 边长 + 小端 float64 高度（Go EncodeF64 格式）
  --edges   F   分界线格边，行 "H x y" / "V x y"，# 开头为注释
  --crest   F   独立脊线像素，行 "x y"（缺省时由本脚本按沿轴局部极大独立重算）
  --tol     N   偏差阈值（格），距离 > N 的像素/格边计入偏差段，默认 2
  --out     F   文本报告输出路径

距离定义：把每条格边取中点（H 边中点 (x, y-0.5)，V 边中点 (x-0.5, y)），
脊线用像素中心 (x, y)；两侧分别做有界 Chamfer 距离变换后按切比雪夫/曼哈顿最近距离比对。
报告给出两组方向（脊线→分界线、分界线→脊线）的偏差像素数、偏差段及其坐标。
"""
import argparse
import collections
import struct
import sys

INF = 10 ** 9


def load_heights(path):
    with open(path, "rb") as f:
        raw = f.read()
    n = struct.unpack_from("<I", raw, 0)[0]
    vals = list(struct.unpack_from("<%dd" % (n * n), raw, 4))
    return n, vals


def load_edges(path):
    horiz, vert = set(), set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            kind, x, y = line.split()
            (horiz if kind == "H" else vert).add((int(x), int(y)))
    return horiz, vert


def load_or_compute_crest(path, n, h):
    if path:
        pts = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                x, y = line.split()
                pts.add((int(x), int(y)))
        return pts
    # 与 Go divide.Crest 同定义的独立检测：四条过格轴线任一上的严格局部极大。
    pts = set()
    for y in range(1, n - 1):
        for x in range(1, n - 1):
            c = h[y * n + x]
            vertical = c > h[y * n + x - 1] and c > h[y * n + x + 1]
            horizontal = c > h[(y - 1) * n + x] and c > h[(y + 1) * n + x]
            diag_main = c > h[(y - 1) * n + x - 1] and c > h[(y + 1) * n + x + 1]
            diag_anti = c > h[(y + 1) * n + x - 1] and c > h[(y - 1) * n + x + 1]
            if vertical or horizontal or diag_main or diag_anti:
                pts.add((x, y))
    return pts


def chamfer_l1(n, seeds):
    """网格点上的 L1(曼哈顿) 有界距离变换，seeds 为 (x,y) 像素坐标集合。"""
    dist = [INF] * (n * n)
    dq = collections.deque()
    for x, y in seeds:
        if 0 <= x < n and 0 <= y < n:
            i = y * n + x
            if dist[i] == INF:
                dist[i] = 0
                dq.append((x, y))
    while dq:
        x, y = dq.popleft()
        cur = dist[y * n + x]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                ni = ny * n + nx
                if dist[ni] > cur + 1:
                    dist[ni] = cur + 1
                    dq.append((nx, ny))
    return dist


def edge_midpoints(horiz, vert):
    """返回 (中点浮点坐标取整网格代理, 原始边) 列表，用最近像素承载距离。"""
    out = []
    for x, y in horiz:
        out.append(((x, y - 0.5), ("H", x, y)))
    for x, y in vert:
        out.append(((x - 0.5, y), ("V", x, y)))
    return out


def nearest_pixel_dist_to_crest(px, py, crest_dist, n):
    """格边中点到最近脊像素的 L1 距离：中点相对像素偏移 0.5，故加 0.5 后取整代理。"""
    cx = min(n - 1, max(0, int(round(px))))
    cy = min(n - 1, max(0, int(round(py))))
    base = crest_dist[cy * n + cx]
    # 中点与代理像素相差半个格，L1 上补 0.5（横或竖一个方向）。
    return base + 0.5


def components(coords):
    """把 8 邻域相接的偏差坐标聚成段，返回按像素数降序的段列表。"""
    pts = set(coords)
    seen = set()
    segs = []
    for p in pts:
        if p in seen:
            continue
        stack = [p]
        seen.add(p)
        seg = []
        while stack:
            x, y = stack.pop()
            seg.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (x + dx, y + dy)
                    if q in pts and q not in seen:
                        seen.add(q)
                        stack.append(q)
        segs.append(sorted(seg, key=lambda c: (c[1], c[0])))
    segs.sort(key=lambda s: (-len(s), s[0][1], s[0][0]))
    return segments_safe(segs)


def segments_safe(segs):
    return segs


def bbox(seg):
    xs = [p[0] for p in seg]
    ys = [p[1] for p in seg]
    return min(xs), max(xs), min(ys), max(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heights", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--crest", default="")
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n, h = load_heights(args.heights)
    horiz, vert = load_edges(args.edges)
    crest = load_or_compute_crest(args.crest, n, h)

    crest_dist = chamfer_l1(n, crest)
    # 以所有格边中点的代理像素作为距离变换种子，供脊像素→分界线方向使用。
    edge_pixels = set()
    for x, y in horiz:
        edge_pixels.add((x, min(n - 1, max(0, y - 1))))
        edge_pixels.add((x, min(n - 1, max(0, y))))
    for x, y in vert:
        edge_pixels.add((min(n - 1, max(0, x - 1)), y))
        edge_pixels.add((min(n - 1, max(0, x)), y))
    edge_dist = chamfer_l1(n, edge_pixels)

    lines = []
    lines.append("画幅 %s  尺寸 %d×%d  容差 %.0f 格" % (args.heights, n, n, args.tol))
    lines.append("分水线格边：水平 %d，竖直 %d，合计 %d" % (len(horiz), len(vert), len(horiz) + len(vert)))
    lines.append("独立脊线像素（对照基准）：%d" % len(crest))
    lines.append("")

    # 第一组：脊线像素 → 分界线。距离超过阈值的脊像素聚成段。
    bad_crest = []
    for x, y in crest:
        if edge_dist[y * n + x] > args.tol:
            bad_crest.append((x, y))
    segs1 = components(bad_crest)
    lines.append("[第一组] 独立脊线 → 分水线：偏离像素 %d，段落 %d" % (len(bad_crest), len(segs1)))
    for si, seg in enumerate(segs1, 1):
        x0, x1, y0, y1 = bbox(seg)
        maxd = max(edge_dist[y * n + x] for x, y in seg)
        lines.append("  段 %3d 像素 %3d，最大偏差 %d，包围盒 x[%d,%d] y[%d,%d]"
                     % (si, len(seg), maxd, x0, x1, y0, y1))
        lines.append("    坐标：" + " ".join("(%d,%d)" % p for p in seg))
    lines.append("")

    # 第二组：分界线格边 → 脊线。中点距离超过阈值的格边聚成段。
    bad_edges = []
    detail = []
    for (px, py), tag in edge_midpoints(horiz, vert):
        d = nearest_pixel_dist_to_crest(px, py, crest_dist, n)
        if d > args.tol:
            # 用边的主网格坐标承载分段，H/V 各自标注。
            ex, ey = tag[1], tag[2]
            bad_edges.append((ex, ey))
            detail.append((tag, d))
    segs2 = components(bad_edges)
    lines.append("[第二组] 分水线 → 独立脊线：偏离格边 %d，段落 %d" % (len(bad_edges), len(segs2)))
    dmap = {(t[1], t[2]): d for t, d in detail}
    for si, seg in enumerate(segs2, 1):
        x0, x1, y0, y1 = bbox(seg)
        maxd = max(int(dmap.get(p, 0)) for p in seg)
        lines.append("  段 %3d 像素 %3d，最大偏差 %d，包围盒 x[%d,%d] y[%d,%d]"
                     % (si, len(seg), maxd, x0, x1, y0, y1))
        lines.append("    坐标：" + " ".join("(%d,%d)" % p for p in seg))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 同时把两组汇总打到 stdout，便于 measure.sh 汇总。
    print("画幅 %s：H=%d V=%d | 脊→线 偏离像素 %d 段 %d | 线→脊 偏离格边 %d 段 %d"
          % (args.heights, len(horiz), len(vert), len(bad_crest), len(segs1), len(bad_edges), len(segs2)))


if __name__ == "__main__":
    sys.exit(main())
