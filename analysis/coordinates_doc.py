"""Generate artifacts/repetition/COORDINATES.md from repeat_coordinates.csv.

Coordinates are output pixels with origin upper-left; bbox is half-open
[x0,y0,x1,y1).  Geometry: center_k = c0 / 2^k, side_k = 256 / 2^k.
Canonical boxes are identical across tiers (a tier just truncates at k<K).
"""
import csv, pathlib, collections

ROOT = pathlib.Path("/home/ys/feedback/uidreczz28eB3n7piyx")
ART = ROOT / "artifacts/repetition"

rows = list(csv.DictReader(open(ART / "repeat_coordinates.csv", newline="")))
for r in rows:
    for k in ("tier", "level", "center_x", "center_y", "x0", "y0", "x1", "y1", "side", "cx0", "cy0"):
        r[k] = int(r[k])

# canonical: use tier=6 rows (contain all levels) keyed by anchor,level
canon = {}
c0 = {}
for r in rows:
    if r["tier"] == 6:
        canon[(r["anchor_id"], r["level"])] = r
        c0[r["anchor_id"]] = (r["cx0"], r["cy0"])

anchors = sorted(c0, key=lambda a: (int(a[2]), int(a[1])))  # a<col? digits aRC


def tiers_for_level(k):
    return [K for K in (2, 4, 6) if K >= k + 1]


lines = []
A = lines.append
A("# 重复片坐标清单\n")
A("坐标系：输出像素，原点在左上角，x 向右、y 向下；包围盒为半开区间 "
  "`[x0,y0,x1,y1)`（宽高 = `side`）。\n")
A("几何（与种子无关，仅由频率比 ρ=2 决定）：\n")
A("```")
A("center_k = center_0 / 2^k      # 整数整除")
A("side_k   = 256 / 2^k")
A("x0,y0    = center_k - side_k/2")
A("```\n")
A("3×3 锚点中心 `center_0 ∈ {256,512,768}²`，锚片边长 256。编号 `aXY`：")
A("第一位是 x 档（列），第二位是 y 档（行），各取 1/2/3，分别映射到 256/512/768。")
A("例如 `a31=(768,256)`（右上角），`a22=(512,512)`（正中）。\n")
A("“出现档位”指该盒在哪些层数档（K=2/4/6）中被标记：第 k 层盒只在 K≥k+1 时存在。\n")

# summary counts
A("## 计数口径\n")
A("| 层数档 K | 锚片 | 重复片（孪生盒，k≥1） | 配对数 9·(K−1) | 标记盒总数 9K |")
A("|---:|---:|---:|---:|---:|")
for K in (2, 4, 6):
    A(f"| {K} | 9 | {9*(K-1)} | {9*(K-1)} | {9*K} |")
A("")

# one big table of all canonical boxes
A("## 全部 54 个规范盒（9 锚 × 6 层；其中 45 个是重复片）\n")
A("| 锚点 | c0=(x,y) | k | 层 | 中心 (x,y) | 包围盒 x0,y0→x1,y1 | side | 类别 | 出现档位 K |")
A("|---|---|---:|---:|---|---|---:|---|---|")
for a in anchors:
    for k in range(6):
        r = canon[(a, k)]
        kind = "锚片" if k == 0 else "**重复片**"
        tiers = ",".join(map(str, tiers_for_level(k))) or "—"
        A(f"| {a} | ({r['cx0']},{r['cy0']}) | {k} | L{k} | "
          f"({r['center_x']},{r['center_y']}) | "
          f"[{r['x0']},{r['y0']},{r['x1']},{r['y1']}) | {r['side']} | {kind} | {tiers} |")
A("")

# per-anchor compact chains of just repeat (k>=1) coordinates
A("## 每个锚点的重复片坐标（仅 k≥1，共 5 片/锚）\n")
for a in anchors:
    cx, cy = c0[a]
    A(f"### {a}  锚点 c0=({cx},{cy})\n")
    A("| k | 中心 | 包围盒 | side |")
    A("|---:|---|---|---:|")
    for k in range(1, 6):
        r = canon[(a, k)]
        A(f"| {k} | ({r['center_x']},{r['center_y']}) | "
          f"[{r['x0']},{r['y0']},{r['x1']},{r['y1']}) | {r['side']} |")
    A("")

(ART / "COORDINATES.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote", ART / "COORDINATES.md", "lines", len(lines))
