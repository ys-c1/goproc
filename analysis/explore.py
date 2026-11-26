#!/usr/bin/env python3
"""探查：容错读取 PGM，统计 8x8 块内方差/标准差分布，辅助选档。"""
import numpy as np

PATH = "/home/ys/feedback/uidreczz28eDSvuZZSv/artifacts/terrain_s7.pgm"
W = 8

raw = open(PATH, "rb").read()
# 容错：生成器把头里的换行写成了字面量 \n
raw = raw.replace(b"\\n", b"\n", 4)
magic, rest = raw[:2], raw[3:]
# 手工解析头: P5\n W H\n MAXV\n data
dims, maxline, payload = rest.split(b"\n", 2)
w, h = (int(x) for x in dims.split())
maxv = int(maxline)
img = np.frombuffer(payload, dtype=np.uint8).reshape(h, w).astype(np.float64)
print("shape", img.shape, "range", img.min(), img.max(), "global mean", img.mean().round(2))

n = h // W
blocks = img.reshape(n, W, n, W).transpose(0, 2, 1, 3).reshape(n, n, W * W)
mean_b = blocks.mean(axis=2)
var_b = blocks.var(axis=2, ddof=1)
std_b = np.sqrt(var_b)

print("block std sigma: min %.2f  p25 %.2f  median %.2f  p75 %.2f  p90 %.2f  max %.2f"
      % tuple(np.percentile(std_b, [0, 25, 50, 75, 90, 100])))
for q in (4, 6, 8, 10, 12, 16, 20, 24):
    k = int((std_b > 3 * q).sum())
    print(f"q={q:2d}  3q={3*q:3d}  预测可见块数(sigma>3q)={k:3d}  像素={k*W*W}")
