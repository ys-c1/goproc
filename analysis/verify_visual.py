#!/usr/bin/env python3
"""
verify_visual.py — 逐行目测复核（任务比值阈值 T 为唯一判据，做几何级人工核对）。

任务规定：阈值 T 由最糊行反解，全图统一套用，不另造判据。本脚本把算法判定
逐条游程对拆开，输出可供肉眼核对的几何量，并做两类边界检查：
  (1) 一致性检查：ratio<T 必判糊、ratio>=T 必不判（期望 0 错误）；
  (2) 边界检查：白隙<=1 的最紧对里，未判糊的是否仅为 ratio 恰好等于 T 的边界对。
另生成 discrepancy 图（红=算法糊，蓝=ratio==T 的等比例边界对）与逐行/逐对 CSV。
"""
import csv
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "analysis")
from contour_blur import read_pgm, render_contours, row_runs, analyze

pgm, interval, tag, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
gray = read_pgm(pgm)
h, w = gray.shape
img = render_contours(gray, interval)
r = analyze(gray, interval)
T = r["threshold"]
EPS = 1e-9

boundary_mask = np.zeros((h, w), bool)
pair_log = []
true_miss = true_false = 0
for y in range(h):
    rr = row_runs(img[y])
    if len(rr) < 2:
        continue
    centers = np.array([(a + b) / 2 for a, b in rr])
    widths = np.array([b - a + 1 for a, b in rr], float)
    pitches = np.diff(centers)
    ratios = pitches / ((widths[:-1] + widths[1:]) / 2)
    gaps = np.array([rr[k + 1][0] - rr[k][1] - 1 for k in range(len(rr) - 1)])
    algo_blur = r["blur_mask"][y]
    for k in range(len(pitches)):
        covered = bool(algo_blur[rr[k][0]:rr[k + 1][1] + 1].all())
        ratio = float(ratios[k])
        pair_log.append([y, k, int(widths[k]), int(widths[k + 1]), int(gaps[k]),
                         round(ratio, 4), covered])
        if ratio < T - EPS and not covered:
            true_miss += 1
        if ratio >= T - EPS and covered:
            true_false += 1
        # 等 T 边界对（白隙 1、ratio 恰为 T），未判糊是 <T 规则的预期行为
        if abs(ratio - T) <= EPS and not covered:
            boundary_mask[y, rr[k][0]:rr[k + 1][1] + 1] = True

n_pairs = len(pair_log)
n_blur = sum(1 for x in pair_log if x[6])
gap_dist: dict[int, int] = {}
for x in pair_log:
    gap_dist[x[4]] = gap_dist.get(x[4], 0) + 1

print(f"=== {tag} interval={interval} T={T:.4f} ===")
print(f"相邻游程对总数={n_pairs} 判糊对数={n_blur}")
print(f"白隙分布(隙px:对数)={dict(sorted(gap_dist.items()))}")
print(f"一致性检查: 真漏标(ratio<T未判)={true_miss}  真误标(ratio>=T却判)={true_false}")
print(f"等T边界对像素(ratio==T,按<T不判,预期保留)={int(boundary_mask.sum())}")

# 逐行：算法糊像素 + 等T边界像素，供与条带目测对照
with open(f"{outdir}/{tag}_visual_audit.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["row", "algo_blur_px", "boundary_ratio_eq_T_px", "blur_segments"])
    bpx = boundary_mask.sum(axis=1)
    for y in range(h):
        wr.writerow([y, int(r["blur_pixels"][y]), int(bpx[y]), int(r["blur_segments"][y])])
with open(f"{outdir}/{tag}_pair_log.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["row", "pair_idx", "width_a", "width_b", "white_gap_px",
                 "pitch_over_width", "judged_blur"])
    wr.writerows(pair_log)

# 核对图：黑=等高线 红=算法糊 蓝=等T边界对
rgb = np.full((h, w, 3), 255, np.uint8)
rgb[img] = (0, 0, 0)
rgb[r["blur_mask"]] = (220, 30, 30)
rgb[boundary_mask] = (30, 80, 220)
Image.fromarray(rgb).resize((w * 4, h * 4), Image.NEAREST).save(
    f"{outdir}/{tag}_discrepancy.png")
