#!/usr/bin/env python3
"""目测辅助：把 128 行分 4 条带输出，每行一条，红色=算法判定糊段；另出最糊行放大图。"""
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "analysis")
from contour_blur import read_pgm, analyze

gray = read_pgm(sys.argv[1])
interval = int(sys.argv[2])
tag = sys.argv[3]
outdir = sys.argv[4]
r = analyze(gray, interval)
img, blur = r["image"], r["blur_mask"]
h, w = img.shape

ROW_H = 26
LABEL_W = 52
SX = 4
for part in range(4):
    rows = range(part * 32, min((part + 1) * 32, h))
    canvas = Image.new("RGB", (LABEL_W + w * SX, 32 * ROW_H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, y in enumerate(rows):
        strip = np.full((ROW_H, w, 3), 255, np.uint8)
        strip[:, img[y]] = (0, 0, 0)
        strip[:, blur[y]] = (220, 30, 30)
        im = Image.fromarray(strip).resize((w * SX, ROW_H), Image.NEAREST)
        canvas.paste(im, (LABEL_W, i * ROW_H))
        color = (200, 0, 0) if y == r["worst_row"] else (0, 0, 180)
        draw.text((2, i * ROW_H + 6), f"r{y:<3d}", fill=color)
    canvas.save(f"{outdir}/{tag}_strip{part}.png")

# 最糊行：12 倍高放大，标像素坐标
y = r["worst_row"]
strip = np.full((1, w, 3), 255, np.uint8)
strip[0, :][img[y]] = (0, 0, 0)
strip[0, :][blur[y]] = (220, 30, 30)
big = Image.fromarray(strip).resize((w * 8, 40), Image.NEAREST)
big.save(f"{outdir}/{tag}_worstrow.png")
print("worst row", y, "runs:", r["per_row_runs"][y])
print("blur pixels on worst row:", int(r["blur_pixels"][y]))
