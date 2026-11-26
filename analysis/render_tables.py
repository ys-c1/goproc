#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对账汇总图（纯 PIL）：每个色阶边界上模型反解数 vs 逐像素统计数。"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BLUE = (42, 120, 214)
ORANGE = (235, 104, 52)
INK = (20, 20, 20)
GRID = (220, 220, 220)
SURF = (252, 252, 251)


def font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def panel(draw: ImageDraw.ImageDraw, ox, oy, w, h, run, title, f_small, f_tiny):
    rows = run["per_boundary"]
    vmax = max(max(r["model"], r["pixel_stat"]) for r in rows) * 1.15
    # 边框与网格
    draw.rectangle([ox, oy, ox + w, oy + h], outline=(180, 180, 180))
    for g in range(1, 4):
        gy = oy + h - h * g / 4
        draw.line([(ox, gy), (ox + w, gy)], fill=GRID)
    nb = len(rows)
    slot = w / nb
    bw = slot * 0.32
    for i, r in enumerate(rows):
        cx = ox + slot * (i + 0.5)
        hm = h * r["model"] / vmax
        hs = h * r["pixel_stat"] / vmax
        draw.rectangle([cx - bw - 1, oy + h - hm, cx - 1, oy + h], fill=BLUE)
        draw.rectangle([cx + 1, oy + h - hs, cx + bw + 1, oy + h], fill=ORANGE)
        draw.text((cx - 8, oy + h + 3), f"k{r['boundary']}", font=f_tiny, fill=INK)
    maxdev = max(r["rel_dev"] for r in rows)
    draw.text((ox + 4, oy + 4), f"{title}  maxdev={maxdev:.3f}",
              font=f_small, fill=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = json.load(open(args.report, encoding="utf-8"))
    runs = report["runs"]
    W, H = 1500, 800
    img = Image.new("RGB", (W, H), SURF)
    d = ImageDraw.Draw(img)
    f_title, f_small, f_tiny = font(22), font(15), font(11)
    d.text((20, 12), "Sign-rule inversion (blue) vs per-pixel statistics (orange): "
                    f"{os.path.basename(args.report)}", font=f_title, fill=INK)
    pw, ph, gap = 460, 300, 30
    for col, L in enumerate((4, 8, 16)):
        for row, mode in enumerate(("floor", "ceil")):
            run = next(r for r in runs if r["levels"] == L and r["mode"] == mode)
            ox = 20 + col * (pw + gap)
            oy = 60 + row * (ph + 70)
            panel(d, ox, oy, pw, ph, run, f"L={L} {mode}", f_small, f_tiny)
    # 图例（置于标题右侧，避免压坐标）
    d.rectangle([1080, 18, 1098, 34], fill=BLUE)
    d.text((1104, 16), "model: same-sign rule inversion", font=f_small, fill=INK)
    d.rectangle([1080, 40, 1098, 56], fill=ORANGE)
    d.text((1104, 38), "pixel statistics: quantized neighbor crossing",
           font=f_small, fill=INK)
    img.save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
