#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
填色留档图：
  1) 分层填色总图（L=4/8/16，floor 档），锯齿像素按模型反解结果红框圈出
  2) L=8 局部放大（最近邻放大，保留像素颗粒），叠加模型圈选与逐像素统计对照
  3) 同一块区域的连续高度灰度对照图
配色为通过 ordinal 校验的单色相绿调 ramp（L=4/8）；L=16 受亮度带物理限制
ΔL 不足 0.06，叠加等值线轮廓作二次编码。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jag_analysis import load_raw, analyze  # noqa: E402

GREEN4 = ["#003a0b", "#2a6736", "#5a9763", "#83c18b"]
GREEN8 = ["#003a0b", "#084c1d", "#205e2e", "#347140",
          "#488452", "#5b9865", "#6fad78", "#83c18b"]
GREEN16 = ["#003a0b", "#004213", "#064b1c", "#135324", "#1e5c2c", "#276434",
           "#316d3c", "#3a7645", "#437f4d", "#4c8856", "#55925e", "#5e9b67",
           "#67a470", "#70ae79", "#79b882", "#83c18b"]
RAMPS = {4: GREEN4, 8: GREEN8, 16: GREEN16}
JAG_RED = (230, 57, 70)
CONTOUR = (12, 12, 12)


def hexrgb(h: str):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def fill_image(qimg: np.ndarray, levels: int) -> Image.Image:
    ramp = [hexrgb(h) for h in RAMPS[levels]]
    rgb = np.zeros((*qimg.shape, 3), dtype=np.uint8)
    for k in range(levels):
        rgb[qimg == k] = ramp[k]
    return Image.fromarray(rgb, "RGB")


def mark_jagged(img: Image.Image, qimg: np.ndarray, scale: int = 1) -> Image.Image:
    """在档位跳变的格边上描红线（锯齿的计数单位就是边）。"""
    img = img.copy()
    d = ImageDraw.Draw(img)
    q = qimg
    h, w = q.shape
    for y in range(h):
        for x in range(w):
            X, Y = x * scale, y * scale
            S = scale
            if x + 1 < w and q[y, x] != q[y, x + 1]:
                d.line([(X + S - 1, Y), (X + S - 1, Y + S - 1)], fill=JAG_RED,
                       width=max(1, scale // 4))
            if y + 1 < h and q[y, x] != q[y + 1, x]:
                d.line([(X, Y + S - 1), (X + S - 1, Y + S - 1)], fill=JAG_RED,
                       width=max(1, scale // 4))
    return img


def mark_jagged_pixels(img: Image.Image, marked: np.ndarray) -> Image.Image:
    """圈像素版本：给模型反解出的锯齿端点像素打红点（小尺寸总图用）。"""
    img = img.copy()
    d = ImageDraw.Draw(img)
    ys, xs = np.where(marked)
    for x, y in zip(xs, ys):
        d.point((x, y), fill=JAG_RED)
    return img


def draw_contours(img: Image.Image, qimg: np.ndarray) -> Image.Image:
    """档位变化处描 1px 深色轮廓（L=16 的二次编码）。"""
    img = img.copy()
    d = ImageDraw.Draw(img)
    q = qimg
    h, w = q.shape
    for y in range(h):
        for x in range(w):
            if x + 1 < w and q[y, x] != q[y, x + 1]:
                d.point((x + 1, y), fill=CONTOUR)
            if y + 1 < h and q[y, x] != q[y + 1, x]:
                d.point((x, y + 1), fill=CONTOUR)
    return img


def upscale(img: Image.Image, factor: int) -> Image.Image:
    return img.resize((img.width * factor, img.height * factor), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--crop", type=int, nargs=4, default=[40, 30, 88, 78],
                    metavar=("X0", "Y0", "X1", "Y1"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    z = load_raw(args.raw)
    x0, y0, x1, y1 = args.crop

    for L in (4, 8, 16):
        info, marked, qimg, _ = analyze(z, L, "floor")
        base = fill_image(qimg, L)
        if L == 16:
            base = draw_contours(base, qimg)
        # 总图：模型反解的锯齿像素用红点表示
        mark_jagged_pixels(base, marked).save(
            os.path.join(args.outdir, f"fill_L{L}_marked.png"))
        base.save(os.path.join(args.outdir, f"fill_L{L}_plain.png"))

        # 局部放大：纯填色 / 锯齿边描红（边即计数单位）
        factor = max(1, 320 // (x1 - x0))
        crop_fill = upscale(base.crop((x0, y0, x1, y1)), factor)
        crop_q = qimg[y0:y1, x0:x1]
        crop_edge = mark_jagged(crop_fill, crop_q, factor)
        crop_fill.save(os.path.join(args.outdir, f"fill_L{L}_crop_zoom.png"))
        crop_edge.save(os.path.join(args.outdir, f"fill_L{L}_crop_zoom_marked.png"))

    # L=8 重点放大：连续灰度对照 + floor/ceil 两档圈选差异
    info_f, mark_f, q_f, _ = analyze(z, 8, "floor")
    info_c, mark_c, q_c, _ = analyze(z, 8, "ceil")
    g8 = (np.clip(z, 0, 1) * 255).astype(np.uint8)
    gray = Image.fromarray(np.dstack([g8, g8, g8]), "RGB").crop((x0, y0, x1, y1))
    factor = max(1, 320 // (x1 - x0))
    upscale(gray, factor).save(os.path.join(args.outdir, "gray_crop_zoom.png"))

    diff = mark_f != mark_c
    n_diff = int(diff.sum())
    img_diff = upscale(fill_image(q_f, 8).crop((x0, y0, x1, y1)), factor)
    d = ImageDraw.Draw(img_diff)
    ys, xs = np.where(diff[y0:y1, x0:x1])
    for x, y in zip(xs, ys):
        d.rectangle([x * factor, y * factor, (x + 1) * factor - 1,
                     (y + 1) * factor - 1], outline=(0, 80, 220), width=2)
    img_diff.save(
        os.path.join(args.outdir, "fill_L8_crop_floor_ceil_diff.png"))

    summary = {
        "crop": [x0, y0, x1, y1], "zoom_factor": factor,
        "floor_ceil_pixel_diff": n_diff,
        "images": sorted(os.listdir(args.outdir)),
    }
    import json
    with open(os.path.join(args.outdir, "fig_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
