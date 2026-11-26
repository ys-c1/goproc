#!/usr/bin/env python3
"""
contour_blur.py — 等高线陡坡糊线分析

流程（严格按任务约定）：
  1. 读取 goproc 生成的 8 位 PGM 高度图（兼容本项目 hmap.go 产出的字面量 \\n 头）。
  2. 对给定等高距，把每条等高线穿越网格边的两个端点置黑，得到标准 1px 二值等高线图；
     陡坡处相邻等高线在栅格化时自然并线，表现为更宽的黑色游程。
  3. 逐行提取黑色游程，以相邻游程的中心距为"线距"，游程像素数为"线宽"。
  4. 最糊行 = 线距最小的行（并列取靠上的行），在该行取最紧游程对的 线距/线宽 作为
     糊判据阈值 T。阈值只由最糊行反解，不遍历全行试参。
  5. 拿 T 回推全图：每行相邻游程对若 线距/均宽 < T，则该对覆盖范围记为糊；按连续像素
     统计每行糊段像素数。
  6. 输出二值等高线图（PGM 留档 + 放大 PNG 供目测）、逐行 CSV、JSON 汇总。

用法：
  python3 -I contour_blur.py <heightmap.pgm> --interval 16 --outdir <dir> --tag c16
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from PIL import Image


def read_pgm(path: str) -> np.ndarray:
    raw = open(path, "rb").read()
    pos = raw.find(b"255")
    if pos < 0:
        raise ValueError("非 8 位 PGM，或头格式异常")
    header = raw[: pos + 3].replace(b"\\n", b"\n")
    parts = header.split()
    if parts[0] != b"P5":
        raise ValueError("仅支持 P5 二进制 PGM")
    w, h = int(parts[1]), int(parts[2])
    data_start = pos + 3
    if raw[data_start : data_start + 2] == b"\\n":  # 本项目 EncodePGM 的字面量 \n
        data_start += 2
    else:
        data_start += 1
    buf = raw[data_start : data_start + w * h]
    if len(buf) < w * h:
        raise ValueError("像素数据不完整")
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w).copy()


def render_contours(gray: np.ndarray, interval: int) -> np.ndarray:
    """标准 1px 等高线：level 穿越网格边时，把边的两个端点置黑。"""
    h, w = gray.shape
    img = np.zeros((h, w), dtype=bool)
    g = gray.astype(np.float64)
    for lv in np.arange(interval, 256, interval):
        s = np.sign(g - lv)
        s[s == 0] = 1.0  # 等值点归入一侧，避免零值被当成两边穿越
        fh = np.diff(s, axis=1) != 0  # 水平网格边（列间）
        fv = np.diff(s, axis=0) != 0  # 垂直网格边（行间）
        ys, xs = np.where(fh)
        img[ys, xs] = True
        img[ys, xs + 1] = True
        ys, xs = np.where(fv)
        img[ys, xs] = True
        img[ys + 1, xs] = True
    return img


def row_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """一行内黑色连续游程 [(start, end_inclusive), ...]。"""
    out: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def analyze(gray: np.ndarray, interval: int) -> dict:
    h, w = gray.shape
    img = render_contours(gray, interval)

    # ---- 逐行游程、线距、线宽 ----
    per_row_runs: list[list[tuple[int, int]]] = []
    min_pitch = np.full(h, np.inf)
    min_width = np.full(h, np.inf)
    for y in range(h):
        rr = row_runs(img[y])
        per_row_runs.append(rr)
        if len(rr) >= 2:
            centers = np.array([(a + b) / 2.0 for a, b in rr])
            widths = np.array([b - a + 1 for a, b in rr], dtype=np.float64)
            pitches = np.diff(centers)
            k = int(np.argmin(pitches))  # argmin 天然并列取靠上/靠前
            min_pitch[y] = pitches[k]
            min_width[y] = (widths[k] + widths[k + 1]) / 2.0

    # ---- 最糊行：线距最小，并列取靠上（argmin 对从上到下的扫描稳定）----
    valid = np.isfinite(min_pitch)
    worst_row = int(np.argmin(np.where(valid, min_pitch, np.inf)))
    worst_pitch = float(min_pitch[worst_row])
    worst_width = float(min_width[worst_row])
    threshold = worst_pitch / worst_width  # 由最糊行反解的唯一阈值

    # ---- 阈值回推全图：逐行判定糊游程对，按连续像素计糊段 ----
    blur_mask = np.zeros((h, w), dtype=bool)
    blur_pixels = np.zeros(h, dtype=int)
    blur_segments = np.zeros(h, dtype=int)
    pair_ratios = np.full(h, np.inf)
    for y in range(h):
        rr = per_row_runs[y]
        if len(rr) < 2:
            continue
        centers = np.array([(a + b) / 2.0 for a, b in rr])
        widths = np.array([b - a + 1 for a, b in rr], dtype=np.float64)
        pitches = np.diff(centers)
        ratios = pitches / ((widths[:-1] + widths[1:]) / 2.0)
        pair_ratios[y] = ratios.min()
        row_blur = np.zeros(w, dtype=bool)
        for k in range(len(pitches)):
            if ratios[k] < threshold:
                row_blur[rr[k][0] : rr[k + 1][1] + 1] = True
        if row_blur.any():
            blur_mask[y] = row_blur
            blur_pixels[y] = int(row_blur.sum())
            blur_segments[y] = len(row_runs(row_blur))

    total_blur = int(blur_pixels.sum())
    total_ink = int(img.sum())
    return {
        "interval": interval,
        "size": h,
        "image": img,
        "blur_mask": blur_mask,
        "per_row_runs": per_row_runs,
        "min_pitch": min_pitch,
        "min_width": min_width,
        "pair_ratios": pair_ratios,
        "blur_pixels": blur_pixels,
        "blur_segments": blur_segments,
        "worst_row": worst_row,
        "worst_pitch_px": worst_pitch,
        "worst_width_px": worst_width,
        "threshold": threshold,
        "total_blur_pixels": total_blur,
        "total_ink_pixels": total_ink,
        "rows_with_blur": int(np.count_nonzero(blur_pixels)),
    }


def save_pgm_bool(path: str, mask: np.ndarray) -> None:
    h, w = mask.shape
    out = np.full((h, w), 255, dtype=np.uint8)
    out[mask] = 0
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(out.tobytes())


def save_overlay_png(path: str, img: np.ndarray, blur: np.ndarray, scale: int = 4) -> None:
    """白底、黑等高线、红糊段，最近邻放大便于目测。"""
    h, w = img.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    rgb[img] = (0, 0, 0)
    rgb[blur] = (220, 30, 30)
    im = Image.fromarray(rgb).resize((w * scale, h * scale), Image.NEAREST)
    im.save(path)


def save_row_strip_png(path: str, img: np.ndarray, blur: np.ndarray, rows, scale_x: int = 4,
                       row_h: int = 14) -> None:
    """把指定行逐条铺开的目测条带：每行一条，标注行号。"""
    w = img.shape[1]
    strips = []
    from PIL import ImageDraw
    for y in rows:
        strip = np.full((row_h, w, 3), 255, dtype=np.uint8)
        line = img[y]
        bl = blur[y]
        strip[line] = (0, 0, 0)
        strip[bl] = (220, 30, 30)
        strips.append((y, strip))
    label_w = 46
    canvas = Image.new("RGB", (label_w + w * scale_x, len(strips) * row_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (y, strip) in enumerate(strips):
        im = Image.fromarray(strip).resize((w * scale_x, row_h), Image.NEAREST)
        canvas.paste(im, (label_w, i * row_h))
        draw.text((2, i * row_h), f"r{y}", fill=(0, 0, 200))
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgm")
    ap.add_argument("--interval", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gray = read_pgm(args.pgm)
    r = analyze(gray, args.interval)

    base = os.path.join(args.outdir, args.tag)
    save_pgm_bool(base + "_contours.pgm", r["image"])
    save_overlay_png(base + "_overlay.png", r["image"], r["blur_mask"])

    # 逐行 CSV
    with open(base + "_rows.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["row", "n_runs", "min_pitch_px", "min_run_width_px",
                     "min_pair_ratio", "blur_pixels", "blur_segments", "is_worst_row"])
        for y in range(r["size"]):
            wr.writerow([
                y, len(r["per_row_runs"][y]),
                "" if not np.isfinite(r["min_pitch"][y]) else round(float(r["min_pitch"][y]), 4),
                "" if not np.isfinite(r["min_width"][y]) else round(float(r["min_width"][y]), 4),
                "" if not np.isfinite(r["pair_ratios"][y]) else round(float(r["pair_ratios"][y]), 4),
                int(r["blur_pixels"][y]), int(r["blur_segments"][y]),
                int(y == r["worst_row"]),
            ])

    summary = {
        "interval": r["interval"],
        "worst_row": r["worst_row"],
        "worst_pitch_px": r["worst_pitch_px"],
        "worst_width_px": r["worst_width_px"],
        "threshold_pitch_over_width": round(r["threshold"], 6),
        "total_blur_pixels": r["total_blur_pixels"],
        "total_ink_pixels": r["total_ink_pixels"],
        "rows_with_blur": r["rows_with_blur"],
    }
    with open(base + "_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
