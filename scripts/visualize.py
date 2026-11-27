#!/usr/bin/env python3
# 读取 Go 实验产物（PGM + report.json），生成标注对比图与相位扫描图。
# 仅用 PIL，不依赖 matplotlib。
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "comb")
OUT_MONTAGE = os.path.join(ART, "montage.png")
OUT_SWEEP = os.path.join(ART, "phase_sweep.png")


def load_pgm(path):
    with open(path, "rb") as f:
        magic = f.readline().strip()
        assert magic == b"P5", magic
        # 跳过注释
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxv = int(f.readline())
        data = f.read(w * h)
    return Image.frombytes("L", (w, h), data)


def get_font(size):
    for cand in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]:
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def montage(rep):
    imgs = [load_pgm(os.path.join(ART, f"group{i}.pgm")) for i in range(3)]
    n = imgs[0].width
    pad, top, bottom = 28, 96, 36
    width = n * 3 + pad * 4
    height = n + top + bottom
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = get_font(18)
    small = get_font(15)

    titles = [
        ("A: r=4/1, delta=0", "counterexample, 16 teeth"),
        ("B: r=sqrt(2), delta=0", "irrational, 0 teeth"),
        ("C: r=4/3, delta=0", "rational, 6 teeth"),
    ]
    red = (214, 39, 40)
    for i, (img, gj) in enumerate(zip(imgs, rep["groups"])):
        x0 = pad + i * (n + pad)
        y0 = top
        canvas.paste(img.convert("RGB"), (x0, y0))
        draw.rectangle([x0, y0, x0 + n - 1, y0 + n - 1], outline=(120, 120, 120), width=1)
        draw.text((x0, 14), titles[i][0], fill=(20, 20, 20), font=font)
        draw.text((x0, 40), titles[i][1], fill=(90, 90, 90), font=small)
        # 红框标注检出的梳齿中心
        for c in (gj.get("tooth_centers_px") or []):
            cx = x0 + int(round(c))
            draw.line([(cx, y0), (cx, y0 + n - 1)], fill=red, width=1)
        sp = gj.get("spacing_pred_px")
        sp_txt = f"{int(sp)}" if sp else "n/a"
        info = f"teeth {gj['teeth']} | pred {gj['predicted']} | diff {gj['diff']} | d {sp_txt}px"
        draw.text((x0, y0 + n + 10), info, fill=(20, 20, 20), font=small)

    canvas.save(OUT_MONTAGE)
    print("wrote", OUT_MONTAGE)


def sweep_plot(rep):
    pts = rep["phase_sweep_r4"]
    W, H = 900, 420
    ml, mr, mt, mb = 70, 30, 50, 60
    pw, ph = W - ml - mr, H - mt - mb
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = get_font(16)
    small = get_font(13)
    ink = (33, 37, 41)
    grid = (220, 220, 220)
    blue = (38, 99, 176)
    red = (214, 39, 40)

    xmax = 360
    ymax = max(p["teeth"] for p in pts) + 2
    # 网格与轴
    for gy in range(0, ymax + 1, 4):
        y = mt + ph - int(gy / ymax * ph)
        d.line([(ml, y), (ml + pw, y)], fill=grid)
        d.text((ml - 34, y - 8), str(gy), fill=ink, font=small)
    for gx in range(0, 361, 45):
        x = ml + int(gx / xmax * pw)
        d.line([(x, mt), (x, mt + ph)], fill=grid)
        d.text((x - 12, mt + ph + 8), f"{gx}", fill=ink, font=small)
    d.line([(ml, mt), (ml, mt + ph)], fill=ink, width=2)
    d.line([(ml, mt + ph), (ml + pw, mt + ph)], fill=ink, width=2)

    # 齿数阶梯线
    prev = None
    for p in pts:
        x = ml + int(p["delta_deg"] / xmax * pw)
        y = mt + ph - int(p["teeth"] / ymax * ph)
        if prev:
            d.line([prev, (x, prev[1])], fill=blue, width=2)
            d.line([(x, prev[1]), (x, y)], fill=blue, width=2)
        prev = (x, y)
    # 标出出齿点
    for p in pts:
        if p["teeth"] > 0:
            x = ml + int(p["delta_deg"] / xmax * pw)
            y = mt + ph - int(p["teeth"] / ymax * ph)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=red)

    d.text((ml, 14), "phase sweep r=4/1: teeth vs global phase delta", fill=ink, font=font)
    d.text((ml + pw - 250, mt + 10), "red dots: teeth detected (1deg tol)", fill=red, font=small)
    d.text((W // 2 - 40, H - 24), "delta (degrees)", fill=ink, font=font)
    d.text((12, mt + ph // 2), "teeth", fill=ink, font=font)
    img.save(OUT_SWEEP)
    print("wrote", OUT_SWEEP)


def main():
    with open(os.path.join(ART, "report.json")) as f:
        rep = json.load(f)
    montage(rep)
    sweep_plot(rep)


if __name__ == "__main__":
    sys.exit(main())
