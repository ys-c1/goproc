#!/usr/bin/env python3
import re
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_pgm(path):
    raw = Path(path).read_bytes()
    m = re.match(rb"P5(?:\\n|\s)+(\d+)(?:\\n|\s)+(\d+)(?:\\n|\s)+(\d+)(?:\\n|\s)", raw)
    w, h, _ = map(int, m.groups())
    i = m.end()
    return np.frombuffer(raw[i:i+w*h], dtype=np.uint8).reshape(h, w)


def smooth(c, sigma=2.5):
    r = int(4*sigma + .5)
    x = np.arange(-r, r+1)
    k = np.exp(-x*x/(2*sigma*sigma)); k /= k.sum()
    return np.convolve(np.pad(c, r, mode="edge"), k, mode="valid")

indir = Path(sys.argv[1]); out = Path(sys.argv[2])
files = sorted(indir.glob("*.pgm"))
gains = []
for p in files:
    g = float(p.stem.split("_g")[1].split("_")[0])
    if g not in gains: gains.append(g)
cellw, cellh = 360, 240
labelh = 28
W, H = cellw*8, (cellh+labelh)*len(gains)
img = Image.new("RGB", (W,H), "white")
d = ImageDraw.Draw(img)
font = ImageFont.load_default()
for row,g in enumerate(gains):
    for octv in range(1,9):
        matches = list(indir.glob(f"*_g{g:.6f}_o{octv}.pgm"))
        p = matches[0]
        a = read_pgm(p); c = np.bincount(a.ravel(), minlength=256).astype(float)
        cs = smooth(c)
        x0=(octv-1)*cellw; y0=row*(cellh+labelh)
        d.rectangle([x0,y0,x0+cellw-1,y0+cellh+labelh-1], outline="#cccccc")
        d.text((x0+8,y0+6), f"gain={g:.4f} K={octv}", fill="black", font=font)
        L,T,R,B = x0+34,y0+labelh+10,x0+cellw-12,y0+cellh-24
        d.line([L,B,R,B], fill="#333")
        d.line([L,T,L,B], fill="#333")
        cmax = max(c.max(),1)
        for x in range(256):
            xx=L+x/255*(R-L)
            bh=(c[x]/cmax)*(B-T)
            d.line([xx,B,xx,B-bh], fill=(220,220,220))
        ymax=max(cs.max(),1)
        pts=[]
        for x,v in enumerate(cs):
            xx=L+x/255*(R-L); yy=B-(v/ymax)*(B-T)
            pts.append((xx,yy))
        d.line(pts, fill=(20,80,180), width=2)
        d.text((L,B+4), "0", fill="#333", font=font)
        d.text((R-18,B+4), "255", fill="#333", font=font)
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out)
print(out)
