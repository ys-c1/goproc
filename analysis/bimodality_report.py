#!/usr/bin/env python3
"""Bimodality table and histogram figures for goproc octave sweep.

The second difference is always the centered, one-gray-bin formula
D2[g] = H[g-1] - 2 H[g] + H[g+1].  A fixed Gaussian pre-smoother (sigma=10
gray bins) is used only to suppress multinomial counting noise; it does not
change the one-bin step of D2.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "artifacts" / "grid_s128_seed7"
ART = ROOT / "artifacts"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#d03b3b"
INK = "#0b0b0b"
MUTED = "#6f6d67"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
BAR = "#d7d5cd"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def read_pgm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    m = re.match(rb"P5(?:\\n|\s)+(\d+)(?:\\n|\s)+(\d+)(?:\\n|\s)+(\d+)(?:\\n|\s)", raw)
    if not m:
        raise ValueError(path)
    w, h, _ = map(int, m.groups())
    i = m.end()
    return np.frombuffer(raw[i:i + w * h], dtype=np.uint8).reshape(h, w)


def gaussian_smooth(c: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    r = int(4 * sigma + 0.5)
    x = np.arange(-r, r + 1)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    return np.convolve(np.pad(c, r, mode="edge"), k, mode="valid")


def d2_center(c: np.ndarray) -> np.ndarray:
    """Return D2[g] for gray g=1..254 at index g-1."""
    return c[:-2] - 2.0 * c[1:-1] + c[2:]


def sign_changes(d2: np.ndarray) -> int:
    s = np.sign(d2).astype(int)
    nz = s[s != 0]
    return int(np.sum(nz[1:] * nz[:-1] < 0)) if nz.size else 0


def major_peaks(cs: np.ndarray, prom_frac: float = 0.15, min_dist: int = 25):
    cand = [g for g in range(3, 253)
            if cs[g] >= cs[g - 1] and cs[g] > cs[g + 1]]
    picked: list[int] = []
    for g in sorted(cand, key=lambda z: -cs[z]):
        if all(abs(g - q) >= min_dist for q in picked):
            lo, hi = max(0, g - 30), min(256, g + 31)
            if cs[g] - cs[lo:hi].min() >= prom_frac * cs.max():
                picked.append(g)
    return sorted(picked)


def params(gain: float, k: int):
    q = gain * gain
    keff = (1 + q) / (1 - q) * (1 - q ** k) / (1 + q ** k)
    w0 = (1 - gain) / (1 - gain ** k)
    return keff, w0


def load_ranges():
    path = ART / "ranges_seed7_s128.txt"
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            g = float(re.search(r"g=([0-9.]+)", line).group(1))
            k = int(re.search(r"k=(\d+)", line).group(1))
            r = float(re.search(r"range=([0-9.]+)", line).group(1))
            out[(round(g, 6), k)] = r
    return out


def collect():
    ranges = load_ranges()
    groups = [
        ("beta5", 5.02, 0.3535533905932738),
        ("beta3", 3.03, 0.7071067811865476),
    ]
    rows = []
    data = {}
    for gid, beta, gain in groups:
        data[gid] = []
        r1 = ranges.get((round(gain, 6), 1))
        for k in range(1, 13):
            p = next(GRID.glob(f"*_g{gain:.6f}_o{k}.pgm"))
            h = read_pgm(p)
            raw = np.bincount(h.ravel(), minlength=256).astype(float)
            sm = gaussian_smooth(raw)
            vr = sign_changes(d2_center(raw))
            vs = sign_changes(d2_center(sm))
            pk = major_peaks(sm)
            gap = pk[1] - pk[0] if len(pk) == 2 else ""
            keff, w0 = params(gain, k)
            pred_n = 2 if keff < 2.0 else 1
            # Calibrated only by the K=1 gray gap; sqrt(p0)=w0 sqrt(Keff).
            pred_gap_spec = int(round(83 * w0 * math.sqrt(keff))) if pred_n == 2 else ""
            rk = ranges.get((round(gain, 6), k))
            pred_gap_r = round(83 * w0 * r1 / rk, 1) if pred_n == 2 and r1 and rk else ""
            rows.append({
                "group": gid, "beta2_hat": beta, "gain": gain, "K": k,
                "Keff": round(keff, 3), "pred_n": pred_n,
                "pred_gap_betaK": pred_gap_spec,
                "pred_gap_range_corrected": pred_gap_r,
                "Vraw": vr, "Vsmooth_sigma10": vs,
                "meas_n": len(pk), "peaks": ";".join(map(str, pk)),
                "meas_gap": gap,
            })
            data[gid].append((k, raw, sm, pk, vr, vs))
    return rows, data


def write_csv(rows):
    out = ART / "bimodality_comparison.csv"
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out


def draw_axes(d, box, ymax, xticks=True, ylabel=""):
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=SURFACE)
    for j in range(5):
        y = y1 - j * (y1 - y0) / 4
        d.line([x0, y, x1, y], fill=GRIDLINE, width=1)
        d.text((x0 - 52, y - 10), f"{int(round(j*ymax/4))}", font=font(15), fill=MUTED)
    d.line([x0, y0, x0, y1], fill=MUTED, width=2)
    d.line([x0, y1, x1, y1], fill=MUTED, width=2)
    for g in [0, 32, 64, 96, 128, 160, 192, 224, 255]:
        x = x0 + g / 255 * (x1 - x0)
        d.line([x, y1, x, y1 + 6], fill=MUTED, width=1)
        if xticks:
            d.text((x - 12, y1 + 10), str(g), font=font(15), fill=MUTED)
    if ylabel:
        d.text((24, y0 - 8), ylabel, font=font(17), fill=INK)


def xg(g, box):
    x0, _, x1, _ = box
    return x0 + g / 255 * (x1 - x0)


def yg(v, ymax, box):
    _, y0, _, y1 = box
    return y1 - max(0, min(v, ymax)) / ymax * (y1 - y0)


def make_single_figure(gid, gain, k, raw, sm, pk, vr, vs, path: Path):
    W, H = 1500, 980
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    beta = 5 if gid == "beta5" else 3
    title = f"2-D spectral slope beta2 ~ {beta}: gain={gain:.6f}, octaves K={k}, seed=7, size=128"
    d.text((50, 34), title, font=font(30, True), fill=INK)
    d.text((50, 76), "Gray histogram (bars), denoised histogram (blue), major curvature modes (orange)", font=font(19), fill=MUTED)

    top = (105, 145, 1440, 560)
    ymax = max(raw.max(), sm.max() * 1.08)
    draw_axes(d, top, ymax, xticks=True, ylabel="")
    d.text((112, 112), "pixel count per gray bin", font=font(18), fill=INK)
    x0, y0, x1, y1 = top
    bw = max(2, (x1 - x0) / 256 * 0.72)
    for g, c in enumerate(raw):
        x = xg(g, top)
        y = yg(c, ymax, top)
        d.rectangle([x - bw / 2, y, x + bw / 2, y1], fill=BAR)
    pts = [(xg(g, top), yg(sm[g], ymax, top)) for g in range(256)]
    d.line(pts, fill=BLUE, width=4, joint="curve")

    for g in pk:
        x = xg(g, top)
        for yy in range(y0 + 10, y1, 14):
            d.line([x, yy, x, min(yy + 7, y1)], fill=ORANGE, width=3)
        yp = yg(sm[g], ymax, top)
        d.ellipse([x - 9, yp - 9, x + 9, yp + 9], fill=ORANGE, outline="white", width=3)
        d.text((x - 18, max(y0 + 4, yp - 38)), str(g), font=font(20, True), fill=ORANGE)

    if len(pk) == 2:
        ga, gb = pk
        xa, xb = xg(ga, top), xg(gb, top)
        ya = y0 + 42
        d.line([xa + 12, ya, xb - 12, ya], fill=INK, width=3)
        d.polygon([(xa, ya), (xa + 14, ya - 8), (xa + 14, ya + 8)], fill=INK)
        d.polygon([(xb, ya), (xb - 14, ya - 8), (xb - 14, ya + 8)], fill=INK)
        d.text(((xa + xb) / 2 - 145, ya - 34), f"peak spacing = {gb-ga} gray levels", font=font(22, True), fill=INK)
    else:
        d.text((x0 + 24, y0 + 28), "one major mode", font=font(22, True), fill=INK)
    d.text((x1 - 250, y1 + 42), "gray level g", font=font(18), fill=MUTED)

    bot = (105, 720, 1440, 875)
    d.text((105, 650), "Centered second difference, one-bin step:  D2[g] = H[g-1] - 2H[g] + H[g+1]", font=font(22, True), fill=INK)
    d2 = d2_center(sm)
    box = bot
    x0b, y0b, x1b, y1b = box
    d.rectangle(box, outline=GRIDLINE, width=2, fill="white")
    m = max(float(np.max(np.abs(d2))), 1e-9)
    zy = (y0b + y1b) / 2
    d.line([x0b, zy, x1b, zy], fill=MUTED, width=2)
    pts = []
    for j, v in enumerate(d2):
        g = j + 1
        x = xg(g, box)
        y = zy - (v / m) * (y1b - y0b) * 0.42
        pts.append((x, y))
    d.line(pts, fill="#52514e", width=3)
    for g in pk:
        j = g - 1
        x = xg(g, box)
        y = zy - (d2[j] / m) * (y1b - y0b) * 0.42
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=ORANGE, outline="white", width=2)
    for g in [0, 64, 128, 192, 255]:
        x = xg(g, box)
        d.text((x - 12, y1b + 8), str(g), font=font(15), fill=MUTED)
    d.text((x0b, y0b - 28), "D2 > 0 convex / valley-side      D2 < 0 concave / mode-side", font=font(17), fill=MUTED)
    d.text((105, 910), f"Raw-bin sign changes Vraw={vr} (sampling noise); after fixed sigma=10 bin denoising V={vs}; major modes={len(pk)}.", font=font(18), fill=INK)
    img.save(path)


def make_sheet(gid, gain, series, path: Path):
    cols, rowsn = 3, 4
    cw, ch = 500, 300
    W, H = cols * cw + 30, rowsn * ch + 90
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    beta = 5 if gid == "beta5" else 3
    d.text((20, 20), f"beta2~{beta}, gain={gain:.6f}: octaves K=1..12 (seed 7, size 128)", font=font(28, True), fill=INK)
    for idx, (k, raw, sm, pk, vr, vs) in enumerate(series):
        r, c = divmod(idx, cols)
        ox, oy = 15 + c * cw, 75 + r * ch
        box = (ox + 45, oy + 35, ox + cw - 20, oy + ch - 45)
        x0, y0, x1, y1 = box
        d.text((ox + 8, oy + 4), f"K={k}  n={len(pk)}" + (f"  Delta={pk[1]-pk[0]}" if len(pk) == 2 else ""), font=font(19, True), fill=INK)
        ymax = max(raw.max(), sm.max() * 1.1)
        d.line([x0, y0, x0, y1], fill=MUTED, width=1)
        d.line([x0, y1, x1, y1], fill=MUTED, width=1)
        for g, v in enumerate(raw):
            x = xg(g, box); y = yg(v, ymax, box)
            d.line([x, y1, x, y], fill=BAR, width=1)
        d.line([(xg(g, box), yg(sm[g], ymax, box)) for g in range(256)], fill=BLUE, width=2)
        for g in pk:
            x = xg(g, box); y = yg(sm[g], ymax, box)
            d.line([x, y0, x, y1], fill=ORANGE, width=1)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=ORANGE)
        d.text((x0, y1 + 8), "0", font=font(13), fill=MUTED)
        d.text((x1 - 24, y1 + 8), "255", font=font(13), fill=MUTED)
    img.save(path)


def main():
    rows, data = collect()
    csv_path = write_csv(rows)
    for gid, gain, k in [("beta5", 0.3535533905932738, 6), ("beta3", 0.7071067811865476, 6)]:
        item = next(x for x in data[gid] if x[0] == k)
        make_single_figure(gid, gain, *item, ART / f"hist_{gid}_K6.png")
        make_sheet(gid, gain, data[gid], ART / f"hist_sheet_{gid}_K1_K12.png")
    print(csv_path)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
