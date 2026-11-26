#!/usr/bin/env python3
import re
import sys
from pathlib import Path
import numpy as np


def read_pgm(path):
    raw = Path(path).read_bytes()
    m = re.match(rb"P5(?:\\n|\s)+(\d+)(?:\\n|\s)+(\d+)(?:\\n|\s)+(\d+)(?:\\n|\s)", raw)
    w, h, _ = map(int, m.groups())
    i = m.end()
    return np.frombuffer(raw[i:i+w*h], dtype=np.uint8).reshape(h, w).astype(float)


def slope(a):
    n = a.shape[0]
    y = a - a.mean()
    wy = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    F = np.fft.fftshift(np.fft.fft2(y * wy))
    P = np.abs(F)**2
    cy = cx = n // 2
    yy, xx = np.indices((n, n))
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2).astype(int)
    bins = np.arange(1, n//2)
    vals = []
    freqs = []
    for rr in bins:
        msk = r == rr
        if msk.any():
            vals.append(P[msk].mean())
            freqs.append(rr / n)
    f = np.array(freqs); p = np.array(vals)
    # fit middle scales: wavelengths 4..64 pixels, avoids Hann/DC and pixel floor
    mask = (f >= 1/64) & (f <= 1/4)
    beta, intercept = np.polyfit(np.log(f[mask]), np.log(p[mask]), 1)
    return -beta

for arg in sys.argv[1:]:
    for path in sorted(Path(arg).glob("*.pgm")):
        g = float(path.stem.split("_g")[1].split("_")[0])
        k = int(path.stem.split("_o")[1])
        b = slope(read_pgm(path))
        print(f"{path.name}\tgain={g:.6f}\tK={k}\tbeta_hat={b:.3f}")
