"""Faithful Python port of goproc/internal/noise + fbm + hmap.

Kept line-for-line comparable to the Go sources so analysis reflects the
shipped generator.  Two table modes:
  - shared:   every octave uses the same perm (the reported bug)
  - layered:  octave i gets its own perm derived from (seed, i)
"""
from __future__ import annotations
import numpy as np

MASK511 = 0x1FF
MASK255 = 0xFF


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def make_perm(seed: int) -> np.ndarray:
    """Returns the doubled [512]uint8 permutation exactly like noise.New."""
    p = list(range(256))
    state = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(255, 0, -1):
        state = splitmix64(state)
        j = int(state % (i + 1))
        p[i], p[j] = p[j], p[i]
    perm = np.empty(512, dtype=np.int64)
    for i in range(512):
        perm[i] = p[i & 255]
    return perm


def derive_layer_seed(seed: int, layer: int) -> int:
    """Mirror of noise.LayerSeed: splitmix64 chain + domain constants."""
    mask = 0xFFFFFFFFFFFFFFFF
    c1 = 0x9E3779B97F4A7C15
    c2 = 0x243F6A8885A308D3
    h = splitmix64((seed + c1) & mask)
    h ^= splitmix64((layer + c2) & mask)
    return splitmix64(h)


def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


P = 256  # permutation period in lattice cells


def hash_grid(perm: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """H(X,Y) exactly as (*Source).grad, broadcasting integer arrays."""
    inner = perm[X & MASK255]
    idx = (inner + Y) & MASK511
    h = perm[idx]
    return h.astype(np.float64) / 127.5 - 1.0


def _floor(v):
    # Go-ish floor for the ranges we use; np.floor is equivalent here.
    return int(np.floor(v))


def value_grid(perm: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Evaluate Value on full coordinate arrays (same scalar formula)."""
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = xs - x0
    fy = ys - y0
    u = fade(fx)
    v = fade(fy)

    def grad(ix, iy):
        # h := perm[(int(perm[ix&255]) + iy) & 511]
        inner = perm[ix & MASK255]
        idx = ((inner + iy) & MASK511)
        h = perm[idx]
        return h / 127.5 - 1.0

    c00 = grad(x0, y0)
    c10 = grad(x0 + 1, y0)
    c01 = grad(x0, y0 + 1)
    c11 = grad(x0 + 1, y0 + 1)
    a = c00 + (c10 - c00) * u
    b = c01 + (c11 - c01) * u
    return a + (b - a) * v


def octave_field(perm, size: int, base_freq: float, octave_index: int,
                 lacunarity: float) -> np.ndarray:
    """Unweighted layer i: Value(pixel * base_freq * lacunarity**i)."""
    idx = np.arange(size, dtype=np.float64)
    px, py = np.meshgrid(idx, idx, indexing="xy")  # row=y, col=x
    f = base_freq * lacunarity ** octave_index
    return value_grid(perm, px * f, py * f)


def fbm_layers(size=256, seed=1, octaves=4, lacunarity=2.0, gain=0.5,
               frequency=0.02, mode="shared"):
    """Return list of weighted? No: raw unweighted octave fields and composite.

    Composite mirrors fbm.Eval exactly: sum(Value*amp)/sum(amp).
    """
    base_perm = make_perm(seed)
    raw = []
    weighted_sum = np.zeros((size, size))
    norm = 0.0
    amp = 1.0
    for i in range(octaves):
        if mode == "shared":
            perm = base_perm
        elif mode == "layered":
            perm = make_perm(derive_layer_seed(seed, i))
        else:
            raise ValueError(mode)
        fld = octave_field(perm, size, frequency, i, lacunarity)
        raw.append(fld)
        weighted_sum += fld * amp
        norm += amp
        amp *= gain
    comp = weighted_sum / norm
    return raw, comp


def normalize(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)
