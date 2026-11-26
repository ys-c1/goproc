import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from goproc_port import make_perm, value_grid, MASK255, MASK511

P = 256

def hash_grid(perm, X, Y):
    """H(X,Y) exactly as grad, broadcasting integer arrays."""
    inner = perm[X & MASK255]
    idx = (inner + Y) & MASK511
    return perm[idx].astype(np.float64) / 127.5 - 1.0

def autocorr_peaks(a, maxlag=200, topn=12, exclude=3):
    """FFT normalized autocorrelation; return top nonzero-lag peaks."""
    a = a - a.mean()
    n = a.shape[0]
    F = np.fft.rfftn(a, s=(2*n, 2*n))
    ac = np.fft.irfftn(F * np.conj(F), s=(2*n, 2*n))
    ac = np.fft.fftshift(ac)
    c = ac.shape[0]//2
    ac = ac / ac[c, c]
    peaks = []
    for dy in range(-maxlag, maxlag+1):
        for dx in range(-maxlag, maxlag+1):
            if dx*dx + dy*dy < exclude*exclude:
                continue
            v = ac[c+dy, c+dx]
            peaks.append((v, dx, dy))
    peaks.sort(reverse=True)
    # non-max suppression-ish: keep separated lags
    kept = []
    for v, dx, dy in peaks:
        if all((dx-kx)**2+(dy-ky)**2 > 25 for _, kx, ky in kept):
            kept.append((v, dx, dy))
        if len(kept) >= topn:
            break
    return kept

def exact_block_dups(field, bh=8, bw=8, maxpairs=20):
    """Hash bhxbw blocks; tally displacement between identical blocks."""
    H, W = field.shape
    table = {}
    disp = {}
    for y in range(0, H-bh+1, 2):
        for x in range(0, W-bw+1, 2):
            blk = field[y:y+bh, x:x+bw]
            key = blk.tobytes()
            if key in table:
                for (x0, y0) in table[key]:
                    d = (x-x0, y-y0)
                    disp[d] = disp.get(d, 0) + 1
                table[key].append((x, y))
            else:
                table[key] = [(x, y)]
    top = sorted(disp.items(), key=lambda kv: -kv[1])[:maxpairs]
    return top

if __name__ == "__main__":
    for seed in (1, 7, 42):
        perm = make_perm(seed)
        # One full table period lattice grid
        xs, ys = np.meshgrid(np.arange(P), np.arange(P))
        H = hash_grid(perm, xs, ys)
        print(f"=== seed {seed} lattice {P}x{P} autocorr peaks ===")
        for v, dx, dy in autocorr_peaks(H, maxlag=128, topn=8):
            print(f"  lag=({dx:4d},{dy:4d}) corr={v:.4f}")
