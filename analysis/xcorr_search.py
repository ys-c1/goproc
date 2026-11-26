import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from goproc_port import make_perm, derive_layer_seed, hash_grid, P

def torus_xcorr_best(A, B, maxlag=64):
    """Best circular cross-correlation of A vs integer shifts of B."""
    a = A - A.mean(); b = B - B.mean()
    denom = np.sqrt((a*a).mean() * (b*b).mean())
    n = P
    Fa = np.fft.rfftn(a)
    Fb = np.fft.rfftn(b)
    xc = np.fft.irfftn(Fa * np.conj(Fb), s=(n, n)) / (n*n) / denom
    # xc[dy,dx] correlation A[y,x] with B[y-dy?, ...]; just scan shifts
    best = []
    for dy in range(-maxlag, maxlag+1):
        for dx in range(-maxlag, maxlag+1):
            v = xc[dy % n, dx % n]
            best.append((v, dx, dy))
    best.sort(reverse=True)
    return best[:10]

for seed in (1, 7, 42, 123):
    perm = make_perm(seed)
    xs, ys = np.meshgrid(np.arange(P), np.arange(P))
    H = hash_grid(perm, xs, ys)
    for r in (2, 3):
        # fine layer lattice at rX,rY (mod P via periodicity)
        F = hash_grid(perm, (r*xs) & 255, (r*ys) & 255)
        top = torus_xcorr_best(H, F, maxlag=128)
        print(f"seed={seed:3d} r={r} SHARED  H(X,Y) vs H(rX+dx,rY+dy) top:")
        for v, dx, dy in top[:5]:
            print(f"    off=({dx:4d},{dy:4d}) corr={v:+.4f}")
    # independent table baseline
    perm2 = make_perm(derive_layer_seed(seed, 1))
    H2 = hash_grid(perm2, xs, ys)
    F2 = hash_grid(perm2, (2*xs) & 255, (2*ys) & 255)
    top = torus_xcorr_best(H, F2, maxlag=64)
    print(f"seed={seed:3d}      LAYERED cross top: {[(dx,dy,round(v,3)) for v,dx,dy in top[:3]]}")
    print()
