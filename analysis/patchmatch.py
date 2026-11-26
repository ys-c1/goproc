import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from goproc_port import fbm_layers, normalize

def best_match_map(field, ps=16, stride=8, exclusion=24):
    """For each patch center on grid, best NCC with a shifted patch.
    Returns records (x,y, bx,by, score). Brute force over integer shifts via
    FFT normalized cross-correlation of patch against whole map is complex;
    here we compare all grid patches pairwise (grid is modest)."""
    H, W = field.shape
    coords = []
    descs = []
    for y in range(0, H-ps, stride):
        for x in range(0, W-ps, stride):
            blk = field[y:y+ps, x:x+ps].astype(np.float64)
            b = blk - blk.mean()
            n = np.sqrt((b*b).sum())
            if n < 1e-9:
                continue
            descs.append(b.ravel()/n)
            coords.append((x, y))
    D = np.array(descs)
    C = D @ D.T
    np.fill_diagonal(C, -2)
    recs = []
    m = len(coords)
    for i in range(m):
        x, y = coords[i]
        best = -2; bj = -1
        for j in range(m):
            if i == j: continue
            x2, y2 = coords[j]
            dx, dy = x2-x, y2-y
            if dx*dx+dy*dy < exclusion*exclusion:
                continue
            if C[i, j] > best:
                best = C[i, j]; bj = j
        x2, y2 = coords[bj]
        recs.append((x, y, x2, y2, best, x2-x, y2-y))
    return recs

if __name__ == "__main__":
    size = 512; F = 0.02
    for mode in ("shared", "layered"):
        raw, comp = fbm_layers(size=size, seed=7, octaves=5, frequency=F, mode=mode)
        z = normalize(comp)
        recs = best_match_map(z, ps=20, stride=20, exclusion=30)
        scores = np.array([r[4] for r in recs])
        print(f"--- {mode}: n={len(recs)} best-NCC "
              f"mean={scores.mean():.3f} p90={np.quantile(scores,.9):.3f} "
              f"p99={np.quantile(scores,.99):.3f} max={scores.max():.3f}")
        hi = [r for r in recs if r[4] > 0.9]
        print(f"    patches with NCC>0.9: {len(hi)}")
        from collections import Counter
        dc = Counter((r[5], r[6]) for r in hi)
        for d, c in dc.most_common(12):
            print("    disp", d, "count", c)
