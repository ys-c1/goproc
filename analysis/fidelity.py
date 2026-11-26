import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from goproc_port import make_perm, derive_layer_seed, octave_field

ROOT = pathlib.Path("/home/ys/feedback/uidreczz28eB3n7piyx")
RAW = ROOT / "artifacts/repetition/raw"
S = 1024; F = 0.02; R = 2.0; SEED = 7
worst = 0.0
for mode in ("shared", "layered"):
    base = make_perm(SEED)
    for i in range(6):
        go = np.fromfile(RAW / f"layer_{mode}_{i}.f64", dtype="<f8").reshape(S, S)
        if mode == "shared":
            perm = base
        else:
            perm = make_perm(derive_layer_seed(SEED, i))
        py = octave_field(perm, S, F, i, R)
        d = float(np.max(np.abs(go - py)))
        worst = max(worst, d)
        print(f"{mode:7s} layer{i} max|go-python|={d:.3e}")
print("WORST", worst)
assert worst == 0.0, "port not bit-exact"
print("PORT_BIT_EXACT")
