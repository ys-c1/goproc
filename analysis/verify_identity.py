import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from goproc_port import (make_perm, derive_layer_seed, value_grid,
                         octave_field, fbm_layers)

size = 512
F = 0.02
r = 2.0
seed = 7
perm = make_perm(seed)

# O1(p) = N(F r p);  O0(r p) = N(F * (r p)) = same coords.
O0 = octave_field(perm, size, F, 0, r)
O1 = octave_field(perm, size, F, 1, r)

# Evaluate O0 at continuous pixel coords q = r*(ix,iy) using its frequency F:
idx = np.arange(size, dtype=np.float64)
px, py = np.meshgrid(idx, idx, indexing="xy")
qxs, qys = r*px, r*py
O0_at_rp = value_grid(perm, qxs*F, qys*F)

diff = np.max(np.abs(O1 - O0_at_rp))
print("SHARED  max|O1(p) - O0(r p)| =", diff)
print("SHARED  corr(O1, O0(rp))     =",
      np.corrcoef(O1.ravel(), O0_at_rp.ravel())[0, 1])

# Layered: O1 from independent sub-table -> identity breaks
perm1 = make_perm(derive_layer_seed(seed, 1))
O1L = octave_field(perm1, size, F, 1, r)
print("LAYERED corr(O1_sub, O0(rp)) =",
      np.corrcoef(O1L.ravel(), O0_at_rp.ravel())[0, 1])
print("LAYERED corr(O1_sub, O0  same-pos) =",
      np.corrcoef(O1L.ravel(), O0.ravel())[0, 1])
print("SHARED  corr(O1, O0 same-pos, different scale) =",
      np.corrcoef(O1.ravel(), O0.ravel())[0, 1])
