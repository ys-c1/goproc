import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from goproc_port import fbm_layers, normalize

OUT = pathlib.Path("/home/ys/feedback/uidreczz28eB3n7piyx/artifacts/repetition")

def hillshade(z, az=315, alt=45):
    ls = LightSource(azdeg=az, altdeg=alt)
    return ls.hillshade(z, vert_exag=3.0)

def ridge(z):
    gx, gy = np.gradient(z)
    return np.hypot(gx, gy)

for F, tag in [(0.02, "f002"), (0.05, "f005")]:
    size = 1024
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, mode in enumerate(("shared", "layered")):
        raw, comp = fbm_layers(size=size, seed=7, octaves=6, frequency=F, mode=mode)
        z = normalize(comp)
        axes[row,0].imshow(z, cmap="terrain", origin="upper"); axes[row,0].set_title(f"{mode} height")
        axes[row,1].imshow(hillshade(z), cmap="gray", origin="upper"); axes[row,1].set_title("hillshade")
        axes[row,2].imshow(ridge(z), cmap="magma", origin="upper"); axes[row,2].set_title("gradient magnitude")
        for c in range(3): axes[row,c].set_xticks([]); axes[row,c].set_yticks([])
    fig.suptitle(f"frequency={F}")
    fig.tight_layout()
    p = OUT / f"look_{tag}.png"; fig.savefig(p, dpi=90); plt.close(fig)
    print("wrote", p)
