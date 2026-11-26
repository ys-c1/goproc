import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from goproc_port import fbm_layers, normalize

OUT = pathlib.Path("/home/ys/feedback/uidreczz28eB3n7piyx/artifacts/repetition")

def render_trial(size, freq, octaves, seed=7, tag="trial"):
    raw, comp = fbm_layers(size=size, seed=seed, octaves=octaves,
                           frequency=freq, mode="shared")
    ncol = octaves + 1
    fig, ax = plt.subplots(1, ncol, figsize=(4*ncol, 4.2))
    for i, fld in enumerate(raw):
        ax[i].imshow(fld, cmap="gray", origin="upper")
        ax[i].set_title(f"oct{i} f={freq*2**i:.4f}")
        ax[i].set_xticks([]); ax[i].set_yticks([])
    ax[-1].imshow(normalize(comp), cmap="terrain", origin="upper")
    ax[-1].set_title("fbm composite")
    ax[-1].set_xticks([]); ax[-1].set_yticks([])
    fig.tight_layout()
    p = OUT / f"{tag}.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p)
    return raw, comp

if __name__ == "__main__":
    # A few regimes: default-like, more base cells, and near table-wrap scale.
    render_trial(256, 0.02, 5, tag="explore_default")
    render_trial(512, 1/32, 6, tag="explore_32cells")   # 16 base cells? freq*size
    render_trial(1024, 1/64, 4, tag="explore_big")
