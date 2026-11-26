"""Supporting proof figures:
 - identity_chain.png : for anchor a22, every adjacent edge A vs ratio-aligned B
 - period_proof.png   : exact additive tile period T5=400px + Vernier formula check
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from make_figures import load_layers, ART, S, LEVEL_COLORS, C_BUG, C_FIX  # font rc too

ROOT = pathlib.Path("/home/ys/feedback/uidreczz28eB3n7piyx")


def level_box(cx0, cy0, k, rho=2, w0=256):
    d = rho ** k
    cx, cy = cx0 // d, cy0 // d
    w = w0 // d
    return cx - w // 2, cy - w // 2, w


def pearson(a, b):
    a = a.ravel().astype(float); b = b.ravel().astype(float)
    a -= a.mean(); b -= b.mean()
    den = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum()/den) if den else np.nan


def chain_figure(layers, cx0=512, cy0=512):
    K = 6
    fig = plt.figure(figsize=(19, 11), dpi=120)
    gs = GridSpec(4, K-1, figure=fig, hspace=0.28, wspace=0.18,
                  left=0.06, right=0.985, top=0.90, bottom=0.05)
    row_titles = ["shared  A = 第 k+1 层细盒", "shared  B = 第 k 层按 ρ 抽稀",
                  "layered  A′ = 第 k+1 层细盒", "layered  B′ = 第 k 层抽稀"]
    for k in range(K-1):
        xk, yk, wk = level_box(cx0, cy0, k)
        xf, yf, wf = level_box(cx0, cy0, k+1)
        As = layers["shared"][k+1][yf:yf+wf, xf:xf+wf]
        Bs = layers["shared"][k][yk:yk+wk, xk:xk+wk][::2, ::2]
        Al = layers["layered"][k+1][yf:yf+wf, xf:xf+wf]
        Bl = layers["layered"][k][yk:yk+wk, xk:xk+wk][::2, ::2]
        assert As.shape == Bs.shape == (wf, wf)
        rs, rl = pearson(As, Bs), pearson(Al, Bl)
        data = [(As, rs, C_BUG), (Bs, rs, C_BUG), (Al, rl, C_FIX), (Bl, rl, C_FIX)]
        for row, (img, r, col) in enumerate(data):
            ax = fig.add_subplot(gs[row, k])
            ax.imshow(img, cmap="gray", origin="upper", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(row_titles[row], fontsize=11, color=col, labelpad=8)
            if row in (0, 2):
                ax.set_title(f"边 L{k}→L{k+1}  {wf}×{wf}\nr={r:+.5f}",
                             fontsize=10.5, color=col)
            for sp in ax.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(1.6)
    fig.suptitle("恒等证据（锚点 a22=(512,512)）：共用置换表时 A 与按频率比 ρ=2 "
                 "对齐抽稀出的 B 逐点相同（r=1.00000）；按层派生子表后 A′、B′ 互不相关",
                 fontsize=15, weight="bold")
    p = ART / "identity_chain.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def overlap_corr_vs_dx(g, shifts):
    out = []
    for dx in shifts:
        if dx >= S:
            out.append(np.nan); continue
        A = g[:, dx:].ravel().astype(float)
        B = g[:, :S-dx].ravel().astype(float)
        out.append(pearson(A, B))
    return np.array(out)


def period_figure(layers):
    g5 = layers["shared"][5]  # f=0.64/px -> T=256/0.64=400
    coarse = np.arange(0, 801, 8)
    rc = overlap_corr_vs_dx(g5, coarse)
    fine = np.arange(392, 409)
    rf = overlap_corr_vs_dx(g5, fine)
    r400 = overlap_corr_vs_dx(g5, [400])[0]
    # exact equality at shift 400 over overlap
    maxdiff = float(np.max(np.abs(g5[:, 400:] - g5[:, :S-400])))

    # Vernier coincidence simulation for integer rho: min t>0 s.t. t = rho*t mod P
    P = 256
    sim_rows = []
    for rho in (2, 3, 4, 5, 9, 17):
        t = 1
        while (rho*t - t) % P != 0:
            t += 1
        formula = P // np.gcd(P, rho-1)
        sim_rows.append((rho, rho-1, int(np.gcd(P, rho-1)), formula, t,
                         "OK" if formula == t else "MISMATCH"))

    fig = plt.figure(figsize=(17, 6.2), dpi=130)
    gs = GridSpec(1, 2, width_ratios=[1.25, 1], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(coarse, rc, "-o", ms=3.5, color="#2a78d6", lw=1.4, label="重叠区 Pearson r(dx)")
    ax.plot(fine, rf, "-o", ms=4, color="#eb6834")
    ax.axvline(400, color=C_BUG, ls="--", lw=1.6)
    ax.plot([400], [r400], "*", ms=22, color=C_BUG, zorder=5,
            label=f"dx=400: r={r400:.6f}, max|Δ|={maxdiff:.1e}")
    ax.set_xlabel("水平平移 dx (px)"); ax.set_ylabel("相关 r")
    ax.set_title("加法置换周期实测：L5, f=0.02·2^5=0.64/px，T=P/f=256/0.64=400 px",
                 fontsize=12.5, weight="bold")
    ax.set_ylim(-0.15, 1.08); ax.grid(alpha=.25); ax.legend(loc="lower right", fontsize=11)

    axt = fig.add_subplot(gs[0, 1]); axt.axis("off")
    axt.set_title("Vernier 重合周期  Π = P/gcd(P, ρ−1)   (P=256)",
                  fontsize=12.5, weight="bold", pad=14)
    cell = [["ρ", "ρ−1", "gcd(256,ρ−1)", "Π 公式", "模拟最小 t", "校验"]]
    for row in sim_rows:
        cell.append([str(x) for x in row])
    tbl = axt.table(cellText=cell, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.7)
    for j in range(6):
        tbl[(0, j)].set_facecolor("#2a78d6")
        tbl[(0, j)].set_text_props(color="white", weight="bold")
    for i in range(1, len(cell)):
        for j in range(6):
            tbl[(i, j)].set_facecolor("#f4f7fb" if i % 2 else "white")
    axt.text(0.02, 0.12,
             "ρ=2（本仓库 lacunarity）：gcd(256,1)=1 ⇒ Π=256 格点，\n"
             "对应基础像素周期 T0=256/0.02=12800px；第 i 层 T_i=T0/2^i。",
             fontsize=11.5, color="#333", va="bottom")
    p = ART / "period_proof.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("wrote", p, "| r400 =", r400, "maxdiff =", maxdiff)
    return sim_rows, r400, maxdiff


def main():
    layers = load_layers()
    chain_figure(layers)
    period_figure(layers)


if __name__ == "__main__":
    main()
