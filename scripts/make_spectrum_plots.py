"""绘制谱预判支撑图：二维功率谱（对数）+ 主轴 + 分径向带角向谱瓣。

每组各向异性地形一张，左为二维功率谱与两条主轴，右为分带角向剖面。
用法: python3 -I make_spectrum_plots.py
"""
import os
import sys

for _p in ("/home/ys/.local/lib/python3.14/site-packages",
           "/usr/local/lib/python3.14/dist-packages",
           "/usr/lib/python3/dist-packages"):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CJK = "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf"
if os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = [
        font_manager.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_core import load_raw, spectrum_parts, radial_band_lobes, _gauss2d_fft

SIZE = 512
TERRAIN_DIR = "artifacts/terrains"
OUT_DIR = "artifacts/results"
BLUE, ORANGE, GREEN, MUTED, INK, SURFACE, GRID = (
    "#2a78d6", "#eb6834", "#0b5a2e", "#8a8a83", "#0b0b0b", "#fcfcfb", "#e3e2dd")
plt.rcParams.update({
    "text.color": INK, "axes.edgecolor": "#c9c8c2",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.unicode_minus": False})


def plot_spectrum(name, title):
    z = load_raw(os.path.join(TERRAIN_DIR, f"terrain_{name}.raw"), SIZE)
    _, _, rho, eta, major, minor, _ = spectrum_parts(z)
    detail, directions, grid = radial_band_lobes(z)

    n = z.shape[0]
    w = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    f = np.fft.fftshift(np.fft.fft2((z - z.mean()) * w))
    p2d = _gauss2d_fft(np.abs(f) ** 2, 1.5)
    show = np.log1p(p2d / p2d.max())
    c = n // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=140,
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    lim = int(0.6 * c)
    ax1.imshow(show, cmap="viridis", extent=(-c, c, -c, c), origin="lower")
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    # 两条主轴
    for ang, col, lab in ((major, ORANGE, f"主轴 {major:.0f}°"),
                          (minor, MUTED, f"次轴 {minor:.0f}°")):
        t = np.radians(ang)
        ax1.plot([-lim * np.cos(t), lim * np.cos(t)],
                 [-lim * np.sin(t), lim * np.sin(t)], color=col, lw=2, label=lab)
    ax1.set_title(f"二维功率谱（对数）  轴比 ρ={rho:.2f}  各向异性度 η={eta:.2f}",
                  fontsize=11)
    ax1.legend(loc="upper right", frameon=False, fontsize=9)
    ax1.set_xlabel("kx"); ax1.set_ylabel("ky")

    # 分带角向剖面（取三个代表带）
    band_prof = {b["band"]: b["lobes"] for b in detail}
    colors = [BLUE, ORANGE, GREEN]
    for b, col in zip((1, 3, 5), colors):
        # 重算该带剖面用于画线
        k = np.arange(n) - c
        KX, KY = np.meshgrid(k, k)
        r = np.hypot(KX, KY)
        ang = np.degrees(np.arctan2(KY, KX)) % 180
        edges = np.linspace(2, 0.6 * n / 2, 7)
        m = (r >= edges[b]) & (r < edges[b + 1])
        prof, _ = np.histogram(ang[m], bins=np.linspace(0, 180, 181), weights=p2d[m])
        from analyze_core import smooth_circular
        prof = smooth_circular(prof, 3.0)
        ax2.plot(np.arange(180), prof / prof.mean(), color=col, lw=1.8,
                 label=f"径向带 {b}")
    ax2.axhline(1.8, color="#b03a2e", lw=1.6, ls="--", label="谱高度阈值 1.8")
    for d in directions:
        ax2.axvline(d["angle"], color=GREEN, lw=1.2, alpha=0.7)
        ax2.annotate(f"{d['angle']:.0f}°", xy=(d["angle"], ax2.get_ylim()[1]),
                     fontsize=8.5, color=GREEN, ha="center", va="top",
                     xytext=(d["angle"], 0))
    ax2.set_title("分径向带角向功率剖面与谱瓣方向", fontsize=11)
    ax2.set_xlim(0, 180); ax2.set_xlabel("波矢方向角（度，模 180°）")
    ax2.set_ylabel("带内功率 / 均值")
    ax2.grid(axis="y", color=GRID, lw=0.8); ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", frameon=False, fontsize=9)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"spectrum_{name}.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for n, t in (
        ("A", "强各向异性地形 A：谱预判（主轴、轴比、分带旁瓣）"),
        ("B", "弱各向异性地形 B：谱预判（主轴、轴比、分带旁瓣）"),
    ):
        print("wrote", plot_spectrum(n, t))
