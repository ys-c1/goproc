#!/usr/bin/env python
"""把 terrainpairs 导出的 PGM 对拼成目视对照图。

上行: 弱派生基线, 种子间隔 1/16/256 的相邻两幅高度图(可视作同一地点)。
下行: 散列扰动 q=1, 间隔 1 与 256 的相邻两幅高度图(已去相关)。
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

font_manager.fontManager.addfont(
    "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf"
)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"


def read_pgm(path):
    with open(path, "rb") as f:
        tokens = []
        # 解析 P5 头: 四个空白分隔 token。
        while len(tokens) < 4:
            line = f.readline()
            if line.startswith(b"#"):
                continue
            tokens.extend(line.split())
        magic, w, h, _maxv = tokens
        assert magic == b"P5"
        w, h = int(w), int(h)
        raw = f.read(w * h)
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w)


def main():
    terrain_dir, out_path = sys.argv[1], sys.argv[2]
    pairs = [
        ("weak_gap1", "弱派生 · 间隔 1"),
        ("weak_gap16", "弱派生 · 间隔 16"),
        ("weak_gap256", "弱派生 · 间隔 256"),
        ("hash_q1_gap1", "散列 q=1 · 间隔 1"),
        ("hash_q1_gap256", "散列 q=1 · 间隔 256"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.4), facecolor=SURFACE)
    for col, (stem, title) in enumerate(pairs):
        a = read_pgm(os.path.join(terrain_dir, stem + "_a.pgm"))
        b = read_pgm(os.path.join(terrain_dir, stem + "_b.pgm"))
        for row, img in enumerate((a, b)):
            ax = axes[row][col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(MUTED)
            if row == 0:
                ax.set_title(title, color=INK, fontsize=11, pad=6)
            label = "种子 S" if row == 0 else "种子 S+间隔"
            ax.set_ylabel(label, color=MUTED, fontsize=9)
    fig.suptitle("弱派生下相邻种子的高度图几乎重合；散列扰动后两幅图完全不同",
                 color=INK, fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
