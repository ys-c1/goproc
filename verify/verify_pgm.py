#!/usr/bin/env python3
"""独立复算 PGM 接缝灰度差：只使用标准库，不依赖 Go 侧任何代码。"""
import pathlib
import sys


def read_pgm(path):
    data = pathlib.Path(path).read_bytes()
    tokens = []
    i = 0
    # P5 头：四个空白分隔的 token，其余为二进制像素
    while len(tokens) < 4:
        j = i
        while j < len(data) and data[j] not in b" \t\n\r":
            j += 1
        if j > i:
            tokens.append(data[i:j])
        i = j + 1
    magic, w, h, maxv = tokens
    assert magic == b"P5" and int(maxv) == 255
    w, h = int(w), int(h)
    px = data[i:i + w * h]
    assert len(px) == w * h
    return w, h, list(px)


def seam_max(w, h, px, vertical_seam):
    if vertical_seam:  # 竖缝：两块各 n 列，接缝在 n-1 与 n 列之间，即同行相邻两列
        c0 = w // 2 - 1
        return max(abs(px[y * w + c0] - px[y * w + c0 + 1]) for y in range(h))
    r0 = h // 2 - 1     # 横缝：两块各 n 行，接缝在 n-1 与 n 行之间，即同列相邻两行
    return max(abs(px[r0 * w + x] - px[(r0 + 1) * w + x]) for x in range(w))


def main(outdir):
    cases = [
        ("horizontal_before.pgm", True),
        ("horizontal_after.pgm", True),
        ("vertical_before.pgm", False),
        ("vertical_after.pgm", False),
    ]
    for name, vert in cases:
        w, h, px = read_pgm(pathlib.Path(outdir) / name)
        print(f"{name:28s} {w}x{h}  seam_max_gray_diff={seam_max(w, h, px, vert)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/seam")
