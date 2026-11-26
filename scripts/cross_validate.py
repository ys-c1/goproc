#!/usr/bin/env python3
"""独立交叉验证：用 scikit-image 的 marching squares 作为第三方权威实现，
与 Go 程序追踪/预判的逐档曲线数比对。

输入（由 `go run ./cmd/contour` 生成）：
  artifacts/contour/heightmap.f64   小端 float64 高度场
  artifacts/contour/levels_d*.csv   Go 的逐档预判与追踪

用法：python3 scripts/cross_validate.py
"""
import csv
import struct
import sys
from pathlib import Path

import numpy as np
from skimage import measure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "contour"


def load_heightmap() -> np.ndarray:
    raw = (OUT / "heightmap.f64").read_bytes()
    n = int(round((len(raw) / 8) ** 0.5))
    assert n * n * 8 == len(raw)
    return np.array(struct.unpack(f"<{n*n}d", raw), dtype=float).reshape(n, n)


def skimage_counts(z: np.ndarray, h: float):
    """返回 (边界跨越 B, 开弧数, 闭合环数)，开/闭按是否触及数组边界判定。"""
    n = z.shape[0]
    contours = measure.find_contours(z, h)
    opens = closed = 0
    for c in contours:
        touches = np.any((c <= 1e-9) | (c >= n - 1 - 1e-9))
        if touches:
            opens += 1
        else:
            closed += 1
    b = 0
    for x in range(n - 1):
        b += (z[0, x] > h) != (z[0, x + 1] > h)
        b += (z[n - 1, x] > h) != (z[n - 1, x + 1] > h)
    for y in range(n - 1):
        b += (z[y, 0] > h) != (z[y + 1, 0] > h)
        b += (z[y, n - 1] > h) != (z[y + 1, n - 1] > h)
    return int(b), opens, closed


def main() -> int:
    z = load_heightmap()
    failures = 0
    rows_checked = 0
    for csv_path in sorted(OUT.glob("levels_d*.csv")):
        print(f"\n=== {csv_path.name} ===")
        print(f"{'h':>9} {'B_go':>5} {'B_sk':>5} {'O_go':>5} {'O_sk':>5} "
              f"{'L_go':>5} {'L_sk':>5}  verdict")
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                h = float(row["h_used"])
                b_sk, o_sk, l_sk = skimage_counts(z, h)
                b_go = int(row["B_boundary_crossings"])
                o_go = int(row["O_trace"])
                l_go = int(row["L_trace"])
                ok = (b_go, o_go, l_go) == (b_sk, o_sk, l_sk)
                failures += 0 if ok else 1
                rows_checked += 1
                print(f"{h:9.6f} {b_go:5d} {b_sk:5d} {o_go:5d} {o_sk:5d} "
                      f"{l_go:5d} {l_sk:5d}  {'ok' if ok else 'MISMATCH'}")
    print(f"\n核对档位 {rows_checked} 个，差异 {failures} 个。")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
