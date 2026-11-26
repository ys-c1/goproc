"""补充实验：钳制（非线线）区的低仰角双光源渲染，检验空间翻转数随方位差的变化。

线性区结论：空间翻转数由地形频率决定、与 Δ 无关；拍频只存在于方位角空间。
本脚本压低仰角使 max(0,·) 钳制大面积发生，观察非线性是否把方位拍频投影成空间条纹。
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_beat import analyze

ROOT = ".."
HM = {
    "fbm": f"{ROOT}/artifacts/heightmaps/fbm_s7.pgm",
    "b1": f"{ROOT}/artifacts/heightmaps/b1_sinusoid.pgm",
    "b2": f"{ROOT}/artifacts/heightmaps/b2_meander.pgm",
}
DELTAS = [15.0, 90.0, 165.0]
LOW_ELEVS = [5.0, 2.0]
PHI_M = 45.0


def main():
    os.makedirs(f"{ROOT}/artifacts/csv", exist_ok=True)
    os.makedirs(f"{ROOT}/artifacts/renders", exist_ok=True)
    rows_out = []
    for tname, path in HM.items():
        for elev in LOW_ELEVS:
            for delta in DELTAS:
                phi1, phi2 = PHI_M - delta / 2, PHI_M + delta / 2
                tag = f"clamped_{tname}_d{int(delta)}_e{int(elev)}"
                res = analyze(path, phi1, phi2, elev, tag, f"{ROOT}/artifacts/renders")
                obs = np.array([r["n_obs"] for r in res["rows"]], float)
                beat = np.array([r["n_beat"] for r in res["rows"]], float)
                rows_out.append({
                    "case": tag, "terrain": tname, "elev": elev, "delta": delta,
                    "clamp1_pct": round(100 * res["clamp1"], 2),
                    "clamp2_pct": round(100 * res["clamp2"], 2),
                    "beat_mean": round(float(beat.mean()), 2),
                    "obs8_mean": round(float(obs.mean()), 2),
                    "obs8_median": round(float(np.median(obs)), 2),
                })
                print(f"{tag:>16} clamp=({100*res['clamp1']:5.1f}%,{100*res['clamp2']:5.1f}%) "
                      f"beat={beat.mean():7.2f} obs8均值={obs.mean():6.2f} 中位={np.median(obs):5.1f}")
    with open(f"{ROOT}/artifacts/csv/clamped_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("\n-> artifacts/csv/clamped_summary.csv，效果图 -> artifacts/renders/clamped/")


if __name__ == "__main__":
    main()
