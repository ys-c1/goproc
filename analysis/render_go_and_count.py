"""用 Go 生产渲染器（cmd/goproc shade）生成留档图，再从 Go 的 PGM 回读逐行计数。

保证"逐行统计"与"双光源效果图留档"同源：计数对象就是 Go 写出的 8-bit PGM。
同时与 Python 渲染器的输出做逐像素一致性核验（量化误差内必须相同）。
"""
import csv
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgmio import read_pgm
from render_beat import render, count_flips, quantize8, normals, strike_angles, beat_periods_row

ROOT = ".."
HM = {
    "fbm": ("fbm_s7", 15.0),
    "b1": ("b1_sinusoid", 20.0),
    "b2": ("b2_meander", 30.0),
}
DELTAS = [15.0, 90.0, 165.0]
PHI_M = 45.0
TOL = 2.0


def main():
    repo = os.path.abspath(ROOT)
    out_dir = os.path.join(repo, "artifacts", "renders")
    csv_dir = os.path.join(repo, "artifacts", "csv")
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    flagged = []
    for tname, (stem, elev) in HM.items():
        h_path = os.path.join(repo, "artifacts", "heightmaps", f"{stem}.pgm")
        _, h = read_pgm(h_path)
        _, _, _, gx, gy = normals(h)
        beta = strike_angles(gx, gy)
        for delta in DELTAS:
            phi1, phi2 = PHI_M - delta / 2, PHI_M + delta / 2
            tag = f"{tname}_d{int(delta)}"
            go_pgm = os.path.join(out_dir, f"go_{tag}.pgm")
            subprocess.run([
                "go", "run", "./cmd/goproc", "shade",
                "--in", h_path, "--out", go_pgm,
                "--phi1", str(phi1), "--phi2", str(phi2), "--elev", str(elev),
            ], cwd=repo, check=True, capture_output=True, text=True)
            _, go_img = read_pgm(go_pgm)
            py_img = quantize8(render(h, phi1, phi2, elev))
            mismatch = int(np.sum(np.abs(go_img - py_img) > 1.0 / 255.0 + 1e-12))

            size = h.shape[0]
            csv_path = os.path.join(csv_dir, f"go_row_compare_{tag}.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["row", "delta_deg", "n_beat_angular", "obs_flips_from_go_pgm",
                            "diff_obs_minus_beat", "flag_gt2"])
                beats, obss = [], []
                for y in range(size):
                    nb = beat_periods_row(beta[y], PHI_M, delta)
                    no = count_flips(go_img[y])
                    beats.append(nb)
                    obss.append(no)
                    if abs(no - nb) > TOL:
                        flagged.append([tag, y, delta, round(nb, 3), no, round(no - nb, 3)])
                    w.writerow([y, delta, round(nb, 4), no, round(no - nb, 4),
                                int(abs(no - nb) > TOL)])
            beats, obss = np.array(beats), np.array(obss)
            summary.append({
                "case": tag, "delta": delta, "elev": elev,
                "go_py_mismatch_pixels": mismatch,
                "beat_mean": round(float(beats.mean()), 2),
                "obs_go_mean": round(float(obss.mean()), 2),
                "obs_go_median": round(float(np.median(obss)), 2),
                "beat_rows_flagged_gt2": int((np.abs(obss - beats) > TOL).sum()),
            })
            print(f"{tag:>9} Go/Python不一致像素={mismatch:3d}  beat均值={beats.mean():7.2f} "
                  f"Go实测翻转均值={obss.mean():6.2f} 中位={np.median(obss):5.1f} "
                  f"beat超差行={int((np.abs(obss-beats)>TOL).sum())}")
    with open(os.path.join(csv_dir, "go_render_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(csv_dir, "go_flagged_rows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "row", "delta_deg", "n_beat", "obs_from_go_pgm", "diff"])
        w.writerows(flagged)
    print(f"\n-> go_render_summary.csv；beat 超差行 {len(flagged)} 条 -> go_flagged_rows.csv")


if __name__ == "__main__":
    main()
