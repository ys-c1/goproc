"""主实验：三档方位差 × 三块地形，拍频/线性预测 vs 渲染后逐行翻转统计。

用法：python3 -I run_experiment.py
输出：
  artifacts/renders/*.pgm           双光源效果图（8-bit，留档）
  artifacts/csv/row_compare_*.csv   逐行对照（按角度差分组：N_beat、N_sp、实测、偏差）
  artifacts/csv/flagged_rows.csv    偏差超过 2 的行（分预测类型）
  artifacts/csv/summary.csv|json    每组合汇总
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_beat import analyze
from pgmio import read_pgm

ROOT = ".."
HM = {
    "fbm": f"{ROOT}/artifacts/heightmaps/fbm_s7.pgm",
    "b1": f"{ROOT}/artifacts/heightmaps/b1_sinusoid.pgm",
    "b2": f"{ROOT}/artifacts/heightmaps/b2_meander.pgm",
}
# 三档方位差：近同向 / 中间 / 近反向。
DELTAS = [15.0, 90.0, 165.0]
# 各地形按坡度探查选定的无钳制仰角（clamp 占比 0%）。
ELEV = {"fbm": 15.0, "b1": 20.0, "b2": 30.0}
PHI_M = 45.0
TOL = 2.0


def main():
    os.makedirs(f"{ROOT}/artifacts/csv", exist_ok=True)
    summary = []
    all_flagged = []
    for tname, path in HM.items():
        elev = ELEV[tname]
        for delta in DELTAS:
            phi1 = PHI_M - delta / 2
            phi2 = PHI_M + delta / 2
            tag = f"{tname}_d{int(delta)}"
            res = analyze(path, phi1, phi2, elev, tag, f"{ROOT}/artifacts/renders")

            n = res["size"]
            beat = np.array([r["n_beat"] for r in res["rows"]])
            nsp = np.array([r["n_sp"] for r in res["rows"]], float)
            nsp8 = np.array([r["n_sp8"] for r in res["rows"]], float)
            obs = np.array([r["n_obs"] for r in res["rows"]], float)
            obsf = np.array([r["n_obs_float"] for r in res["rows"]], float)
            obsdb = np.array([r["n_obs_db2"] for r in res["rows"]], float)

            with open(f"{ROOT}/artifacts/csv/row_compare_{tag}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["row", "delta_deg", "n_beat_angular", "n_sp_linear_float",
                            "n_sp_linear_8bit", "obs_flips_8bit", "obs_flips_float",
                            "obs_flips_deadband2", "diff_obs_minus_beat",
                            "diff_obs_minus_sp8", "flag_beat_gt2", "flag_sp8_gt2"])
                for r in res["rows"]:
                    db_beat = abs(r["n_obs"] - r["n_beat"]) > TOL
                    db_sp = abs(r["n_obs"] - r["n_sp8"]) > TOL
                    if db_beat:
                        all_flagged.append([tag, r["row"], delta, "beat",
                                            round(r["n_beat"], 3), r["n_obs"], round(r["n_obs"] - r["n_beat"], 3)])
                    if db_sp:
                        all_flagged.append([tag, r["row"], delta, "linear8",
                                            r["n_sp8"], r["n_obs"], round(r["n_obs"] - r["n_sp8"], 3)])
                    w.writerow([r["row"], delta, round(r["n_beat"], 4), r["n_sp"], r["n_sp8"],
                                r["n_obs"], r["n_obs_float"], r["n_obs_db2"],
                                round(r["n_obs"] - r["n_beat"], 4),
                                round(r["n_obs"] - r["n_sp8"], 4),
                                int(db_beat), int(db_sp)])

            summary.append({
                "case": tag,
                "terrain": tname,
                "delta": delta,
                "phi1": phi1,
                "phi2": phi2,
                "elev": elev,
                "clamp1_pct": round(100 * res["clamp1"], 4),
                "clamp2_pct": round(100 * res["clamp2"], 4),
                "beat_mean": round(float(beat.mean()), 2),
                "sp_float_mean": round(float(nsp.mean()), 2),
                "sp8_mean": round(float(nsp8.mean()), 2),
                "obs8_mean": round(float(obs.mean()), 2),
                "obsfloat_mean": round(float(obsf.mean()), 2),
                "obsdb2_mean": round(float(obsdb.mean()), 2),
                "bias_obs_minus_sp8_mean": round(float((obs - nsp8).mean()), 2),
                "absdiff_sp8_max": round(float(np.abs(obs - nsp8).max()), 2),
                "absdiff_beat_max": round(float(np.abs(obs - beat).max()), 2),
                "sp8_rows_flagged": int((np.abs(obs - nsp8) > TOL).sum()),
                "beat_rows_flagged": int((np.abs(obs - beat) > TOL).sum()),
                "rows_total": n,
                "sp8_pass_rate_pct": round(100 * (np.abs(obs - nsp8) <= TOL).mean(), 2),
            })
            print(f"{tag:>9} Δ={delta:5.1f} clamp=({res['clamp1']:.1e},{res['clamp2']:.1e}) "
                  f"beat={beat.mean():7.2f} sp8={nsp8.mean():6.2f} obs8={obs.mean():6.2f} "
                  f"max|obs-sp8|={np.abs(obs-nsp8).max():3.0f} sp8超差行={int((np.abs(obs-nsp8)>TOL).sum())} "
                  f"beat超差行={int((np.abs(obs-beat)>TOL).sum())}")

    with open(f"{ROOT}/artifacts/csv/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(f"{ROOT}/artifacts/csv/flagged_rows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "row", "delta_deg", "pred_type", "pred", "obs", "diff_obs_minus_pred"])
        w.writerows(all_flagged)
    with open(f"{ROOT}/artifacts/csv/summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nsummary -> artifacts/csv/summary.csv；超差记录 {len(all_flagged)} 条 -> flagged_rows.csv")


if __name__ == "__main__":
    main()
