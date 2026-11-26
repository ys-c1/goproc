"""峰数预判与实测比对流水线。

流程（严格先谱后直方图，预判不使用任何 HOG 信息）：
  1. 谱：惯性张量 -> 两条主轴、轴比 rho、各向异性度 eta；
  2. 谱：分径向带瓣检测 -> 旁瓣方向与强度，跨带一致性 + 双阈值，
     并结合 eta 门控判定网格谐波伪影，得到预判峰数与方向；
  3. HOG：幅度加权梯度方向直方图，双阈值（高度+间距）实测峰；
  4. 按各向异性档比对预判与实测，多认/少认均给方向角。

用法: python3 -I run_analysis.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from analyze_core import (
    load_raw, spectrum_parts, radial_band_lobes, low_band_profiles,
    hog_parts, pick_peaks, _circdist,
)

# ---- 双阈值（由各向同性标定场 iso 数据驱动确定，见报告）----
HOG_HEIGHT_THR = 1.25      # HOG 峰/均值；iso 最高 1.09，留 ~15% 余量
HOG_SEP_THR = 15.0         # HOG 峰最小间距（度）
SPEC_HEIGHT_THR = 1.8      # 谱分带瓣强度/带内均值
SPEC_SEP_THR = 15.0        # 谱瓣最小间距（度），与 HOG 一致
SIZE = 512
TERRAIN_DIR = "artifacts/terrains"
OUT_DIR = "artifacts/results"


def predict_from_spectrum(z):
    """纯谱预判，返回预判峰与诊断量。不接触 HOG。"""
    res = spectrum_parts(z)
    rho, eta, major, minor = res[2], res[3], res[4], res[5]
    detail, directions, grid = radial_band_lobes(
        z, height_thr=SPEC_HEIGHT_THR, min_sep_deg=SPEC_SEP_THR)

    # 区分真实脊系瓣与采样网格谐波伪影。
    # 物理本质：真实定向带在低频带即具有明显角宽；采样谐波是窄谱线，
    # 在 band1 仅表现为角度很窄的尖峰。对 grid_like 簇测其 band1 半高宽
    # （以 1.3 为底），>=WIDTH_THR 判真实，否则判谐波。
    WIDTH_THR = 18.0
    WIDTH_BASE = 1.3
    low_profiles = low_band_profiles(z, (1,))
    prof1 = low_profiles[1]

    def _band1_width(angle0):
        idx = int(round(angle0)) % 180
        pk, best = idx, 0.0
        for k in range(-15, 16):
            j = (idx + k) % 180
            if prof1[j] > best:
                best, pk = prof1[j], j
        if best < WIDTH_BASE:
            return 0.0
        lo = pk
        while prof1[lo % 180] > WIDTH_BASE and pk - lo < 45:
            lo -= 1
        hi = pk
        while prof1[hi % 180] > WIDTH_BASE and hi - pk < 45:
            hi += 1
        return float(hi - lo)

    lobes, harmonic = [], []
    for d in directions:
        is_real = True
        if d["grid_like"]:
            w = _band1_width(d["angle"])
            is_real = w >= WIDTH_THR
        if is_real:
            lobes.append(d)
        else:
            harmonic.append(d["angle"])
    gate_note = (
        f"网格轴瓣测 band1 角宽（底{WIDTH_BASE}）：>={WIDTH_THR}°判真实脊系，"
        f"否则判窄谱线谐波；剔除 {sorted(harmonic) if harmonic else '无'}，"
        f"eta={eta:.2f}, rho={rho:.2f}")

    predicted = [
        {"angle": d["angle"], "strength": d["strength"], "bands": d["bands"]}
        for d in lobes
    ]
    return {
        "axis_ratio": round(rho, 3),
        "anisotropy": round(eta, 3),
        "major_axis_deg": round(major, 1),
        "minor_axis_deg": round(minor, 1),
        "grid_harmonic_angles": grid,
        "removed_harmonic_angles": sorted(harmonic),
        "gate_note": gate_note,
        "predicted_peaks": predicted,
        "predicted_count": len(predicted),
        "band_detail": detail,
    }


def measure_hog(z):
    """HOG 双阈值实测峰。"""
    h_mean, h_max, _, _ = hog_parts(z)
    accepted, rej_height, rej_sep = pick_peaks(
        h_mean, HOG_HEIGHT_THR, HOG_SEP_THR)
    return {
        "measured_peaks": accepted,
        "measured_count": len(accepted),
        "rejected_by_height": rej_height,
        "rejected_by_spacing": rej_sep,
        "profile_mean": np.round(h_mean, 4).tolist(),
        "profile_max": np.round(h_max, 4).tolist(),
    }


def match_angles(pred, meas, tol=10.0):
    """按角度最近邻匹配预判与实测峰，返回多认、少认、配对。"""
    used = set()
    pairs, extra_pred, missing = [], [], []
    for p in pred:
        best, bd = None, tol
        for j, m in enumerate(meas):
            if j in used:
                continue
            d = _circdist(p["angle"], m["angle"])
            if d <= bd:
                best, bd = j, d
        if best is None:
            extra_pred.append(p["angle"])
        else:
            used.add(best)
            pairs.append((p["angle"], meas[best]["angle"], round(bd, 1)))
    for j, m in enumerate(meas):
        if j not in used:
            missing.append(m["angle"])
    return pairs, sorted(extra_pred), sorted(missing)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"thresholds": {
        "hog_height": HOG_HEIGHT_THR, "hog_min_sep_deg": HOG_SEP_THR,
        "spectrum_height": SPEC_HEIGHT_THR, "spectrum_min_sep_deg": SPEC_SEP_THR,
        "grid_harmonic_width_base": 1.3, "grid_harmonic_width_thr_deg": 18.0,
    }, "terrains": {}}

    for name in ("A", "B"):
        z = load_raw(os.path.join(TERRAIN_DIR, f"terrain_{name}.raw"), SIZE)
        spec = predict_from_spectrum(z)
        hog = measure_hog(z)
        pairs, extra, missing = match_angles(
            spec["predicted_peaks"], hog["measured_peaks"])
        verdict = "一致" if spec["predicted_count"] == hog["measured_count"] else \
            ("预判多认" if spec["predicted_count"] > hog["measured_count"] else "预判少认")
        entry = {
            "spectrum": {k: v for k, v in spec.items() if k != "band_detail"},
            "hog": {k: v for k, v in hog.items()
                    if k not in ("profile_mean", "profile_max")},
            "comparison": {
                "predicted_count": spec["predicted_count"],
                "measured_count": hog["measured_count"],
                "verdict": verdict,
                "matched_pairs(pred,meas,delta)": pairs,
                "extra_predicted_angles": extra,
                "missing_in_prediction_angles": missing,
            },
        }
        report["terrains"][name] = entry
        with open(os.path.join(OUT_DIR, f"{name}_profiles.json"), "w") as fp:
            json.dump({
                "hog_profile_mean": hog["profile_mean"],
                "hog_profile_max": hog["profile_max"],
                "band_detail": spec["band_detail"],
            }, fp, ensure_ascii=False, indent=2)

    # 各向同性标定场单独记录，用于阈值溯源
    z = load_raw(os.path.join(TERRAIN_DIR, "terrain_iso.raw"), SIZE)
    report["calibration_iso"] = predict_from_spectrum(z)
    report["calibration_iso"].pop("band_detail", None)
    hm, _, _, _ = hog_parts(z)
    report["calibration_iso"]["hog_max_peak_to_mean"] = round(float(hm.max()), 3)

    with open(os.path.join(OUT_DIR, "report.json"), "w") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
