"""角空间拍频的直接数值验证（证明一）。

固定像素（固定山脊走向 beta），让等强度双灯绕 z 轴旋转：
    phi1=phi_m-Δ/2, phi2=phi_m+Δ/2，phi_m 从 0 扫到 180°。
理论：无钳制线性区，双灯响应中与朝向有关的分量为
    I_var(phi_m) = -(cos eps / M)·cos(Δ/2)·[gx sin phi_m + gy cos phi_m]
    = A·cos(phi_m - beta)·cos(Δ/2)，
即每 180° 扫描恰好 1 个完整余弦周期（一次亮暗翻转），与 Δ 无关（Δ 只改幅度）。
而两灯“差异信号” dF=I2-I1：
    dF(phi_m) = -(cos eps / M)·2 sin(Δ/2)·cos(beta-phi_m 方向几何) ...
其随 phi_m 的相位周期同样由走向决定；沿走向分布累加、对 Δ 求包络即得
    N_beat = sin Δ · Σ |cos gamma|。
本脚本逐像素扫描，实测角周期数，并与逐像素解析期望对照。
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_beat import normals, strike_angles
from pgmio import read_pgm

ROOT = ".."
HM = {
    "fbm": f"{ROOT}/artifacts/heightmaps/fbm_s7.pgm",
    "b1": f"{ROOT}/artifacts/heightmaps/b1_sinusoid.pgm",
    "b2": f"{ROOT}/artifacts/heightmaps/b2_meander.pgm",
}
ELEV = {"fbm": 15.0, "b1": 20.0, "b2": 30.0}
DELTAS = [15.0, 90.0, 165.0]
SWEEP = np.linspace(0.0, 360.0, 7201)  # 0.05° 步长，扫一个完整方位周期


def count_sign_flips(seq):
    s = np.sign(np.diff(seq))
    s = s[s != 0]
    if s.size < 2:
        return 0
    return int((s[:-1] != s[1:]).sum())


def main():
    out_rows = []
    for tname, path in HM.items():
        elev = ELEV[tname]
        size, h = read_pgm(path)
        nx, ny, nz, gx, gy = normals(h)
        beta = strike_angles(gx, gy)
        e = np.deg2rad(elev)
        ce, se = np.cos(e), np.sin(e)
        # 抽样像素（每 16 像素一个，避开零梯度）
        ys, xs = np.mgrid[8:size:16, 8:size:16]
        samples = [(int(y), int(x)) for y, x in zip(ys.ravel(), xs.ravel())
                   if np.hypot(gx[int(y), int(x)], gy[int(y), int(x)]) > 1e-6]
        for delta in DELTAS:
            errs = []
            for (y, x) in samples:
                nxx, nyy, nzz = nx[y, x], ny[y, x], nz[y, x]
                ivar = np.empty_like(SWEEP)
                for i, pm in enumerate(SWEEP):
                    p1, p2 = np.deg2rad(pm - delta / 2), np.deg2rad(pm + delta / 2)
                    l1x, l1y, l1z = ce * np.sin(p1), ce * np.cos(p1), se
                    l2x, l2y, l2z = ce * np.sin(p2), ce * np.cos(p2), se
                    d1 = max(0.0, nxx * l1x + nyy * l1y + nzz * l1z)
                    d2 = max(0.0, nxx * l2x + nyy * l2y + nzz * l2z)
                    ivar[i] = (d1 + d2) / 2.0
                # 去掉直流（减去均值），数朝向分量翻转；扫满 360° = 1 个余弦周期 = 2 次过零翻转
                flips = count_sign_flips(ivar - ivar.mean())
                errs.append(flips - 2)  # 期望 2 次翻转（一个完整余弦）
            errs = np.array(errs)
            out_rows.append({
                "terrain": tname, "delta": delta, "elev": elev,
                "samples": len(samples),
                "exact_2flips_pct": round(100 * (errs == 0).mean(), 2),
                "max_abs_err": int(np.abs(errs).max()),
            })
            print(f"{tname} Δ={delta:5.1f} 抽样 {len(samples):3d} 像素: "
                  f"恰好2次翻转占比={100*(errs==0).mean():6.2f}%  最大误差={np.abs(errs).max()}")

    # 第二步：沿整条山脊走向场累加，验证 N_beat 随 Δ 的包络形状
    print("\n== 包络检验：固定地形，N_beat(Δ) 应正比于 sin Δ ==")
    tname = "b1"
    size, h = read_pgm(HM[tname])
    _, _, _, gx, gy = normals(h)
    beta = strike_angles(gx, gy)
    phi_m = 45.0
    gamma_factor = np.nansum(np.abs(np.sin(np.deg2rad(beta - phi_m))))
    table = []
    for delta in np.arange(5.0, 180.0, 5.0):
        nb = np.sin(np.deg2rad(delta)) * gamma_factor
        table.append((delta, nb))
    # 与 sin Δ 的相关系数（应=1），以及峰值位置（应在 Δ=90°）
    ds = np.array([t[0] for t in table])
    nb = np.array([t[1] for t in table])
    corr = np.corrcoef(nb, np.sin(np.deg2rad(ds)))[0, 1]
    peak = ds[int(np.argmax(nb))]
    print(f"corr(N_beat, sinΔ)={corr:.8f}  峰值位于 Δ={peak:.0f}°（理论 90°），"
          f"Δ=15/90/165 取值比 = {nb[2]:.1f}:{nb[17]:.1f}:{nb[32]:.1f}")
    with open(f"{ROOT}/artifacts/csv/angular_beat_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["terrain", "delta", "elev", "samples", "exact_2flips_pct", "max_abs_err"])
        for r in out_rows:
            w.writerow([r["terrain"], r["delta"], r["elev"], r["samples"],
                        r["exact_2flips_pct"], r["max_abs_err"]])
        w.writerow([])
        w.writerow(["envelope_corr_with_sin_delta", round(corr, 8), "peak_delta_deg", peak, "", ""])


if __name__ == "__main__":
    main()
