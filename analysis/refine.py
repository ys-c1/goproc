"""细扫与跨种子稳健性检查（不产图，只打印数值）。

运行：cd /tmp && python3 -P <repo>/analysis/refine.py
"""

import sys

import numpy as np

sys.path.insert(0, "/home/ys/feedback/uidreczz28eCcwNSxAr/analysis")
import terrainlib as tl  # noqa: E402

REPO = "/home/ys/feedback/uidreczz28eCcwNSxAr"


def analyze(seed: int) -> None:
    h = tl.load_raw(f"{REPO}/artifacts/raw/seed{seed}.raw")
    kappa = tl.curvature(h)
    samp = tl.interior_curvature(kappa)
    mode, q_mode = tl.curvature_mode(samp)
    print(f"\n===== seed {seed} =====")
    print(f"κ*={mode:.6f}  F(κ*)={q_mode:.4f}  内部样本={samp.size}")

    # 细扫 45 档（等分位铺档，保证每档样本量稳定）
    qs = np.linspace(0.0, 0.98, 45)
    taus = np.quantile(samp, qs)
    taus[0] = 0.0
    sw = tl.sweep(h, kappa, taus)

    # g(tau) 线性度：对 RMSE 与 Emax 分别拟合 RMSE ≈ a*tau（过原点）与一次项
    nz = sw["tau"] > 0
    for name, col in (("RMSE", "rmse"), ("Emax", "emax")):
        e = sw[col][nz]
        t = sw["tau"][nz]
        a0 = float(np.dot(t, e) / np.dot(t, t))
        resid0 = e - a0 * t
        a1, b1 = np.polyfit(t, e, 1)
        resid1 = e - (a1 * t + b1)
        print(f"  g_{name}: 过原点 a={a0:.4f} 残差RMS={np.sqrt((resid0**2).mean()):.5f}; "
              f"一次 a={a1:.4f} b={b1:.5f} 残差RMS={np.sqrt((resid1**2).mean()):.5f}")

    pred_idx = int(np.argmin(np.abs(taus - mode)))
    print(f"  预测档位 idx={pred_idx} q={qs[pred_idx]:.4f} tau={taus[pred_idx]:.6f} "
          f"N_pred={tl.retained_count(kappa, taus[pred_idx])}")
    for col in ("rmse", "emax"):
        x, y = sw[col], sw["nvert"].astype(float)
        kg = tl.discrete_curvature(x, y, smooth=3)
        d2 = np.full_like(kg, np.nan)
        d2[1:-1] = kg[2:] - 2 * kg[1:-1] + kg[:-2]
        flips = [i for i in range(2, len(kg) - 1)
                 if np.isfinite(d2[i - 1]) and np.isfinite(d2[i]) and d2[i - 1] * d2[i] < 0]
        pick = min(flips, key=lambda i: abs(i - pred_idx))
        # 另报标准 L 曲线最大曲率点作参照
        kg_abs = np.abs(kg)
        maxc = int(np.nanargmax(kg_abs))
        print(f"  [{col}] 变号点={flips[:8]}{'...' if len(flips) > 8 else ''}")
        print(f"    规则选中 idx={pick} (Δ档={pick - pred_idx:+d}) E={x[pick]:.6f} N={int(y[pick])}; "
              f"最大|曲率|参照 idx={maxc} E={x[maxc]:.6f} N={int(y[maxc])}")


for s in (7, 1, 42):
    analyze(s)
