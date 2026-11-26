"""探索性检查：曲率分布形态与 N(E) 曲线拐点是否存在。

运行（在仓库根目录，cwd 不含任何待导入数据）：
  cd /tmp && python3 -P /home/ys/feedback/uidreczz28eCcwNSxAr/analysis/explore.py <raw 路径>
"""

import sys

import numpy as np

sys.path.insert(0, "/home/ys/feedback/uidreczz28eCcwNSxAr/analysis")
import terrainlib as tl  # noqa: E402


def main(path: str) -> None:
    h = tl.load_raw(path)
    n = h.shape[0]
    kappa = tl.curvature(h)
    samp = tl.interior_curvature(kappa)
    print(f"grid={n}  interior samples={samp.size}")
    print("曲率分位: " + ", ".join(f"q{q}={np.quantile(samp, q):.5f}" for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)))
    mode, q_at_mode = tl.curvature_mode(samp)
    print(f"KDE 众数 κ*={mode:.5f}  对应分位 F(κ*)={q_at_mode:.4f}")

    # 粗扫 30 档，看曲线形态
    taus = np.linspace(0.0, float(np.quantile(samp, 0.98)), 30)
    taus[0] = 0.0
    sw = tl.sweep(h, kappa, taus)
    print("\n tau       N       RMSE      Emax")
    for i in range(len(taus)):
        print(f"{sw['tau'][i]:8.5f} {sw['nvert'][i]:6d} {sw['rmse'][i]:9.6f} {sw['emax'][i]:9.6f}")

    pred_idx = int(np.argmin(np.abs(taus - mode)))
    for metric in ("emax", "rmse"):
        x = sw[metric]
        y = sw["nvert"].astype(float)
        kg = tl.discrete_curvature(x, y, smooth=3)
        d2 = np.full_like(kg, np.nan)
        d2[1:-1] = kg[2:] - 2 * kg[1:-1] + kg[:-2]
        flips = [i for i in range(2, len(kg) - 1) if np.isfinite(d2[i - 1]) and np.isfinite(d2[i]) and d2[i - 1] * d2[i] < 0]
        print(f"\n[{metric}] 预测档位 idx={pred_idx} tau={taus[pred_idx]:.5f}; 变号点 idx={flips}")
        if flips:
            pick = min(flips, key=lambda i: abs(i - pred_idx))
            print(f"  选中 idx={pick} tau={taus[pick]:.5f} E={x[pick]:.6f} N={int(y[pick])}")


if __name__ == "__main__":
    main(sys.argv[1])
