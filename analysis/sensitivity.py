"""平滑窗口敏感性：验证离散曲率二阶差分变号判据在不同平滑窗口下的稳定性。

运行：cd /tmp && python3 -P <repo>/analysis/sensitivity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrainlib as tl  # noqa: E402
from run_experiment import SEEDS, make_schedules, run_seed  # noqa: E402

REPO = "/home/ys/feedback/uidreczz28eCcwNSxAr"


def main() -> None:
    print(f"{'seed':>5} {'schedule':>18} {'metric':>5} {'pred':>5} | "
          f"{'w=1':>5} {'w=3':>5} {'w=5':>5} {'w=7':>5}")
    unstable = 0
    total = 0
    for seed in SEEDS:
        res = run_seed(seed)
        for sched_name in make_schedules(res["samp"]):
            sched = res["schedules"][sched_name]
            sw = sched["sweep"]
            pt = sched["pred_tier"]
            for metric in ("rmse", "emax"):
                picks = []
                for w in (1, 3, 5, 7):
                    try:
                        knee, _ = tl.curvature_knee(sw[metric], sw["nvert"].astype(float),
                                                    predicted_index=pt, smooth=w)
                    except ValueError:
                        knee = -1
                    picks.append(knee)
                total += 1
                spread = max(picks) - min(picks)
                if spread > 1:
                    unstable += 1
                flag = "" if spread <= 1 else "  <-- 窗口间漂移 >1 档"
                print(f"{seed:>5} {sched_name:>18} {metric:>5} {pt:>5} | "
                      f"{picks[0]:>5} {picks[1]:>5} {picks[2]:>5} {picks[3]:>5}{flag}")
    print(f"\n共 {total} 组，窗口间漂移超过 1 档的组数：{unstable}")


if __name__ == "__main__":
    main()
