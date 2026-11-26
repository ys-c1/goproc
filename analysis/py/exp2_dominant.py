"""实验二：证明多倍频叠加时混叠由最高层频率决定，而不是基础频率。

做法：fBm 第 k 层（k=0..K-1）频率 f_k=f0 L^k、归一化振幅 G^k/ΣG^j。
  E_alias(K) = Σ_k (G^k/ΣG^j)^2 · E2(f_k)
其中 E2(f_k) 是单倍频 2D 奈奎斯特尾能量（解析 PSD 积分，无量纲）。
对比“完整和”与“仅最高层”，并报告最高层占比。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import alias_energy_2d, variance_analytic

F0 = 0.025                 # 细网格单位（像素当量 0.2）
L = 2.0
G = 0.5
STEP = 8
DELTA = float(STEP)
FN = 0.5 / DELTA           # 粗网格奈奎斯特（细单位）=0.0625
KS = (3, 4, 5)


def octave_alias(k: int) -> float:
    fk = F0 * L ** k
    if fk <= FN:
        return 0.0
    return alias_energy_2d(fk, DELTA) / variance_analytic(fk)


def main():
    rows = []
    for K in KS:
        norm = sum(G ** k for k in range(K))
        per = [(G ** k / norm) ** 2 * octave_alias(k) for k in range(K)]
        total = sum(per)
        top = per[-1]
        rows.append(dict(K=K, f_top=F0 * L ** (K - 1), total=total,
                         top=top, top_fraction=top / total if total else 0.0,
                         per_octave=per))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp2_rows.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)

    print(f"{'K':>2} {'f_top':>7} {'总混叠能量':>12} {'仅最高层':>12} {'最高层占比':>10}")
    for r in rows:
        print(f"{r['K']:>2} {r['f_top']:>7.3f} {r['total']:>12.5f} "
              f"{r['top']:>12.5f} {r['top_fraction']:>10.2%}")
    print("\n逐层贡献（按倍频层）:")
    for r in rows:
        print(f"  K={r['K']}: " + " ".join(f"{v:.5f}" for v in r["per_octave"]))


if __name__ == "__main__":
    main()
