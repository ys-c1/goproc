import sys, math
sys.path.insert(0, "/home/ys/feedback/uidreczz28eCcwNSdbd/scripts")
import numpy as np
from coastline_analysis import ruler_walk

# 1) 直线长 100, 各 eps 下应得 L≈100, 步数≈100/eps
chain = [(float(i), 0.0) for i in range(101)]
for eps in [10, 5, 2, 1]:
    L, q = ruler_walk(chain, False, float(eps))
    print(f"straight eps={eps}: L={L:.3f} expect 100, npts={len(q)} expect ~{100//eps+1}")

# 2) 锯齿: 每段上升 3 再水平? 用已知折线: 直角折线 (0,0)->(10,0)->(10,10), 弧长20
chain2 = [(0.0,0.0),(10.0,0.0),(10.0,10.0)]
for eps in [5.0, 2.0]:
    L, q = ruler_walk(chain2, False, eps)
    print(f"L-shape eps={eps}: L={L:.3f} npts={len(q)}")
    print("   pts:", np.round(q,2).tolist())

# 3) 圆 (闭合), 半径 r=50, 真值周长=2pi r; 圆规测得应略小
th = np.linspace(0, 2*np.pi, 4000, endpoint=False)
circ = [(50+50*math.cos(t), 50+50*math.sin(t)) for t in th]
for eps in [20.0, 10.0, 5.0, 2.0]:
    L, q = ruler_walk(circ, True, eps)
    print(f"circle r50 eps={eps}: L={L:.2f} true={2*math.pi*50:.2f} ratio={L/(2*math.pi*50):.4f} npts={len(q)}")
