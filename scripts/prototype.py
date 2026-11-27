#!/usr/bin/env python3
# 梳齿（同相叠加）数值原型 v2：修正相位差定义 + 相邻列合并成齿。
# 两层余弦 + 弱二维值噪声。正式实现以 Go 为准。
import math
import numpy as np
from fractions import Fraction

N = 256
F1 = 1.0 / 16.0          # 第一层频率（周期/像素），周期 16px
A = 1.0
W = 0.10                 # 值噪声调制幅度
NOISE_SCALE = 4.0        # 值噪声特征尺度（像素）
T = 1.70                 # 点亮阈值（原始场 h）：无噪声同相峰值 2A，留 0.30 余量
H_MIN = 64               # 段高阈值：同列连续点亮像素 >= 64（256 的 25%）
MERGE_GAP = 8            # 相邻 <=8 列的合格亮段合并为同一个梳齿
PHI_TOL = math.radians(1.0)  # 相位判据容差：由 sqrt2 组最小非零残差(1.10deg)标定

def splitmix64(x):
    x = (x + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)

def make_perm(seed):
    p = list(range(256))
    state = seed
    for i in range(255, 0, -1):
        state = splitmix64(state)
        j = state % (i + 1)
        p[i], p[j] = p[j], p[i]
    return p + p

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def value_noise(Wsize, Hsize, scale, seed):
    perm = make_perm(seed)
    def grad(ix, iy):
        h = perm[(perm[ix & 255] + iy) & 511]
        return h / 127.5 - 1.0
    xs = np.arange(Wsize) / scale
    ys = np.arange(Hsize) / scale
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = xs - x0
    fy = ys - y0
    u = fade(fx)
    v = fade(fy)
    out = np.empty((Hsize, Wsize))
    for j in range(Hsize):
        vy = v[j]
        c00 = np.array([grad(x0[i], y0[j]) for i in range(Wsize)])
        c10 = np.array([grad(x0[i] + 1, y0[j]) for i in range(Wsize)])
        c01 = np.array([grad(x0[i], y0[j] + 1) for i in range(Wsize)])
        c11 = np.array([grad(x0[i] + 1, y0[j] + 1) for i in range(Wsize)])
        top = c00 + (c10 - c00) * u
        bot = c01 + (c11 - c01) * u
        out[j] = top + (bot - top) * vy
    return out

noise2d = value_noise(N, N, NOISE_SCALE, 11)
xx = np.arange(N)
# 几何阈值 T 对应的理论相位容差：2A cos(|delta|/2) - w >= T
# 即 |delta| <= 2 acos((T+w)/(2A))。仅用于相位扫描的理论对比，不用于分类。
EPS_GEOM = 2 * math.acos((T + W) / (2 * A))
# 分类用精确相位判据容差，由无理数组的最小近重合残差标定。
EPS = PHI_TOL
print(f"几何容差 eps_geom={math.degrees(EPS_GEOM):.2f}deg  分类容差 eps={math.degrees(EPS):.2f}deg  "
      f"H_MIN={H_MIN}, T={T}, w={W}\n")

def build(r, delta):
    f2 = r * F1
    l1 = np.cos(2 * math.pi * F1 * xx)
    l2 = np.cos(2 * math.pi * f2 * xx + delta)
    col = A * (l1 + l2)
    return col[None, :] + W * noise2d

def lit_runs(h):
    """每列高度 >= H_MIN 的连续点亮段：[(列, 段高), ...]"""
    lit = h >= T
    segs = []
    for c in range(N):
        run = 0
        for j in range(N):
            if lit[j, c]:
                run += 1
            else:
                if run >= H_MIN:
                    segs.append((c, run))
                run = 0
        if run >= H_MIN:
            segs.append((c, run))
    return segs

def phase_diff(c, r, delta):
    """两层在第 c 列的相位差，wrap 到 [-pi, pi]。"""
    d = 2 * math.pi * (r - 1) * F1 * c + delta
    return (d + math.pi) % (2 * math.pi) - math.pi

def detect(r, delta):
    h = build(r, delta)
    segs = lit_runs(h)
    aligned = [(c, hgt) for c, hgt in segs if abs(phase_diff(c, r, delta)) <= EPS]
    # 合并相邻列（同一亮峰）成梳齿
    clusters = []
    for c, hgt in aligned:
        if clusters and c - clusters[-1][-1][0] <= MERGE_GAP:
            clusters[-1].append((c, hgt))
        else:
            clusters.append([(c, hgt)])
    # 边界瞬态排除：若全图只有一个同相簇且贴在 c=0 边上，它是 delta=0 在原点的
    # 平凡重合，窗口内不存在公共周期的第二个证据，不构成周期性梳齿。
    teeth = clusters
    if len(clusters) == 1 and clusters[0][0][0] <= MERGE_GAP:
        teeth = []
    centers = [sum(c for c, _ in cl) / len(cl) for cl in teeth]
    return len(segs), len(teeth), centers, h

def predict_rational(r):
    fr = Fraction(r).limit_denominator(10000)
    p, q = fr.numerator, fr.denominator
    spacing = q / F1
    # 预测内部同相列：c 是 spacing 的整数倍，且要求至少两列（排除边界瞬态）
    cols = [c for c in range(N) if abs(phase_diff(c, float(fr), 0.0)) < 1e-9]
    if len(cols) < 2:
        cols = []
    return p, q, spacing, cols

groups = [
    ("A counterexample r=4/1 delta=0", 4.0, 0.0),
    ("B irrational  r=sqrt2 delta=0", math.sqrt(2), 0.0),
    ("C rational    r=4/3 delta=0", 4.0 / 3.0, 0.0),
]

for name, r, delta in groups:
    raw, teeth, centers, h = detect(r, delta)
    if abs(r - 4.0) < 1e-12 or abs(r * 3 - 4) < 1e-9:
        p, q, spacing, cols = predict_rational(r)
        pred = len(cols)
        gaps = np.diff(centers) if len(centers) > 1 else np.array([])
        print(f"{name}")
        print(f"  raw_runs={raw} teeth={teeth} predicted={pred} diff={abs(teeth-pred)} "
              f"spacing_pred={spacing} spacing_meas={np.round(gaps,2)}")
        print(f"  centers={np.round(centers,1)}")
    else:
        # c=0 在 delta=0 时恒为同相，统计最小残差时排除（无信息量）
        resid = sorted(abs(phase_diff(c, r, delta)) for c in range(1, N))
        minres = resid[0]
        print(f"{name}")
        print(f"  raw_runs={raw} teeth={teeth} predicted=0 diff={teeth} "
              f"min|phase diff|(c>=1)={math.degrees(minres):.2f} deg (eps={math.degrees(EPS):.2f})")
        print(f"  5 smallest residuals(deg): {[round(math.degrees(x),2) for x in resid[:5]]}")
        print(f"  被相位判据拒绝的几何亮段数: {raw - teeth}")
    print()

# 相位扫描：固定 r=4/1，5 度步进（前 30 度加密到 2 度以定位过渡带）
print("phase sweep r=4/1 (分类容差 eps=1deg；几何亮段容差 eps_geom=51.7deg):")
degs = list(range(0, 31, 2)) + list(range(45, 361, 15))
for deg in degs:
    delta = math.radians(deg)
    raw, teeth, centers, _ = detect(4.0, delta)
    dwrapped = abs((delta + math.pi) % (2*math.pi) - math.pi)
    flag = "ALIGNED" if dwrapped <= EPS else "off"
    print(f"  delta={deg:4d}deg raw={raw:3d} teeth={teeth:3d} {flag}")
