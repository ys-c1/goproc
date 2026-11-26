"""各向异性地形的谱分析与梯度方向直方图核心计算（不含绘图）。

角度约定：方向角模 180 度；谱角与梯度角共用同一坐标系，
走向为 strike 的脊线，其波矢/梯度主方向为 strike+90（模 180）。

用法: python3 -I analyze_core.py terrain.raw size
输出 JSON 到 stdout。
"""
import json
import sys

import numpy as np

BINS = 180
ANGS = np.arange(BINS, dtype=float)  # 0..179 度


def load_raw(path, size):
    a = np.fromfile(path, dtype="<f8")
    if a.size != size * size:
        raise SystemExit(f"raw 尺寸不符: {a.size} != {size*size}")
    return a.reshape(size, size)


def smooth_circular(p, sigma_deg, bins=BINS):
    """周期 180 度的高斯平滑，标准循环卷积，不错位。"""
    step = 180.0 / bins
    radius = int(np.ceil(3 * sigma_deg / step))
    x = np.arange(-radius, radius + 1) * step
    kernel = np.exp(-0.5 * (x / sigma_deg) ** 2)
    kernel /= kernel.sum()
    out = np.zeros(bins)
    for k, w in zip(np.arange(-radius, radius + 1), kernel):
        out += w * np.roll(p, k)
    return out


def _gauss2d_fft(p, sigma_px):
    """频域高斯平滑，避免 -I 模式下依赖 scipy。"""
    n = p.shape[0]
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k)
    g = np.exp(-2 * np.pi ** 2 * sigma_px ** 2 * (kx ** 2 + ky ** 2) / n ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(p) * g))


def spectrum_parts(z, blur_sigma=1.5, smooth_sigma=3.0):
    """返回角向谱剖面（均值归一化）、惯性张量特征量与二维平滑功率谱。"""
    n = z.shape[0]
    w = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    f = np.fft.fftshift(np.fft.fft2((z - z.mean()) * w))
    p2d = _gauss2d_fft(np.abs(f) ** 2, blur_sigma)

    cy = cx = n // 2
    kx = np.arange(n) - cx
    ky = np.arange(n) - cy
    KX, KY = np.meshgrid(kx, ky)  # 行=ky、列=kx，与 z[y,x] 一致
    r = np.hypot(KX, KY)
    rmax = 0.6 * n / 2
    mask = (r > 2) & (r < rmax)

    # 谱惯性张量 J = Σ P kkᵀ；两条特征轴即谱的两条主轴
    j = np.array([
        [np.sum(p2d[mask] * KX[mask] ** 2), np.sum(p2d[mask] * KX[mask] * KY[mask])],
        [np.sum(p2d[mask] * KX[mask] * KY[mask]), np.sum(p2d[mask] * KY[mask] ** 2)],
    ])
    vals, vecs = np.linalg.eigh(j)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    v1 = vecs[:, 0]
    major_axis = float(np.degrees(np.arctan2(v1[1], v1[0])) % 180.0)
    minor_axis = (major_axis + 90.0) % 180.0
    rho = float(np.sqrt(vals[0] / vals[1]))                    # 谱轴比
    eta = float((vals[0] - vals[1]) / (vals[0] + vals[1]))    # 各向异性度

    # 功率加权角向剖面，再周期平滑、按均值归一化（跨场可比）
    ang = np.degrees(np.arctan2(KY[mask], KX[mask])) % 180.0
    prof, _ = np.histogram(ang, bins=np.linspace(0, 180, BINS + 1), weights=p2d[mask])
    prof = smooth_circular(prof, smooth_sigma)
    prof_mean = prof / prof.mean()
    prof_max = prof / prof.max()
    return prof_mean, prof_max, rho, eta, major_axis, minor_axis, p2d


def low_band_profiles(z, low_bands=(0, 1), rmax_frac=0.6,
                      blur_sigma=1.5, smooth_sigma=3.0):
    """返回指定低频径向带的均值归一化角向剖面，供谐波回查。"""
    n = z.shape[0]
    w = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    f = np.fft.fftshift(np.fft.fft2((z - z.mean()) * w))
    p2d = _gauss2d_fft(np.abs(f) ** 2, blur_sigma)
    c = n // 2
    k = np.arange(n) - c
    KX, KY = np.meshgrid(k, k)
    r = np.hypot(KX, KY)
    ang = np.degrees(np.arctan2(KY, KX)) % 180.0
    rmax = rmax_frac * n / 2
    edges = np.linspace(2, rmax, 7)
    out = {}
    for b in low_bands:
        m = (r >= edges[b]) & (r < edges[b + 1])
        prof, _ = np.histogram(ang[m], bins=np.linspace(0, 180, BINS + 1),
                               weights=p2d[m])
        prof = smooth_circular(prof, smooth_sigma)
        out[b] = prof / prof.mean()
    return out


def _circdist(a, b):
    return abs((a - b + 90) % 180 - 90)


def radial_band_lobes(z, nbands=6, rmax_frac=0.6, blur_sigma=1.5,
                      smooth_sigma=3.0, height_thr=1.8,
                      cluster_tol=5.0, min_bands=2, min_sep_deg=15.0):
    """分径向带的谱瓣检测 + 跨带一致性 + 间距聚类。

    全局角向求和会把高频旁瓣淹没在主脊系功率里；分带后旁瓣在自身
    频带内按相对强度显形。确认规则：
      1. 每个径向带内做角向剖面、均值归一化，取越高度阈值的局部极大；
      2. 跨带把角度在 cluster_tol 内的命中聚成一簇，簇须在 >=min_bands
         个径向带复现（压制单带散斑）；
      3. 簇代表方向再过间距阈值 min_sep_deg（与 HOG 同一间距阈值），
         贪心合并过近簇；
      4. 识别 90 度等距的网格谐波伪影并标记，不计入主方向。
    返回 (带明细, 确认方向 list[dict], 网格谐波角 list[float])。
    """
    n = z.shape[0]
    w = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    f = np.fft.fftshift(np.fft.fft2((z - z.mean()) * w))
    p2d = _gauss2d_fft(np.abs(f) ** 2, blur_sigma)
    c = n // 2
    k = np.arange(n) - c
    KX, KY = np.meshgrid(k, k)
    r = np.hypot(KX, KY)
    ang = np.degrees(np.arctan2(KY, KX)) % 180.0
    rmax = rmax_frac * n / 2
    edges = np.linspace(2, rmax, nbands + 1)

    band_detail, all_hits, band_profiles = [], [], []
    for b in range(nbands):
        m = (r >= edges[b]) & (r < edges[b + 1])
        prof, _ = np.histogram(ang[m], bins=np.linspace(0, 180, BINS + 1),
                               weights=p2d[m])
        prof = smooth_circular(prof, smooth_sigma)
        prof = prof / prof.mean()
        band_profiles.append(prof)
        hits = [
            (float(i), float(prof[i]))
            for i in range(BINS)
            if prof[i] >= height_thr
            and prof[i] >= prof[(i - 1) % BINS] and prof[i] >= prof[(i + 1) % BINS]
        ]
        band_detail.append({
            "band": b,
            "r_range": [round(float(edges[b]), 1), round(float(edges[b + 1]), 1)],
            "lobes": [{"angle": a, "strength": round(s, 3)} for a, s in hits],
        })
        all_hits.extend([(a, s, b) for a, s in hits])

    # 跨带角度聚类（周期 180 度）
    clusters = []  # [代表角, 最大强度, 出现带集合]
    for a, s, b in sorted(all_hits):
        placed = False
        for cl in clusters:
            if _circdist(a, cl[0]) <= cluster_tol:
                cl[1] = max(cl[1], s)
                cl[2].add(b)
                # 代表角取强度加权均值（周期处理）
                cl[0] = _weighted_angle([(cl[0], cl[1]), (a, s)])
                placed = True
                break
        if not placed:
            clusters.append([a, s, {b}])
    confirmed = [cl for cl in clusters if len(cl[2]) >= min_bands]

    # 间距阈值贪心合并（与 HOG 同阈值），强度高者保留
    confirmed.sort(key=lambda c: -c[1])
    kept = []
    for ang0, s0, bands0 in confirmed:
        if any(_circdist(ang0, k[0]) < min_sep_deg for k in kept):
            continue
        kept.append([ang0, s0, bands0])
    grid_angles = _grid_harmonics(kept)

    def _width_at(angle0, b):
        """簇在带 b 的半高宽（度）；该带无命中返回 None。"""
        prof = band_profiles[b]
        idx = int(round(angle0)) % BINS
        peak_v = prof[idx]
        if peak_v < height_thr:
            return None
        half = 1.0 + (peak_v - 1.0) * 0.5
        lo = idx
        while prof[lo % BINS] > half and (idx - lo) < 45:
            lo -= 1
        hi = idx
        while prof[hi % BINS] > half and (hi - idx) < 45:
            hi += 1
        return float(hi - lo)

    directions = []
    for a, s, bs in kept:
        widths = [w for b in sorted(bs) if (w := _width_at(a, b)) is not None]
        directions.append({
            "angle": round(a, 1),
            "strength": round(s, 3),
            "bands": sorted(bs),
            "grid_like": any(_circdist(a, g) <= 6.0 for g in grid_angles),
            "min_width_deg": round(min(widths), 1) if widths else 0.0,
            "median_width_deg": round(float(np.median(widths)), 1) if widths else 0.0,
        })
    directions.sort(key=lambda d: d["angle"])
    return band_detail, directions, sorted(grid_angles)


def _weighted_angle(weighted):
    """周期 180 度的强度加权平均角。"""
    sw, sx, sy = 0.0, 0.0, 0.0
    for a, wgt in weighted:
        t = np.radians(2 * a)  # 周期 180 -> 周期 360
        sx += wgt * np.cos(t)
        sy += wgt * np.sin(t)
        sw += wgt
    return float((np.degrees(np.arctan2(sy, sx)) / 2.0) % 180.0)


def _grid_harmonics(confirmed):
    """标记彼此近似 90 度等距且高频复现的网格谐波方向。"""
    if len(confirmed) < 2:
        return []
    angles = sorted(c[0] for c in confirmed)
    grid = []
    for a in angles:
        partner = (a + 90.0) % 180.0
        if any(_circdist(partner, b) <= 6.0 for b in angles):
            grid.extend([round(a, 1), round(partner, 1)])
    # 去重
    out = []
    for g in grid:
        if not any(_circdist(g, q) <= 6.0 for q in out):
            out.append(g)
    return out


def hog_parts(z, smooth_sigma=4.0):
    """幅度加权梯度方向直方图，方向模 180 度，返回均值/最大值两种归一化。"""
    gy, gx = np.gradient(z)
    gx, gy = gx[1:-1, 1:-1], gy[1:-1, 1:-1]
    mag = np.hypot(gx, gy)
    theta = np.degrees(np.arctan2(gy, gx)) % 180.0
    h, _ = np.histogram(theta, bins=np.linspace(0, 180, BINS + 1), weights=mag)
    h = smooth_circular(h, smooth_sigma)
    return h / h.mean(), h / h.max(), theta, mag


def pick_peaks(profile, height_thr, min_sep_deg, bins=BINS):
    """双阈值峰判定：高度阈值与间距阈值必须同时满足。

    候选先过高度阈值，再按强度贪心、周期 180 度上做间距抑制。
    返回 (保留峰, 被高度否决的局部极大, 被间距否决的候选)，均附方向角。
    """
    step = 180.0 / bins
    local_max = [
        (i * step, float(profile[i]))
        for i in range(bins)
        if profile[i] >= profile[(i - 1) % bins] and profile[i] >= profile[(i + 1) % bins]
    ]
    rejected_height = [
        {"angle": round(a, 1), "height": round(h, 3)}
        for a, h in local_max if h < height_thr
    ]
    tall = [(a, h) for a, h in local_max if h >= height_thr]
    accepted, rejected_sep = [], []
    for ang, hgt in sorted(tall, key=lambda t: -t[1]):
        if all(min((ang - a + 180) % 180, (a - ang + 180) % 180) >= min_sep_deg
               for a, _ in accepted):
            accepted.append((ang, hgt))
        else:
            rejected_sep.append({"angle": round(ang, 1), "height": round(hgt, 3)})
    accepted.sort()
    return (
        [{"angle": round(a, 1), "height": round(h, 3)} for a, h in accepted],
        rejected_height,
        rejected_sep,
    )


def main():
    path, size = sys.argv[1], int(sys.argv[2])
    z = load_raw(path, size)
    sp_mean, sp_max, rho, eta, major, minor, _ = spectrum_parts(z)
    h_mean, h_max, _, _ = hog_parts(z)
    out = {
        "spectrum_profile_mean": np.round(sp_mean, 4).tolist(),
        "spectrum_profile_max": np.round(sp_max, 4).tolist(),
        "hog_profile_mean": np.round(h_mean, 4).tolist(),
        "hog_profile_max": np.round(h_max, 4).tolist(),
        "axis_ratio": round(rho, 4),
        "anisotropy": round(eta, 4),
        "major_axis_deg": round(major, 2),
        "minor_axis_deg": round(minor, 2),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
