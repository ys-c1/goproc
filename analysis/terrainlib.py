"""地形简化分析公共库。

约定
----
* 高度图来自 Go 生成器导出的 GORAW001 二进制（24 字节头 + 行优先小端 float64）。
* 顶点曲率 κ 取离散拉普拉斯算子的幅值：κ(x,y) = |h[x+1,y]+h[x-1,y]+h[x,y+1]+h[x,y-1]-4h[x,y]|。
* 边界顶点强制保留（曲率置为 +inf，不进入曲率分布统计），简化只在内部顶点上做。
* 简化网格 = 保留顶点的 Delaunay 三角剖分 + 逐三角形线性插值，在完整网格上重采样算误差。

所有函数只依赖 numpy / scipy，无第三方数据读取。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import gaussian_kde

MAGIC = b"GORAW001"
HEADER = 24


def load_raw(path: str) -> np.ndarray:
    """读取 GORAW001 浮点高度图，返回 (n, n) 行优先数组。"""
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:8] != MAGIC:
        raise ValueError(f"非法文件头: {blob[:8]!r}，期望 {MAGIC!r}")
    nx = int(np.frombuffer(blob, dtype="<i8", offset=8, count=1)[0])
    ny = int(np.frombuffer(blob, dtype="<i8", offset=16, count=1)[0])
    if nx != ny:
        raise ValueError(f"仅支持正方形高度图，收到 {nx}x{ny}")
    h = np.frombuffer(blob, dtype="<f8", offset=HEADER, count=nx * ny)
    return h.reshape(ny, nx).copy()


def curvature(h: np.ndarray) -> np.ndarray:
    """逐顶点曲率（离散拉普拉斯幅值），边界置 +inf。"""
    n = h.shape[0]
    lap = np.zeros_like(h)
    lap[1:-1, 1:-1] = (
        h[2:, 1:-1] + h[:-2, 1:-1] + h[1:-1, 2:] + h[1:-1, :-2] - 4.0 * h[1:-1, 1:-1]
    )
    kappa = np.abs(lap)
    kappa[0, :] = np.inf
    kappa[-1, :] = np.inf
    kappa[:, 0] = np.inf
    kappa[:, -1] = np.inf
    return kappa


def interior_curvature(kappa: np.ndarray) -> np.ndarray:
    """返回内部顶点（有限曲率）的一维曲率样本。"""
    return kappa[np.isfinite(kappa)]


def point_grid(n: int) -> np.ndarray:
    """完整网格的 (x, y) 坐标数组，行优先，形状 (n*n, 2)。"""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    return np.column_stack([xs.ravel(), ys.ravel()])


def simplify_and_error(
    h: np.ndarray, kappa: np.ndarray, tau: float
) -> tuple[int, float, float, int]:
    """按曲率阈值 tau 简化一次，返回 (保留顶点数, RMSE, 最大绝对误差, 三角形数)。

    保留规则：边界顶点全部保留；内部顶点 κ >= tau 保留。
    tau=0 时内部顶点也全部保留（完整网格），误差为 0。
    """
    n = h.shape[0]
    keep = (kappa >= tau) | ~np.isfinite(kappa)
    pts = point_grid(n)[keep.ravel()]
    vals = h.ravel()[keep.ravel()]
    if tau <= 0.0:
        return int(keep.sum()), 0.0, 0.0, 2 * (n - 1) * (n - 1)
    tri = Delaunay(pts)
    interp = _LinearBarycentric(pts, vals, tri)
    grid = point_grid(n)
    approx = interp(grid)
    diff = approx - h.ravel()
    rmse = float(np.sqrt(np.mean(diff**2)))
    emax = float(np.max(np.abs(diff)))
    return int(keep.sum()), rmse, emax, tri.simplices.shape[0]


class _LinearBarycentric:
    """Delaunay 三角网上的逐三角形线性插值（向量化重心坐标）。"""

    def __init__(self, pts: np.ndarray, vals: np.ndarray, tri: Delaunay) -> None:
        self.tri = tri
        self.pts = pts
        self.vals = vals

    def __call__(self, q: np.ndarray) -> np.ndarray:
        idx = self.tri.find_simplex(q)
        if np.any(idx < 0):
            # 保留边界后查询点必在凸包内；出现外部点说明逻辑有误。
            raise ValueError("存在落在三角网凸包外的查询点")
        t = self.tri.transform[idx, :2, :]
        b12 = np.einsum("ijk,ik->ij", t, q - self.tri.transform[idx, 2, :])
        b0 = 1.0 - b12[:, 0] - b12[:, 1]
        verts = self.tri.simplices[idx]
        return (
            b0 * self.vals[verts[:, 0]]
            + b12[:, 0] * self.vals[verts[:, 1]]
            + b12[:, 1] * self.vals[verts[:, 2]]
        )


def sweep(h: np.ndarray, kappa: np.ndarray, taus: np.ndarray) -> dict[str, np.ndarray]:
    """对一组曲率阈值做完整简化扫描。"""
    rows = [simplify_and_error(h, kappa, float(t)) for t in taus]
    nvert, rmse, emax, ntri = zip(*rows)
    return {
        "tau": np.asarray(taus, dtype=float),
        "nvert": np.asarray(nvert, dtype=np.int64),
        "rmse": np.asarray(rmse, dtype=float),
        "emax": np.asarray(emax, dtype=float),
        "ntri": np.asarray(ntri, dtype=np.int64),
    }


def discrete_curvature(x: np.ndarray, y: np.ndarray, smooth: int = 3) -> np.ndarray:
    """平面曲线 (x, y) 的离散有向曲率 κ_g，端点为 NaN。

    κ_g = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)，中心差分。
    smooth>1 时先对 x、y 做宽度为 smooth 的简单滑动平均，抑制扫描档上的锯齿。
    """
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        x = np.convolve(x, kernel, mode="same")
        y = np.convolve(y, kernel, mode="same")
    xp = np.gradient(x)
    yp = np.gradient(y)
    xpp = np.gradient(xp)
    ypp = np.gradient(yp)
    den = (xp**2 + yp**2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        kg = (xp * ypp - yp * xpp) / den
    kg[[0, -1]] = np.nan
    return kg


def curvature_knee(
    x: np.ndarray, y: np.ndarray, predicted_index: int, smooth: int = 3
) -> tuple[int, np.ndarray]:
    """曲率拐点检测：离散曲率二阶差分变号点中，取距预测档位最近的一处。

    按任务约定，“曲率的二阶差分变号”作用在离散曲率序列 κ_g 上：
    d2[i] = κ_g[i+1] - 2κ_g[i] + κ_g[i-1]，变号即 d2[i-1]*d2[i] < 0（含由零跨出）。
    返回 (选中档位下标, 离散曲率序列)。
    """
    kg = discrete_curvature(x, y, smooth=smooth)
    d2 = np.full_like(kg, np.nan)
    inner = slice(1, len(kg) - 1)
    d2[inner] = kg[2:] - 2.0 * kg[1:-1] + kg[:-2]
    sign_changes: list[int] = []
    for i in range(2, len(kg) - 1):
        a, b = d2[i - 1], d2[i]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0.0:
            sign_changes.append(i)
    if not sign_changes:
        raise ValueError("未检测到曲率二阶差分变号点")
    chosen = min(sign_changes, key=lambda i: abs(i - predicted_index))
    return chosen, kg


def curvature_mode(samples: np.ndarray, grid: int = 1024) -> tuple[float, float]:
    """用高斯核密度估计曲率分布的众数，返回 (众数曲率, 对应经验分位)。"""
    kde = gaussian_kde(samples)
    lo = float(np.quantile(samples, 0.001))
    hi = float(np.quantile(samples, 0.999))
    xs = np.linspace(lo, hi, grid)
    dens = kde(xs)
    mode = float(xs[int(np.argmax(dens))])
    q = float(np.mean(samples <= mode))
    return mode, q


def ecdf_quantile(samples: np.ndarray, q: float) -> float:
    """经验分布函数的分位数（线性插值）。"""
    return float(np.quantile(samples, q))


def retained_count(kappa: np.ndarray, tau: float) -> int:
    """不做三角剖分，仅凭曲率阈值预测保留顶点数（边界 + κ>=tau 的内部顶点）。"""
    n_boundary = 4 * (kappa.shape[0] - 1)
    n_interior_keep = int(np.count_nonzero(kappa[np.isfinite(kappa)] >= tau))
    return n_boundary + n_interior_keep
