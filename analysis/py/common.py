"""共用工具：裸 float64 读取、窗口、解析谱与混叠能量。

符号约定（全部以“粗像素”为空间单位，格距 Δ）：
  f0 : 值噪声格点基频（周期/像素），连续场为 v(f0*x, f0*y)
  fN = 1/(2Δ) 奈奎斯特
  ξ  = f0 - fN : 频率超出量（仅在 f0>fN 时非零）
"""
from __future__ import annotations

import os

import numpy as np


_PYDIR = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(_PYDIR), "raw")


def read_raw(path: str, n: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def dft2(z: np.ndarray) -> np.ndarray:
    """中心化的幅度谱 |Z|（未归一化，调用方自行归一）。"""
    return np.abs(np.fft.fftshift(np.fft.fft2(z)))


def radial_profile(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = spec.shape[0]
    cy = cx = n // 2
    fy, fx = np.mgrid[0:n, 0:n]
    r = np.hypot(fy - cy, fx - cx)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), spec.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1), np.arange(len(tbin))


# ---------------------------------------------------------------- 解析值噪声谱
# 1D 值噪声：格点 i.i.d. 单位方差，smoothstep fade(t)=t^3(t(6t-15)+10)。
# 内插核 w 分段三次，[-1,1] 紧支；其傅里叶变换 W(f) 数值积分得到（采样无关的常数核）。

def _interp_kernel(ts: np.ndarray) -> np.ndarray:
    """值噪声基函数 B(u)：格点 i 的单位权重对位置 x 的插值贡献，u=x-i。

    x∈[i,i+1] 时 v=(1-fade(t))g_i+fade(t)g_{i+1}（t=x-i），故
      u∈[0,1] : B(u)=1-fade(u)
      u∈[-1,0]: B(u)=fade(1+u)
    紧支 [-1,1]，整数格点处 B(0)=1、B(±1)=0。
    """
    w = np.zeros_like(ts)
    m = (ts >= 0) & (ts <= 1)
    t = ts[m]
    w[m] = 1.0 - (t * t * t * (t * (t * 6 - 15) + 10))   # 1 - fade(t)
    m = (ts >= -1) & (ts < 0)
    t = ts[m] + 1.0                                        # fade(1+u)
    w[m] = t * t * t * (t * (t * 6 - 15) + 10)
    return w


_TS = np.linspace(-1, 1, 4001)
_WT = _interp_kernel(_TS)

# 高分辨率 FFT 预算基函数谱 B(f)：在细时间格采样紧支基函数，一次 FFT。
# 基函数在 |u|>~1 严格为零、谱随频率快速衰减，插值误差在目标频段可忽略。
_B_DT = 1 / 8192.0
_B_TGRID = np.arange(-2, 2, _B_DT)
_B_VAL = np.interp(_B_TGRID, _TS, _WT, left=0.0, right=0.0)
_B_WF = np.fft.fft(np.fft.ifftshift(_B_VAL)) * _B_DT
_B_FREQ = np.fft.fftfreq(len(_B_TGRID), d=_B_DT)
_B_ORDER = np.argsort(_B_FREQ)
_B_FREQ_S = _B_FREQ[_B_ORDER]
_B_WF_S = _B_WF[_B_ORDER]


def kernel_spectrum(f: np.ndarray) -> np.ndarray:
    """值噪声基函数谱 B(f)=∫B(u)e^{-i2πfu}du，FFT 预算 + 线性插值。"""
    f = np.atleast_1d(f).astype(float)
    re = np.interp(f, _B_FREQ_S, _B_WF_S.real, left=0.0, right=0.0)
    im = np.interp(f, _B_FREQ_S, _B_WF_S.imag, left=0.0, right=0.0)
    return re + 1j * im


def psd1d_analytic(f: np.ndarray, f0: float) -> np.ndarray:
    """S_1(f; f0) = (1/f0)|W(f/f0)|^2。"""
    return (1.0 / f0) * np.abs(kernel_spectrum(f / f0)) ** 2


def psd2d_analytic(fx: np.ndarray, fy: np.ndarray, f0: float) -> np.ndarray:
    wfx = kernel_spectrum((fx / f0).ravel())
    wfy = kernel_spectrum((fy / f0).ravel())
    w = (wfx[:, None] * wfy[None, :]).reshape(fx.shape)
    return (1.0 / f0) ** 2 * np.abs(w) ** 2


def variance_analytic(f0: float) -> float:
    """R(0)=∫S(f)df，2D 方差 = 1D 方差的平方（可分离）。"""
    f = np.linspace(-3 * f0, 3 * f0, 6001)
    s = psd1d_analytic(f, f0)
    v1 = np.trapz(s, f)
    return float(v1 * v1)


# ---------------------------------------------------------------- 解析混叠能量
def alias_energy_1d(f0: float, delta: float, fmax: float = 20.0) -> float:
    """E_alias^1D = ∫_{|f|>fN} S_1(f) df，fN=1/(2Δ)。"""
    fN = 1.0 / (2 * delta)
    if f0 <= fN:
        return 0.0
    f = np.linspace(fN, fmax, 12001)
    return float(2 * np.trapz(psd1d_analytic(f, f0), f))


def alias_energy_2d(f0: float, delta: float, fmax: float = 20.0) -> float:
    """E_alias^2D = 正方形奈奎斯特区外的 2D 谱积分。"""
    fN = 1.0 / (2 * delta)
    if f0 <= fN:
        return 0.0
    f = np.linspace(fN, fmax, 12001)
    tail1 = np.trapz(psd1d_analytic(f, f0), f)          # 单轴单边尾
    v1 = variance_analytic(f0) ** 0.5                    # 1D 总能量
    return float(v1 * v1 - (v1 - tail1) ** 2)


def folded_spectrum_1d(f0: float, delta: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """把连续 S_1 按采样折叠到离散 DFT 频率格，返回 (freqs, 折叠后 PSD 采样)。"""
    fs = 1.0 / delta
    freqs = np.fft.fftfreq(n, d=delta)
    folded = np.zeros(n)
    fgrid = np.linspace(-fs * 4, fs * 4, 80001)
    s = psd1d_analytic(fgrid, f0)
    binw = fs / n
    idx = np.round(((fgrid + fs / 2) % fs - fs / 2) / binw).astype(int) % n
    np.add.at(folded, idx, s * (fgrid[1] - fgrid[0]))
    return freqs, folded
