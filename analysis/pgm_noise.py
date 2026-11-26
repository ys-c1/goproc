"""PGM P5 读取与 Go 噪声管线的独立 Python 复刻，用于交叉验证。

只依赖标准库与 numpy；复刻 internal/noise、internal/fbm、internal/stretch
的算法，不读取 Go 侧任何代码路径之外的数据。
"""
import numpy as np

MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


class ValueNoise:
    """与 internal/noise/noise.go 逐位对应的值噪声。"""

    def __init__(self, seed: int):
        p = list(range(256))
        state = seed
        for i in range(255, 0, -1):
            state = splitmix64(state)
            j = state % (i + 1)
            p[i], p[j] = p[j], p[i]
        self.perm = np.array([p[i & 255] for i in range(512)], dtype=np.int64)

    def _grad(self, ix, iy):
        h = self.perm[(self.perm[ix & 255] + iy) & 511]
        return h / 127.5 - 1.0

    def value(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx = x - x0
        fy = y - y0
        u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
        c00 = self._grad(x0, y0)
        c10 = self._grad(x0 + 1, y0)
        c01 = self._grad(x0, y0 + 1)
        c11 = self._grad(x0 + 1, y0 + 1)
        nx0 = c00 + (c10 - c00) * u
        nx1 = c01 + (c11 - c01) * u
        return nx0 + (nx1 - nx0) * v


def fbm_eval(src: ValueNoise, x, y, octaves: int, lacunarity: float, gain: float) -> np.ndarray:
    amp = 1.0
    freq = 1.0
    total = np.zeros_like(np.asarray(x, dtype=np.float64), dtype=np.float64)
    norm = 0.0
    for _ in range(octaves):
        total += src.value(np.asarray(x) * freq, np.asarray(y) * freq) * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


def warp_xy(x, y, theta: float, ratio: float):
    """与 internal/stretch/stretch.go 的 Warp.Apply 对应。"""
    if ratio == 0:
        return x, y
    c, s = np.cos(theta), np.sin(theta)
    u = c * x + s * y
    v = -s * x + c * y
    u = u + ratio * v
    return c * u - s * v, s * u + c * v


def read_pgm(path: str) -> np.ndarray:
    """读取八位 P5 PGM，返回 float64 二维数组（0..255 原值）。"""
    with open(path, "rb") as fh:
        tokens = []
        # 解析三个头 token：魔数、宽、高、最大值（P5 共 4 个）
        while len(tokens) < 4:
            line = fh.readline()
            if not line:
                raise ValueError("PGM 头不完整")
            if line.startswith(b"#"):
                continue
            tokens.extend(line.split())
        if tokens[0] != b"P5":
            raise ValueError("仅支持 P5")
        width, height, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
        if maxval != 255:
            raise ValueError("仅支持 maxval=255")
        raw = fh.read(width * height)
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width).astype(np.float64)
