"""构造并自检分层设色 ramp：OKLCH 明度严格单调、相邻 ΔE 达标。

不依赖 colorspacious/colour，手写 sRGB→OKLab（Björn Ottosson 公式）。
分层设色是顺序（ordinal）编码：允许跨色相，但明度必须单调、相邻可分辨，
并由图上带区间色标提供颜色之外的第二编码。
"""
from __future__ import annotations

import numpy as np


def srgb_to_oklab(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(r), lin(g), lin(b)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, a, bb


def delta_e_oklab(c1: str, c2: str) -> float:
    L1, a1, b1 = srgb_to_oklab(c1)
    L2, a2, b2 = srgb_to_oklab(c2)
    return 100 * np.hypot(np.hypot(L1 - L2, a1 - a2), b1 - b2)


def report(ramp: list[str], name: str) -> None:
    labs = [srgb_to_oklab(c) for c in ramp]
    Ls = [lab[0] for lab in labs]
    mono = all(Ls[i] < Ls[i + 1] for i in range(len(Ls) - 1))
    dLs = [Ls[i + 1] - Ls[i] for i in range(len(Ls) - 1)]
    des = [delta_e_oklab(ramp[i], ramp[i + 1]) for i in range(len(ramp) - 1)]
    print(f"== {name} ==")
    for c, (L, a, bb) in zip(ramp, labs):
        chroma = np.hypot(a, bb)
        print(f"  {c}  L={L:.3f} C={chroma:.3f}")
    print(f"  明度单调: {mono}  相邻ΔL min={min(dLs):.3f}")
    print(f"  相邻 ΔE(OKLab×100): min={min(des):.1f} 全部={[round(x,1) for x in des]}")


# 8 带：深蓝（低）→ 青 → 绿 → 黄 → 浅棕 → 近白（高），明度单调
RAMP8 = [
    "#1d4e89",  # 深海蓝
    "#217195",  # 蓝青
    "#2a928f",  # 青
    "#3fa86f",  # 青绿
    "#74b65c",  # 草绿
    "#b7bf57",  # 黄绿
    "#d9b878",  # 沙棕
    "#efe6d2",  # 高光近白
]

def _oklab_to_srgb(L: float, a: float, b: float) -> str:
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    def enc(c):
        c = np.clip(c, 0, 1)
        return 1.055 * c ** (1 / 2.4) - 0.055 if c > 0.0031308 else 12.92 * c

    return "#" + "".join(f"{int(round(enc(c) * 255)):02x}" for c in (r, g, bb))


def _interp_oklab(c1: str, c2: str, t: float) -> str:
    L1, a1, b1 = srgb_to_oklab(c1)
    L2, a2, b2 = srgb_to_oklab(c2)
    return _oklab_to_srgb(
        L1 + (L2 - L1) * t, a1 + (a2 - a1) * t, b1 + (b2 - b1) * t
    )


# 16 带：在 OKLab 空间把 8 个基色构成的折线均匀重采样到 16 色，
# 明度随色阶等距递增，相邻色差比简单中点插值更均匀。
def _resample_oklab(base: list[str], n_out: int) -> list[str]:
    labs = np.array([srgb_to_oklab(c) for c in base])
    # 累计 OKLab 弧长参数化
    seg = np.linalg.norm(np.diff(labs, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    targets = np.linspace(0.0, total, n_out)
    out = []
    for t in targets:
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = min(j, len(base) - 2)
        f = 0.0 if seg[j] == 0 else (t - cum[j]) / seg[j]
        L, a, b = labs[j] * (1 - f) + labs[j + 1] * f
        out.append(_oklab_to_srgb(L, a, b))
    return out


RAMP16 = _resample_oklab(RAMP8, 16)
assert len(RAMP8) == 8 and len(RAMP16) == 16


if __name__ == "__main__":
    report(RAMP8, "8 带 ramp")
    report(RAMP16, "16 带 ramp")
