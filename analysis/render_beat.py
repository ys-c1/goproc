"""双光源 Lambert 渲染、拍频预测与逐行明暗翻转统计（Python 参考实现）。

渲染公式与 Go internal/shade 完全一致：
    gx,gy 为中心差分梯度，n=(-gx,-gy,1)/M，M=sqrt(1+gx^2+gy^2)
    I = (max(0,n·l1)+max(0,n·l2))/2，l=(cos eps sin phi, cos eps cos phi, sin eps)
坐标系 x 向右、y 向下，方位角 phi 从 +y 轴顺时针为正。

两个预测量（均无需渲染双光源）：
  N_beat  角空间拍频周期数，仅需山脊走向场 beta 与方位差 delta：
          N_beat = sin(delta) * Σ_pixel |cos gamma|，
          gamma 为两灯方向差连线(方位 phi_m±90)与山脊走向 beta 的夹角。
  N_sp    线性（无钳制）Lambert 解析翻转数，仅需地形梯度场：
          I_lin = nz sin eps - (cos eps/M)(gx sin phi_m + gy cos phi_m) cos(delta/2)
          在其逐行信号上数亮暗翻转，含近反向时 cos(delta/2) 对比度塌陷效应。
"""
import numpy as np

from pgmio import read_pgm, write_pgm


def gradients(h):
    """与 Go shade.Render 一致的中心差分（边界单侧）。"""
    gx = np.zeros_like(h)
    gy = np.zeros_like(h)
    gx[:, 1:-1] = h[:, 2:] - h[:, :-2]
    gx[:, 0] = h[:, 1] - h[:, 0]
    gx[:, -1] = h[:, -1] - h[:, -2]
    gy[1:-1, :] = h[2:, :] - h[:-2, :]
    gy[0, :] = h[1, :] - h[0, :]
    gy[-1, :] = h[-1, :] - h[-2, :]
    return gx, gy


def normals(h):
    gx, gy = gradients(h)
    M = np.sqrt(1.0 + gx * gx + gy * gy)
    return -gx / M, -gy / M, 1.0 / M, gx, gy


def light_vec(phi_deg, elev_deg):
    p, e = np.deg2rad(phi_deg), np.deg2rad(elev_deg)
    return np.cos(e) * np.sin(p), np.cos(e) * np.cos(p), np.sin(e)


def render(h, phi1, phi2, elev):
    """双光源灰度图（float64, 0..1），角度为度。"""
    nx, ny, nz, _, _ = normals(h)
    l1 = light_vec(phi1, elev)
    l2 = light_vec(phi2, elev)
    d1 = np.maximum(0.0, nx * l1[0] + ny * l1[1] + nz * l1[2])
    d2 = np.maximum(0.0, nx * l2[0] + ny * l2[1] + nz * l2[2])
    return np.clip((d1 + d2) / 2.0, 0.0, 1.0)


def render_linear(h, phi_m, delta, elev):
    """无钳制线性 Lambert 解析场（只用地形梯度，不产生双光源渲染图）。"""
    nx, ny, nz, gx, gy = normals(h)
    e = np.deg2rad(elev)
    pm = np.deg2rad(phi_m)
    envelope = np.cos(np.deg2rad(delta / 2.0))
    base = nz * np.sin(e)
    grad_term = (np.cos(e) / np.sqrt(1.0 + gx * gx + gy * gy)) * (
        gx * np.sin(pm) + gy * np.cos(pm))
    return base - envelope * grad_term


def clamp_fraction(h, phi, elev):
    """单灯在 (phi,elev) 下被 Lambert 钳制（n·l<0）的像素占比。"""
    nx, ny, nz, _, _ = normals(h)
    lx, ly, lz = light_vec(phi, elev)
    return float((nx * lx + ny * ly + nz * lz < 0).mean())


def strike_angles(gx, gy):
    """逐像素山脊走向 beta（等值线切向，度，模 180）；零梯度像素为 nan。

    梯度方位 alpha=atan2(gx,gy)，走向 beta=alpha+90 (mod 180)。
    """
    s = np.hypot(gx, gy)
    beta = (np.rad2deg(np.arctan2(gx, gy)) + 90.0) % 180.0
    beta[s < 1e-9] = np.nan
    return beta


def beat_periods_row(beta_row, phi_m, delta):
    """N_beat：角空间拍频周期数（单行）。

    两灯方向差连线方位为 phi_m-90（模 180 即 phi_m+90），
    gamma = beta-(phi_m-90)，几何因子 |cos gamma| = |sin(beta-phi_m)|。
    """
    gamma_factor = np.abs(np.sin(np.deg2rad(beta_row - phi_m)))
    return float(np.nansum(gamma_factor) * np.sin(np.deg2rad(delta)))


def count_flips(row, dead_band=0.0):
    """一行内亮暗翻转次数：差分符号改变即一次翻转。

    dead_band: 相邻像素差绝对值不超过该阈值视为平台，不计（用于量化稳健性）。
    """
    d = np.diff(np.asarray(row, dtype=np.float64))
    s = np.sign(d)
    s[np.abs(d) <= dead_band] = 0
    nz = s[s != 0]
    if nz.size < 2:
        return 0
    return int((nz[:-1] != nz[1:]).sum())


def quantize8(a):
    """与 Go hmap.EncodePGM 相同的 8-bit 量化后再映射回 0..1。"""
    return np.clip(np.floor(a * 255.0 + 0.5), 0, 255).astype(np.uint8).astype(np.float64) / 255.0


def analyze(heightmap_path, phi1, phi2, elev, tag, out_prefix):
    size, h = read_pgm(heightmap_path)
    _, _, _, gx, gy = normals(h)
    beta = strike_angles(gx, gy)
    delta = abs(phi2 - phi1)
    phi_m = (phi1 + phi2) / 2.0

    gray = render(h, phi1, phi2, elev)
    gray8 = quantize8(gray)
    write_pgm(f"{out_prefix}_{tag}.pgm", gray8)

    lin = render_linear(h, phi_m, delta, elev)
    lin8 = quantize8(lin)

    c1 = clamp_fraction(h, phi1, elev)
    c2 = clamp_fraction(h, phi2, elev)

    rows = []
    for y in range(size):
        n_beat = beat_periods_row(beta[y, :], phi_m, delta)
        n_sp = count_flips(lin[y, :])
        n_sp8 = count_flips(lin8[y, :])  # 与留档 PGM 同量化规则的解析预测（无渲染）
        n_obs = count_flips(gray8[y, :])
        n_obs_float = count_flips(gray[y, :])
        n_obs_db2 = count_flips(gray8[y, :], dead_band=2.0 / 255.0)
        rows.append({
            "row": y,
            "n_beat": n_beat,
            "n_sp": n_sp,
            "n_sp8": n_sp8,
            "n_obs": n_obs,
            "n_obs_float": n_obs_float,
            "n_obs_db2": n_obs_db2,
        })
    return {
        "tag": tag,
        "size": size,
        "phi1": phi1,
        "phi2": phi2,
        "phi_m": phi_m,
        "delta": delta,
        "elev": elev,
        "clamp1": c1,
        "clamp2": c2,
        "rows": rows,
        "gray": gray,
        "gray8": gray8,
        "lin": lin,
        "beta": beta,
    }
