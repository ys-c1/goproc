#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slope_powerlaw.py — 谱斜率 β -> 坡度分布形状/尾部 -> 可通行面积幂律 的完整复核流水线。

输入：cmd/goproc gen --format f64 --raw 产出的小端 float64 高度场（行优先，边长 N）。
输出（默认 artifacts/slope/）：
  - slope_map_<id>.png                 每组连续坡度图（分别存档）
  - walk_<id>_2tier.png / _4tier.png   两档 / 四档阈值的可通行掩膜分面图
  - fig_spectral.png / fig_shape.png / fig_powerlaw.png / fig_tail.png / fig_rows.png
  - threshold_rows.csv                 全部阈值逐行数据
  - group_metrics.json                 全部机器可读数值
  - tables.md                          生成的 Markdown 表格片段
并把 analysis/report_template.md 填充为 artifacts/slope_powerlaw_report.md。

所有随机性都在 Go 端由固定种子决定；本脚本只做确定性计算。
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager

# ---------------------------------------------------------------- 调色板（dataviz 参考色板前 3 色）
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, RED, INK, MUTED, GRID, SURFACE = "#6250d6", "#e34948", "#1b1b1a", "#6b6a65", "#e3e1d8", "#fcfcfb"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]

def setup_cjk():
    candidates = [
        "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.edgecolor"] = "#c9c7bd"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

setup_cjk()

# ---------------------------------------------------------------- 固定实验设计
N = 2048
SEED = 7
F0 = 1.0 / 256.0
LAC = 2.0
FIT_LAM = (12.0, 120.0)        # 谱斜率 β 的共同拟合波段（像素波长）
U_FIT = (0.10, 0.40)           # 幂律双对数拟合区间，单位 u = θ/σ̂_β
N_UFIT = 25
U4 = [0.15, 0.225, 0.30, 0.375]          # 四档阈值（u）
U2 = [0.225, 0.375]                      # 两档阈值（u），为 U4 的嵌套子集
TOL_FIT = 0.05                 # 对“最小二乘幂律”的相对偏差容差
TOL_BETA = 0.15                # 对“β 先验幂律”的相对偏差容差
V_ANGLE = 100.0                # 坡度角显示用垂直比例（仅标注，不影响面积）

GROUPS = [
    dict(id="G1", title="G1 基准", file="work/G1_k6_g050.f64", K=6, gain=0.50, color=BLUE,
         note="g=0.50（H=1），K=6：每层梯度等权，理论 β=4"),
    dict(id="G2", title="G2 改谱斜率（粗糙）", file="work/G2_k6_g065.f64", K=6, gain=0.65, color=ORANGE,
         note="g=0.65（H≈0.62），K=6：高频层占优，理论 β≈3.24"),
    dict(id="G3", title="G3 加层数", file="work/G3_k8_g050.f64", K=8, gain=0.50, color=AQUA,
         note="g=0.50，K=8：比 G1 多两个高频层，β 不变而尾部延展"),
]

# ---------------------------------------------------------------- 基础工具
def load_field(path: Path, n: int) -> np.ndarray:
    a = np.fromfile(path, dtype="<f8")
    if a.size != n * n:
        raise ValueError(f"{path}: 期望 {n*n} 个 float64，实得 {a.size}")
    return a.reshape(n, n)

def central_slopes(h: np.ndarray):
    gx = np.empty_like(h); gy = np.empty_like(h)
    gx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5
    gx[:, 0] = h[:, 1] - h[:, 0]
    gx[:, -1] = h[:, -1] - h[:, -2]
    gy[1:-1, :] = (h[2:, :] - h[:-2, :]) * 0.5
    gy[0, :] = h[1, :] - h[0, :]
    gy[-1, :] = h[-1, :] - h[-2, :]
    return gx, gy, np.hypot(gx, gy)

def radial_spectrum(h: np.ndarray):
    n = h.shape[0]
    w1 = np.hanning(n)
    win = w1[:, None] * w1[None, :]
    wcorr = float((win ** 2).mean())
    H = np.fft.fftshift(np.fft.fft2((h - h.mean()) * win))
    P = (H.real ** 2 + H.imag ** 2) / (n * n) / wcorr
    k1 = np.fft.fftshift(np.fft.fftfreq(n)) * 2.0 * np.pi
    KX, KY = np.meshgrid(k1, k1)
    K = np.hypot(KX, KY)
    edges = np.geomspace(2 * np.pi / n, np.pi, 90)
    idx = np.digitize(K.ravel(), edges)
    km, pm = [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() > 20:
            km.append(math.sqrt(edges[b - 1] * edges[b]))
            pm.append(float(P.ravel()[m].mean()))
    return np.asarray(km), np.asarray(pm)

def fit_beta(km, pm, lam_lo, lam_hi):
    lam = 2 * np.pi / km
    m = (lam >= lam_lo) & (lam <= lam_hi)
    x, y = np.log(km[m]), np.log(pm[m])
    sl, ic = np.polyfit(x, y, 1)
    yhat = sl * x + ic
    npt = int(m.sum())
    r2 = 1 - float(((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    resid = y - yhat
    se_sl = math.sqrt(float((resid ** 2).sum() / max(npt - 2, 1)) / float(((x - x.mean()) ** 2).sum()))
    return dict(beta=float(-sl), intercept=float(ic), A=float(math.exp(ic)),
                r2=r2, n_bins=npt, beta_se=float(se_sl),
                lam_used=(float(lam[m].min()), float(lam[m].max())))

# ---- β,K -> 每分量坡度标准差的先验闭式（五次 fade 值噪声核常数）
SIGMA_C2 = (256.0 ** 2 - 1.0) / 12.0 / (127.5 ** 2)
M2 = 181.0 / 231.0
SIGMA1_2 = (20.0 / 7.0) * SIGMA_C2 * M2     # 单层值噪声解析梯度方差 ≈0.752

def g_from_beta(beta: float) -> float:
    return 2.0 ** (1.0 - beta / 2.0)

def sigma_prior(beta: float, K: int, f0: float = F0):
    g = g_from_beta(beta)
    AK = sum(g ** i for i in range(K))
    octsum = sum((2.0 * g) ** (2 * i) for i in range(K))
    sig = math.sqrt(SIGMA1_2) * f0 / AK * math.sqrt(octsum)
    return sig, g, AK, octsum

def _fade(t):
    return t ** 3 * (t * (6 * t - 15) + 10)

def _phi(d):
    d = np.abs(np.asarray(d, dtype=float))
    return np.where(d <= 1.0, 1.0 - _fade(np.clip(d, 0.0, 1.0)), 0.0)

def D_fd(delta: float, n: int = 60001) -> float:
    """中心差分核能量 D(δ)=∫[(φ(t+δ)-φ(t-δ))/2]^2 dt；δ→0 时 D/δ²→20/7。"""
    T = 1.0 + delta + 1e-12
    t = np.linspace(-T, T, n)
    z = (_phi(t + delta) - _phi(t - delta)) * 0.5
    return float(np.trapezoid(z * z, t))

def sigma_prior_fd(beta: float, K: int, f0: float = F0):
    """含精确中心差分/混叠核修正的 β,K 先验坡度标准差。"""
    g = g_from_beta(beta)
    AK = sum(g ** i for i in range(K))
    var = 0.0
    for i in range(K):
        delta = f0 * LAC ** i
        var += g ** (2 * i) * SIGMA_C2 * M2 * D_fd(delta)
    return math.sqrt(var) / AK, g, AK, None

def beta_theory(gain: float) -> float:
    return 2.0 - 2.0 * math.log(gain, 2.0)

def ols_powerlaw(theta, A):
    x = np.log(np.asarray(theta)); y = np.log(np.asarray(A))
    p, c = np.polyfit(x, y, 1)
    yhat = p * x + c
    npt = len(x)
    r2 = 1 - float(((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    se = math.sqrt(float(((y - yhat) ** 2).sum() / max(npt - 2, 1)) / float(((x - x.mean()) ** 2).sum()))
    return float(p), float(c), r2, se, npt

def fmt_pct(x, d=2): return f"{100*x:.{d}f}%"
def rel(a, b): return float(a / b - 1.0)

# ---------------------------------------------------------------- 图件
def style_ax(ax):
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def save_slope_map(h, s, theta_hi, path: Path, title: str):
    """连续坡度图，2 倍抽稀。"""
    d = 2
    sd = s[::d, ::d]
    vmax = float(np.quantile(s, 0.995))
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUES)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(sd, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=13, weight="normal")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"坡度模  $|\nabla h|$ （rise/run，像元=1）", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def block_mean(mask: np.ndarray, b: int = 2):
    n = mask.shape[0]
    return mask.reshape(n // b, b, n // b, b).mean(axis=(1, 3))

def save_walk_panels(s, thetas, us, Ameas, path: Path, group_title: str, ncol: int):
    n = s.shape[0]; b = 2
    panels = []
    for th in thetas:
        panels.append(block_mean((s <= th).astype(float), b))
    k = len(thetas)
    nrow = math.ceil(k / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.4 * nrow), squeeze=False)
    for j, ax in enumerate(axes.ravel()):
        if j >= k:
            ax.axis("off"); continue
        ax.imshow(panels[j], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(rf"阈值 $\theta$={thetas[j]:.5f}（u={us[j]:g}）  可通行 {100*Ameas[j]:.2f}%", fontsize=11.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(group_title + "：可通行掩膜（蓝=坡度≤阈值）", fontsize=14, weight="normal", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=120); plt.close(fig)

# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--template", default="analysis/report_template.md")
    args = ap.parse_args()
    root = Path(args.root)
    outdir = root / "artifacts" / "slope"
    outdir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    csv_rows = []
    table_tokens = {}

    for cfg in GROUPS:
        gid, color = cfg["id"], cfg["color"]
        h = load_field(root / cfg["file"], N)
        gx, gy, s = central_slopes(h)
        Npix = N * N

        # ---------- 谱 β
        km, pm = radial_spectrum(h)
        bs = fit_beta(km, pm, *FIT_LAM)
        beta = bs["beta"]
        sig_beta, geff, AK, _ = sigma_prior_fd(beta, cfg["K"])
        sig_lead, _, _, octsum = sigma_prior(beta, cfg["K"])
        sig_param, gp, _, _ = sigma_prior_fd(beta_theory(cfg["gain"]), cfg["K"])
        sig_meas = math.sqrt(0.5 * (gx.var() + gy.var()))
        sig_mle = math.sqrt(float((s ** 2).mean()) / 2.0)
        kurt = lambda z: float(((z - z.mean()) ** 4).mean() / z.var() ** 2)

        # ---------- 幂律拟合（u 以先验 σβ 为尺）
        u_grid = np.geomspace(U_FIT[0], U_FIT[1], N_UFIT)
        th_grid = u_grid * sig_beta
        A_grid = np.array([(s <= th).mean() for th in th_grid])
        p_fit, logC, r2pl, p_se, npl = ols_powerlaw(th_grid, A_grid)
        C_fit = math.exp(logC)
        th_lo, th_hi = float(th_grid[0]), float(th_grid[-1])

        # ---------- 阈值行（两档 + 四档）
        rows = []
        for tier, us in (("两档", U2), ("四档", U4)):
            for u in us:
                th = u * sig_beta
                n_walk = int((s <= th).sum())
                Am = n_walk / Npix
                A_fit = C_fit * th ** p_fit
                A_beta = 0.5 * (th / sig_beta) ** 2     # = u^2/2
                rf, rb = rel(Am, A_fit), rel(Am, A_beta)
                ppb = 100 * (Am - A_beta)
                fit_bad = abs(rf) > TOL_FIT
                beta_bad = abs(rb) > TOL_BETA
                ang = math.degrees(math.atan(V_ANGLE * th))
                row = dict(group=gid, tier=tier, u=u, theta=th, angle=ang, n_walk=n_walk,
                           A_meas=Am, A_fit=float(A_fit), rel_fit=rf,
                           A_beta=float(A_beta), rel_beta=rb, pp_beta=ppb,
                           fit_bad=fit_bad, beta_bad=beta_bad)
                rows.append(row); csv_rows.append(row)

        # ---------- 尾部 / 极值
        R = Npix / 1000.0
        uR = math.sqrt(2.0 * math.log(R)); qR = 1.0 - 1.0 / R
        sR_emp = float(np.quantile(s, qR))
        smax = float(s.max())
        u_max = math.sqrt(2.0 * math.log(Npix))
        q999 = float(np.quantile(s, 0.999))

        # ---------- β 失准受控诊断（固定物理阈值=四档；换 β 预测）
        diag = []
        th4 = [u * sig_beta for u in U4]
        Am4 = [float((s <= t).mean()) for t in th4]
        for label, bw in [("β̂（实测）", beta),
                          ("β̂−0.5（低估谱陡度）", max(2.05, beta - 0.5)),
                          ("β̂+0.5（高估谱陡度）", beta + 0.5),
                          ("照搬默认 β=4.00", 4.0)]:
            sw, gw, _, _ = sigma_prior_fd(bw, cfg["K"])
            rels, flags, preds = [], 0, []
            for t, Am in zip(th4, Am4):
                Ap = 0.5 * (t / sw) ** 2
                r = rel(Am, Ap); rels.append(r); preds.append(Ap)
                if abs(r) > TOL_BETA: flags += 1
            diag.append(dict(label=label, beta=bw, g=gw, sigma=sw,
                             max_abs_rel=float(np.max(np.abs(rels))),
                             mean_abs_rel=float(np.mean(np.abs(rels))), n_flag=flags))

        m = dict(cfg)
        m.update(dict(
            beta=beta, beta_se=bs["beta_se"], beta_r2=bs["r2"], beta_bins=bs["n_bins"],
            beta_lam_lo=bs["lam_used"][0], beta_lam_hi=bs["lam_used"][1],
            beta_theory=beta_theory(cfg["gain"]), g_eff=geff,
            sig_beta=sig_beta, sig_lead=sig_lead, sig_param=sig_param,
            sig_meas=sig_meas, sig_mle=sig_mle,
            ratio_beta_meas=sig_beta / sig_meas,
            kurt_x=kurt(gx), kurt_y=kurt(gy),
            corr=float(np.corrcoef(gx.ravel()[::8], gy.ravel()[::8])[0, 1]),
            p_fit=p_fit, p_se=p_se, r2_pl=r2pl, n_pl=npl, C_fit=C_fit,
            theta_lo=th_lo, theta_hi=th_hi,
            A_lo=float(A_grid[0]), A_hi=float(A_grid[-1]),
            rows=rows, diag=diag,
            tail=dict(R=R, uR=uR, sR_emp=sR_emp, sR_beta=sig_beta * uR, sR_mle=sig_mle * uR,
                      q999=q999, smax=smax, u_max_obs=smax / sig_meas, u_max_gumbel=u_max),
        ))
        all_metrics.append(m)

        # ================= 图件（组内地图） =================
        save_slope_map(h, s, None, outdir / f"slope_map_{gid}.png",
                       rf"{cfg['title']}  坡度模图（{cfg['K']} 层, g={cfg['gain']:g}, $\hat\beta$={beta:.2f}）")
        A2 = [next(r["A_meas"] for r in rows if r["tier"] == "两档" and abs(r["u"] - u) < 1e-9) for u in U2]
        save_walk_panels(s, [u * sig_beta for u in U2], U2, A2,
                         outdir / f"walk_{gid}_2tier.png", cfg["title"], ncol=2)
        A4 = [next(r["A_meas"] for r in rows if r["tier"] == "四档" and abs(r["u"] - u) < 1e-9) for u in U4]
        save_walk_panels(s, th4, U4, A4,
                         outdir / f"walk_{gid}_4tier.png", cfg["title"], ncol=2)

        # 暂存汇总图数据
        m["_hist"] = np.histogram(s, bins=220, density=True)
        m["_km"] = km; m["_pm"] = pm; m["_fit"] = (bs["intercept"], beta)
        m["_u_pl"] = u_grid; m["_A_pl"] = A_grid
        usurv = np.geomspace(0.8, 5.2, 44)
        m["_usurv"] = usurv
        m["_surv"] = np.array([max((s > u * sig_meas).mean(), 1e-9) for u in usurv])

        print(f"[{gid}] beta={beta:.3f} (th {m['beta_theory']:.2f}) R2={bs['r2']:.5f} | "
              f"sigβ={sig_beta:.5f} meas={sig_meas:.5f} ratio={sig_beta/sig_meas:.3f} | "
              f"p={p_fit:.3f}±{p_se:.3f} R2pl={r2pl:.6f} | rows={len(rows)}")

    # ================================================================ CSV
    csv_path = outdir / "threshold_rows.csv"
    fields = ["group", "tier", "u", "theta", "angle", "n_walk", "N", "A_meas",
              "A_fitPL", "rel_fit", "A_betaPL", "rel_beta", "pp_beta", "fit_flag", "beta_flag"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(fields)
        for r in csv_rows:
            w.writerow([r["group"], r["tier"], f"{r['u']:.3f}", f"{r['theta']:.8f}", f"{r['angle']:.3f}",
                        r["n_walk"], N * N, f"{r['A_meas']:.8f}", f"{r['A_fit']:.8f}", f"{r['rel_fit']:.5f}",
                        f"{r['A_beta']:.8f}", f"{r['rel_beta']:.5f}", f"{r['pp_beta']:.4f}",
                        "BAD" if r["fit_bad"] else "ok", "BAD" if r["beta_bad"] else "ok"])

    # ================================================================ 汇总图
    # ---- 谱拟合
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for ax, m in zip(axes, all_metrics):
        km, pm = m["_km"], m["_pm"]; ic, beta = m["_fit"]
        lam = 2 * np.pi / km
        band = (lam >= FIT_LAM[0]) & (lam <= FIT_LAM[1])
        ax.loglog(2 * np.pi / km, pm, "o", ms=3.5, color=m["color"], alpha=0.55)
        kk = np.geomspace(km[band].min(), km[band].max(), 50)
        ax.loglog(2 * np.pi / kk, math.exp(ic) * kk ** (-beta), "-", color=INK, lw=2.2,
                  label=rf"$\beta={beta:.3f}$" + "\n" + rf"$R^2={m['beta_r2']:.4f}$")
        ax.axvspan(FIT_LAM[0], FIT_LAM[1], color=m["color"], alpha=0.08)
        ax.invert_xaxis(); style_ax(ax)
        ax.set_title(f"{m['id']} 径向功率谱", fontsize=12.5, weight="normal")
        ax.set_xlabel(r"波长 $\lambda$（像素）"); ax.set_ylabel("径向平均 P(k)")
        ax.legend(frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout(); fig.savefig(outdir / "fig_spectral.png", dpi=140); plt.close(fig)

    # ---- 坡度分布形状：原始尺度 + 标准化坍缩
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    uu = np.linspace(0, 4.2, 400)
    for m in all_metrics:
        cnt, edges = m["_hist"]; ctr = 0.5 * (edges[:-1] + edges[1:]); w = edges[1] - edges[0]
        ax[0].plot(ctr, cnt, color=m["color"], lw=1.8, label=rf"{m['id']} $\sigma$={m['sig_meas']:.5f}")
        # 标准化密度：把密度坐标换算到 u=s/σ
        ax[1].plot(ctr / m["sig_meas"], cnt * m["sig_meas"], color=m["color"], lw=1.8, label=m["id"])
    ray = uu / 1.0 ** 2 * np.exp(-uu ** 2 / 2.0)
    ax[1].plot(uu, ray, "--", color=INK, lw=2.2, label="Rayleigh(1) 理论")
    ax[0].set_title("(a) 坡度分布（原始尺度）：谱斜率/层数决定宽度", fontsize=12.5, weight="normal")
    ax[1].set_title("(b) 按实测尺度标准化后坍缩为 Rayleigh", fontsize=12.5, weight="normal")
    for a in ax:
        style_ax(a); a.set_xlim(0, None); a.legend(frameon=False, fontsize=10)
    ax[0].set_xlabel(r"坡度模 $|\nabla h|$"); ax[0].set_ylabel("概率密度")
    ax[1].set_xlabel(r"$u=|\nabla h|/\sigma_g$"); ax[1].set_ylabel(r"标准化密度 $\sigma_g\,p(s)$")
    ax[1].set_xlim(0, 4.2)
    fig.tight_layout(); fig.savefig(outdir / "fig_shape.png", dpi=140); plt.close(fig)

    # ---- 幂律拟合
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    for ax, m in zip(axes, all_metrics):
        ug, Ag = m["_u_pl"], m["_A_pl"]
        ax.loglog(ug, Ag, "o", ms=5, color=m["color"], label="实测 A(θ)")
        xx = np.geomspace(U_FIT[0], U_FIT[1], 50)
        # 用拟合 p,C（C 针对原始 θ），转换到 u 坐标：A=C (u σβ)^p
        ax.loglog(xx, m["C_fit"] * (xx * m["sig_beta"]) ** m["p_fit"], "-", color=INK, lw=2.2,
                  label=f"LS 幂律 p={m['p_fit']:.3f}")
        ax.loglog(xx, xx ** 2 / 2.0, "--", color=MUTED, lw=1.8, label="理论 u²/2（p=2）")
        for u in U4:
            Am = next(r["A_meas"] for r in m["rows"] if r["tier"] == "四档" and abs(r["u"] - u) < 1e-9)
            ax.loglog(u, Am, "s", ms=9, mfc="white", mec=m["color"], mew=1.8)
        for u in U2:
            Am = next(r["A_meas"] for r in m["rows"] if r["tier"] == "两档" and abs(r["u"] - u) < 1e-9)
            ax.loglog(u, Am, "D", ms=8, color=m["color"])
        ax.axvspan(U_FIT[0], U_FIT[1], color=m["color"], alpha=0.07)
        style_ax(ax)
        ax.set_title(f"{m['id']}  p={m['p_fit']:.3f}±{m['p_se']:.3f}, R²={m['r2_pl']:.5f}",
                     fontsize=12, weight="normal")
        ax.set_xlabel(r"$u=\theta/\hat{\sigma}_\beta$"); ax.set_ylabel("可通行面积占比 A")
        ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout(); fig.savefig(outdir / "fig_powerlaw.png", dpi=140); plt.close(fig)

    # ---- 尾部
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for m in all_metrics:
        ax.semilogy(m["_usurv"], m["_surv"], "-o", ms=3.5, color=m["color"], label=m["id"])
    uu = np.linspace(0.8, 5.2, 200)
    ax.semilogy(uu, np.exp(-uu ** 2 / 2), "--", color=INK, lw=2.0, label="Rayleigh 尾 exp(−u²/2)")
    style_ax(ax); ax.set_xlabel(r"$u=|\nabla h|/\sigma_g$（实测）"); ax.set_ylabel("超越概率  P(S>s)")
    ax.set_title(r"坡度分布尾部：标准化后近 Rayleigh，绝对尺度由谱斜率 $\beta$ 与层数 K 决定", fontsize=12.5, weight="normal")
    ax.legend(frameon=False, fontsize=10.5); ax.set_ylim(1e-6, 0.6)
    fig.tight_layout(); fig.savefig(outdir / "fig_tail.png", dpi=140); plt.close(fig)

    # ---- 逐行比对
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
    for ax, m in zip(axes, all_metrics):
        rs = m["rows"]; xs = np.arange(len(rs))
        ax.semilogy(xs, [r["A_meas"] * 100 for r in rs], "o", ms=8, color=INK, label="实测")
        ax.semilogy(xs, [r["A_fit"] * 100 for r in rs], "x", ms=9, mew=2, color=BLUE, label="LS 幂律预测")
        ax.semilogy(xs, [r["A_beta"] * 100 for r in rs], "^", ms=8, mfc="none", mew=1.8,
                    color=ORANGE, label=r"$\beta$ 先验 $\theta^2/(2\hat\sigma_\beta^2)$")
        ticklabels = ["两1", "两2", "四1", "四2", "四3", "四4"]
        ax.set_xticks(xs, ticklabels[:len(rs)], fontsize=9)
        style_ax(ax); ax.set_title(f"{m['id']} 逐阈值", fontsize=12, weight="normal")
        ax.set_xlabel("阈值（两档×2 + 四档×4）")
        if ax is axes[0]: ax.set_ylabel("可通行面积占比（%，对数轴）")
        ax.legend(frameon=False, fontsize=8.6, loc="lower right")
    fig.tight_layout(); fig.savefig(outdir / "fig_rows.png", dpi=140); plt.close(fig)

    # ================================================================ JSON（去掉私有大块）
    metrics_clean = []
    for m in all_metrics:
        mm = {k: v for k, v in m.items() if not k.startswith("_")}
        metrics_clean.append(mm)
    max_se = math.sqrt(0.25 / Npix)  # p=.5 最大二项 SD（占比绝对值）
    payload = dict(seed=SEED, N=N, Npix=Npix, f0=F0, lacunarity=LAC, V_angle=V_ANGLE,
                   fit_lambda_px=list(FIT_LAM), u_fit=list(U_FIT), U2=U2, U4=U4,
                   tol_fit=TOL_FIT, tol_beta=TOL_BETA,
                   binomial_sd_max_pp=100 * max_se, binomial_ci95_max_pp=100 * 1.96 * max_se,
                   groups=metrics_clean)
    (outdir / "group_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ================================================================ Markdown 片段
    def row_table(m):
        head = ["组", "档", "u=θ/σ̂β", "θ（rise/run）", f"坡角* V={V_ANGLE:g}", "可通行像元",
                "实测 A", "LS幂律预测", "幂律残差", "β先验预测", "β偏差", "判定(β)"]
        lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
        for r in m["rows"]:
            verdict = "⚠ 超差" if r["beta_bad"] else "合格"
            lines.append("| {g} | {t} | {u:.3f} | {th:.5f} | {ang:.2f}° | {nw:,} | {am} | {af} | {rf:+.1f}% | {ab} | {rb:+.1f}% | {vd} |".format(
                g=r["group"], t=r["tier"], u=r["u"], th=r["theta"], ang=r["angle"], nw=r["n_walk"],
                am=fmt_pct(r["A_meas"], 3), af=fmt_pct(r["A_fit"], 3), rf=100 * r["rel_fit"],
                ab=fmt_pct(r["A_beta"], 3), rb=100 * r["rel_beta"], vd=verdict))
        return "\n".join(lines)

    for m in all_metrics:
        table_tokens[f"ROWS_{m['id']}"] = row_table(m)

    # 总览表
    head = ["组", "g", "K", "β理论", "β̂(谱拟合)", "R²谱", "σ̂β", "σ实测", "σ̂β/σ", "峰度 x/y", "p_LS", "p SE", "R²幂律"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for m in all_metrics:
        lines.append("| {id} | {gain:g} | {K} | {bt:.2f} | {b:.3f}±{se:.3f} | {br:.4f} | {sb:.5f} | {sm:.5f} | {ratio:.3f} | {kx:.2f}/{ky:.2f} | {p:.3f} | {pse:.3f} | {r2:.5f} |".format(
            id=m["id"], gain=m["gain"], K=m["K"], bt=m["beta_theory"], b=m["beta"], se=m["beta_se"], br=m["beta_r2"],
            sb=m["sig_beta"], sm=m["sig_meas"], ratio=m["ratio_beta_meas"], kx=m["kurt_x"], ky=m["kurt_y"],
            p=m["p_fit"], pse=m["p_se"], r2=m["r2_pl"]))
    table_tokens["SUMMARY_TABLE"] = "\n".join(lines)

    # 拟合区间表
    head = ["组", "拟合 u 区间", "θ 区间（rise/run）", "对应 A 区间", "LS 点数", "p", "R²", "C"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for m in all_metrics:
        lines.append(f"| {m['id']} | [{U_FIT[0]:.2f}, {U_FIT[1]:.2f}] | [{m['theta_lo']:.5f}, {m['theta_hi']:.5f}] "
                     f"| [{100*m['A_lo']:.2f}%, {100*m['A_hi']:.2f}%] | {m['n_pl']} | {m['p_fit']:.4f}±{m['p_se']:.4f} "
                     f"| {m['r2_pl']:.6f} | {m['C_fit']:.3e} |")
    table_tokens["FIT_TABLE"] = "\n".join(lines)

    # 尾部表
    head = ["组", "σ实测", "s_R 实测(R=N/1000)", "s_R β预测", "比值", "99.9%分位", "最大值", "u_max实测", "Gumbel √(2lnN)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for m in all_metrics:
        t = m["tail"]
        lines.append(f"| {m['id']} | {m['sig_meas']:.5f} | {t['sR_emp']:.5f} | {t['sR_beta']:.5f} "
                     f"| {t['sR_emp']/t['sR_beta']:.3f} | {t['q999']:.5f} | {t['smax']:.5f} "
                     f"| {t['u_max_obs']:.2f} | {t['u_max_gumbel']:.2f} |")
    table_tokens["TAIL_TABLE"] = "\n".join(lines)

    # β 失准诊断表
    head = ["组", "采用的 β", "反演 g", "σ̂", "四档最大相对偏差", "平均相对偏差", f"超差档数(τβ={int(100*TOL_BETA)}%)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for m in all_metrics:
        for d in m["diag"]:
            lines.append(f"| {m['id']} | {d['label']} → {d['beta']:.3f} | {d['g']:.3f} | {d['sigma']:.5f} "
                         f"| {100*d['max_abs_rel']:.1f}% | {100*d['mean_abs_rel']:.1f}% | {d['n_flag']}/4 |")
    table_tokens["WRONGBETA_TABLE"] = "\n".join(lines)

    table_tokens["TOL_FIT_PCT"] = str(int(100 * TOL_FIT))
    table_tokens["TOL_BETA_PCT"] = str(int(100 * TOL_BETA))
    table_tokens["MAXSE_PP"] = f"{payload['binomial_ci95_max_pp']:.3f}"
    table_tokens["SPECTRAL_INTERVAL"] = f"{FIT_LAM[0]:g}–{FIT_LAM[1]:g}"

    # 结论速览（自动取数，避免正文与表格数字漂移）
    hb = ["- 三组共同谱拟合波段 λ∈[{}] px，径向周期图 + Hann 窗、对数 bin、双对数 OLS：".format(table_tokens["SPECTRAL_INTERVAL"])]
    for m in all_metrics:
        hb.append(
            f"  - **{m['id']}**（g={m['gain']:g}, K={m['K']}）：理论 β=2−2log₂g={m['beta_theory']:.2f}，"
            f"实测 **β̂={m['beta']:.3f}±{m['beta_se']:.3f}**（R²={m['beta_r2']:.4f}）；"
            f"幂律指数 **p={m['p_fit']:.3f}±{m['p_se']:.3f}**（拟合 u∈[{U_FIT[0]:.2f},{U_FIT[1]:.2f}]，"
            f"R²={m['r2_pl']:.5f}）；β,K 先验 σ̂β/实测 σ={m['ratio_beta_meas']:.3f}。")
    n_beta_bad = sum(1 for r in csv_rows if r["beta_bad"])
    n_fit_bad = sum(1 for r in csv_rows if r["fit_bad"])
    wb_bad = sum(d["n_flag"] for m in all_metrics for d in m["diag"] if d["label"].startswith(("β̂−", "β̂+", "照搬")))
    table_tokens["HEADLINE_BULLETS"] = "\n".join(hb)
    table_tokens["N_BETA_BAD"] = str(n_beta_bad)
    table_tokens["N_FIT_BAD"] = str(n_fit_bad)
    table_tokens["N_ROWS"] = str(len(csv_rows))
    table_tokens["WRONGBETA_BAD"] = str(wb_bad)

    (outdir / "tables.md").write_text("\n\n".join(
        f"### {k}\n\n{v}" for k, v in table_tokens.items() if k.startswith(("ROWS", "SUMMARY", "FIT", "TAIL", "WRONGBETA"))),
        encoding="utf-8")

    # ================================================================ 填充报告模板
    tpl_path = root / args.template
    report_path = root / "artifacts" / "slope_powerlaw_report.md"
    if tpl_path.exists():
        text = tpl_path.read_text(encoding="utf-8")
        for k, v in table_tokens.items():
            text = text.replace("@@" + k + "@@", str(v))
        leftover = [tok for tok in set(__import__("re").findall(r"@@[A-Z0-9_]+@@", text))]
        if leftover:
            print("WARNING 未填充占位符:", leftover)
        report_path.write_text(text, encoding="utf-8")
        print("report ->", report_path)
    print("metrics ->", outdir / "group_metrics.json")
    print("csv ->", csv_path)

if __name__ == "__main__":
    main()
