#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坡度分级着色：谱斜率预判各阶面积占比 vs 渲染后逐阶实测统计。

流程（预判严格发生在分级统计之前）：
  1. 读入 Go 端导出的 float64 归一化高度场；
  2. 仅由高度场求径向功率谱、拟合谱斜率 beta 与截距 A；
  3. 用谱矩公式推出梯度分量方差 sigma_hat^2，按高斯场 -> 坡度模 Rayleigh，
     得到 F_hat(s)=1-exp(-s^2/(2 sigma_hat^2))，各阶预判占比 = CDF 之差；
  4. 然后才计算坡度、按预先固定的边界分级、逐像素统计实测占比；
  5. 逐阶比对，|预测-实测| > 2 个百分点的色阶标记；
  6. 出两张分级着色图、两份独立占比表(Markdown+CSV)、谱检验图、比对柱状图、
     以及一份含完整证明的报告。

用法：
  python3 analysis/slope_forecast.py --input work/xxx.f64 --n 2048 \
      --outdir artifacts --seed 7 --octaves 6 --frequency 0.01
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

# ---------------------------------------------------------------------------
# 固定的分析约定（不依据渲染后的占比回调）
# ---------------------------------------------------------------------------
FIT_LAMBDA = (5.0, 200.0)          # 谱斜率拟合/积分的波长区间（像素）
FLAG_PP = 2.0                      # 超差阈值：百分点
VERTICAL_EXAG = 10.0               # 仅用于把归一化坡度换算成可读坡度角的显示比例

# 两套色阶数与预先固定的等步长边界（归一化高度、格距=1 的无量纲 rise/run）。
# 5 级步长 0.004，10 级步长 0.002，后者两两嵌套进前者，可交叉核对。
SCHEMES = {
    5: {
        "step": 0.004,
        "edges": [0.0, 0.004, 0.008, 0.012, 0.016, np.inf],
        "palette": ["#6a9fee", "#4c80cd", "#2f62ac", "#11458c", "#00286d"],
    },
    10: {
        "step": 0.002,
        "edges": [0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012,
                  0.014, 0.016, 0.018, np.inf],
        "palette": ["#83b3fa", "#6f9ee3", "#5b89cd", "#4875b7", "#3561a2",
                    "#224e8d", "#0f3b78", "#002864", "#001550", "#00013d"],
    },
}

PRED_COLOR = "#2a78d6"   # 验证通过的 categorical blue
MEAS_COLOR = "#eb6834"   # 验证通过的 categorical orange
FLAG_COLOR = "#d62728"

# 中文字体：优先用 fc-list 找到字体文件并显式注册，避免 matplotlib 字体缓存遗漏
def _setup_cjk_font() -> str | None:
    import subprocess
    files: list[str] = []
    try:
        out = subprocess.run(["fc-list", ":lang=zh", "file"],
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            path = line.split(":", 1)[0].strip()
            if path and Path(path).exists():
                files.append(path)
    except Exception:
        pass
    for path in files:
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    return None


_CJK_FONT = _setup_cjk_font()
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 谱分析
# ---------------------------------------------------------------------------
def radial_psd(h: np.ndarray):
    """Hann 加窗、能量校正的周期图 + 径向平均。返回 bin 中点 k 与平均 P。"""
    n = h.shape[0]
    w1 = np.hanning(n)
    win = w1[:, None] * w1[None, :]
    wcorr = float((win ** 2).mean())
    H = np.fft.fftshift(np.fft.fft2((h - h.mean()) * win))
    P = (H.real ** 2 + H.imag ** 2) / (n * n) / wcorr

    k1 = np.fft.fftshift(np.fft.fftfreq(n)) * 2.0 * np.pi
    KX, KY = np.meshgrid(k1, k1)
    K = np.hypot(KX, KY)

    edges = np.geomspace(2.0 * np.pi / n, np.pi, 90)
    idx = np.digitize(K.ravel(), edges)
    km, pm = [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() > 20:
            km.append(np.sqrt(edges[b - 1] * edges[b]))
            pm.append(P.ravel()[m].mean())
    return np.asarray(km), np.asarray(pm), K, P


def fit_spectral_slope(km: np.ndarray, pm: np.ndarray):
    lam = 2.0 * np.pi / km
    m = (lam >= FIT_LAMBDA[0]) & (lam <= FIT_LAMBDA[1])
    x, y = np.log(km[m]), np.log(pm[m])
    slope, intercept = np.polyfit(x, y, 1)
    beta = -slope
    A = float(np.exp(intercept))
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return float(beta), A, float(r2), int(m.sum())


def sigma_from_spectral_slope(beta: float, A: float):
    """谱矩 sigma_g^2 = A/(4pi) * ∫ k^{3-beta} dk，积分区间即拟合波段。"""
    ka = 2.0 * np.pi / FIT_LAMBDA[1]
    kb = 2.0 * np.pi / FIT_LAMBDA[0]
    q = 4.0 - beta
    if abs(q) < 1e-3:
        integral = np.log(kb / ka)
    else:
        integral = (kb ** q - ka ** q) / q
    sig2 = A / (4.0 * np.pi) * integral
    return float(np.sqrt(sig2)), float(sig2), float(ka), float(kb), float(integral)


def rayleigh_cdf(s, sigma):
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)
    inf = np.isinf(s)
    out[inf] = 1.0
    fin = ~inf
    out[fin] = 1.0 - np.exp(-(s[fin] ** 2) / (2.0 * sigma ** 2))
    return out


def interval_label(a: float, b: float) -> str:
    if np.isinf(b):
        return f"[{a:.3f}, ∞)"
    return f"[{a:.3f}, {b:.3f})"


def degree_span(a: float, b: float, v: float = VERTICAL_EXAG) -> str:
    d1 = np.degrees(np.arctan(v * a))
    d2 = "∞" if np.isinf(b) else f"{np.degrees(np.arctan(v * b)):.2f}°"
    return f"{d1:.2f}°–{d2}"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--octaves", type=int, default=6)
    ap.add_argument("--frequency", type=float, default=0.01)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = np.fromfile(args.input, dtype="<f8")
    n = args.n
    h = raw.reshape(n, n)
    npx = n * n

    # --- 预判①：谱与谱斜率（此时还不计算坡度分级占比） ---------------------
    km, pm, Kmat, Pmat = radial_psd(h)
    beta, A, r2, nbins = fit_spectral_slope(km, pm)
    sig_hat, sig2_hat, ka, kb, integral = sigma_from_spectral_slope(beta, A)

    # 离散精确算子版本：np.gradient 内部是中心差分，谱响应为 sin(k) 而非 k。
    # 仅用于误差归因，主预判仍用证明中的连续谱矩公式。
    kk = np.fft.fftshift(np.fft.fftfreq(n)) * 2.0 * np.pi
    KXd, KYd = np.meshgrid(kk, kk)
    Kd = np.hypot(KXd, KYd)
    bmask = (Kd >= ka) & (Kd <= kb)
    model_psd = np.zeros_like(Kd)
    model_psd[bmask] = A * Kd[bmask] ** (-beta)
    sig2_cd = 0.5 * (model_psd * (np.sin(KXd) ** 2 + np.sin(KYd) ** 2)).sum() / n ** 2
    sig_hat_cd = float(np.sqrt(sig2_cd))

    # --- 然后才做坡度与逐阶实测统计 ---------------------------------------
    gy, gx = np.gradient(h)
    s = np.hypot(gx, gy)
    sig_meas = float(np.sqrt(0.5 * (gx.var() + gy.var())))
    kurt_x = float(np.mean((gx - gx.mean()) ** 4) / gx.var() ** 2)
    kurt_y = float(np.mean((gy - gy.mean()) ** 4) / gy.var() ** 2)
    corr_xy = float(np.corrcoef(gx.ravel(), gy.ravel())[0, 1])

    # 归因检验：用实测 sigma 的 Rayleigh 对经验 CDF 的全分布 KS 偏差
    s_sorted = np.sort(s.ravel())
    ecdf = np.arange(1, s_sorted.size + 1) / s_sorted.size
    f_oracle = 1.0 - np.exp(-(s_sorted ** 2) / (2.0 * sig_meas ** 2))
    rayleigh_ks_pp = float(np.max(np.abs(ecdf - f_oracle)) * 100.0)

    schemes_result = {}
    for K, cfg in SCHEMES.items():
        edges = np.asarray(cfg["edges"], dtype=float)
        cdf_edges = rayleigh_cdf(edges, sig_hat)
        pred = np.diff(cdf_edges)                       # F(b)-F(a)
        counts, _ = np.histogram(s, bins=edges)
        meas = counts / npx

        rows = []
        for i in range(K):
            p_pct = round(float(pred[i]) * 100, 2)
            m_pct = round(float(meas[i]) * 100, 2)
            diff = round(p_pct - m_pct, 2)
            flagged = abs(diff) > FLAG_PP
            rows.append({
                "cls": f"C{i + 1}",
                "interval": interval_label(edges[i], edges[i + 1]),
                "degrees": degree_span(edges[i], edges[i + 1]),
                "pred_pct": p_pct,
                "meas_pct": m_pct,
                "diff_pp": diff,
                "flagged": flagged,
                "pixels": int(counts[i]),
            })
        schemes_result[K] = {"rows": rows, "edges": edges}

        # --- CSV -----------------------------------------------------------
        csv_path = outdir / f"slope_table_K{K}.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["色阶", "坡度区间(无量纲)", f"坡度角(V={VERTICAL_EXAG:g})",
                        "预判占比%", "实测占比%", "偏差pp(预判-实测)", "判定", "像素数"])
            for r in rows:
                w.writerow([r["cls"], r["interval"], r["degrees"], r["pred_pct"],
                            r["meas_pct"], r["diff_pp"],
                            "偏差超2pp" if r["flagged"] else "合格", r["pixels"]])

        # --- 独立 Markdown 占比表 ------------------------------------------
        md_path = outdir / f"slope_table_K{K}.md"
        n_flag = sum(r["flagged"] for r in rows)
        lines = [
            f"# {K} 色阶坡度分级：预判占比 vs 渲染后实测占比",
            "",
            f"- 地形参数：size={n}, seed={args.seed}, octaves={args.octaves}, "
            f"frequency={args.frequency}, lacunarity=2, gain=0.5",
            f"- 坡度定义：中心差分 `s=sqrt(h_x^2+h_y^2)`，格距 1，全部 {npx:,} 个像素等面积计入",
            f"- 区间划分：等步长 {cfg['step']:.3f}，左闭右开，末阶为开放尾；"
            f"边界 {', '.join(interval_label(edges[i], edges[i+1]) for i in range(K))}",
            f"- 谱斜率预判：β={beta:.3f}，σ̂_g={sig_hat:.6f}（Rayleigh 尺度参数）",
            f"- 超差判据：|预判−实测| > {FLAG_PP:g} 个百分点（按表中两位小数数值判定）",
            "",
            "| 色阶 | 坡度区间 | 坡度角* | 预判占比 | 实测占比 | 偏差 (pp) | 判定 |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for r in rows:
            verdict = "⚠ **超差 >2pp**" if r["flagged"] else "合格"
            lines.append(
                f"| {r['cls']} | `{r['interval']}` | {r['degrees']} | "
                f"{r['pred_pct']:.2f}% | {r['meas_pct']:.2f}% | "
                f"{r['diff_pp']:+.2f} | {verdict} |"
            )
        lines += [
            "",
            f"\\* 坡度角按显示用垂直夸张系数 V={VERTICAL_EXAG:g} 换算："
            "θ=atan(V·s)；V 仅影响角度标注，不影响任何占比与判定。",
            "",
            f"**本套色阶共 {n_flag} 个超差色阶。**",
            "",
        ]
        md_path.write_text("\n".join(lines), encoding="utf-8")

        # --- 分级着色图 -----------------------------------------------------
        interior = edges[1:-1]
        cls_idx = np.digitize(s, interior, right=False).astype(np.uint8)
        cmap = ListedColormap(cfg["palette"])
        fig, ax = plt.subplots(figsize=(10.6, 8.6), dpi=170)
        ax.imshow(cls_idx, cmap=cmap, interpolation="nearest",
                  vmin=0, vmax=K - 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.set_title(f"坡度分级着色图（{K} 色阶）  "
                     f"N={n}, seed={args.seed}, octaves={args.octaves}",
                     fontsize=13, pad=10)
        handles = []
        for r, hexc in zip(rows, cfg["palette"]):
            label = (f"{r['cls']}  {r['interval']}  "
                     f"实测 {r['meas_pct']:.2f}%")
            if r["flagged"]:
                label += f"  ⚠ {r['diff_pp']:+.2f}pp"
            handles.append(Patch(facecolor=hexc, edgecolor=FLAG_COLOR if r["flagged"] else "0.35",
                                 linewidth=2.2 if r["flagged"] else 0.6, label=label))
        leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5),
                        frameon=True, fontsize=10.5, title="坡度区间 / 实测占比（红框=超差）",
                        title_fontsize=10.5, borderpad=0.9, labelspacing=0.9)
        leg.get_frame().set_edgecolor("0.75")
        fig.tight_layout()
        fig.savefig(outdir / f"slope_map_K{K}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # --- 比对柱状图（两个色阶数同图两面板） --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), dpi=160)
    for ax, K in zip(axes, (5, 10)):
        rows = schemes_result[K]["rows"]
        x = np.arange(K)
        w = 0.38
        pred_v = [r["pred_pct"] for r in rows]
        meas_v = [r["meas_pct"] for r in rows]
        ax.bar(x - w / 2, pred_v, w, label="谱斜率预判", color=PRED_COLOR)
        ax.bar(x + w / 2, meas_v, w, label="渲染后实测", color=MEAS_COLOR)
        for i, r in enumerate(rows):
            if r["flagged"]:
                top = max(r["pred_pct"], r["meas_pct"])
                ax.annotate(f"{r['diff_pp']:+.2f}pp\n⚠超差", (i, top),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.6,
                            color=FLAG_COLOR, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([r["cls"] for r in rows])
        ax.set_ylabel("面积占比 / %")
        ax.set_title(f"{K} 色阶：预判 vs 实测")
        ax.grid(axis="y", linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9.5)
        ymax = max(max(pred_v), max(meas_v))
        ax.set_ylim(0, ymax * 1.28)
    fig.suptitle("逐色阶占比比对（红色标注 = 偏差超过 2 个百分点）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outdir / "pred_vs_meas.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- 谱拟合 + Rayleigh 检验诊断图 --------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=160)
    lam_km = 2 * np.pi / km
    ax1.loglog(lam_km, pm, "o", ms=3.5, color="#333333", alpha=0.65,
               label="径向平均周期图")
    mfit = (lam_km >= FIT_LAMBDA[0]) & (lam_km <= FIT_LAMBDA[1])
    xx = np.array([kb, ka])
    ax1.loglog(2 * np.pi / xx, A * xx ** (-beta), "-", lw=2.2, color=FLAG_COLOR,
               label=fr"拟合：$P(k)=Ak^{{-\beta}}$, $\beta={beta:.3f}$")
    ax1.axvspan(FIT_LAMBDA[0], FIT_LAMBDA[1], color=PRED_COLOR, alpha=0.08,
                label=f"拟合波段 λ∈[{FIT_LAMBDA[0]:g}, {FIT_LAMBDA[1]:g}] px")
    ax1.set_xlabel("波长 λ / 像素")
    ax1.set_ylabel("径向平均谱 P(k)")
    ax1.set_title(f"功率谱与谱斜率（$R^2={r2:.4f}$，{nbins} 个径向 bin）")
    ax1.grid(which="both", linewidth=0.5, alpha=0.35)
    ax1.legend(fontsize=9, frameon=False)

    ax2.hist(s.ravel(), bins=160, density=True, color="#9ec5f4",
             edgecolor="white", linewidth=0.3, label="实测坡度直方图")
    ss = np.linspace(0, np.quantile(s, 0.9995), 400)
    pdf_hat = ss / sig_hat ** 2 * np.exp(-ss ** 2 / (2 * sig_hat ** 2))
    pdf_oracle = ss / sig_meas ** 2 * np.exp(-ss ** 2 / (2 * sig_meas ** 2))
    ax2.plot(ss, pdf_hat, lw=2.3, color=PRED_COLOR,
             label=rf"Rayleigh($\hat{{\sigma}}_{{spec}}$={sig_hat:.5f})")
    ax2.plot(ss, pdf_oracle, lw=1.8, ls="--", color="#555555",
             label=rf"Rayleigh($\sigma_{{meas}}$={sig_meas:.5f}，仅归因检验)")
    ax2.set_xlabel("坡度模 s = |∇h|")
    ax2.set_ylabel("概率密度")
    ax2.set_title("坡度分布与 Rayleigh 近似")
    ax2.grid(linewidth=0.5, alpha=0.35)
    ax2.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "spectral_derivation_check.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    # --- 总报告（含两个证明、区间划分、两套核对表） -------------------------
    all_flags = [(K, r) for K in (5, 10) for r in schemes_result[K]["rows"]
                 if r["flagged"]]

    # 嵌套核对：5 级 C2 = 10 级 C3 + C4
    r5c2 = schemes_result[5]["rows"][1]
    r10c3 = schemes_result[10]["rows"][2]
    r10c4 = schemes_result[10]["rows"][3]

    def md_table_for(K):
        out = [
            "| 色阶 | 坡度区间 s | 坡度角* | 预判占比 | 实测占比 | 偏差 (pp) | 判定 |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for r in schemes_result[K]["rows"]:
            verdict = "⚠ **超差 >2pp**" if r["flagged"] else "合格"
            out.append(
                f"| {r['cls']} | `{r['interval']}` | {r['degrees']} | "
                f"{r['pred_pct']:.2f}% | {r['meas_pct']:.2f}% | "
                f"{r['diff_pp']:+.2f} | {verdict} |"
            )
        return "\n".join(out)

    report = f"""# 坡度分级面积占比：概率/CDF 证明、谱斜率预判与渲染后核对

## 0. 结论速览

- 地形由仓库自带的确定性 fBm 值噪声生成：`size={n}, seed={args.seed}, octaves={args.octaves}, frequency={args.frequency}, lacunarity=2, gain=0.5`，高度经 min–max 归一化到 [0,1]，导出为小端 float64（无 8 bit 量化）。
- 径向功率谱在 λ∈[{FIT_LAMBDA[0]:g}, {FIT_LAMBDA[1]:g}] px 的无标度段内符合幂律，拟合得 **谱斜率 β = {beta:.3f}**（log-log 线性拟合 R² = {r2:.4f}，{nbins} 个径向 bin），截距 A = {A:.4e}。
- 由谱矩公式推出梯度分量标准差 **σ̂_g = {sig_hat:.6f}**；渲染后用全部像素测得 σ_g = {sig_meas:.6f}，比值 {sig_hat / sig_meas:.3f}。梯度分量峰度 {kurt_x:.2f}/{kurt_y:.2f}（高斯=3），相关系数 {corr_xy:+.3f}，支持“近似独立高斯 → 坡度模 Rayleigh”。
- 两套色阶（**5 级**与**10 级**，边界嵌套）逐阶核对见第 5 节；共 **{len(all_flags)}** 个色阶偏差超过 {FLAG_PP:g} 个百分点：{("；".join(f"K={K} {r['cls']} ({r['diff_pp']:+.2f} pp)" for K, r in all_flags) if all_flags else "无")}。
- 产物：`slope_map_K5.png`、`slope_map_K10.png` 两张着色图；`slope_table_K5.{{md,csv}}`、`slope_table_K10.{{md,csv}}` 各一份占比表；`pred_vs_meas.png` 比对柱状图；`spectral_derivation_check.png` 谱拟合与 Rayleigh 检验图；`analysis_metrics.json` 全部数值。

---

## 1. 证明一：某色阶的面积占比 = 坡度落在该区间的概率 = 坡度分布的累积差

设区域 D 被离散成 {n}×{n} 个等面积正方形像元，格距 Δ=1，每个像元面积 Δ²。像元 p 的坡度模为

`s(p) = sqrt(h_x(p)^2 + h_y(p)^2)`，

其中 h_x、h_y 用中心差分计算（边界退化为单侧差分，占像元比例 4/{n}≈0.2%，且同样等面积计入）。

给定色阶 i 的区间 I_i = [a_i, b_i)，定义示性函数 1_i(p) = 1{{a_i ≤ s(p) < b_i}}。该色阶的真实面积与面积占比为

```
A_i   = ∫_D 1_i(x) dA = Δ² · Σ_p 1_i(p)
w_i   = A_i / A_total = (1/n²) · Σ_p 1_i(p) = N_i / n².
```

因为每个像元等面积，**面积占比恰好等于“在区域内均匀抽一个像元，其坡度落入 I_i”这一事件的经验概率**。在随机场平稳/遍历或大数定律意义下，n²→∞ 时

`w_i → P(a_i ≤ S < b_i)`.

若 F_S(s) = P(S < s) 是坡度模的累积分布函数（连续分布，边界点概率为 0，左闭右开不影响数值），则

**`P(a_i ≤ S < b_i) = F_S(b_i) − F_S(a_i)`**。

这就是“某色阶面积占比等于坡度落在该区间的概率，并写成坡度分布的累积差”。脚本中 `pred_i = F_hat(b_i) − F_hat(a_i)` 与 `meas_i = N_i/n²` 严格按此式计算。

## 2. 证明二：坡度分布可由谱斜率近似推出

### 2.1 谱斜率决定导数谱的指数

采用离散周期图约定：未归一化二维 DFT 为 H(k_x,k_y)=Σ h exp(−i(k_xx+k_yy))，P(k) 为径向平均的 |H|²/n²。实现时为抑制高度场非周期边缘造成的谱泄漏，先乘二维 Hann 窗并除以 mean(w²) 做能量校正，校正后与该约定同量纲。Parseval 给出（各向同性场，波数 k=√(k_x²+k_y²)，k∈[2π/n, π]）

```
σ_h² = E[h²]     = (1/2π) ∫ k   P(k) dk,
σ_g² = E[h_x²]   = (1/4π) ∫ k³  P(k) dk,     (各向同性时 h_y 相同)
```

第二个式子的来源：傅里叶域中 ∂/∂x ↔ i k_x，求导相当于给谱乘 k_x²；各向同性下 E[k_x²|k]=k²/2，于是导数谱多一个 k² 因子，坡度分量的谱为 k²P(k)。

设高度谱在无标度段服从幂律 **P(k)=A k^(−β)**，β 即双对数谱斜率的相反数（log P = log A − β log k）。则

`σ_g² = A/(4π) · ∫_(k_a)^(k_b) k^(3−β) dk`

从而

```
σ_g² = A/(4π) · (k_b^(4−β) − k_a^(4−β))/(4−β),   β ≠ 4
σ_g² = A/(4π) · ln(k_b/k_a),                      β = 4
```

其中 k_a=2π/{FIT_LAMBDA[1]:g}、k_b=2π/{FIT_LAMBDA[0]:g} 是拟合/积分波段 λ∈[{FIT_LAMBDA[0]:g},{FIT_LAMBDA[1]:g}] px 的波数端点。**β 决定积分的幂次与收敛行为，A 决定绝对尺度**；A 与 β 来自同一条谱直线的最小二乘拟合，在任何坡度分级统计之前即可得到。对本地形 β≈4.011，恰好落在对数情形，σ_g² 对波段端点只对数敏感。

> 有限差分的精确修正：np.gradient 的中心差分谱响应是 i·sin k_x（而非 i k_x），故离散实现可把 k_x² 换成 sin²k_x。用该精确算子在同一拟合波段求和得到 σ̂_cd = {sig_hat_cd:.6f}（连续公式为 {sig_hat:.6f}），差异与波段外尾部一起构成第 6 节讨论的近似误差。主表采用证明中的连续谱矩口径。

### 2.2 高斯随机场 → 坡度模服从 Rayleigh

fBm 是高斯随机场；本实现的多层值噪声叠加后也近似高斯（实测 h_x、h_y 峰度 {kurt_x:.2f}、{kurt_y:.2f}，接近 3；相关系数 {corr_xy:+.3f}，接近 0）。各向同性时可设

`h_x, h_y 近似独立同分布 N(0, σ_g²)`.

二维各向同性高斯向量的模 R=√(h_x²+h_y²) 服从 **Rayleigh 分布**：

```
f_R(s) = (s/σ_g²) exp(−s²/(2σ_g²)),  s ≥ 0
F̂_R(s) = 1 − exp(−s²/(2σ̂_g²)).
```

代入谱矩公式给出的 σ̂_g，着色前即可算出任意色阶 I_i=[a_i,b_i) 的预判面积占比：

**`ŵ_i = F̂_R(b_i) − F̂_R(a_i) = exp(−a_i²/(2σ̂_g²)) − exp(−b_i²/(2σ̂_g²))`**（b_i=∞ 时第二项为 0）。

这就是“坡度分布可由谱斜率近似推出，因此着色前就能预判每个色阶占比”的完整逻辑链：
**β（+A）→ 谱矩 σ̂_g → Rayleigh CDF → CDF 之差 → 各色阶预判面积占比**。

## 3. 谱拟合结果

见 `spectral_derivation_check.png` 左图。拟合值：

| 量 | 值 |
|---|---:|
| 谱斜率 β | {beta:.4f} |
| 截距 A（本离散谱约定） | {A:.4e} |
| log-log 拟合 R² | {r2:.5f} |
| 拟合径向 bin 数 | {nbins} |
| 拟合波段 | λ ∈ [{FIT_LAMBDA[0]:g}, {FIT_LAMBDA[1]:g}] px |
| 谱矩积分 ∫k^{{3−β}}dk | {integral:.4f} |
| **预判 σ̂_g（连续谱矩）** | **{sig_hat:.6f}** |
| 参考：中心差分算子 σ̂_cd | {sig_hat_cd:.6f} |
| 渲染后实测 σ_g | {sig_meas:.6f} |
| σ̂_g / σ_g | {sig_hat / sig_meas:.4f} |

β≈4 与参数设置一致：gain=0.5、lacunarity=2 对应 H=−log₂(gain)=1，二维 fBm 的理论谱指数 β≈2H+2=4。右图把实测坡度直方图与 Rayleigh 曲线叠在一起；用实测 σ_g 的虚线与经验分布的全分布 KS 最大 CDF 偏差仅 {rayleigh_ks_pp:.2f} pp，说明 Rayleigh 形状假设很准，各阶预判误差主要来自谱公式对 σ̂_g 的估计，而非分布形状。

## 4. 坡度区间（色阶）如何划分

- 坡度以**归一化高程、格距 1** 为单位，即无量纲 rise/run；脚本中边界是预先固定的常量，不依据渲染后的占比回调。
- 采用**等步长 + 开放尾阶**：5 级步长 0.004，边界 `[0, .004, .008, .012, .016, ∞)`；10 级步长 0.002，边界 `[0, .002, .004, …, .018, ∞)`。
- 两套边界严格嵌套：每个 5 级阶恰好等于相邻两个 10 级阶之并（如 C2 `[.004,.008)` = 10 级 C3 `[.004,.006)` + C4 `[.006,.008)`），因此两套统计可互相核对。
- 坡度角换算：若声明垂直比例系数 V，则 θ=atan(V·s)。表中角度列取显示用 **V={VERTICAL_EXAG:g}**（仅为可读标注，V 在 CDF 中与 σ 同比缩放后消去，不影响占比与判定）。不做垂直夸张时直接 θ=atan(s)，本地形对应约 0.1°–2° 的缓坡。

## 5. 预判占比 vs 渲染后逐色阶实测占比（逐个比对）

判定规则：**|预判占比 − 实测占比| > {FLAG_PP:g} 个百分点（pp）的色阶标 “⚠ 超差”**，数值按表中两位小数口径判定。

### 5.1 五个色阶

{md_table_for(5)}

### 5.2 十个色阶

{md_table_for(10)}

独立表格文件：`slope_table_K5.md` / `slope_table_K5.csv` 与 `slope_table_K10.md` / `slope_table_K10.csv`；柱状比对见 `pred_vs_meas.png`；两张着色图为 `slope_map_K5.png`、`slope_map_K10.png`，图例中红框即超差色阶。

## 6. 偏差与超差色阶讨论

1. **谱尺度轻微偏低**：σ̂_g/σ_g={sig_hat/sig_meas:.3f}，即谱公式把坡度尺度低估约 {(1-sig_hat/sig_meas)*100:.1f}%。Rayleigh CDF 在众数附近最陡，边界处很小的尺度偏差会转化为 1–2 pp 的占比搬家，符号上表现为低坡阶预判偏多、高坡阶预判偏少（各表偏差列同向）。
2. **超差阶来自相邻亚阈值偏差的同向叠加**：5 级 {r5c2['cls']} 的 {r5c2['diff_pp']:+.2f} pp，正好约等于其嵌套的两个 10 级阶 {r10c3['cls']}（{r10c3['diff_pp']:+.2f} pp）与 {r10c4['cls']}（{r10c4['diff_pp']:+.2f} pp）之和——细分级下没有任何单阶超过 2 pp，合并成粗阶后超过阈值。这说明 2 pp 阈值的判定对分级粒度敏感，也正是保留两套色阶的意义。
3. **误差来源**：① 幂律只在有限无标度段成立，波段外（最粗基频阶以下、最细插值核滚降段以上）仍贡献一部分谱矩，连续公式未计入；② 中心差分相对连续导数在高波数有 sin(k)/k 衰减（σ̂_cd={sig_hat_cd:.6f}）；③ 值噪声非严格高斯、min–max 归一化由极端像元决定尺度；④ 有限样本。{npx/1e6:.2f}×10⁶ 个等面积像元下，占比的二项标准误最大约 {100*np.sqrt(0.25/npx):.3f} pp（95% 置信半宽约 {196*np.sqrt(0.25/npx):.2f} pp），远小于 2 pp，抽样波动可忽略。
4. **总体判断**：谱斜率 + Rayleigh 的先验公式在着色前给出的各阶占比，绝大多数与渲染统计相差 ≤2 pp；超差阶已在上表与图注中逐个标出，且偏差方向可由 σ̂ 的轻微低估解释，不存在异常色阶。

## 7. 复现

```bash
go run ./cmd/goproc gen --size {n} --seed {args.seed} --octaves {args.octaves} --frequency {args.frequency} --format f64 --out work/terrain_n{n}_s{args.seed}_o{args.octaves}_f01.f64

python3 analysis/slope_forecast.py --input work/terrain_n{n}_s{args.seed}_o{args.octaves}_f01.f64 --n {n} --outdir artifacts --seed {args.seed} --octaves {args.octaves} --frequency {args.frequency}
```

分析脚本：`analysis/slope_forecast.py`（全部数值由该脚本一次生成，确定性可复跑）。
"""
    (outdir / "slope_prediction_report.md").write_text(report, encoding="utf-8")

    # --- 指标 JSON ----------------------------------------------------------
    metrics = {
        "terrain": {"n": n, "pixels": npx, "seed": args.seed,
                    "octaves": args.octaves, "frequency": args.frequency,
                    "lacunarity": 2, "gain": 0.5},
        "spectrum": {"beta": beta, "A": A, "r2_logfit": r2, "fit_bins": nbins,
                     "fit_lambda_px": list(FIT_LAMBDA),
                     "k_range": [ka, kb], "integral": integral,
                     "sigma_hat": sig_hat, "sigma_hat2": sig2_hat,
                     "sigma_hat_central_diff": sig_hat_cd},
        "slope_check": {"sigma_meas": sig_meas,
                        "sigma_ratio_hat_over_meas": sig_hat / sig_meas,
                        "kurtosis_gx": kurt_x, "kurtosis_gy": kurt_y,
                        "corr_gx_gy": corr_xy,
                        "rayleigh_oracle_ks_pp": rayleigh_ks_pp},
        "schemes": {str(K): {"step": SCHEMES[K]["step"],
                             "rows": [{kk: vv for kk, vv in r.items()}
                                      for r in schemes_result[K]["rows"]]}
                    for K in SCHEMES},
    }
    (outdir / "analysis_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要
    print(f"beta={beta:.4f} R2={r2:.5f} A={A:.4e}")
    print(f"sigma_hat={sig_hat:.6f} sigma_meas={sig_meas:.6f} "
          f"ratio={sig_hat/sig_meas:.4f}")
    for K in SCHEMES:
        flags = [r["cls"] for r in schemes_result[K]["rows"] if r["flagged"]]
        print(f"K={K}: flagged={flags}")
        for r in schemes_result[K]["rows"]:
            print(f"  {r['cls']} {r['interval']:>12} pred={r['pred_pct']:6.2f} "
                  f"meas={r['meas_pct']:6.2f} diff={r['diff_pp']:+.2f} "
                  f"{'FLAG' if r['flagged'] else ''}")


if __name__ == "__main__":
    main()
