#!/usr/bin/env python3
"""八位量化直方图空隙分析。

数据流（全部确定性、可复现）：
  Go  api.Generate -> Normalize -> EncodePGM 同款量化规则
      analysis/godump   导出 floats_*.bin / map_*.pgm
      analysis/gobatch  导出 60 个种子的 batch_*.jsonl
  本脚本:
      1. 读 floats，复算量化字节并与 PGM 负载逐字节核对
      2. 统计 256 档计数，空隙 = 1..254 档中计数为零者（首尾两档不计）
      3. 期望表达式求值：
         - 均匀占位模型      E[G] = W exp(-n/W)
         - 形状感知 iid 模型 E[G] = sum_{k=1}^{254}(1-p_k)^n   （p_k 由高斯 KDE 积分）
         - 相关性修正        以 n_eff 代 n，n_eff 由 60 种子均值标定
      4. 每个画幅输出一张量化直方图（红档=空隙），输出 metrics.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 显式注册用户目录下的中文字体（matplotlib 缓存可能未收录）
_FONT_PATH = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.ttf"
if _FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH))

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
SEED = 7
SIZES = (64, 256)
NBATCH = 60

# dataviz 参考调色板（light，已过 validate_palette）
BLUE = "#2a78d6"
RED = "#e34948"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK = "#0b0b0b"
MUTED = "#898781"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "grid.color": GRID,
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
})

EDGES = np.arange(257, dtype=float) - 0.5          # 档 k 的边界 k±0.5（量化单位 1/255）
EDGES[0], EDGES[-1] = -1e9, 1e9                    # 首尾档含截断区


def load_frame(size: int):
    """返回 (浮点高度 n, 量化字节 n, 256 档计数)。"""
    x = np.fromfile(ART / f"floats_{size}.bin", dtype="<f8")
    n = x.size
    g = np.clip((x * 255.0 + 0.5).astype(np.int64), 0, 255).astype(np.uint8)
    # 与 PGM 负载逐字节核对（仓库 PGM 头含字面 \n，按 255\n 字面串定位负载）
    raw = (ART / f"map_{size}.pgm").read_bytes()
    payload = np.frombuffer(raw.split(b"255\\n", 1)[1], dtype=np.uint8)
    assert payload.size == n and np.array_equal(g, payload), "量化结果与 PGM 负载不一致"
    counts = np.bincount(g, minlength=256)
    return x, g, counts


def kde_bin_probs(x: np.ndarray) -> tuple[np.ndarray, float]:
    """高斯 KDE（Scott 带宽）在 256 个量化档区间上的闭式积分概率。"""
    n = x.size
    bw = 1.06 * x.std(ddof=1) * n ** -0.2
    c = math.sqrt(0.5) / bw
    z = EDGES / 255.0
    cdf = np.empty(257)
    for i, e in enumerate(z):
        d = (e - x) * c
        s = 0.0
        for j in range(0, n, 8192):
            s += sum(math.erf(v) for v in d[j:j + 8192])
        cdf[i] = 0.5 * (1.0 + s / n)
    p = np.diff(cdf)
    return p / p.sum(), bw


def expected_gaps(p: np.ndarray, n_eff: float) -> float:
    """形状感知期望：内部 254 档各档空置概率之和。"""
    return float(np.sum((1.0 - p[1:255]) ** n_eff))


def calibrate_n_eff(size: int, n: int, held_out_seed: int) -> tuple[float, float, float]:
    """用除 held_out 外的种子，令 E[G] 均值等于实测均值，二分标定 n_eff。"""
    rows = [json.loads(line) for line in (ART / f"batch_{size}.jsonl").read_text().splitlines()]
    cal = [r for r in rows if r["seed"] != held_out_seed]
    probs, measured = [], []
    for r in cal:
        c = np.asarray(r["counts"], dtype=np.int64)
        xs = np.repeat(np.arange(256) / 255.0, c)
        p, _ = kde_bin_probs(xs)
        probs.append(p)
        measured.append(len(r["gaps"] or []))
    target = float(np.mean(measured))
    lo, hi = 1.0, float(n)
    mid = 0.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        v = float(np.mean([expected_gaps(p, mid) for p in probs]))
        if v > target:
            lo = mid            # E 随 n_eff 单调递减
        else:
            hi = mid
    return mid, target, float(np.mean([expected_gaps(p, mid) for p in probs]))


def plot_histogram(size: int, counts: np.ndarray, gaps: list[int], metrics: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    colors = [RED if k in gaps else BLUE for k in range(256)]
    ax.bar(range(256), counts, width=1.0, color=colors, linewidth=0)
    ax.set_xlim(-0.5, 255.5)
    ax.set_xlabel("灰度档 k（量化值 0–255）")
    ax.set_ylabel("像素计数")
    ax.set_title(f"八位量化灰度直方图  {size}×{size}（n={size*size}，种子 {SEED}）\n"
                 f"内部空隙 {len(gaps)} 个：{gaps if gaps else '无'}（红档，首尾两档不计）",
                 fontsize=12)
    ax.grid(axis="y", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # 空隙档：零高红柱不可见，用红三角在基线上标出，并在上方标注档号
    ymax = counts.max()
    for k in gaps:
        ax.plot(k, 0, marker="v", color=RED, markersize=7, zorder=5, clip_on=False)
        ax.annotate(str(k), xy=(k, 0), xytext=(k, ymax * 0.04),
                    ha="center", fontsize=9, color=RED)
    txt = (f"n = {metrics['n']}\n"
           f"W（参与率宽度）= {metrics['W_participation']:.1f} 档\n"
           f"均匀模型 W·exp(−n/W) = {metrics['pred_uniform']:.3f}\n"
           f"形状感知 E[G]（iid）= {metrics['pred_iid']:.3f}\n"
           f"相关性修正 E[G]（n_eff）= {metrics['pred_corr']:.3f}\n"
           f"实测空隙 = {metrics['measured_gaps']}")
    ax.text(0.995, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color="#52514e",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#e1e0d9", alpha=0.92))
    ax.legend(handles=[
        Patch(facecolor=BLUE, label="有像素落入的档"),
        Patch(facecolor=RED, label="空隙档（计数为零，不含首尾）"),
    ], loc="upper center", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    all_metrics = {}
    for size in SIZES:
        x, g, counts = load_frame(size)
        n = x.size
        gaps = [k for k in range(1, 255) if counts[k] == 0]
        p, bw = kde_bin_probs(x)
        p_emp = counts / n
        W_part = float(1.0 / np.sum(p_emp ** 2))                 # 参与率/有效支撑宽度
        n_eff, cal_meas_mean, cal_pred_mean = calibrate_n_eff(size, n, SEED)

        metrics = {
            "size": size,
            "n": n,
            "seed": SEED,
            "measured_gaps": len(gaps),
            "gap_bins": gaps,
            "occupied_bins": int((counts > 0).sum()),
            "bin0_count": int(counts[0]),
            "bin255_count": int(counts[255]),
            "kde_bandwidth_Scott": bw,
            "W_participation": W_part,
            "pred_uniform": W_part * math.exp(-n / W_part),
            "pred_iid": expected_gaps(p, float(n)),
            "n_eff_calibrated": n_eff,
            "n_eff_over_n": n_eff / n,
            "pred_corr": expected_gaps(p, n_eff),
            "calibration_seeds": NBATCH - 1,
            "calib_measured_mean": cal_meas_mean,
            "calib_predicted_mean": cal_pred_mean,
        }
        all_metrics[f"{size}x{size}"] = metrics
        plot_histogram(size, counts, gaps, metrics, ART / f"histogram_{size}.png")
        print(json.dumps(metrics, ensure_ascii=False, indent=1))

    (ART / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print("\nwrote artifacts/metrics.json and artifacts/histogram_{64,256}.png")


if __name__ == "__main__":
    main()
