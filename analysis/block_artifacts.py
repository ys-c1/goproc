#!/usr/bin/env python3
"""
块状压缩伪影可预判性验证。

压缩模型: 8x8 块均值压缩 + 均匀量化, 量化步长 q。
块 b 内像素记为 x_k, 块均值 mu_b, 压缩后块内全部像素取 q*round(mu_b/q)。

理论链:
  1. 压缩后块内误差 e_k = x_k - q*round(mu_b/q)。
     块内 RMS 误差 rms_b = sqrt(mean e_k^2)。
  2. 记量化偏差 d_b = mu_b - q*round(mu_b/q), |d_b| <= q/2, 则
        e_k = (x_k - mu_b) + d_b
        rms_b^2 = sigma_b^2 + d_b^2                                  (交叉项和为 0)
     其中 sigma_b 为压缩前即可算出的块内标准差。
     => rms_b 与 sigma_b 之差不超过 q/2, 压缩后的误差能量由压缩前的
        局部方差 sigma_b^2 主导, 量化只贡献 O(q^2) 的修正。
  3. 可见判据: 归一化反差 r_b = sigma_b / q > T(T=3)。
     这是纯压缩前量; 压缩后对应 rms_b/q > T, 两判据在阈值附近相差
     至多约 1/(2T) 个阈值单位, 故可在压缩前预判可见块及其集中区域。

实测(独立, 必须压缩后才能得到):
  - 块级: rms_b/q > 3 的块为实测可见块;
  - 像素级(辅助): |e| > 3q 的极端像素。
输出: results.json/results.csv 与两张标注图。
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PGM = "/home/ys/feedback/uidreczz28eDSvuZZSv/artifacts/terrain_s7.pgm"
OUT = "/home/ys/feedback/uidreczz28eDSvuZZSv/artifacts"
W = 8            # 局部方差窗口 / 压缩块边长(格)
T = 3.0          # 可见判定阈值
STEPS = (4, 8)   # 两个量化步长档


def read_pgm_tolerant(path):
    """容错读取 P5 PGM(本仓库生成器把头部换行写成了字面量 \\n)。"""
    raw = open(path, "rb").read().replace(b"\\n", b"\n", 4)
    assert raw[:2] == b"P5"
    dims, maxline, payload = raw[3:].split(b"\n", 2)
    w, h = (int(x) for x in dims.split())
    assert int(maxline) == 255
    return np.frombuffer(payload, dtype=np.uint8).reshape(h, w).astype(np.float64)


def block_view(img):
    n = img.shape[0] // W
    return n, img.reshape(n, W, n, W).transpose(0, 2, 1, 3).reshape(n, n, W * W)


def component_centers(mask):
    """4-邻域连通域 -> (面积, 格心(行,列), 成员), 面积降序。"""
    n = mask.shape[0]
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for i in range(n):
        for j in range(n):
            if not mask[i, j] or seen[i, j]:
                continue
            stack, members = [(i, j)], []
            seen[i, j] = True
            while stack:
                r, c = stack.pop()
                members.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            cr = sum(p[0] for p in members) / len(members)
            cc = sum(p[1] for p in members) / len(members)
            comps.append((len(members), (round(cr, 2), round(cc, 2)), members))
    return sorted(comps, key=lambda t: -t[0])


def main():
    img = read_pgm_tolerant(PGM)
    H = img.shape[0]
    n, blk = block_view(img)
    mean_b = blk.mean(axis=2)
    var_b = blk.var(axis=2, ddof=0)          # 总体方差, 与 RMS 误差同一定义(除以 64)
    std_b = np.sqrt(var_b)

    rows, detail = [], {}
    for q in STEPS:
        # ---- 压缩前预判 ----
        pred_block = (std_b / q) > T
        comps_pred = component_centers(pred_block)

        # ---- 压缩 + 压缩后实测 ----
        mean_q = np.round(mean_b / q) * q
        rec = np.repeat(np.repeat(mean_q, W, axis=0), W, axis=1)
        err = np.abs(rec - img)
        rms_b = np.sqrt((err ** 2).reshape(n, W, n, W).transpose(0, 2, 1, 3)
                        .reshape(n, n, -1).mean(axis=2))
        actual_block = (rms_b / q) > T                       # 块级实测
        actual_px_strict = err > T * q                       # 像素级辅助
        comps_actual = component_centers(actual_block)

        pred_px = np.repeat(np.repeat(pred_block, W, 0), W, 1)
        actual_px = np.repeat(np.repeat(actual_block, W, 0), W, 1)

        tp = int((pred_block & actual_block).sum())
        fp = int((pred_block & ~actual_block).sum())
        fn = int((~pred_block & actual_block).sum())
        inter = int((pred_px & actual_px).sum())
        union = int((pred_px | actual_px).sum())
        # 恒等式 rms^2 = sigma^2 + d^2 的数值核验
        d = mean_b - mean_q
        ident_err = float(np.abs(rms_b ** 2 - (var_b + d ** 2)).max())

        row = {
            "q": q,
            "pred_blocks": int(pred_block.sum()),
            "pred_pixels": int(pred_px.sum()),
            "actual_blocks": int(actual_block.sum()),
            "actual_pixels": int(actual_px.sum()),
            "strict_pixel_e_gt_3q": int(actual_px_strict.sum()),
            "block_TP": tp, "block_FP": fp, "block_FN": fn,
            "block_recall": round(tp / max(1, tp + fn), 4),
            "pixel_iou": round(inter / union, 4),
            "pixel_accuracy": round(float((pred_px == actual_px).mean()), 4),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
            "identity_maxerr": f"{ident_err:.2e}",
            "pred_region_centers_rc": [c[1] for c in comps_pred],
            "actual_region_centers_rc": [c[1] for c in comps_actual],
        }
        rows.append(row)
        detail[q] = dict(pred_block=pred_block, actual_block=actual_block,
                         strict=actual_px_strict, comps=comps_pred)

        # ---- 标注图 ----
        fig, ax = plt.subplots(figsize=(8.4, 8))
        ax.imshow(img, cmap="gray", extent=(0, H, H, 0))
        ov = np.zeros((H, H, 4))
        ov[actual_px_strict] = (1, 0, 0, 0.5)          # 红: 实测极端误差像素
        ax.imshow(ov, extent=(0, H, H, 0))
        for idx, (area, center, members) in enumerate(comps_pred, 1):
            for (r, c) in members:
                ax.add_patch(Rectangle((c * W, r * W), W, W, fill=False,
                                       edgecolor="#ffd400", lw=1.5))
            ax.text(center[1] * W + W / 2, center[0] * W + W / 2, str(idx),
                    color="#ffd400", fontsize=10, fontweight="bold",
                    ha="center", va="center")
        ax.set_title(f"q={q}: yellow=predicted visible blocks (pre-compression sigma/q>3), "
                     f"red=measured pixels |e|>3q")
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        fig.tight_layout()
        fig.savefig(f"{OUT}/artifact_map_q{q}.png", dpi=130)
        plt.close(fig)

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(f"{OUT}/results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["q", "pred_blocks", "pred_pixels", "actual_blocks",
                     "actual_pixels", "strict_pixel_e_gt_3q",
                     "TP", "FP", "FN", "block_recall", "pixel_iou",
                     "pixel_accuracy", "rmse", "identity_maxerr",
                     "pred_centers_rc", "actual_centers_rc"])
        for r in rows:
            wr.writerow([r["q"], r["pred_blocks"], r["pred_pixels"],
                         r["actual_blocks"], r["actual_pixels"],
                         r["strict_pixel_e_gt_3q"], r["block_TP"],
                         r["block_FP"], r["block_FN"], r["block_recall"],
                         r["pixel_iou"], r["pixel_accuracy"], r["rmse"],
                         r["identity_maxerr"],
                         json.dumps(r["pred_region_centers_rc"]),
                         json.dumps(r["actual_region_centers_rc"])])
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
