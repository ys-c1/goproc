#!/usr/bin/env python3
"""
极值最近邻间距：正式测量与预判比对管线。

口径（与报告一致）：
- 数据全部由 analysis/dump/main.go 调用仓库 api 生成（manifest.json）。
- 极值 = 严格 8 邻域局部极大/极小；最外一圈不参与判定。
- 边界点剔除：距任一边 < margin 的极值不进入间距样本；
  计算 NN 时邻居集仍是全部已检出极值（guard 邻居，消除边缘上偏）。
  margin 与边长成比例：768 -> 50px（= 基础晶格距 1/F0），128 -> 8px。
- 间距 = cKDTree 最近邻距离（排除自身），极大+极小池化为主口径。
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "artifacts" / "extrema" / "data"
OUT = ROOT / "artifacts" / "extrema"
F0 = 0.02
LAC = 2.0


def extrema_points(z):
    n = z.shape[0]
    zc = z[1:-1, 1:-1]
    mx = np.ones(zc.shape, bool)
    mn = np.ones(zc.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            w = z[1 + dy:n - 1 + dy, 1 + dx:n - 1 + dx]
            mx &= zc > w
            mn &= zc < w
    ys, xs = np.nonzero(mx)
    maxp = np.column_stack([xs + 1, ys + 1]).astype(float)
    ys, xs = np.nonzero(mn)
    minp = np.column_stack([xs + 1, ys + 1]).astype(float)
    return maxp, minp


def nn_guarded(points, margin, size):
    """保留边界内点作为样本；在全部点中查最近邻。返回距离与保留数。"""
    if len(points) < 2:
        return np.array([]), 0
    keep = ((points[:, 0] >= margin) & (points[:, 0] <= size - 1 - margin) &
            (points[:, 1] >= margin) & (points[:, 1] <= size - 1 - margin))
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return np.array([]), 0
    d, _ = cKDTree(points).query(points[idx], k=2)
    return d[:, 1], len(idx)


def nn_naive_kept_only(points, margin, size):
    """对照：先剔除边界点，再只在剩余点集内找 NN。"""
    keep = ((points[:, 0] >= margin) & (points[:, 0] <= size - 1 - margin) &
            (points[:, 1] >= margin) & (points[:, 1] <= size - 1 - margin))
    p = points[keep]
    if len(p) < 2:
        return np.array([])
    d, _ = cKDTree(p).query(p, k=2)
    return d[:, 1]


def measure_entry(e):
    z = np.fromfile(DATA / e["file"], dtype="<f8").reshape(e["size"], e["size"])
    size = e["size"]
    margin = max(2, round(size * 50 / 768))
    maxp, minp = extrema_points(z)
    allp = np.vstack([maxp, minp]) if len(maxp) + len(minp) else np.empty((0, 2))
    d_all, nkeep = nn_guarded(allp, margin, size)
    d_max, nmaxkeep = nn_guarded(maxp, margin, size)
    d_min, nminkeep = nn_guarded(minp, margin, size)
    d_naive = nn_naive_kept_only(allp, margin, size)
    # 边界宽度稳健性（仅大尺寸）
    robust = {}
    if size == 768:
        for m in (25, 75):
            dd, _ = nn_guarded(allp, float(m), size)
            robust[m] = float(dd.mean()) if len(dd) else None
    return {
        "tag": e["tag"], "role": e["role"], "size": size, "seed": e["seed"],
        "octaves": e["octaves"], "gain": e["gain"], "frequency": e["frequency"],
        "margin": margin,
        "n_max": int(len(maxp)), "n_min": int(len(minp)), "n_keep": int(nkeep),
        "mean": float(d_all.mean()) if len(d_all) else None,
        "sd": float(d_all.std(ddof=1)) if len(d_all) > 1 else None,
        "var": float(d_all.var(ddof=1)) if len(d_all) > 1 else None,
        "dists": d_all,
        "mean_max": float(d_max.mean()) if len(d_max) else None,
        "mean_min": float(d_min.mean()) if len(d_min) else None,
        "mean_naive": float(d_naive.mean()) if len(d_naive) else None,
        "robust": robust,
    }


def pool(recs):
    """合并同配置跨种子的 NN 距离。"""
    arrs = [r["dists"] for r in recs if r["dists"] is not None and len(r["dists"])]
    if not arrs:
        return None
    d = np.concatenate(arrs)
    seed_means = np.array([r["mean"] for r in recs if r["mean"] is not None])
    return {
        "n_seeds": len(recs), "n_dists": int(len(d)),
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)),
        "var": float(d.var(ddof=1)),
        "se_mean": float(seed_means.std(ddof=1) / math.sqrt(len(seed_means))) if len(seed_means) > 1 else 0.0,
        "seed_mean_spread": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
        "mean_max": float(np.mean([r["mean_max"] for r in recs if r["mean_max"]])),
        "mean_min": float(np.mean([r["mean_min"] for r in recs if r["mean_min"]])),
        "mean_naive": float(np.mean([r["mean_naive"] for r in recs if r["mean_naive"]])),
        "robust25": float(np.mean([r["robust"][25] for r in recs if r["robust"].get(25)])),
        "robust75": float(np.mean([r["robust"][75] for r in recs if r["robust"].get(75)])),
    }


def f_centroid(g, n, f0=F0, lac=LAC):
    return f0 * sum((g * lac) ** k for k in range(n)) / sum(g ** k for k in range(n))


def main():
    entries = json.load(open(DATA / "manifest.json"))
    recs = [measure_entry(e) for e in entries]

    # ---------- 配置分组 ----------
    groups = defaultdict(list)
    for r in recs:
        if r["role"] in ("A", "B"):
            key = (r["role"], r["gain"], r["octaves"])
        elif r["role"] == "cal":
            key = ("cal", r["frequency"], 1)
        elif r["role"] == "product":
            key = ("product", round(r["gain"], 6), r["octaves"])
        else:
            key = ("small128", 0.5, r["octaves"])
        groups[key].append(r)
    agg = {f"{k[0]}|{k[1]}|{k[2]}": pool(v) for k, v in groups.items()}

    # ---------- 标定 C：d = C / f ----------
    cal_rows = []
    for (role, f, n), rec in sorted(groups.items()):
        if role != "cal":
            continue
        p = pool(rec)
        cal_rows.append({"f": f, "ell": 1 / f, **p,
                         "C": p["mean"] * f, "kappa": p["var"] / p["mean"] ** 2})
    # 加权过原点拟合 C = sum(n f d)/sum(n f^2) 等价对 d=C/f
    num = den = 0.0
    xs, ys, ws = [], [], []
    for row in cal_rows:
        C_i = row["C"]
        w = row["n_dists"]
        num += w * C_i
        den += w
        xs.append(math.log(row["f"]))
        ys.append(math.log(row["mean"]))
        ws.append(w)
    C = num / den
    slope = float(np.polyfit(np.array(xs), np.array(ys), 1,
                             w=np.sqrt(np.array(ws, float)))[0])
    C_med = float(np.median([r["C"] for r in cal_rows]))

    # ---------- 主组逐层预判 ----------
    table = {}
    for role, g in (("A", 0.5), ("B", 0.8)):
        rows = []
        for n in range(1, 9):
            p = agg[f"{role}|{g}|{n}"]
            f_top = F0 * LAC ** (n - 1)
            d_top = C / f_top
            f_c = f_centroid(g, n)
            d_c = C / f_c
            f_p = F0 * g * n
            d_p = C / f_p
            rows.append({
                "n": n, "g": g, "gN": g * n,
                "f_top": f_top, "ell_top": 1 / f_top,
                "d_top_pred": d_top,
                "d_centroid_pred": d_c, "f_centroid": f_c,
                "d_product_pred": d_p, "f_product": f_p,
                "meas": p["mean"], "var": p["var"], "sd": p["sd"],
                "se_mean": p["se_mean"], "n_dists": p["n_dists"],
                "kappa": p["var"] / p["mean"] ** 2,
                "err_top": p["mean"] - d_top,
                "abserr_top": abs(p["mean"] - d_top),
                "abserr_product": abs(p["mean"] - d_p),
                "abserr_centroid": abs(p["mean"] - d_c),
                "var_pred_order": 0.20 * d_top ** 2,
                **{kk: p[kk] for kk in ("mean_max", "mean_min", "mean_naive",
                                        "robust25", "robust75")},
            })
        # 饱和地板：N>=7 实测均值的渐近值
        floor = float(np.mean([r["meas"] for r in rows if r["n"] >= 7]))
        for r in rows:
            r["floor_g"] = floor
            r["d_sat_pred"] = max(r["d_top_pred"], floor)
            r["abserr_sat"] = abs(r["meas"] - r["d_sat_pred"])
            r["var_sat_pred"] = 0.20 * r["d_sat_pred"] ** 2
        table[role] = rows

    # 产品模型的全局最优 C（给该假设最佳拟合机会）
    gp_num = gp_den = 0.0
    for role, g in (("A", 0.5), ("B", 0.8)):
        for n in range(1, 9):
            r = next(x for x in table[role] if x["n"] == n)
            x = 1 / (F0 * g * n)  # d = Cp * x
            w = r["n_dists"]
            gp_num += w * x * r["meas"]
            gp_den += w * x * x
    Cp_best = gp_num / gp_den

    # ---------- 等 gN 乘积 / 固定 N 变增益 ----------
    prod_rows = []
    lookup = {}
    for (role, g, n), rec in groups.items():
        if role in ("A", "B", "product"):
            lookup[(round(g, 6), n)] = pool(rec)
    for (g, n), p in sorted(lookup.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        prod_rows.append({"g": g, "n": n, "gN": g * n,
                          "meas": p["mean"], "var": p["var"], "n_dists": p["n_dists"],
                          "ell_top": 1 / (F0 * LAC ** (n - 1))})

    # ---------- 128 默认尺寸 ----------
    small_rows = []
    for n in range(1, 9):
        p = agg[f"small128|0.5|{n}"]
        if p:
            small_rows.append({"n": n, **p, "d_top_pred": C / (F0 * LAC ** (n - 1))})

    # ---------- 留出种子验证：训练 {1,7,42}，测试 {99,2024,31337} ----------
    train_s, test_s = {1, 7, 42}, {99, 2024, 31337}
    heldout = {}
    for role, g in (("A", 0.5), ("B", 0.8)):
        rr = [r for r in recs if r["role"] == role]

        def split_dists(n, seeds):
            arrs = [r["dists"] for r in rr if r["octaves"] == n and r["seed"] in seeds]
            return np.concatenate(arrs) if arrs else np.array([])

        floor_tr = float(np.mean([split_dists(n, train_s).mean() for n in (7, 8)]))
        rows = []
        for n in range(1, 9):
            dte = split_dists(n, test_s)
            ftop = F0 * LAC ** (n - 1)
            d_top = C / ftop
            d_sat = max(d_top, floor_tr)
            rows.append({"n": n, "test_mean": float(dte.mean()),
                         "test_var": float(dte.var(ddof=1)), "n_dists": len(dte),
                         "d_top_pred": d_top, "d_sat_pred": d_sat,
                         "err_top": float(dte.mean() - d_top),
                         "err_sat": float(dte.mean() - d_sat)})
        heldout[role] = {"floor_train": floor_tr, "rows": rows}

    # ---------- CSV 导出 ----------
    import csv
    with open(OUT / "spacing_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "gain", "N", "gN", "meas_mean_px", "meas_sd_px",
                    "meas_var_px2", "kappa_var_over_mean2", "se_of_mean",
                    "pred_top_px", "abserr_top_px", "pred_satfloor_px",
                    "abserr_sat_px", "pred_product_C_over_f0gN_px",
                    "abserr_product_px", "n_distances"])
        for role in ("A", "B"):
            for r in table[role]:
                w.writerow([role, r["g"], r["n"], r["gN"],
                            f"{r['meas']:.3f}", f"{r['sd']:.3f}", f"{r['var']:.3f}",
                            f"{r['kappa']:.4f}", f"{r['se_mean']:.4f}",
                            f"{r['d_top_pred']:.3f}", f"{r['abserr_top']:.3f}",
                            f"{r['d_sat_pred']:.3f}", f"{r['abserr_sat']:.3f}",
                            f"{r['d_product_pred']:.2f}", f"{r['abserr_product']:.2f}",
                            r["n_dists"]])

    result = {
        "C_weighted": C, "C_median": C_med, "loglog_slope": slope,
        "Cp_best_product": Cp_best,
        "calibration": cal_rows,
        "A": table["A"], "B": table["B"],
        "heldout": heldout,
        "product_factorial": prod_rows,
        "small128": small_rows,
    }

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    # dists 不写进 json
    json.dump(result, open(OUT / "results.json", "w"), indent=1, default=default)

    # ---------- 打印摘要 ----------
    print(f"CALIBRATION C_weighted={C:.4f} C_median={C_med:.4f} loglog_slope={slope:.4f}")
    print("\n单倍频标定:")
    for r in cal_rows:
        print(f"  f={r['f']:.6f} ell={r['ell']:6.2f} mean={r['mean']:6.2f} "
              f"C={r['C']:.3f} kappa={r['kappa']:.3f} n={r['n_dists']}")

    for role in ("A", "B"):
        print(f"\n===== {role} 组 =====")
        print(" N  gN   meas   d_top  errTop >2?  d_sat  errSat >2? | d_prod errP >2? | kappa  n")
        for r in table[role]:
            flag = lambda e: "超2px" if abs(e) > 2 else ""
            ep = r["meas"] - r["d_product_pred"]
            es = r["meas"] - r["d_sat_pred"]
            print(f"{r['n']:2d} {r['gN']:4.1f} {r['meas']:6.2f} "
                  f"{r['d_top_pred']:6.2f} {r['err_top']:+6.2f}  {flag(r['err_top']):5s}  "
                  f"{r['d_sat_pred']:6.2f} {es:+6.2f}   {flag(es):5s}   "
                  f"{r['d_product_pred']:6.1f} {ep:+7.1f} {flag(ep):5s} | "
                  f"{r['kappa']:.3f} {r['n_dists']}")
        print(f" floor_g={table[role][0]['floor_g']:.2f}")

    print(f"\n产品模型全局最优 Cp={Cp_best:.3f}")
    print("\n等 gN 乘积 / 变增益:")
    for r in prod_rows:
        print(f"  g={r['g']:.4f} N={r['n']} gN={r['gN']:.3f} meas={r['meas']:6.2f} "
              f"ell_top={r['ell_top']:6.2f} n={r['n_dists']}")
    print("\n128px 默认尺寸:")
    for r in small_rows:
        print(f"  N={r['n']} mean={r['mean']:6.2f} sd={r['sd']:5.2f} n={r['n_dists']} "
              f"pred={r['d_top_pred']:6.2f} naive={r['mean_naive']:6.2f}")

    print("\n留出种子验证（地板仅由训练种子 N=7,8 标定）:")
    for role in ("A", "B"):
        h = heldout[role]
        print(f" {role} floor_train={h['floor_train']:.2f}")
        for r in h["rows"]:
            flag = " 超2px" if abs(r["err_sat"]) > 2 else ""
            print(f"   N={r['n']} test={r['test_mean']:6.2f} "
                  f"pred_sat={r['d_sat_pred']:6.2f} err={r['err_sat']:+5.2f}{flag}")


if __name__ == "__main__":
    main()
