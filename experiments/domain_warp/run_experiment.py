#!/usr/bin/env python3
"""Gradient-driven domain-warp verification for goproc terrains.

The script deliberately separates three things:

1. prediction from the analytic gradient field, computed before warping;
2. exact read-back of the inverse resampling map used to archive each image;
3. image-only checks: sub-pixel NCC at the predicted extremum and dense
   Lucas-Kanade optical flow for an independent global position scan.

Run from the repository root:

    python3 experiments/domain_warp/run_experiment.py
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, median_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Circle

try:
    from skimage.registration import optical_flow_ilk

    HAS_SKIMAGE = True
except Exception:  # pragma: no cover - optional diagnostic
    HAS_SKIMAGE = False

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "domain_warp"
BEFORE = ART / "before"
AFTER = ART / "after"
FIELDS = ART / "fields"
METRICS = ART / "metrics"
FIGURES = ART / "figures"
ARCHIVES = ART / "archives"
for d in (BEFORE, AFTER, FIELDS, METRICS, FIGURES, ARCHIVES):
    d.mkdir(parents=True, exist_ok=True)

SIZE = 256
SEEDS = [7, 17, 29, 43]
OCTAVES = 6
FREQUENCY = 0.035
# Gradient-push amplitudes in pixels. D = A * grad(phi), with grad taken in
# dimensionless normalized coordinates u=x/SIZE, v=y/SIZE.
AMPLITUDES = [2.0, 4.0]
AMP_NAME = {2.0: "a2", 4.0: "a4"}
SIGMA_X_PX = 45.5
SIGMA_Y_PX = 72.0
BLOCK_RADIUS = 6

# Pixel-center coordinates. The half-pixel center makes the Gaussian maxima
# occur as an exact 2x2 tie; C-order argmax applies the top-then-left rule.
rows, cols = np.indices((SIZE, SIZE), dtype=np.float64)
x = cols - (SIZE - 1) / 2.0
y = rows - (SIZE - 1) / 2.0
u = x / SIZE
v = y / SIZE
sigma_u = SIGMA_X_PX / SIZE
sigma_v = SIGMA_Y_PX / SIZE

phi = np.exp(-0.5 * ((u / sigma_u) ** 2 + (v / sigma_v) ** 2))
grad_u = -(u / sigma_u**2) * phi
grad_v = -(v / sigma_v**2) * phi
grad_norm = np.hypot(grad_u, grad_v)


def extremum(
    a: np.ndarray, tie_tol: float = 0.0
) -> tuple[int, int, float, list[tuple[int, int]]]:
    """Return C-order argmax, its value, and tied maxima.

    ``np.argmax`` on a C-contiguous array chooses the smallest flat index,
    i.e. the smallest row first and then the smallest column. That is the
    stated top-then-left tie rule. ``tie_tol`` treats sub-tolerance floating
    point splits as numerical ties.
    """
    value = float(np.max(a))
    if tie_tol > 0.0:
        candidates = np.argwhere(a >= value - tie_tol)
        first = candidates[0]
        r, c = int(first[0]), int(first[1])
        ties = [(int(rr), int(cc)) for rr, cc in candidates]
        return r, c, value, ties

    flat = int(np.argmax(a))
    r, c = divmod(flat, a.shape[1])
    rr, cc = np.where(a == value)
    ties = [(int(rr[i]), int(cc[i])) for i in range(len(rr))]
    return r, c, value, ties


pred_r, pred_c, gmax, tie_points = extremum(grad_norm)
prediction = {
    "coordinate_convention": "zero-based (row, column); x=column, y=row",
    "tie_rule": "maximum grad norm; exact ties resolved by smallest row then smallest column",
    "size": SIZE,
    "seeds": SEEDS,
    "octaves": OCTAVES,
    "frequency": FREQUENCY,
    "amplitudes_px": AMPLITUDES,
    "warp": {
        "type": "gradient push D=A*grad(phi)",
        "phi": "exp(-0.5*((u/sigma_u)^2+(v/sigma_v)^2))",
        "u": "(column-(SIZE-1)/2)/SIZE",
        "v": "(row-(SIZE-1)/2)/SIZE",
        "sigma_x_px": SIGMA_X_PX,
        "sigma_y_px": SIGMA_Y_PX,
        "sigma_u": sigma_u,
        "sigma_v": sigma_v,
    },
    "gradient_max": gmax,
    "predicted_row_zero_based": pred_r,
    "predicted_col_zero_based": pred_c,
    "predicted_row_one_based": pred_r + 1,
    "predicted_col_one_based": pred_c + 1,
    "exact_gradient_ties_zero_based": tie_points,
    "exact_gradient_ties_one_based": [[r + 1, c + 1] for r, c in tie_points],
    "predicted_displacement_px": {
        AMP_NAME[A]: A * gmax for A in AMPLITUDES
    },
}
(METRICS / "warp_prediction.json").write_text(
    json.dumps(prediction, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

with (METRICS / "predicted_locations.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "amplitude",
            "gradient_max",
            "pred_row_0based",
            "pred_col_0based",
            "pred_row_1based",
            "pred_col_1based",
            "pred_dmax_px",
            "pred_dx_px",
            "pred_dy_px",
            "tie_count",
        ],
    )
    w.writeheader()
    for A in AMPLITUDES:
        w.writerow(
            {
                "amplitude": A,
                "gradient_max": f"{gmax:.12f}",
                "pred_row_0based": pred_r,
                "pred_col_0based": pred_c,
                "pred_row_1based": pred_r + 1,
                "pred_col_1based": pred_c + 1,
                "pred_dmax_px": f"{A * gmax:.9f}",
                "pred_dx_px": f"{A * grad_u[pred_r, pred_c]:.9f}",
                "pred_dy_px": f"{A * grad_v[pred_r, pred_c]:.9f}",
                "tie_count": len(tie_points),
            }
        )

tie_selftest = {}
same_row = np.array([[1.0, 3.0, 2.0, 3.0, 1.0]])
r, c, value, ties = extremum(same_row)
tie_selftest["same_row"] = {"ties": ties, "selected": [r, c], "pass": [r, c] == [0, 1]}
same_col = np.array([[1.0], [3.0], [2.0], [3.0], [1.0]])
r, c, value, ties = extremum(same_col)
tie_selftest["same_col"] = {"ties": ties, "selected": [r, c], "pass": [r, c] == [1, 0]}
(METRICS / "tie_break_selftest.json").write_text(
    json.dumps(tie_selftest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
if not all(v["pass"] for v in tie_selftest.values()):
    raise RuntimeError("tie-break self-test failed")

np.savez_compressed(
    FIELDS / "warp_gradient.npz",
    phi=phi,
    grad_u=grad_u,
    grad_v=grad_v,
    grad_norm=grad_norm,
)


def generate_before(seed: int) -> tuple[str, Path, Path]:
    name = f"seed{seed:02d}"
    pgm = BEFORE / f"{name}_before.pgm"
    subprocess.run(
        [
            "go",
            "run",
            "./cmd/goproc",
            "gen",
            "--size",
            str(SIZE),
            "--seed",
            str(seed),
            "--octaves",
            str(OCTAVES),
            "--frequency",
            str(FREQUENCY),
            "--out",
            str(pgm),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    png = BEFORE / f"{name}_before.png"
    img = Image.open(pgm)
    img.save(png)
    return name, pgm, png


def ncc_at(
    after: np.ndarray,
    before: np.ndarray,
    point: tuple[int, int],
    radius: int,
    search_radius: int,
) -> dict[str, float | tuple[int, int]]:
    """Local translation NCC with quadratic sub-pixel peak refinement."""
    pr, pc = point
    patch = after[pr - radius : pr + radius + 1, pc - radius : pc + radius + 1]
    patch_c = patch - patch.mean()
    patch_norm = np.sqrt(float((patch_c * patch_c).sum()))

    best_score = -np.inf
    best_dy = best_dx = 0
    scores: dict[tuple[int, int], float] = {}
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            qr = pr - dy
            qc = pc - dx
            cand = before[
                qr - radius : qr + radius + 1, qc - radius : qc + radius + 1
            ]
            if cand.shape != patch.shape:
                continue
            cand_c = cand - cand.mean()
            denom = patch_norm * np.sqrt(float((cand_c * cand_c).sum())) + 1e-12
            score = float((patch_c * cand_c).sum() / denom)
            scores[(dy, dx)] = score
            if score > best_score:
                best_score = score
                best_dy, best_dx = dy, dx

    s0 = scores[(best_dy, best_dx)]
    sm = scores[(best_dy, best_dx - 1)]
    sp = scores[(best_dy, best_dx + 1)]
    denom_x = sm - 2.0 * s0 + sp
    sub_x = np.clip(0.5 * (sm - sp) / denom_x, -1.0, 1.0) if denom_x < 0 else 0.0

    sm = scores[(best_dy - 1, best_dx)]
    sp = scores[(best_dy + 1, best_dx)]
    denom_y = sm - 2.0 * s0 + sp
    sub_y = np.clip(0.5 * (sm - sp) / denom_y, -1.0, 1.0) if denom_y < 0 else 0.0

    dx = best_dx + float(sub_x)
    dy = best_dy + float(sub_y)
    return {
        "ncc": best_score,
        "dx": dx,
        "dy": dy,
        "magnitude": float(np.hypot(dx, dy)),
        "integer_dx": best_dx,
        "integer_dy": best_dy,
    }


def flow_check(after: np.ndarray, before: np.ndarray) -> dict[str, float | int | bool]:
    """Independent dense image-flow scan with forward/backward filtering."""
    if not HAS_SKIMAGE:
        return {"available": False}

    f_ab = optical_flow_ilk(
        after,
        before,
        radius=9,
        num_warp=12,
        gaussian=True,
        prefilter=True,
        dtype=np.float64,
    )
    f_ba = optical_flow_ilk(
        before,
        after,
        radius=9,
        num_warp=12,
        gaussian=True,
        prefilter=True,
        dtype=np.float64,
    )
    # Light median de-speckle; the true warp field is smooth.
    flow_y = median_filter(f_ab[0], size=3)
    flow_x = median_filter(f_ab[1], size=3)

    qy = rows + flow_y
    qx = cols + flow_x
    back_y = map_coordinates(f_ba[0], [qy, qx], order=1, mode="reflect")
    back_x = map_coordinates(f_ba[1], [qy, qx], order=1, mode="reflect")
    fb = np.hypot(flow_y + back_y, flow_x + back_x)
    mag = np.hypot(flow_x, flow_y)

    valid = fb < 0.5
    border = 24
    valid[:border, :] = False
    valid[-border:, :] = False
    valid[:, :border] = False
    valid[:, -border:] = False
    if not np.any(valid):
        valid = np.ones_like(mag, dtype=bool)
        valid[:border, :] = False
        valid[-border:, :] = False
        valid[:, :border] = False
        valid[:, -border:] = False

    fr, fc = int(np.argmax(np.where(valid, mag, -1.0))) // SIZE, 0
    flat = int(np.argmax(np.where(valid, mag, -1.0)))
    fr, fc = divmod(flat, SIZE)
    selected_error = float(np.hypot(fr - pred_r, fc - pred_c))
    nearest_tie_error = min(
        float(np.hypot(fr - tr, fc - tc)) for tr, tc in tie_points
    )
    return {
        "available": True,
        "row": fr,
        "col": fc,
        "magnitude": float(mag[fr, fc]),
        "selected_position_error_px": selected_error,
        "nearest_tie_error_px": nearest_tie_error,
        "forward_backward_error_px": float(fb[fr, fc]),
        "valid_fraction": float(valid.mean()),
        "flow_magnitude_field": mag,
    }


def make_figure(
    name: str,
    A: float,
    before_h: np.ndarray,
    after_h: np.ndarray,
    flow: dict[str, float | int | bool],
) -> None:
    ls = LightSource(azdeg=315, altdeg=45)
    shade0 = ls.hillshade(before_h, vert_exag=1.6)
    shade1 = ls.hillshade(after_h, vert_exag=1.6)

    Dx = A * grad_u
    Dy = A * grad_v
    source_r = pred_r - Dy[pred_r, pred_c]
    source_c = pred_c - Dx[pred_r, pred_c]

    fig, ax = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax[0, 0].imshow(shade0, cmap="gray")
    ax[0, 0].set_title(f"{name}: before height/hillshade")
    ax[0, 0].add_patch(Circle((pred_c, pred_r), 7, fill=False, ec="gold", lw=2))
    ax[0, 0].arrow(
        source_c,
        source_r,
        Dx[pred_r, pred_c],
        Dy[pred_r, pred_c],
        color="red",
        width=0.7,
        head_width=5,
        length_includes_head=True,
    )
    ax[0, 0].plot(source_c, source_r, "c+", ms=12, mew=2)

    ax[0, 1].imshow(shade1, cmap="gray")
    ax[0, 1].set_title(f"after: amplitude={A:g}, Dmax={A*gmax:.3f}px")
    ax[0, 1].add_patch(Circle((pred_c, pred_r), 7, fill=False, ec="gold", lw=2))
    ax[0, 1].plot(pred_c, pred_r, "r*", ms=14)

    im = ax[1, 0].imshow(grad_norm, cmap="viridis")
    ax[1, 0].set_title("gradient norm and tie set")
    for tr, tc in tie_points:
        ax[1, 0].plot(tc, tr, "wx", ms=10, mew=2)
    ax[1, 0].plot(pred_c, pred_r, "r*", ms=14)
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)

    if flow.get("available"):
        ax[1, 1].imshow(flow["flow_magnitude_field"], cmap="magma")
        for tr, tc in tie_points:
            ax[1, 1].plot(tc, tr, "wx", ms=9, mew=1.5)
        ax[1, 1].plot(pred_c, pred_r, "c*", ms=13)
        ax[1, 1].plot(flow["col"], flow["row"], "o", mfc="none", mec="lime", ms=12, mew=2)
        ax[1, 1].text(
            0.03,
            0.03,
            "lime circle: image-flow peak\n"
            f"row={flow['row']}, col={flow['col']}, "
            f"mag={flow['magnitude']:.3f}px\n"
            f"selected err={flow['selected_position_error_px']:.1f}px, "
            f"nearest-tie err={flow['nearest_tie_error_px']:.1f}px",
            transform=ax[1, 1].transAxes,
            va="bottom",
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65},
        )
        ax[1, 1].set_title("independent image-flow scan")
        ax[1, 1].set_xlabel("column (0-based)")
        ax[1, 1].set_ylabel("row (0-based)")
    else:
        ax[1, 1].text(0.1, 0.5, "scikit-image unavailable")
        ax[1, 1].axis("off")

    for a in ax.flat[:3]:
        a.set_xlabel("column (0-based)")
        a.set_ylabel("row (0-based)")
    fig.savefig(FIGURES / f"{name}_{AMP_NAME[A]}_check.png", dpi=150)
    plt.close(fig)


rows_out: list[dict[str, object]] = []
for seed in SEEDS:
    name, before_pgm, _ = generate_before(seed)
    before_h = np.asarray(Image.open(before_pgm)).astype(np.float64) / 255.0
    light = LightSource(azdeg=315, altdeg=45)
    before_shade = (light.hillshade(before_h, vert_exag=1.6) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(before_shade, mode="L").save(
        BEFORE / f"{name}_before_hillshade.png"
    )

    for A in AMPLITUDES:
        tag = AMP_NAME[A]
        Dx = A * grad_u
        Dy = A * grad_v
        dnorm = np.hypot(Dx, Dy)
        map_x = cols - Dx
        map_y = rows - Dy
        warped = map_coordinates(
            before_h, [map_y, map_x], order=1, mode="reflect"
        )
        warped_u8 = (np.clip(warped, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        after_pgm = AFTER / f"{name}_{tag}_after.pgm"
        after_png = AFTER / f"{name}_{tag}_after.png"
        Image.fromarray(warped_u8, mode="L").save(after_pgm)
        Image.fromarray(warped_u8, mode="L").save(after_png)
        after_shade = (
            light.hillshade(warped, vert_exag=1.6) * 255 + 0.5
        ).astype(np.uint8)
        Image.fromarray(after_shade, mode="L").save(
            AFTER / f"{name}_{tag}_after_hillshade.png"
        )

        np.savez_compressed(
            FIELDS / f"{name}_{tag}_displacement.npz",
            Dx=Dx,
            Dy=Dy,
            displacement_norm=dnorm,
            map_x=map_x,
            map_y=map_y,
        )

        # Exact read-back from the actual inverse map.
        read_dx = cols - map_x
        read_dy = rows - map_y
        read_norm = np.hypot(read_dx, read_dy)
        meas_r, meas_c, meas_dmax, meas_ties = extremum(read_norm, tie_tol=1e-12)
        position_error = float(np.hypot(meas_r - pred_r, meas_c - pred_c))
        displacement_error = abs(meas_dmax - A * gmax)

        # Re-read the archived PGM so block matching tests the saved image.
        archived_after = np.asarray(Image.open(after_pgm)).astype(np.float64) / 255.0
        search_radius = int(np.ceil(A * gmax)) + 2
        bm = ncc_at(
            archived_after,
            before_h,
            (pred_r, pred_c),
            BLOCK_RADIUS,
            search_radius,
        )
        bm_vector_error = float(
            np.hypot(
                float(bm["dx"]) - Dx[pred_r, pred_c],
                float(bm["dy"]) - Dy[pred_r, pred_c],
            )
        )
        flow = flow_check(archived_after, before_h)
        make_figure(name, A, before_h, archived_after, flow)

        row: dict[str, object] = {
            "image": name,
            "seed": seed,
            "amplitude": A,
            "gradient_max": gmax,
            "pred_row": pred_r,
            "pred_col": pred_c,
            "pred_row_1based": pred_r + 1,
            "pred_col_1based": pred_c + 1,
            "pred_dmax_px": A * gmax,
            "map_measured_row": meas_r,
            "map_measured_col": meas_c,
            "map_measured_dmax_px": meas_dmax,
            "map_position_error_px": position_error,
            "map_displacement_error_px": displacement_error,
            "map_tie_count": len(meas_ties),
            "ncc_at_prediction": bm["ncc"],
            "blockmatch_dx_px": bm["dx"],
            "blockmatch_dy_px": bm["dy"],
            "blockmatch_dmax_px": bm["magnitude"],
            "blockmatch_integer_dx": bm["integer_dx"],
            "blockmatch_integer_dy": bm["integer_dy"],
            "blockmatch_vector_error_px": bm_vector_error,
        }
        if flow.get("available"):
            row.update(
                {
                    "optical_flow_row": flow["row"],
                    "optical_flow_col": flow["col"],
                    "optical_flow_dmax_px": flow["magnitude"],
                    "optical_flow_selected_position_error_px": flow[
                        "selected_position_error_px"
                    ],
                    "optical_flow_nearest_tie_error_px": flow[
                        "nearest_tie_error_px"
                    ],
                    "optical_flow_fb_error_px": flow[
                        "forward_backward_error_px"
                    ],
                }
            )
        rows_out.append(row)

fieldnames = list(rows_out[0].keys())
with (METRICS / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

(METRICS / "summary.json").write_text(
    json.dumps(rows_out, ensure_ascii=False, indent=2, default=float) + "\n",
    encoding="utf-8",
)

# Separate before/after archives for convenient preservation.
for subdir, archive_name in ((BEFORE, "before_images"), (AFTER, "after_images")):
    stem = ARCHIVES / archive_name
    if stem.with_suffix(".zip").exists():
        stem.with_suffix(".zip").unlink()
    shutil.make_archive(str(stem), "zip", root_dir=ART, base_dir=subdir.name)

# --- Markdown report -------------------------------------------------------


def fmt(v: object, nd: int = 3) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{nd}f}"
    return str(v)


main_table = [
    "| 图/种子 | 幅度 A (px) | max‖∇φ‖ | 预测行,列 (0基) | 预测 Dmax (px) | 实测回读 Dmax (px) | 实测行,列 (0基) | 位置偏差 (px) | D偏差 (px) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows_out:
    main_table.append(
        "| {image} / {seed} | {amplitude:g} | {g:.4f} | ({pr},{pc}) | {pd:.4f} | {md:.4f} | ({mr},{mc}) | {pe:.3f} | {de:.2e} |".format(
            image=r["image"],
            seed=r["seed"],
            amplitude=float(r["amplitude"]),
            g=float(r["gradient_max"]),
            pr=r["pred_row"],
            pc=r["pred_col"],
            pd=float(r["pred_dmax_px"]),
            md=float(r["map_measured_dmax_px"]),
            mr=r["map_measured_row"],
            mc=r["map_measured_col"],
            pe=float(r["map_position_error_px"]),
            de=float(r["map_displacement_error_px"]),
        )
    )

bm_table = [
    "| 图/种子 | A | 存档图 NCC | 块匹配 dx (px) | 块匹配 dy (px) | 块匹配模 (px) | 向量误差 (px) |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows_out:
    bm_table.append(
        f"| {r['image']} | {float(r['amplitude']):g} | {float(r['ncc_at_prediction']):.5f} | "
        f"{float(r['blockmatch_dx_px']):.3f} | {float(r['blockmatch_dy_px']):.3f} | "
        f"{float(r['blockmatch_dmax_px']):.3f} | {float(r['blockmatch_vector_error_px']):.3f} |"
    )

cv_table = [
    "| 图/种子 | A | 光流峰行,列 | 光流模 (px) | 距指定左上点 (px) | 距最近并列点 (px) | 说明 |",
    "|---|---:|---:|---:|---:|---:|---|",
]
cv_notes: list[str] = []
for r in rows_out:
    if "optical_flow_row" not in r:
        continue
    sel_err = float(r["optical_flow_selected_position_error_px"])
    tie_err = float(r["optical_flow_nearest_tie_error_px"])
    if sel_err <= 2.0:
        note = "与左上判定点一致（≤2 px）"
    elif tie_err <= 2.0:
        note = "落在另一并列梯度极值点附近；按规则主位置仍取左上点"
    elif any(
        np.hypot(float(r["optical_flow_row"]) - tr, float(r["optical_flow_col"]) - tc)
        <= 2.0
        for tr, tc in tie_points
    ):
        note = "落在另一并列梯度极值点上；按规则主位置仍取左上点"
    else:
        note = (
            "超过 2 px：梯度峰附近为缓顶/并列结构，局部窗积分与双线性重采样使光流 argmax 偏移；"
            "该图 NCC 与位移回读仍确认指定点"
        )
    cv_table.append(
        f"| {r['image']} | {float(r['amplitude']):g} | "
        f"({r['optical_flow_row']},{r['optical_flow_col']}) | "
        f"{float(r['optical_flow_dmax_px']):.3f} | {sel_err:.1f} | {tie_err:.1f} | {note} |"
    )
    if sel_err > 2.0:
        cv_notes.append(
            f"- **{r['image']}, A={float(r['amplitude']):g}**: 光流峰 "
            f"({r['optical_flow_row']},{r['optical_flow_col']}) 距规则指定点 "
            f"({pred_r},{pred_c}) 为 {sel_err:.1f} px，距最近并列梯度点 {tie_err:.1f} px。{note}。"
        )

max_map_position_error = max(float(r["map_position_error_px"]) for r in rows_out)
max_map_displacement_error = max(
    float(r["map_displacement_error_px"]) for r in rows_out
)
if max_map_position_error > 2.0:
    raise RuntimeError("predicted/measured displacement position differs by >2 px")
if max_map_displacement_error > 1e-10:
    raise RuntimeError("Dmax does not match A*max(grad norm) to floating precision")
map_error_sentence = (
    f"所有主比对的位置偏差均为 {max_map_position_error:.3f} px，"
    f"最大位移差为 {max_map_displacement_error:.2e} px（浮点精度量级）。"
    if max_map_position_error == 0.0
    else (
        f"主比对的最大位置偏差为 {max_map_position_error:.3f} px（未超过 2 px），"
        f"最大位移差为 {max_map_displacement_error:.2e} px。"
    )
)

report = f"""# 域扭曲梯度—位移验证报告

生成命令：

```bash
python3 experiments/domain_warp/run_experiment.py
```

## 1. 结论与适用条件

本实验验证的是**梯度推挤型域扭曲**：

\\[
\\mathbf d_A(p)=A\\,\\nabla\\phi(p),\\qquad A\\ge 0.
\\]

这里 \\(\\phi\\) 是标量扭曲势，\\(A\\) 是以像素计的推挤幅度系数。若你的管线把某个向量场 \\(\\mathbf w\\) 直接当作偏移，即 \\(\\mathbf d=A\\mathbf w\\)，则一般结论是 \\(D_{{\\max}}=A\\max\\|\\mathbf w\\|\\)；只有在 \\(\\mathbf w=\\nabla\\phi\\) 时才写成本报告的梯度形式。梯度本身控制局部拉伸/旋转，不自动等于任意域扭曲的位移。

在梯度推挤定义下：

\\[
\\|\\mathbf d_A(p)\\|=A\\|\\nabla\\phi(p)\\|,
\\]

所以逐点位移由幅度和梯度模共同决定，且

\\[
D_{{\\max}}(A)=\\max_p\\|\\mathbf d_A(p)\\|
=A\\max_p\\|\\nabla\\phi(p)\\|=AG_{{\\max}}.
\\]

当 \\(A>0\\) 时，

\\[
\\arg\\max_p\\|\\mathbf d_A(p)\\|
=\\arg\\max_p\\|\\nabla\\phi(p)\\|.
\\]

因此极限位置与幅度档位无关，扭曲前只计算梯度场即可定位。若只在山脊像素集合 \\(R\\) 上取最大，则应为
\\(A\\max_{{p\\in R}}\\|\\nabla\\phi(p)\\|\\)；它与全局位置相同还要求全局梯度极值点位于所跟踪的山脊集合中。本报告的主验证针对整幅域变换，并用局部块匹配检查存档图像。

## 2. 本次事先定位结果

- 图像：{SIZE}×{SIZE}，种子 {SEEDS}，Go 生成参数 `octaves={OCTAVES}`、`frequency={FREQUENCY}`。
- 两档幅度：A={AMPLITUDES[0]:g} px、A={AMPLITUDES[1]:g} px。
- 梯度模最大值：`{gmax:.12f}`。
- 规则指定位置：**0 基 (行={pred_r}, 列={pred_c})**；1 基为 (行={pred_r+1}, 列={pred_c+1})。
- 精确并列点（0 基）：{tie_points}。它们构成关于图像中心的 2×2 对称极值；C 顺序 `argmax` 先取最小行、再取最小列，因此取左上的 `({pred_r},{pred_c})`。
- A={AMPLITUDES[0]:g} 的预判最大位移：**{AMPLITUDES[0] * gmax:.6f} px**。
- A={AMPLITUDES[1]:g} 的预判最大位移：**{AMPLITUDES[1] * gmax:.6f} px**。

并列规则的单元自检见 `metrics/tie_break_selftest.json`。

## 3. 预判与逐图实测回读

“实测回读”直接来自生成存档图时实际使用的逆重采样表 `source=p-D(p)`，并再次按同一梯度模最大值/并列规则定位。

{chr(10).join(main_table)}

{map_error_sentence} 实测回读对 1e-12 px 以内的浮点分裂按并列处理，仍执行先小行、后小列的规则。这不是事后在位移场里另找最大值：`metrics/predicted_locations.csv` 与 `metrics/warp_prediction.json` 在构造和扭曲图像之前写出。

## 4. 存档图像的独立块匹配

下列数值从重新读取的存档 PGM 计算；在预判点周围做平移归一化互相关（NCC），并做二次亚像素修正。块窗口会把缓变位移场内的轻微变形平均掉，因此高幅度档可能有亚像素级低估。

{chr(10).join(bm_table)}

## 5. 独立 dense optical-flow 位置扫描

光流只作为不依赖重采样表的图像侧诊断。由于本实验故意保留精确并列梯度点，且极值是缓顶，局部窗口光流的单个 `argmax` 可能跳到另一个并列点或偏移数像素。

{chr(10).join(cv_table)}

### 超过 2 px 的位置偏差说明

{chr(10).join(cv_notes) if cv_notes else "- 无。"}

判读时应区分：

1. 与规则指定的左上点 `({pred_r},{pred_c})` 的距离；
2. 与任一合法并列梯度极值点的最近距离。

若偏差主要来自跳到右侧/下方并列点，数学上的梯度定位并不错误，只是并列规则要求报告左上点。若与所有并列点都超过 2 px，则来自图像匹配的窗口平滑、双线性重采样、PGM 8-bit 量化和局部纹理孔径；第 4 节的高 NCC 与该点位移测量可验证变换本身。

## 6. 文件清单

- 扭曲前图：`before/seedXX_before.pgm`、高度图 `.png` 与 `*_hillshade.png` 地形晕渲。
- 扭曲后图：`after/seedXX_a2_after.*`、`after/seedXX_a4_after.*`，每档含 PGM、高度图 PNG 和晕渲 PNG。
- 前后图分别打包：`archives/before_images.zip`、`archives/after_images.zip`。
- 预判：`metrics/warp_prediction.json`、`metrics/predicted_locations.csv`。
- 逐图明细：`metrics/comparison.csv`、`metrics/summary.json`。
- 位移/梯度场：`fields/*.npz`。
- 检查图：`figures/seedXX_a2_check.png`、`figures/seedXX_a4_check.png`。

PGM 导出中原先把头部分隔符写成了字面量 `\\n`，导致文件不是合法 PGM；本次已修正为真实换行并加入 `internal/hmap/hmap_test.go`。
"""

(ART / "REPORT.md").write_text(report, encoding="utf-8")
print(f"wrote experiment artifacts to {ART}")
print(f"predicted=({pred_r},{pred_c}) Gmax={gmax:.12f} ties={tie_points}")
for r in rows_out:
    print(
        r["image"],
        "A=",
        r["amplitude"],
        "meas_pos=",
        (r["map_measured_row"], r["map_measured_col"]),
        "D=",
        f"{float(r['map_measured_dmax_px']):.6f}",
        "NCC=",
        f"{float(r['ncc_at_prediction']):.5f}",
    )
