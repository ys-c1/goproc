#!/usr/bin/env python3
"""Finite phase-candidate peak audit for goproc fBm maps.

The predictor intentionally evaluates only:
  1. the finite set of lowest-frequency lattice extrema;
  2. finest-frequency lattice nodes and a 3x3 pixel phase check around their
     local refinement.

The full raster is evaluated afterwards only to audit the prediction.
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import generate_binary_structure, label, maximum_filter
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "peak_analysis"
DATA = OUT / "data"
IMG = OUT / "images"

SEEDS = [7, 42]
SIZE = 128
OCTAVES = 4
LACUNARITY = 2.0
GAIN = 0.5
F0 = 0.02
# One lowest-frequency lattice wavelength.  This is the explicit audit radius;
# the derivative/Hessian argument in PEAK_REPORT.md explains the localization.
ZONE_RADIUS_PX = 1.0 / F0
MATCH_TOL_PX = 1.5


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


class Noise:
    """Exact Python port of internal/noise/noise.go for positive/negative nodes."""

    def __init__(self, seed: int):
        p = list(range(256))
        state = int(seed)
        for i in range(255, 0, -1):
            state = splitmix64(state)
            j = state % (i + 1)
            p[i], p[j] = p[j], p[i]
        self.perm = np.array(p + p, dtype=np.int64)

    def corner(self, ix: int, iy: int) -> float:
        return float(self.perm[(int(self.perm[ix & 255]) + int(iy)) & 511] / 127.5 - 1.0)

    @staticmethod
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def value_scalar(self, x: float, y: float) -> float:
        x0 = math.floor(x)
        y0 = math.floor(y)
        fx = x - x0
        fy = y - y0
        u = self.fade(fx)
        v = self.fade(fy)
        c00 = self.corner(x0, y0)
        c10 = self.corner(x0 + 1, y0)
        c01 = self.corner(x0, y0 + 1)
        c11 = self.corner(x0 + 1, y0 + 1)
        a = c00 + (c10 - c00) * u
        b = c01 + (c11 - c01) * u
        return a + (b - a) * v

    def value_grid(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx = x - x0
        fy = y - y0
        u = self.fade(fx)
        v = self.fade(fy)
        perm = self.perm

        def corners(ix, iy):
            return perm[(perm[ix & 255] + iy) & 511] / 127.5 - 1.0

        c00 = corners(x0, y0)
        c10 = corners(x0 + 1, y0)
        c01 = corners(x0, y0 + 1)
        c11 = corners(x0 + 1, y0 + 1)
        a = c00 + (c10 - c00) * u
        b = c01 + (c11 - c01) * u
        return a + (b - a) * v


def frequencies() -> np.ndarray:
    return np.array([F0 * LACUNARITY**k for k in range(OCTAVES)], dtype=float)


def weights() -> np.ndarray:
    a = np.array([GAIN**k for k in range(OCTAVES)], dtype=float)
    return a / a.sum()


FS = frequencies()
W = weights()


def fbm_scalar(n: Noise, x: float, y: float) -> float:
    return float(sum(W[k] * n.value_scalar(x * FS[k], y * FS[k]) for k in range(OCTAVES)))


def fbm_layers_scalar(n: Noise, x: float, y: float) -> list[float]:
    return [float(n.value_scalar(x * FS[k], y * FS[k])) for k in range(OCTAVES)]


def actual_field(seed: int) -> tuple[np.ndarray, np.ndarray, Noise]:
    n = Noise(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    layers = np.stack([n.value_grid(xx * f, yy * f) for f in FS])
    field = np.tensordot(W, layers, axes=(0, 0))
    return field, layers, n


@dataclass
class Peak:
    x: float
    y: float
    ix: int
    iy: int
    value: float
    area: int


def actual_peaks(field: np.ndarray) -> list[Peak]:
    neighborhood = maximum_filter(field, size=3, mode="nearest")
    mask = field >= neighborhood - 1e-15
    lab, count = label(mask, structure=generate_binary_structure(2, 2))
    peaks: list[Peak] = []
    for component in range(1, count + 1):
        ys, xs = np.where(lab == component)
        ix = int(round(float(xs.mean())))
        iy = int(round(float(ys.mean())))
        peaks.append(
            Peak(
                x=float(xs.mean()),
                y=float(ys.mean()),
                ix=ix,
                iy=iy,
                value=float(field[ys[0], xs[0]]),
                area=len(xs),
            )
        )
    peaks.sort(key=lambda p: p.value, reverse=True)
    return peaks


def low_candidates(n: Noise, pad: int = 1) -> list[dict]:
    """Strict axial maxima of octave 0.  Diagonal neighbours are not required."""
    qmax_index = int(math.floor((SIZE - 1) * F0)) + 1
    out = []
    cid = 1
    for i in range(-pad, qmax_index + pad + 1):
        for j in range(-pad, qmax_index + pad + 1):
            c = n.corner(i, j)
            axial = [n.corner(i + 1, j), n.corner(i - 1, j), n.corner(i, j + 1), n.corner(i, j - 1)]
            if all(c > z for z in axial):
                out.append(
                    {
                        "id": f"C{cid:02d}",
                        "i": i,
                        "j": j,
                        "x": i / F0,
                        "y": j / F0,
                        "low_value": c,
                        "in_image": 0 <= i / F0 <= SIZE - 1 and 0 <= j / F0 <= SIZE - 1,
                    }
                )
                cid += 1
    return out


def phase_fractions(x: float, y: float) -> list[dict]:
    rows = []
    for k, f in enumerate(FS):
        qx, qy = f * x, f * y
        px, py = qx - math.floor(qx), qy - math.floor(qy)
        # Exact lattice starts produce harmless 1e-16 residues.
        if abs(px) < 1e-12 or abs(px - 1) < 1e-12:
            px = 0.0
        if abs(py) < 1e-12 or abs(py - 1) < 1e-12:
            py = 0.0
        rows.append(
            {
                "layer": k,
                "frequency": f,
                "phi_x": px,
                "phi_y": py,
                "theta_x_cycles": px,
                "theta_y_cycles": py,
                "theta_x_rad": 2 * math.pi * px,
                "theta_y_rad": 2 * math.pi * py,
            }
        )
    return rows


def predict_finest_phase_peaks(seed: int) -> list[dict]:
    """Finite phase-lattice candidates, accepted only as integer 8-ne maxima."""
    n = Noise(seed)
    h = 1.0 / FS[-1]
    imax = int(math.ceil((SIZE - 1) * FS[-1])) + 1
    accepted: dict[tuple[int, int], dict] = {}

    for a in range(-1, imax + 1):
        for b in range(-1, imax + 1):
            x0, y0 = a * h, b * h
            if x0 + h / 2 < 0 or x0 - h / 2 > SIZE - 1:
                continue
            if y0 + h / 2 < 0 or y0 - h / 2 > SIZE - 1:
                continue

            bounds = [
                (max(0.0, x0 - h / 2), min(float(SIZE - 1), x0 + h / 2)),
                (max(0.0, y0 - h / 2), min(float(SIZE - 1), y0 + h / 2)),
            ]
            result = minimize(
                lambda z: -fbm_scalar(n, z[0], z[1]),
                [x0, y0],
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 300},
            )
            rx, ry = float(result.x[0]), float(result.x[1])
            ix, iy = int(round(rx)), int(round(ry))
            if not (0 <= ix < SIZE and 0 <= iy < SIZE):
                continue

            xlo, xhi = max(0, ix - 1), min(SIZE - 1, ix + 1)
            ylo, yhi = max(0, iy - 1), min(SIZE - 1, iy + 1)
            center_value = fbm_scalar(n, float(ix), float(iy))
            neighbour_values = [
                fbm_scalar(n, float(xx), float(yy))
                for xx in range(xlo, xhi + 1)
                for yy in range(ylo, yhi + 1)
                if (xx, yy) != (ix, iy)
            ]
            if centre_is_peak := center_value + 1e-13 >= max(neighbour_values):
                key = (ix, iy)
                rec = {
                    "x": ix,
                    "y": iy,
                    "continuous_x": rx,
                    "continuous_y": ry,
                    "refined_value": float(-result.fun),
                    "pixel_value": center_value,
                    "start_x": x0,
                    "start_y": y0,
                }
                if key not in accepted or rec["pixel_value"] > accepted[key]["pixel_value"]:
                    accepted[key] = rec

    preds = list(accepted.values())
    preds.sort(key=lambda p: p["pixel_value"], reverse=True)
    for rank, p in enumerate(preds, 1):
        p["rank"] = rank
    return preds


def read_go_pgm_lenient(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    blob = path.read_bytes()
    # The current Go format string emits literal "\n" two-character sequences.
    match = re.match(rb"P5(?:\\n|\s)+(\d+)(?:\\n|\s)+(\d+)(?:\\n|\s)+(\d+)(?:\\n|\s)", blob)
    if match is None:
        # Also accept a standards-compliant PGM.
        match = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", blob)
    if match is None:
        return None
    w, h, maxv = map(int, match.groups())
    data_start = match.end()
    return np.frombuffer(blob[data_start : data_start + w * h], dtype=np.uint8).reshape(h, w)


def write_valid_pgm(path: Path, normalized: np.ndarray) -> None:
    gray = np.clip(np.floor(normalized * 255.0 + 0.5), 0, 255).astype(np.uint8)
    header = f"P5\n{SIZE} {SIZE}\n255\n".encode("ascii")
    path.write_bytes(header + gray.tobytes())


def make_overlay(seed: int, field: np.ndarray, preds: list[dict], peaks: list[Peak],
                 candidates: list[dict], missed: list[dict], winner_candidate: str,
                 path: Path) -> None:
    norm = (field - field.min()) / (field.max() - field.min())
    fig, ax = plt.subplots(figsize=(9.2, 8.0), dpi=170)
    ax.imshow(norm, cmap="terrain", origin="upper", extent=(-0.5, SIZE - 0.5, SIZE - 0.5, -0.5))

    ax.scatter([p.x for p in peaks], [p.y for p in peaks], marker="+", s=22,
               c="white", linewidths=0.55, alpha=0.72, label=f"actual 8-neighbour peaks (n={len(peaks)})")
    ax.scatter([p["x"] for p in preds], [p["y"] for p in preds], marker="x", s=24,
               c="cyan", linewidths=0.8, alpha=0.95, label=f"phase-predicted peaks (n={len(preds)})")
    if missed:
        ax.scatter([m["x"] for m in missed], [m["y"] for m in missed], marker="o", s=115,
                   facecolors="none", edgecolors="magenta", linewidths=1.5,
                   label=f"missed actual peaks (n={len(missed)})")

    for c in candidates:
        if -30 <= c["x"] <= SIZE + 29 and -30 <= c["y"] <= SIZE + 29:
            ax.scatter([c["x"]], [c["y"]], marker="D", s=58, c="#1f77b4",
                       edgecolors="white", linewidths=0.8, zorder=5)
            ax.annotate(c["id"], (c["x"], c["y"]), xytext=(4, 4), textcoords="offset points",
                        color="white", fontsize=7.5, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.16", fc="#1f3b73", ec="none", alpha=0.78))

    gp = preds[0]
    gy, gx = np.unravel_index(np.argmax(field), field.shape)
    ax.scatter([gx], [gy], marker="o", s=260, facecolors="none", edgecolors="red",
               linewidths=2.2, zorder=8, label="actual raw maximum")
    ax.scatter([gp["x"]], [gp["y"]], marker="*", s=190, c="gold", edgecolors="black",
               linewidths=0.65, zorder=9, label="phase-predicted global")

    # Draw association from winning low-frequency candidate to predicted global.
    wc = next(c for c in candidates if c["id"] == winner_candidate)
    ax.plot([wc["x"], gp["x"]], [wc["y"], gp["y"]], color="yellow", lw=1.2, ls="--", alpha=0.9)

    ax.set_xlim(-0.5, SIZE - 0.5)
    ax.set_ylim(SIZE - 0.5, -0.5)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(f"seed={seed}: low-frequency candidates, phase prediction and actual maxima")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=2, fontsize=8.5, frameon=True)
    ax.grid(color="white", alpha=0.16, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    candidate_rows = []
    phase_rows = []
    missed_rows = []
    manifest = {"config": {
        "size": SIZE, "octaves": OCTAVES, "lacunarity": LACUNARITY, "gain": GAIN,
        "base_frequency": F0, "frequencies": FS.tolist(), "weights": W.tolist(),
        "zone_radius_px": ZONE_RADIUS_PX, "match_tolerance_px": MATCH_TOL_PX,
    }, "images": []}

    for seed in SEEDS:
        # Prediction stage: no full field is used by predict_finest_phase_peaks.
        preds = predict_finest_phase_peaks(seed)
        n_pred_noise = Noise(seed)
        candidates = low_candidates(n_pred_noise, pad=1)
        (DATA / f"phase_prediction_seed{seed}.json").write_text(
            json.dumps({
                "seed": seed,
                "config": {
                    "size": SIZE, "octaves": OCTAVES, "lacunarity": LACUNARITY,
                    "gain": GAIN, "base_frequency": F0,
                    "actual_frequencies_per_px": FS.tolist(),
                    "weights": W.tolist(),
                    "zone_radius_px": ZONE_RADIUS_PX,
                },
                "low_frequency_candidates": candidates,
                "predicted_peaks": preds,
                "predicted_global": preds[0] if preds else None,
                "predicted_peak_count": len(preds),
            }, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8",
        )

        field, layers, _ = actual_field(seed)
        peaks = actual_peaks(field)
        gy, gx = np.unravel_index(np.argmax(field), field.shape)
        raw_global = {"x": int(gx), "y": int(gy)}

        normalized = (field - field.min()) / (field.max() - field.min())
        write_valid_pgm(IMG / f"valid_seed{seed}.pgm", normalized)
        np.savez_compressed(DATA / f"fields_seed{seed}.npz", field=field, layers=layers,
                            frequencies=FS, weights=W)

        go_pgm = read_go_pgm_lenient(IMG / f"go_seed{seed}.pgm")
        go_exact_byte_match = None
        pgm_max_pixels = []
        pgm_first = None
        if go_pgm is not None:
            expected_gray = np.clip(np.floor(normalized * 255 + 0.5), 0, 255).astype(np.uint8)
            go_exact_byte_match = bool(np.array_equal(go_pgm, expected_gray))
            max_gray = int(go_pgm.max())
            ys, xs = np.where(go_pgm == max_gray)
            pgm_max_pixels = [{"x": int(x), "y": int(y)} for x, y in zip(xs, ys)]
            first_index = int(np.argmax(go_pgm))
            pgm_first = {"x": first_index % SIZE, "y": first_index // SIZE}

        pred_keys = {(p["x"], p["y"]) for p in preds}
        missed = []
        for p in peaks:
            nearest_pred_dist = min(math.hypot(p.x - q["x"], p.y - q["y"]) for q in preds)
            if (p.ix, p.iy) not in pred_keys and nearest_pred_dist > MATCH_TOL_PX:
                missed.append({
                    "seed": seed, "x": p.ix, "y": p.iy, "x_centroid": p.x, "y_centroid": p.y,
                    "value": p.value, "nearest_prediction_distance_px": nearest_pred_dist,
                })
        missed.sort(key=lambda r: r["nearest_prediction_distance_px"], reverse=True)
        missed_rows.extend(missed)

        # Associate each low candidate with its strongest phase-predicted peak
        # inside one lowest-frequency wavelength.
        for c in candidates:
            nearby = [p for p in preds
                      if math.hypot(c["x"] - p["x"], c["y"] - p["y"]) <= ZONE_RADIUS_PX]
            nearby.sort(key=lambda p: p["pixel_value"], reverse=True)
            best = nearby[0] if nearby else None

            nearest_actual = min(peaks, key=lambda p: math.hypot(c["x"] - p.x, c["y"] - p.y))
            anchor_error = math.hypot(c["x"] - nearest_actual.x, c["y"] - nearest_actual.y)
            if best is not None:
                matched_actual = min(peaks, key=lambda p: math.hypot(best["x"] - p.x, best["y"] - p.y))
                phase_error = math.hypot(best["x"] - matched_actual.x, best["y"] - matched_actual.y)
                status = "phase_hit" if phase_error <= MATCH_TOL_PX else "phase_false_positive"
                pred_x, pred_y = best["x"], best["y"]
                pred_cx, pred_cy = best["continuous_x"], best["continuous_y"]
                matched_x, matched_y = matched_actual.ix, matched_actual.iy
            else:
                status = "empty_candidate"
                pred_x = pred_y = pred_cx = pred_cy = ""
                matched_x = matched_y = ""
                phase_error = ""

            q_layers = fbm_layers_scalar(n_pred_noise, c["x"], c["y"])
            q_score = float(sum(W[k] * q_layers[k] for k in range(OCTAVES)))
            row = {
                "seed": seed, "candidate": c["id"], "i": c["i"], "j": c["j"],
                "anchor_x_px": round(c["x"], 3), "anchor_y_px": round(c["y"], 3),
                "in_image": c["in_image"], "low_value": round(c["low_value"], 6),
                "phase_score_at_anchor": round(q_score, 6),
                "zone_best_phase_value": round(best["pixel_value"], 6) if best is not None else "",
                "pred_x_px": pred_x, "pred_y_px": pred_y,
                "pred_continuous_x": round(pred_cx, 3) if pred_cx != "" else "",
                "pred_continuous_y": round(pred_cy, 3) if pred_cy != "" else "",
                "matched_actual_x_px": matched_x, "matched_actual_y_px": matched_y,
                "phase_position_error_px": round(phase_error, 3) if phase_error != "" else "",
                "nearest_actual_to_anchor_x": nearest_actual.ix,
                "nearest_actual_to_anchor_y": nearest_actual.iy,
                "anchor_to_nearest_actual_px": round(anchor_error, 3),
                "status": status,
            }
            candidate_rows.append(row)

            for ph in phase_fractions(c["x"], c["y"]):
                k = ph["layer"]
                phase_rows.append({
                    "seed": seed, "candidate": c["id"], "i": c["i"], "j": c["j"],
                    "layer": k, "actual_frequency_per_px": ph["frequency"],
                    "phi_x": round(ph["phi_x"], 6), "phi_y": round(ph["phi_y"], 6),
                    "theta_x_rad": round(ph["theta_x_rad"], 6),
                    "theta_y_rad": round(ph["theta_y_rad"], 6),
                    "layer_value_at_anchor": round(q_layers[k], 6),
                    "normalized_weight": W[k],
                    "weighted_contribution": round(W[k] * q_layers[k], 6),
                })

        predicted_global = preds[0]
        # Find candidate owning predicted global; nearest candidate is the label.
        winner_candidate = min(candidates, key=lambda c: math.hypot(
            c["x"] - predicted_global["x"], c["y"] - predicted_global["y"]))
        global_error = math.hypot(predicted_global["x"] - gx, predicted_global["y"] - gy)
        continuous_global_error = math.hypot(
            predicted_global["continuous_x"] - gx,
            predicted_global["continuous_y"] - gy,
        )
        pgm_error = None
        if pgm_first is not None:
            pgm_error = math.hypot(predicted_global["x"] - pgm_first["x"],
                                   predicted_global["y"] - pgm_first["y"])

        expanded_count = len(candidates)
        in_image_count = sum(c["in_image"] for c in candidates)
        empty = [r for r in candidate_rows if r["seed"] == seed and r["status"] == "empty_candidate"]
        false_pred = [r for r in candidate_rows if r["seed"] == seed and r["status"] == "phase_false_positive"]

        summary_rows.append({
            "image": f"seed_{seed}",
            "seed": seed,
            "size": SIZE,
            "actual_frequencies": ";".join(f"{f:.2f}" for f in FS),
            "weights": ";".join(f"{w:.5f}" for w in W),
            "low_candidates_expanded": expanded_count,
            "low_candidates_in_image": in_image_count,
            "winner_candidate": winner_candidate["id"],
            "winner_lattice_ij": f"({winner_candidate['i']},{winner_candidate['j']})",
            "winner_anchor_xy": f"({winner_candidate['x']:.1f},{winner_candidate['y']:.1f})",
            "pred_global_xy": f"({predicted_global['x']},{predicted_global['y']})",
            "actual_raw_global_xy": f"({gx},{gy})",
            "raw_pixel_error_px": round(global_error, 6),
            "continuous_to_raw_error_px": round(continuous_global_error, 6),
            "pgm_first_max_xy": "" if pgm_first is None else f"({pgm_first['x']},{pgm_first['y']})",
            "pgm_max_tie_pixels": len(pgm_max_pixels),
            "pgm_vs_pred_error_px": "" if pgm_error is None else round(pgm_error, 6),
            "go_pgm_byte_exact_match": go_exact_byte_match,
            "pred_peak_count": len(preds),
            "actual_peak_count": len(peaks),
            "count_delta_pred_minus_actual": len(preds) - len(peaks),
            "count_abs_delta": abs(len(preds) - len(peaks)),
            "global_multiplicity_pred": 1,
            "global_multiplicity_actual_raw": int(np.sum(np.isclose(field, field.max(), rtol=0, atol=1e-14))),
            "missed_actual_peak_count": len(missed),
            "false_phase_prediction_count": len(false_pred),
            "phase_hit_low_candidate_count": sum(
                1 for r in candidate_rows
                if r["seed"] == seed and r["status"] in {"phase_hit", "phase_false_positive"}
            ),
            "empty_low_candidate_count": len(empty),
            "zone_radius_px": ZONE_RADIUS_PX,
        })

        manifest["images"].append({
            "seed": seed,
            "predicted_global": {k: predicted_global[k] for k in
                                 ["x", "y", "continuous_x", "continuous_y", "pixel_value"]},
            "actual_raw_global": raw_global,
            "pred_peak_count": len(preds),
            "actual_peak_count": len(peaks),
            "predicted_peaks": preds,
            "pgm_max_pixels": pgm_max_pixels,
        })

        make_overlay(seed, field, preds, peaks, candidates, missed,
                     winner_candidate["id"], IMG / f"overlay_seed{seed}.png")

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(DATA / "image_summary.csv", summary_rows)
    write_csv(DATA / "candidate_report.csv", candidate_rows)
    write_csv(DATA / "phase_vectors.csv", phase_rows)
    write_csv(DATA / "missed_actual_peaks.csv", missed_rows)
    (DATA / "prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # Human-readable markdown tables.
    def md_table(rows: list[dict], columns: list[str] | None = None) -> str:
        if not rows:
            return "_无_\n"
        columns = columns or list(rows[0].keys())
        out = ["| " + " | ".join(columns) + " |",
               "| " + " | ".join(["---"] * len(columns)) + " |"]
        for r in rows:
            out.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
        return "\n".join(out) + "\n"

    summary_cols = ["image", "low_candidates_expanded", "phase_hit_low_candidate_count",
                    "empty_low_candidate_count", "winner_candidate", "pred_global_xy",
                    "actual_raw_global_xy", "raw_pixel_error_px", "pgm_first_max_xy",
                    "pgm_max_tie_pixels", "pred_peak_count", "actual_peak_count",
                    "count_delta_pred_minus_actual", "count_abs_delta", "missed_actual_peak_count"]
    (DATA / "image_summary.md").write_text(md_table(summary_rows, summary_cols), encoding="utf-8")

    candidate_md = "# Candidate rows\n\n"
    for seed in SEEDS:
        rows = [r for r in candidate_rows if r["seed"] == seed]
        cols = ["candidate", "i", "j", "anchor_x_px", "anchor_y_px", "phase_score_at_anchor",
                "zone_best_phase_value", "pred_x_px", "pred_y_px", "matched_actual_x_px", "matched_actual_y_px",
                "phase_position_error_px", "nearest_actual_to_anchor_x",
                "nearest_actual_to_anchor_y", "anchor_to_nearest_actual_px", "status"]
        candidate_md += f"\n## seed {seed}\n\n" + md_table(rows, cols)
    (DATA / "candidate_report.md").write_text(candidate_md, encoding="utf-8")

    missed_md = "# Missed actual peaks\n\n" + md_table(missed_rows)
    (DATA / "missed_actual_peaks.md").write_text(missed_md, encoding="utf-8")

    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
