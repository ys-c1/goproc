#!/usr/bin/env python3
"""Predict and verify permutation-period autocorrelation peaks after rotation.

The prediction file is written before the Go dumper is executed, so the peak
position is derived from source constants rather than from the autocorrelation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]

# Light-mode reference palette (dataviz skill).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def directional_acf(z: np.ndarray, max_lag: int) -> np.ndarray:
    """Unbiased, row-averaged ACF along output columns; R(0) normalized to 1."""
    x = z - z.mean(axis=1, keepdims=True)
    n = x.shape[1]
    fft_len = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, fft_len, axis=1)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), fft_len, axis=1)[:, :n]
    sums = corr.sum(axis=0)
    lags = np.arange(n)
    unbiased = sums / ((n - lags) * x.shape[0])
    acf = unbiased / unbiased[0]
    return acf[: max_lag + 1]


def first_accepted_peak(
    acf: np.ndarray, *, guard: int, min_interval_px: int, prominence: float
) -> tuple[int, dict, list[dict]]:
    """First strict local maximum accepted by the explicit peak thresholds."""
    peaks, props = find_peaks(
        acf[guard:],
        distance=min_interval_px,
        prominence=prominence,
    )
    all_peaks: list[dict] = []
    prominences = props["prominences"]
    for j, p in enumerate(peaks):
        lag = int(p + guard)
        all_peaks.append(
            {
                "lag_px": lag,
                "acf": float(acf[lag]),
                "prominence": float(prominences[j]),
            }
        )
    if not all_peaks:
        raise RuntimeError("no accepted local maximum found")
    first = all_peaks[0]
    return first["lag_px"], first, all_peaks


def subpixel_quadratic(acf: np.ndarray, i: int) -> float | None:
    y0, y1, y2 = acf[i - 1], acf[i], acf[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return None
    delta = 0.5 * (y0 - y2) / denom
    if abs(delta) > 1:
        return None
    return i + float(delta)


def make_plot(
    path_png: Path,
    path_svg: Path,
    angle: dict,
    acf: np.ndarray,
    predicted: float,
    measured: int,
) -> None:
    lags = np.arange(acf.size)
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(lags, acf, color=BLUE, linewidth=2.0, label="directional ACF")
    ax.axvline(
        predicted,
        color=ORANGE,
        linestyle="--",
        linewidth=1.8,
        label=f"predicted {predicted:.2f} px",
    )
    ax.scatter(
        [measured],
        [acf[measured]],
        s=82,
        color=INK,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        label=f"first local maximum {measured} px",
    )

    ax.set_xlim(0, acf.size - 1)
    ax.set_ylim(-0.12, 1.08)
    ax.set_xlabel("horizontal lag (pixels)", color=INK)
    ax.set_ylabel("normalized autocorrelation", color=INK)
    ax.set_title(
        f"Rotation θ = {angle['degrees']:.5f}°  (primitive lattice vector "
        f"({angle['a']},{angle['b']}), Cθ = √({angle['a']}²+{angle['b']}²) = "
        f"{angle['closure']:.6f})",
        color=INK,
        pad=12,
    )
    ax.grid(True, color=GRID, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=SECONDARY)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)

    # Narrow inset: integer peak versus non-integer geometric prediction.
    left = max(0, int(round(predicted)) - 12)
    right = min(acf.size - 1, int(round(predicted)) + 12)
    inset = ax.inset_axes([0.13, 0.50, 0.31, 0.34])
    inset.set_facecolor(SURFACE)
    inset.plot(lags[left : right + 1], acf[left : right + 1], color=BLUE, lw=1.7, marker="o", ms=3)
    inset.axvline(predicted, color=ORANGE, ls="--", lw=1.4)
    inset.scatter([measured], [acf[measured]], s=36, color=INK, edgecolor="white", linewidth=0.8, zorder=5)
    inset.set_xlim(left, right)
    pad = max(0.02, 0.12 * float(np.max(acf[left : right + 1])))
    inset.set_ylim(float(np.min(acf[left : right + 1])) - pad, 1.05)
    inset.tick_params(labelsize=7, colors=SECONDARY)
    inset.grid(True, color=GRID, linewidth=0.6)
    for spine in ("top", "right"):
        inset.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(path_png, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(path_svg, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frequency", type=float, default=0.5)
    parser.add_argument("--octaves", type=int, default=1)
    parser.add_argument("--max-lag", type=int, default=1300)
    parser.add_argument("--guard", type=int, default=4)
    parser.add_argument("--min-interval-px", type=int, default=20)
    parser.add_argument("--prominence", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = ROOT / "artifacts" / "autocorr"
    raw_dir = out_dir / "raw"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    perm_period = 256
    h = args.frequency
    fs = 1.0 / h  # samples per noise-lattice unit
    angles = [
        {
            "label": "theta45",
            "degrees": 45.0,
            "cli_degrees": "45",
            "a": 1,
            "b": 1,
        },
        {
            "label": "theta26p565",
            "degrees": float(np.degrees(np.arctan(0.5))),
            "cli_degrees": repr(float(np.degrees(np.arctan(0.5)))),
            "a": 2,
            "b": 1,
        },
    ]
    for spec in angles:
        spec["closure"] = float(np.hypot(spec["a"], spec["b"]))
        spec["predicted_lag_px"] = perm_period * spec["closure"] / h
        spec["reciprocal_product_cycles_per_px"] = 1.0 / (
            perm_period * spec["closure"] * fs
        )
        # Requested reciprocal-of-a-product form for the pixel lag itself:
        # reciprocal permutation period times effective sampling step h/C.
        spec["pixel_lag_reciprocal_product"] = 1.0 / (
            (1.0 / perm_period) * (h / spec["closure"])
        )

    criteria = {
        "definition": "strict local maximum R[i] > R[i-1] and R[i] >= R[i+1]",
        "guard_lags_lt": args.guard,
        "minimum_peak_interval_px": args.min_interval_px,
        "minimum_prominence": args.prominence,
        "measurement": "integer lag; no subpixel fit used for pass/fail",
        "deviation_tolerance_px": 2.0,
    }
    prediction = {
        "seed": args.seed,
        "size": args.size,
        "octaves": args.octaves,
        "permutation_period_lattice_units": perm_period,
        "sampling_step_lattice_units_per_pixel": h,
        "sampling_frequency_samples_per_lattice_unit": fs,
        "direction": "row-averaged horizontal autocorrelation after coordinate rotation",
        "criteria": criteria,
        "angles": angles,
    }
    prediction_path = out_dir / "prediction.json"
    prediction_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2) + "\n")

    angle_arg = ",".join(f"{s['label']}:{s['cli_degrees']}" for s in angles)
    cmd = [
        "go",
        "run",
        "./analysis/autocorr",
        f"--out-dir={raw_dir}",
        f"--size={args.size}",
        f"--seed={args.seed}",
        f"--frequency={h}",
        f"--octaves={args.octaves}",
        f"--angles={angle_arg}",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)

    rows = []
    result_angles = []
    for spec in angles:
        raw_path = raw_dir / f"{spec['label']}.f64"
        z = np.fromfile(raw_path, dtype="<f8").reshape(args.size, args.size)
        acf = directional_acf(z, args.max_lag)
        np.save(out_dir / f"{spec['label']}_acf.npy", acf)
        with (out_dir / f"{spec['label']}_acf.csv").open(
            "w", newline="", encoding="utf-8"
        ) as curve_file:
            writer = csv.writer(curve_file)
            writer.writerow(["lag_px", "normalized_acf"])
            for lag, value in enumerate(acf):
                writer.writerow([lag, f"{value:.12g}"])

        measured, first_info, all_peaks = first_accepted_peak(
            acf,
            guard=args.guard,
            min_interval_px=args.min_interval_px,
            prominence=args.prominence,
        )
        subpixel = subpixel_quadratic(acf, measured)
        predicted = spec["predicted_lag_px"]
        error = abs(measured - predicted)
        row = {
            "angle_label": spec["label"],
            "angle_degrees": round(spec["degrees"], 6),
            "primitive_vector": f"({spec['a']},{spec['b']})",
            "closure_Ctheta": round(spec["closure"], 6),
            "predicted_lag_px": round(predicted, 6),
            "reciprocal_product_cycles_per_px": round(
                spec["reciprocal_product_cycles_per_px"], 9
            ),
            "measured_first_peak_px": measured,
            "subpixel_estimate_px": None if subpixel is None else round(subpixel, 4),
            "absolute_error_px": round(error, 6),
            "within_2_px": error <= 2.0,
            "acf_at_peak": round(first_info["acf"], 6),
            "prominence": round(first_info["prominence"], 6),
        }
        rows.append(row)
        result_angles.append(
            {**spec, "measurement": row, "accepted_peaks": all_peaks}
        )
        make_plot(
            fig_dir / f"{spec['label']}_acf.png",
            fig_dir / f"{spec['label']}_acf.svg",
            spec,
            acf,
            predicted,
            measured,
        )

    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {"criteria": criteria, "rows": rows, "angles": result_angles}
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
