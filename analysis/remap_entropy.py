#!/usr/bin/env python3
"""Entropy remapping analysis for the deterministic goproc heightmaps.

The mathematical conclusion is deliberately not hidden by the code:

* minimizing fixed-bin Shannon entropy over all monotone endpoint-fixed maps
  gives a degenerate two-level map, not a finite power of the empirical CDF;
* maximizing entropy gives the CDF/histogram-equalization map;
* an identity-at-gamma=1 CDF-power family is the pullback
  T_gamma = Q(F(x)^gamma).  Its finite max-entropy exponent is reported as
  the optional scalar power law.

Run from the repository root:  python3 analysis/remap_entropy.py
"""

from __future__ import annotations

import csv
import json
import math
import struct
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATA = ART / "data"
OCTAVES = [1, 4, 12]
SIZE = 128
SEED = 1
FREQUENCY = 0.02
BINS = 32
QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK = "#0b0b0b"
MUTED = "#52514e"


def shannon(values: np.ndarray, bins: int = BINS) -> float:
    counts = np.histogram(values, bins=bins, range=(0.0, 1.0))[0]
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks in [0, n-1], preserving equal values."""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x), dtype=float)
    j = 0
    while j < len(x):
        k = j + 1
        while k < len(x) and sx[k] == sx[j]:
            k += 1
        avg = 0.5 * (j + k - 1)
        ranks[order[j:k]] = avg
        j = k
    return ranks


def load_height(octave: int) -> np.ndarray:
    DATA.mkdir(parents=True, exist_ok=True)
    raw = DATA / f"height_o{octave}.bin"
    subprocess.run(
        [
            "go", "run", "./cmd/dumpheight",
            "--size", str(SIZE),
            "--seed", str(SEED),
            "--frequency", str(FREQUENCY),
            "--octaves", str(octave),
            "--out", str(raw),
        ],
        cwd=ROOT,
        check=True,
    )
    blob = raw.read_bytes()
    got_octave, n = struct.unpack_from("<QQ", blob, 0)
    assert got_octave == octave
    values = np.array(struct.unpack_from("<" + "d" * n, blob, 16))
    assert len(values) == SIZE * SIZE
    return values


def binary_entropy(count: int, n: int) -> float:
    p = count / n
    if p == 0 or p == 1:
        return 0.0
    return float(-p * math.log(p) - (1 - p) * math.log(1 - p))


def optimize_pullback(x: np.ndarray, f: np.ndarray, qx: np.ndarray) -> dict:
    """Maximize fixed-bin entropy over T=Q(F(x)^gamma), gamma>0."""

    grid = np.arange(len(x)) / (len(x) - 1)

    def entropy_at(log_gamma: float) -> float:
        gamma = math.exp(log_gamma)
        y = np.interp(np.power(f, gamma), grid, qx)
        return shannon(y)

    coarse = np.linspace(-6.0, 6.0, 2401)
    h_coarse = np.array([entropy_at(t) for t in coarse])
    i = int(np.argmax(h_coarse))
    lo = coarse[max(0, i - 2)]
    hi = coarse[min(len(coarse) - 1, i + 2)]
    fine = np.linspace(lo, hi, 4001)
    h_fine = np.array([entropy_at(t) for t in fine])
    j = int(np.argmax(h_fine))
    log_gamma = float(fine[j])
    gamma = math.exp(log_gamma)
    y = np.interp(np.power(f, gamma), grid, qx)
    return {
        "gamma": gamma,
        "log_gamma": log_gamma,
        "entropy_nats": float(h_fine[j]),
        "values": y,
    }


def write_pgm(path: Path, x: np.ndarray, size: int = SIZE) -> None:
    gray = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
    header = f"P5\n{size} {size}\n255\n".encode()
    path.write_bytes(header + gray.tobytes())


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def bar_hist(ax: plt.Axes, x: np.ndarray, color: str, title: str, h: float) -> None:
    counts, edges = np.histogram(x, bins=BINS, range=(0, 1))
    p = counts / len(x)
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.92
    ax.bar(centers, p, width=width, color=color, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_title(f"{title}  ·  H={h:.4f} nats ({h / math.log(2):.4f} bits)", fontsize=10, color=INK)
    ax.set_xlabel("remapped height")
    ax.set_ylabel("probability")
    style_ax(ax)


def make_figures(tiers: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    # Two principal versions: linear baseline and CDF equalization.
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.0), sharex=True, sharey=True, constrained_layout=True)
    for row, tier in enumerate(tiers):
        bar_hist(axes[row, 0], tier["linear"], BLUE, f"octaves={tier['octave']} · linear stretch", tier["h_linear"])
        bar_hist(axes[row, 1], tier["equalized"], ORANGE, f"octaves={tier['octave']} · CDF equalized, γ=1", tier["h_equalized"])
    fig.suptitle("Linear heightmap vs entropy-maximizing CDF remap", fontsize=14, color=INK)
    fig.savefig(ART / "hist_before_after.png", dpi=180)
    fig.savefig(ART / "hist_before_after.svg")
    plt.close(fig)

    # Separate two versions, one file per version, with three facets.
    for key, label, color, stem in [
        ("linear", "Linear stretch (identity after normalization)", BLUE, "histograms_linear"),
        ("equalized", "CDF equalized (F^1)", ORANGE, "histograms_equalized"),
        ("power", "Identity-preserving CDF pullback power", GREEN, "histograms_power"),
        ("literal_min", "Literal entropy minimum (degenerate step)", "#e34948", "histograms_literal_min"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharex=True, sharey=True, constrained_layout=True)
        for ax, tier in zip(axes, tiers):
            gamma_text = ""
            if key == "power":
                gamma_text = f", γ*={tier['gamma_power']:.3f}"
            if key == "literal_min":
                gamma_text = ", γ=∞"
            bar_hist(ax, tier[key], color, f"K={tier['octave']}{gamma_text}", tier[f"h_{key}"])
        fig.suptitle(label, fontsize=13, color=INK)
        fig.savefig(ART / f"{stem}.png", dpi=180)
        fig.savefig(ART / f"{stem}.svg")
        plt.close(fig)

    # Item-by-item quantile comparison in small multiples to keep each panel <=3 series.
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharex=True, sharey=True, constrained_layout=True)
    qgrid = np.array(QS)
    for ax, tier in zip(axes, tiers):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, alpha=0.55, label="uniform")
        ax.plot(qgrid, tier["q_linear"], "o-", color=BLUE, lw=2, ms=5, label="linear")
        ax.plot(qgrid, tier["q_equalized"], "^-", color=ORANGE, lw=2, ms=5, label="CDF γ=1")
        ax.plot(qgrid, tier["q_power"], "s--", color=GREEN, lw=2, ms=5,
                label=f"pullback γ={tier['gamma_power']:.2f}")
        ax.set_title(f"octaves={tier['octave']}", fontsize=11, color=INK)
        ax.set_xlabel("requested quantile")
        style_ax(ax)
    axes[0].set_ylabel("height quantile")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    fig.suptitle("Quantile-by-quantile comparison (exact values in quantile_comparison.csv)", fontsize=13, color=INK)
    fig.savefig(ART / "quantile_comparison.png", dpi=180)
    fig.savefig(ART / "quantile_comparison.svg")
    plt.close(fig)


def main() -> None:
    ART.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    tiers = []
    hist_rows = []
    quant_rows = []

    for octave in OCTAVES:
        x = load_height(octave)
        n = len(x)
        ranks = average_ranks(x)
        f = ranks / (n - 1)
        qx = np.sort(x)
        grid = np.arange(n) / (n - 1)

        h_linear = shannon(x)
        equalized = f.copy()
        h_equalized = shannon(equalized)

        power = optimize_pullback(x, f, qx)
        y_power = power["values"]
        h_power = shannon(y_power)

        n_min = int(np.count_nonzero(x == x.min()))
        n_max = int(np.count_nonzero(x == x.max()))
        # Put every non-maximum pixel in bin zero and every maximum pixel at 1.
        literal = np.zeros_like(x)
        literal[x == x.max()] = 1.0
        h_literal = shannon(literal)
        h_global_lower = min(binary_entropy(n_min, n), binary_entropy(n_max, n))

        # Finite CDF-power threshold for the same fixed-bin entropy floor.
        f_second_max = (n - 2) / (n - 1)  # unique endpoint maximum in these runs
        gamma_bin_threshold = math.log(1.0 / BINS) / math.log(f_second_max)
        f_second_min = 1.0 / (n - 1)
        gamma_zero_threshold = math.log((BINS - 1) / BINS) / math.log(f_second_min)

        q_linear = np.quantile(x, QS)
        q_equalized = np.quantile(equalized, QS)
        q_power = np.quantile(y_power, QS)
        q_literal = np.quantile(literal, QS)

        tier = {
            "octave": octave,
            "n": n,
            "bins": BINS,
            "h_linear": h_linear,
            "h_equalized": h_equalized,
            "h_power": h_power,
            "h_literal_min": h_literal,
            "h_global_lower_nats": h_global_lower,
            "gamma_cdf_max_entropy": 1.0,
            "gamma_power": power["gamma"],
            "gamma_literal_min": math.inf,
            "gamma_dark_bin_threshold": gamma_bin_threshold,
            "gamma_bright_bin_threshold": gamma_zero_threshold,
            "n_min_tied": n_min,
            "n_max_tied": n_max,
            "linear": x,
            "equalized": equalized,
            "power": y_power,
            "literal_min": literal,
            "q_linear": q_linear,
            "q_equalized": q_equalized,
            "q_power": q_power,
            "q_literal": q_literal,
        }
        tiers.append(tier)

        write_pgm(ART / f"height_o{octave}_linear.pgm", x)
        write_pgm(ART / f"height_o{octave}_equalized.pgm", equalized)
        write_pgm(ART / f"height_o{octave}_power.pgm", y_power)
        write_pgm(ART / f"height_o{octave}_literal_min.pgm", literal)

        for q, a, b, c, d in zip(QS, q_linear, q_equalized, q_power, q_literal):
            quant_rows.append({
                "octaves": octave,
                "quantile": q,
                "linear": a,
                "cdf_equalized_g1": b,
                "pullback_power": c,
                "literal_min": d,
            })

        for version, values in [
            ("linear", x),
            ("cdf_equalized", equalized),
            ("pullback_power", y_power),
            ("literal_min", literal),
        ]:
            counts, edges = np.histogram(values, bins=BINS, range=(0, 1))
            for k, count in enumerate(counts):
                hist_rows.append({
                    "octaves": octave,
                    "version": version,
                    "bin": k,
                    "left": edges[k],
                    "right": edges[k + 1],
                    "count": int(count),
                    "probability": float(count / n),
                })

    summary_fields = [
        "octaves", "n", "bins", "h_linear_nats", "h_linear_bits",
        "gamma_cdf_max_entropy", "h_equalized_nats", "h_equalized_bits",
        "gamma_pullback_max_entropy", "h_power_nats", "h_power_bits",
        "gamma_literal_min_ideal", "gamma_dark_bin_threshold",
        "gamma_bright_bin_threshold", "h_literal_nats", "h_literal_bits",
        "h_global_lower_nats", "h_global_lower_bits",
    ]
    with (ART / "remap_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=summary_fields)
        w.writeheader()
        for t in tiers:
            w.writerow({
                "octaves": t["octave"],
                "n": t["n"],
                "bins": BINS,
                "h_linear_nats": t["h_linear"],
                "h_linear_bits": t["h_linear"] / math.log(2),
                "gamma_cdf_max_entropy": 1.0,
                "h_equalized_nats": t["h_equalized"],
                "h_equalized_bits": t["h_equalized"] / math.log(2),
                "gamma_pullback_max_entropy": t["gamma_power"],
                "h_power_nats": t["h_power"],
                "h_power_bits": t["h_power"] / math.log(2),
                "gamma_literal_min_ideal": "Infinity",
                "gamma_dark_bin_threshold": t["gamma_dark_bin_threshold"],
                "gamma_bright_bin_threshold": t["gamma_bright_bin_threshold"],
                "h_literal_nats": t["h_literal_min"],
                "h_literal_bits": t["h_literal_min"] / math.log(2),
                "h_global_lower_nats": t["h_global_lower_nats"],
                "h_global_lower_bits": t["h_global_lower_nats"] / math.log(2),
            })

    with (ART / "quantile_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(quant_rows[0].keys()))
        w.writeheader()
        w.writerows(quant_rows)

    with (ART / "histograms.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist_rows[0].keys()))
        w.writeheader()
        w.writerows(hist_rows)

    json_payload = {
        "config": {
            "size": SIZE, "seed": SEED, "frequency": FREQUENCY,
            "octaves": OCTAVES, "bins": BINS, "quantiles": QS,
            "entropy_unit": "nats unless bits explicitly named",
        },
        "tiers": [
            {k: v for k, v in t.items() if not isinstance(v, np.ndarray)}
            | {
                "q_linear": t["q_linear"].tolist(),
                "q_equalized": t["q_equalized"].tolist(),
                "q_power": t["q_power"].tolist(),
                "q_literal": t["q_literal"].tolist(),
            }
            for t in tiers
        ],
    }
    # Infinity is not strict JSON; replace it for the JSON artifact.
    json_payload["tiers"] = [
        {k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in tier.items()}
        for tier in json_payload["tiers"]
    ]
    (ART / "remap_metrics.json").write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    make_figures(tiers)

    for t in tiers:
        print(
            f"K={t['octave']:2d} H_lin={t['h_linear']:.9f} "
            f"gamma_pb={t['gamma_power']:.9f} H_pb={t['h_power']:.9f} "
            f"H_eq={t['h_equalized']:.9f} H_min={t['h_literal_min']:.9f} "
            f"gamma_bin>{t['gamma_dark_bin_threshold']:.3f}"
        )


if __name__ == "__main__":
    main()
