#!/usr/bin/env python3
"""Render one mirror-correlation curve per frame from axisdemo CSV output."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "symmetry"

BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def read_curves(path: Path, case_no: int = 1):
    cols, rs = [], []
    true_axis = pred_axis = actual_axis = None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["case_no"]) != case_no:
                continue
            cols.append(int(row["column"]))
            rs.append(float(row["r"]))
            true_axis = int(row["true_axis"])
            pred_axis = int(row["pred_axis"])
            actual_axis = int(row["actual_axis"])
    return cols, rs, true_axis, pred_axis, actual_axis


def render(frame: str, size_label: str) -> None:
    csv_path = ART / f"curves_{frame}.csv"
    cols, rs, true_axis, pred_axis, actual_axis = read_curves(csv_path)
    if not cols:
        raise SystemExit(f"no curve data in {csv_path}")

    fig, ax = plt.subplots(figsize=(8.2, 5.5), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(cols, rs, color=BLUE, linewidth=2.2, label=r"mirror correlation $\rho(c)$")
    ax.fill_between(cols, rs, min(rs) - 0.03, color=BLUE, alpha=0.10, linewidth=0)
    ax.axvline(
        pred_axis,
        color=ORANGE,
        linewidth=2.0,
        linestyle="--",
        label=f"phase-sum prediction c={pred_axis}",
    )
    peak_r = rs[cols.index(actual_axis)]
    ax.scatter(
        [actual_axis],
        [peak_r],
        s=72,
        color=INK,
        edgecolor="white",
        linewidth=1.4,
        zorder=5,
        label=f"column-scan maximum c={actual_axis}",
    )
    ax.annotate(
        f"peak ({actual_axis}, {peak_r:.6f})",
        xy=(actual_axis, peak_r),
        xytext=(12, -22),
        textcoords="offset points",
        fontsize=9.5,
        color=SECONDARY,
        arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": 0.9},
    )

    ax.set_title(f"Frame {frame}: {size_label}, case 01", color=INK, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("candidate symmetry column c (zero-based; edge self-pairs excluded)", color=INK, fontsize=10.5)
    ax.set_ylabel(r"Pearson mirror correlation $\rho(c)$", color=INK, fontsize=10.5)
    ax.set_xlim(min(cols), max(cols))
    upper = min(1.012, max(1.002, peak_r + 0.004))
    ax.set_ylim(min(rs) - 0.04, upper)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=SECONDARY, labelsize=9.5)

    handles, labels = ax.get_legend_handles_labels()
    # Keep the marker plus phase line and curve identity explicit.
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=1,
        frameon=False,
        fontsize=9.5,
    )

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for suffix in ("png", "svg"):
        out = ART / f"correlation_{frame}.{suffix}"
        fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    render("A", "128×128")
    render("B", "96×96")
    print(f"wrote {ART / 'correlation_A.png'}")
    print(f"wrote {ART / 'correlation_B.png'}")


if __name__ == "__main__":
    main()
