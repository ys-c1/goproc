#!/usr/bin/env python3
"""One-shot ponding criterion versus round-by-round D8 accumulation.

The script regenerates the deterministic heightmap through the Go API/CLI,
computes D8 slopes and contributing-cell counts, compares a direct predicate
with a synchronous iterative accumulation, writes cell/coordinate tables and
renders one annotated PNG per slope threshold.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ponding"
PGM = OUT / "height_seed7.pgm"

SIZE = 128
SEED = 7
OCTAVES = 4
F_INFILTRATION = 16
# Dimensionless D8 slope thresholds (vertical grid units per horizontal cell).
GROUPS = [
    {"tag": "tau_0p005", "label": "阈值组 A：坡度阈值 0.005", "tau": 0.005},
    {"tag": "tau_0p010", "label": "阈值组 B：坡度阈值 0.010", "tau": 0.010},
]

# Okabe-Ito/reference-dataviz-derived annotation colors.  Mismatch also gets an
# X marker, so color is not the only encoding.
WATER = (42, 120, 214, 185)
BOUNDARY = (235, 104, 52, 255)
MISMATCH = (208, 59, 59, 255)
INK = (11, 11, 11, 255)
MUTED = (82, 81, 78, 255)
PANEL = (252, 252, 251, 255)
PANEL_FILL = (241, 240, 235, 255)
GRID = (225, 224, 217, 255)

D8 = (
    (-1, -1, 2.0**0.5),
    (-1, 0, 1.0),
    (-1, 1, 2.0**0.5),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, 2.0**0.5),
    (1, 0, 1.0),
    (1, 1, 2.0**0.5),
)
D4 = tuple(d for d in D8 if d[2] == 1.0)


def load_font(size: int):
    candidates = [
        "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in candidates:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def regenerate_heightmap() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "go", "run", "./cmd/goproc", "gen",
            "--size", str(SIZE),
            "--seed", str(SEED),
            "--octaves", str(OCTAVES),
            "--out", str(PGM.relative_to(ROOT)),
        ],
        cwd=ROOT,
        check=True,
    )


def read_pgm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    match = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", raw)
    if match is None:
        raise ValueError(f"not a binary P5 PGM: {path}")
    width, height, max_value = map(int, match.groups())
    if max_value != 255:
        raise ValueError(f"only 8-bit PGM supported, got max={max_value}")
    start = match.end()
    pixels = raw[start : start + width * height]
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width).astype(float) / 255.0


def build_d8(z: np.ndarray):
    height, width = z.shape
    slope = np.zeros_like(z)
    receiver = np.full((height, width), -1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            best_slope = 0.0
            best_receiver = -1
            for dy, dx, distance in D8:
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width:
                    downhill = (z[y, x] - z[yy, xx]) / distance
                    if downhill > best_slope:
                        best_slope = downhill
                        best_receiver = yy * width + xx
            slope[y, x] = best_slope
            receiver[y, x] = best_receiver
    return slope, receiver


def direct_accumulation(z: np.ndarray, receiver: np.ndarray) -> np.ndarray:
    """One descending-elevation sweep; equivalent to the ancestor-set sum."""
    height, width = z.shape
    flat_receiver = receiver.ravel()
    acc = np.ones(height * width, dtype=np.int64)
    order = np.argsort((-z).ravel(), kind="stable")
    for cell in order:
        target = int(flat_receiver[int(cell)])
        if target >= 0:
            acc[target] += acc[int(cell)]
    return acc.reshape(height, width)


def iterative_accumulation(receiver: np.ndarray, slope: np.ndarray):
    """Synchronous Jacobi rounds A^(k+1)=1+sum child A^(k), until fixed."""
    flat_receiver = receiver.ravel()
    valid = flat_receiver >= 0
    acc = np.ones_like(flat_receiver)
    round_rows = []
    rounds = 0
    while True:
        nxt = np.ones_like(acc)
        np.add.at(nxt, flat_receiver[valid], acc[valid])
        if np.array_equal(nxt, acc):
            break
        acc = nxt
        rounds += 1
        row = {"round": rounds}
        flat_acc = acc.reshape(slope.shape)
        for group in GROUPS:
            mask = (slope < group["tau"]) & (flat_acc > F_INFILTRATION)
            row[group["tag"] + "_area"] = int(mask.sum())
        round_rows.append(row)
    return acc.reshape(slope.shape), rounds, round_rows


def boundary4(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def coord_list(mask: np.ndarray) -> list[dict[str, int]]:
    ys, xs = np.where(mask)
    return [{"x": int(x), "y": int(y)} for y, x in zip(ys, xs)]


def write_coordinate_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hillshade(z: np.ndarray, vertical_exaggeration: float = 35.0) -> np.ndarray:
    gy, gx = np.gradient(z)
    gx *= vertical_exaggeration
    gy *= vertical_exaggeration
    slope_angle = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = np.radians(315.0)
    altitude = np.radians(45.0)
    shade = (
        np.sin(altitude) * np.cos(slope_angle)
        + np.cos(altitude) * np.sin(slope_angle) * np.cos(azimuth - aspect)
    )
    shade = (shade - float(shade.min())) / max(float(np.ptp(shade)), 1e-12)
    gray = np.rint(62 + 178 * shade).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def render_map(
    path: Path,
    z: np.ndarray,
    mask: np.ndarray,
    boundary: np.ndarray,
    mismatch: np.ndarray,
    group: dict[str, object],
    stats: dict[str, object],
) -> None:
    height, width = z.shape
    scale = 5
    map_w, map_h = width * scale, height * scale
    left, top = 76, 112
    right_panel_w = 300
    canvas_w = left + map_w + 28 + right_panel_w + 28
    canvas_h = top + map_h + 78
    image = Image.new("RGB", (canvas_w, canvas_h), tuple(PANEL[:3]))
    draw = ImageDraw.Draw(image, "RGBA")

    base = hillshade(z)
    rgba = np.dstack([base, np.full((height, width), 255, dtype=np.uint8)])
    rgba[mask] = WATER
    terrain = Image.fromarray(rgba, "RGBA").resize((map_w, map_h), Image.Resampling.NEAREST)
    image.paste(terrain, (left, top), terrain)

    # Exposed 4-neighbour edges only; avoids drawing shared cell borders.
    for y, x in zip(*np.where(boundary)):
        px, py = left + x * scale, top + y * scale
        if y == 0 or not mask[y - 1, x]:
            draw.line((px, py, px + scale, py), fill=BOUNDARY, width=2)
        if y == height - 1 or not mask[y + 1, x]:
            draw.line((px, py + scale, px + scale, py + scale), fill=BOUNDARY, width=2)
        if x == 0 or not mask[y, x - 1]:
            draw.line((px, py, px, py + scale), fill=BOUNDARY, width=2)
        if x == width - 1 or not mask[y, x + 1]:
            draw.line((px + scale, py, px + scale, py + scale), fill=BOUNDARY, width=2)

    # Mismatches are intentionally symbolized by an X as well as red.
    for y, x in zip(*np.where(mismatch)):
        cx = left + x * scale + scale // 2
        cy = top + y * scale + scale // 2
        radius = max(4, scale)
        draw.line((cx - radius - 1, cy - radius - 1, cx + radius + 1, cy + radius + 1), fill=(255, 255, 255, 255), width=5)
        draw.line((cx - radius - 1, cy + radius + 1, cx + radius + 1, cy - radius - 1), fill=(255, 255, 255, 255), width=5)
        draw.line((cx - radius, cy - radius, cx + radius, cy + radius), fill=MISMATCH, width=2)
        draw.line((cx - radius, cy + radius, cx + radius, cy - radius), fill=MISMATCH, width=2)

    draw.rectangle((left, top, left + map_w, top + map_h), outline=INK, width=1)

    font_title = load_font(27)
    font_h = load_font(20)
    font = load_font(17)
    font_small = load_font(14)
    draw.text((left, 28), str(group["label"]), font=font_title, fill=INK)
    draw.text(
        (left, 68),
        f"判据：s(x,y) < {group['tau']:.3f}  且  A(x,y) > F = {F_INFILTRATION}    "
        "坐标 (x,y)：x 自左向右、y 自上向下，均从 0 开始",
        font=font,
        fill=MUTED,
    )

    # Coordinate axes and ticks every 16 cells.
    tick = 16
    for x in range(0, width + 1, tick):
        xx = left + x * scale
        draw.line((xx, top + map_h, xx, top + map_h + 6), fill=INK, width=1)
        draw.text((xx, top + map_h + 10), str(x), font=font_small, fill=MUTED, anchor="ma")
    for y in range(0, height + 1, tick):
        yy = top + y * scale
        draw.line((left - 6, yy, left, yy), fill=INK, width=1)
        draw.text((left - 11, yy), str(y), font=font_small, fill=MUTED, anchor="rm")
    draw.text((left + map_w // 2, canvas_h - 28), "x / 列", font=font, fill=INK, anchor="mm")
    y_label = "y / 行"
    label_box = font.getbbox(y_label)
    label_w = label_box[2] - label_box[0] + 10
    label_h = label_box[3] - label_box[1] + 10
    label_layer = Image.new("RGBA", (label_w, label_h), (0, 0, 0, 0))
    ImageDraw.Draw(label_layer).text((5, 3), y_label, font=font, fill=INK)
    label_layer = label_layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(
        label_layer,
        (10, top + map_h // 2 - label_layer.height // 2),
        label_layer,
    )

    panel_x0 = left + map_w + 28
    panel_x1 = canvas_w - 28
    panel_y0 = top
    panel_y1 = top + 360
    draw.rounded_rectangle((panel_x0, panel_y0, panel_x1, panel_y1), radius=12, fill=PANEL_FILL, outline=(195, 194, 183, 255), width=1)
    lines = [
        ("结果分组", font_h, INK),
        (f"τ = {group['tau']:.3f}（无量纲坡度）", font, INK),
        (f"F = {F_INFILTRATION:g}（常数下渗阈值）", font, INK),
        ("", font_small, MUTED),
        (f"预判面积：{stats['predicted_area_cells']} 格", font, INK),
        (f"逐轮面积：{stats['iterative_area_cells']} 格", font, INK),
        (f"全格 XOR：{stats['mask_mismatch_count']} 格", font, INK),
        (f"4 邻域边界：{stats['boundary_pred_count']} 格", font, INK),
        (f"边界集合差异：{stats['boundary_set_mismatch_count']} 格", font, INK),
        (f"边界标签差异：{stats['boundary_label_mismatch_count']} 格", font, INK),
        (f"汇水收敛轮数：{stats['iterative_rounds']}", font, INK),
    ]
    yy = panel_y0 + 18
    for text, item_font, color in lines:
        draw.text((panel_x0 + 18, yy), text, font=item_font, fill=color)
        yy += 29 if item_font is font else 24
        if text == "":
            yy -= 8

    legend_y0 = panel_y1 + 24
    legend_y1 = legend_y0 + 154
    draw.rounded_rectangle((panel_x0, legend_y0, panel_x1, legend_y1), radius=12, fill=PANEL, outline=(195, 194, 183, 255), width=1)
    yy = legend_y0 + 16
    draw.rectangle((panel_x0 + 18, yy + 3, panel_x0 + 42, yy + 21), fill=WATER, outline=INK)
    draw.text((panel_x0 + 52, yy), "积水格（两种方法重合）", font=font_small, fill=INK)
    yy += 38
    draw.line((panel_x0 + 18, yy + 10, panel_x0 + 42, yy + 10), fill=BOUNDARY, width=4)
    draw.text((panel_x0 + 52, yy), "积水区边界", font=font_small, fill=INK)
    yy += 38
    draw.line((panel_x0 + 21, yy + 2, panel_x0 + 39, yy + 20), fill=MISMATCH, width=3)
    draw.line((panel_x0 + 21, yy + 20, panel_x0 + 39, yy + 2), fill=MISMATCH, width=3)
    draw.text((panel_x0 + 52, yy), f"不一致格（{stats['mask_mismatch_count']}）", font=font_small, fill=INK)

    image.save(path)


def main() -> None:
    regenerate_heightmap()
    z = read_pgm(PGM)
    slope, receiver = build_d8(z)
    acc_direct = direct_accumulation(z, receiver)
    acc_iter, rounds, round_rows = iterative_accumulation(receiver, slope)

    with (OUT / "iterative_area_rounds.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["round"] + [g["tag"] + "_area" for g in GROUPS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(round_rows)

    common_fields = [
        "x", "y", "height", "slope", "upstream_cells_direct", "upstream_cells_iter",
    ]
    for group in GROUPS:
        tag = group["tag"]
        common_fields += [
            f"pred_{tag}", f"iter_{tag}", f"mask_diff_{tag}",
            f"boundary_pred_{tag}", f"boundary_iter_{tag}",
            f"boundary_set_diff_{tag}", f"boundary_label_diff_{tag}",
        ]

    group_results = []
    full_rows = []
    masks: dict[str, dict[str, np.ndarray]] = {}

    for group in GROUPS:
        tag = group["tag"]
        tau = float(group["tau"])
        predicted = (slope < tau) & (acc_direct > F_INFILTRATION)
        iterative = (slope < tau) & (acc_iter > F_INFILTRATION)
        mask_diff = predicted != iterative
        boundary_pred = boundary4(predicted)
        boundary_iter = boundary4(iterative)
        boundary_set_diff = boundary_pred != boundary_iter
        boundary_union = boundary_pred | boundary_iter
        boundary_label_diff = boundary_union & mask_diff
        masks[tag] = {
            "predicted": predicted,
            "iterative": iterative,
            "mask_diff": mask_diff,
            "boundary_pred": boundary_pred,
            "boundary_iter": boundary_iter,
            "boundary_set_diff": boundary_set_diff,
            "boundary_label_diff": boundary_label_diff,
        }

        mask_mismatch_rows = [
            {"threshold_group": tag, "x": c["x"], "y": c["y"], "difference": "mask_xor"}
            for c in coord_list(mask_diff)
        ]
        write_coordinate_csv(
            OUT / f"mask_mismatches_{tag}.csv",
            mask_mismatch_rows,
            ["threshold_group", "x", "y", "difference"],
        )

        boundary_rows = []
        for c in coord_list(boundary_set_diff):
            boundary_rows.append({"threshold_group": tag, "x": c["x"], "y": c["y"], "difference": "boundary_set_xor"})
        for c in coord_list(boundary_label_diff):
            boundary_rows.append({"threshold_group": tag, "x": c["x"], "y": c["y"], "difference": "boundary_label_xor"})
        write_coordinate_csv(
            OUT / f"boundary_mismatches_{tag}.csv",
            boundary_rows,
            ["threshold_group", "x", "y", "difference"],
        )
        write_coordinate_csv(
            OUT / f"boundary_pred_{tag}.csv",
            [{"threshold_group": tag, **c} for c in coord_list(boundary_pred)],
            ["threshold_group", "x", "y"],
        )

        stats = {
            "tag": tag,
            "label": group["label"],
            "slope_threshold": tau,
            "infiltration_constant": F_INFILTRATION,
            "predicted_area_cells": int(predicted.sum()),
            "iterative_area_cells": int(iterative.sum()),
            "area_equal": int(predicted.sum()) == int(iterative.sum()),
            "mask_mismatch_count": int(mask_diff.sum()),
            "boundary_pred_count": int(boundary_pred.sum()),
            "boundary_iter_count": int(boundary_iter.sum()),
            "boundary_set_mismatch_count": int(boundary_set_diff.sum()),
            "boundary_label_mismatch_count": int(boundary_label_diff.sum()),
            "boundary_mismatch_coordinates": coord_list(boundary_set_diff | boundary_label_diff),
            "mask_mismatch_coordinates": coord_list(mask_diff),
            "map_png": f"ponding_{tag}.png",
            "boundary_coordinates_csv": f"boundary_pred_{tag}.csv",
            "boundary_mismatches_csv": f"boundary_mismatches_{tag}.csv",
        }
        group_results.append(stats)

        map_path = OUT / stats["map_png"]
        render_map(map_path, z, predicted, boundary_pred, mask_diff, group, {
            **stats,
            "iterative_rounds": rounds,
        })

    for y in range(SIZE):
        for x in range(SIZE):
            row = {
                "x": x,
                "y": y,
                "height": f"{z[y, x]:.17g}",
                "slope": f"{slope[y, x]:.17g}",
                "upstream_cells_direct": int(acc_direct[y, x]),
                "upstream_cells_iter": int(acc_iter[y, x]),
            }
            for group in GROUPS:
                tag = group["tag"]
                arrays = masks[tag]
                row.update({
                    f"pred_{tag}": int(arrays["predicted"][y, x]),
                    f"iter_{tag}": int(arrays["iterative"][y, x]),
                    f"mask_diff_{tag}": int(arrays["mask_diff"][y, x]),
                    f"boundary_pred_{tag}": int(arrays["boundary_pred"][y, x]),
                    f"boundary_iter_{tag}": int(arrays["boundary_iter"][y, x]),
                    f"boundary_set_diff_{tag}": int(arrays["boundary_set_diff"][y, x]),
                    f"boundary_label_diff_{tag}": int(arrays["boundary_label_diff"][y, x]),
                })
            full_rows.append(row)

    with (OUT / "cell_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=common_fields)
        writer.writeheader()
        writer.writerows(full_rows)

    summary = {
        "coordinate_convention": "x=column from left, y=row from top, zero-based",
        "parameters": {
            "size": SIZE,
            "seed": SEED,
            "octaves": OCTAVES,
            "rainfall_runoff_per_cell_units": 1,
            "infiltration_F_units": F_INFILTRATION,
            "slope_definition": "maximum positive D8 downhill gradient, orthogonal cell length 1",
            "boundary_definition": "ponding cell with a 4-neighbor outside mask or on grid edge",
            "domain_boundary": "closed D8 domain; no receiver outside grid",
        },
        "iterative_convergence_rounds": rounds,
        "direct_and_iterative_accumulation_equal": bool(np.array_equal(acc_direct, acc_iter)),
        "groups": group_results,
    }
    group_fields = [
        "tag", "slope_threshold", "infiltration_constant",
        "predicted_area_cells", "iterative_area_cells", "mask_mismatch_count",
        "boundary_pred_count", "boundary_iter_count",
        "boundary_set_mismatch_count", "boundary_label_mismatch_count",
        "boundary_mismatch_coordinates", "mask_mismatch_coordinates",
        "map_png", "boundary_mismatches_csv",
    ]
    with (OUT / "groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_fields)
        writer.writeheader()
        for group in group_results:
            row = {key: group[key] for key in group_fields if key in group}
            row["boundary_mismatch_coordinates"] = json.dumps(row["boundary_mismatch_coordinates"], ensure_ascii=False)
            row["mask_mismatch_coordinates"] = json.dumps(row["mask_mismatch_coordinates"], ensure_ascii=False)
            writer.writerow(row)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
