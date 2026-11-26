#!/usr/bin/env python3
"""Candidate-offset block audit for goproc.

Reads raw float64 heightmaps exported by `go run ./cmd/blockaudit`, partitions
them into 32x32 blocks, checks the theorem-derived candidate offsets, and
independently groups every block by its exact bytes as a ground-truth audit.
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "blockdup"
BLOCK = 32
G = 16  # 512 / 32
SEED = 7

SPECS = [
    {
        "key": "period4",
        "label": "I",
        "frequency": 2.0,
        "step_px_per_lattice": 0.5,
        "period_px": 128,
        "q": 4,
        "image": "marked_period4.png",
    },
    {
        "key": "period8",
        "label": "II",
        "frequency": 1.0,
        "step_px_per_lattice": 1.0,
        "period_px": 256,
        "q": 8,
        "image": "marked_period8.png",
    },
]

# Eight predeclared marker sources. They are fixed before any comparison.
PROBE_ROWS = list(range(8))
PROBE_SOURCE = [(r, 0) for r in PROBE_ROWS]
CANONICAL_DIRECTION = (1, 1)  # nearest down-right structural displacement


def font(size: int, bold: bool = True):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_height(spec: dict) -> np.ndarray:
    path = OUT / f"{spec['key']}.f64"
    a = np.fromfile(path, dtype="<f8")
    if a.size != (G * BLOCK) ** 2:
        raise RuntimeError(f"unexpected raw size {a.size} in {path}")
    return a.reshape(G * BLOCK, G * BLOCK)


def block_view(a: np.ndarray) -> np.ndarray:
    # Shape: [block_row, block_col, inner_row, inner_col].
    return a.reshape(G, BLOCK, G, BLOCK).transpose(0, 2, 1, 3)


def candidate_offsets(q: int) -> tuple[list[int], list[tuple[int, int]]]:
    m = (G - 1) // q
    components = [k * q for k in range(-m, m + 1)]
    offsets = [
        (dr, dc)
        for dr in components
        for dc in components
        if dr != 0 or dc != 0
    ]
    return components, offsets


def pixel_top_left(rc: tuple[int, int]) -> tuple[int, int]:
    r, c = rc
    return c * BLOCK, r * BLOCK


def predict_probe_coordinates(q: int):
    dr, dc = CANONICAL_DIRECTION
    predicted = []
    for i, src in enumerate(PROBE_SOURCE, start=1):
        delta = (dr * q, dc * q)
        target = (src[0] + delta[0], src[1] + delta[1])
        if not (0 <= target[0] < G and 0 <= target[1] < G):
            raise AssertionError(f"predicted target outside grid: {target}")
        predicted.append(
            {
                "i": i,
                "source_block": list(src),
                "predicted_target_block": list(target),
                "predicted_delta_block": list(delta),
                "source_pixel_xy": list(pixel_top_left(src)),
                "predicted_target_pixel_xy": list(pixel_top_left(target)),
            }
        )
    return predicted


def exact_ground_truth(blocks: np.ndarray):
    """Group all blocks by exact float64 bytes; equivalent to exhaustive lookup."""
    groups: dict[bytes, list[tuple[int, int]]] = defaultdict(list)
    for r in range(G):
        for c in range(G):
            groups[blocks[r, c].tobytes()].append((r, c))

    offset_pairs: set[tuple[int, int]] = set()
    ordered_block_pairs = 0
    for pts in groups.values():
        n = len(pts)
        ordered_block_pairs += n * n
        for i, (r1, c1) in enumerate(pts):
            for r2, c2 in pts[i + 1 :]:
                dr, dc = r2 - r1, c2 - c1
                offset_pairs.add((dr, dc))
                offset_pairs.add((-dr, -dc))
    return groups, offset_pairs, ordered_block_pairs


def candidate_scan(blocks: np.ndarray, q: int, offsets: list[tuple[int, int]]):
    rows = []
    equal_checks = 0
    exact_matches = 0
    max_abs_diff = 0.0
    valid_check_positions = 0
    for r in range(G):
        for c in range(G):
            for dr, dc in offsets:
                tr, tc = r + dr, c + dc
                if not (0 <= tr < G and 0 <= tc < G):
                    continue
                valid_check_positions += 1
                equal_checks += 1
                x = blocks[r, c]
                y = blocks[tr, tc]
                same = np.array_equal(x, y)
                d = float(np.max(np.abs(x - y)))
                max_abs_diff = max(max_abs_diff, d)
                exact_matches += int(same)
                if same:
                    rows.append((r, c, tr, tc, dr, dc, d))
    return {
        "valid_comparisons": valid_check_positions,
        "exact_matches": exact_matches,
        "max_abs_diff": max_abs_diff,
        "match_rows": rows,
    }


def actual_probe_coordinates(predicted: list[dict], match_set: set[tuple[int, int, int, int]]):
    actual = []
    for p in predicted:
        sr, sc = p["source_block"]
        pr, pc = p["predicted_target_block"]
        if (sr, sc, pr, pc) not in match_set:
            raise RuntimeError(f"canonical prediction failed for probe {p['i']}")
        found = (pr, pc)
        er = found[0] - pr
        ec = found[1] - pc
        actual.append(
            {
                **p,
                "actual_target_block": list(found),
                "delta_actual_minus_predicted_block": [er, ec],
                "position_error_chebyshev_blocks": max(abs(er), abs(ec)),
                "error_gt_one_block": max(abs(er), abs(ec)) > 1,
                "exact_float64_equal": True,
            }
        )
    return actual


def render_annotation(a: np.ndarray, spec: dict, comparisons: list[dict]):
    q = spec["q"]
    scale = 2
    h = w = G * BLOCK
    gray = np.clip(np.rint(a * 255.0 + 0.5), 0, 255).astype(np.uint8)
    terrain = Image.fromarray(gray, mode="L").convert("RGB").resize(
        (w * scale, h * scale), Image.Resampling.NEAREST
    )

    top = 58
    bottom = 214
    canvas = Image.new("RGB", (w * scale, h * scale + top + bottom), (18, 20, 26))
    canvas.paste(terrain, (0, top))
    draw = ImageDraw.Draw(canvas)
    f_title = font(22)
    f = font(18)
    f_small = font(15)

    title = (
        f"Period {spec['label']}: "
        f"δ={spec['frequency']:g} noise-px/px, s=1/δ={spec['step_px_per_lattice']:g} px/cell, "
        f"P·s={spec['period_px']}px={q} blocks"
    )
    draw.text((12, 8), title, fill=(245, 247, 250), font=f_title)
    draw.text(
        (12, 36),
        "blue=predicted source/target pair labels; yellow arrow=(+q,+q); "
        "thick gold lines=P·s period; thin gray=32-cell block grid",
        fill=(210, 216, 226),
        font=f_small,
    )

    ox, oy = 0, top

    def X(px: int) -> int:
        return ox + scale * px

    def Y(py: int) -> int:
        return oy + scale * py

    # Thin 32-cell block grid.
    for i in range(G + 1):
        p = i * BLOCK
        color = (95, 101, 112) if i % q else (255, 214, 64)
        width = 1 if i % q else 3
        draw.line([(X(p), Y(0)), (X(p), Y(h))], fill=color, width=width)
        draw.line([(X(0), Y(p)), (X(w), Y(p))], fill=color, width=width)

    for row in comparisons:
        i = row["i"]
        sr, sc = row["source_block"]
        tr, tc = row["actual_target_block"]
        x0, y0 = sc * BLOCK, sr * BLOCK
        x1, y1 = tc * BLOCK, tr * BLOCK

        def rect(x, y, color, width=4):
            draw.rectangle(
                [X(x) + 1, Y(y) + 1, X(x + BLOCK) - 2, Y(y + BLOCK) - 2],
                outline=color,
                width=width,
            )

        rect(x0, y0, (0, 176, 255), 4)
        rect(x1, y1, (255, 72, 72), 4)
        c0 = (X(x0 + BLOCK // 2), Y(y0 + BLOCK // 2))
        c1 = (X(x1 + BLOCK // 2), Y(y1 + BLOCK // 2))
        draw.line([c0, c1], fill=(255, 235, 59), width=4)
        for (cx, cy), col in [(c0, (0, 176, 255)), (c1, (255, 72, 72))]:
            r = 7
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=(255, 255, 255), width=2)

        lx, ly = X(x0) + 5, Y(y0) + 4
        text = str(i)
        tb = draw.textbbox((lx, ly), text, font=f)
        draw.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill=(8, 10, 16))
        draw.text((lx, ly), text, fill=(255, 255, 255), font=f)

    footer_y = h * scale + top + 8
    draw.text(
        (12, footer_y),
        f"Seed {SEED}, size {G*BLOCK}², octaves=4, lacunarity=2, gain=.5; "
        "all marked pairs are exact float64-equal.",
        fill=(210, 216, 226),
        font=f_small,
    )

    # Enlarged patch proof for marker #1.
    proof = comparisons[0]
    sr, sc = proof["source_block"]
    tr, tc = proof["actual_target_block"]
    patch_scale = 3
    patch_px = BLOCK * patch_scale

    def patch_image(rr: int, cc: int) -> Image.Image:
        arr = a[rr * BLOCK : (rr + 1) * BLOCK, cc * BLOCK : (cc + 1) * BLOCK]
        u8 = np.clip(np.rint(arr * 255.0 + 0.5), 0, 255).astype(np.uint8)
        return Image.fromarray(u8, mode="L").convert("RGB").resize(
            (patch_px, patch_px), Image.Resampling.NEAREST
        )

    panel_y = footer_y + 58
    source_pos = (48, panel_y)
    target_pos = (220, panel_y)
    canvas.paste(patch_image(sr, sc), source_pos)
    canvas.paste(patch_image(tr, tc), target_pos)
    draw.rectangle(
        [source_pos[0] - 3, source_pos[1] - 3, source_pos[0] + patch_px + 2, source_pos[1] + patch_px + 2],
        outline=(0, 176, 255), width=3,
    )
    draw.rectangle(
        [target_pos[0] - 3, target_pos[1] - 3, target_pos[0] + patch_px + 2, target_pos[1] + patch_px + 2],
        outline=(255, 72, 72), width=3,
    )
    draw.text(
        (source_pos[0], panel_y - 24),
        f"S #1 @ ({sr},{sc})",
        fill=(120, 210, 255), font=f,
    )
    draw.text(
        (target_pos[0], panel_y - 24),
        f"T @ ({tr},{tc})",
        fill=(255, 130, 130), font=f,
    )
    draw.text((154, panel_y + patch_px // 2 - 16), "=", fill=(245, 247, 250), font=font(42))
    draw.text(
        (350, panel_y + 18),
        "3× enlarged 32×32 patches\n"
        "np.array_equal = True\n"
        "max|source-target| = 0.0\n"
        "PGM quantization classes identical",
        fill=(235, 240, 248),
        font=f,
        spacing=8,
    )

    path = OUT / spec["image"]
    canvas.save(path)
    return path


def write_outputs(all_results):
    OUT.mkdir(parents=True, exist_ok=True)

    # Machine-readable full summary.
    summary = []
    for res in all_results:
        spec = res["spec"]
        summary.append(
            {
                key: spec[key]
                for key in ["key", "label", "frequency", "step_px_per_lattice", "period_px", "q"]
            }
            | {
                "grid_blocks_per_side": G,
                "block_size_cells": BLOCK,
                "component_values_block": res["components"],
                "candidate_offsets_including_zero_count": len(res["components"]) ** 2,
                "candidate_offsets_nonzero_count": len(res["offsets"]),
                "directed_unoriented_half_count": len(res["offsets"]) // 2,
                "candidate_valid_comparisons": res["scan"]["valid_comparisons"],
                "candidate_exact_matches": res["scan"]["exact_matches"],
                "candidate_max_abs_diff": res["scan"]["max_abs_diff"],
                "equivalence_class_count": len(res["groups"]),
                "brute_distinct_offset_count": len(res["brute_offsets"]),
                "brute_ordered_block_pair_lookups": res["ordered_pairs"],
                "brute_offset_set_equals_candidate_set": res["brute_offsets"] == set(res["offsets"]),
                "off_candidate_exact_offset_count": len(res["brute_offsets"] - set(res["offsets"])),
                "full_search_all_ordered_block_pairs": G**4,
                "candidate_nonzero_comparisons": res["scan"]["valid_comparisons"],
                "canonical_diagonal_pairs": len(res["canonical"]),
                "canonical_diagonal_exact": sum(x["exact_float64_equal"] for x in res["canonical"]),
                "ground_truth_seconds": res["brute_seconds"],
                "candidate_scan_seconds": res["candidate_seconds"],
            }
        )
    (OUT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    pred_payload = [
        {"period": r["spec"]["key"], "q": r["spec"]["q"], "predictions": r["predicted"]}
        for r in all_results
    ]
    (OUT / "predicted_coordinates.json").write_text(
        json.dumps(pred_payload, ensure_ascii=False, indent=2) + "\n"
    )
    actual_payload = [
        {"period": r["spec"]["key"], "q": r["spec"]["q"], "actual": r["actual"]}
        for r in all_results
    ]
    (OUT / "actual_coordinates.json").write_text(
        json.dumps(actual_payload, ensure_ascii=False, indent=2) + "\n"
    )

    # Long CSV: one row per probe per period.
    with (OUT / "coordinate_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "period", "q_blocks", "i", "source_r", "source_c",
                "predicted_r", "predicted_c", "actual_r", "actual_c",
                "error_dr", "error_dc", "chebyshev_error_blocks",
                "gt_one_block", "exact_float64_equal",
                "source_pixel_x", "source_pixel_y",
                "target_pixel_x", "target_pixel_y",
            ]
        )
        for res in all_results:
            key, q = res["spec"]["key"], res["spec"]["q"]
            for a in res["actual"]:
                sr, sc = a["source_block"]
                pr, pc = a["predicted_target_block"]
                ar, ac = a["actual_target_block"]
                sx, sy = a["source_pixel_xy"]
                tx, ty = a["predicted_target_pixel_xy"]
                edr, edc = a["delta_actual_minus_predicted_block"]
                w.writerow(
                    [key, q, a["i"], sr, sc, pr, pc, ar, ac, edr, edc,
                     a["position_error_chebyshev_blocks"], a["error_gt_one_block"],
                     a["exact_float64_equal"], sx, sy, tx, ty]
                )

    # Two-column coordinate arrangement.
    lines = [
        "# 两个周期的坐标两栏对照",
        "",
        "坐标为块坐标 `(row,column)`，原点左上；像素左上角为 `(x,y)=(32c,32r)`。",
        "每格写作 `源块 ⇒ 实测重复块 (Δr,Δc)`；预判目标与实测完全相同。",
        "",
        "| # | 周期 I：δ=2, s=0.5, q=4 | 周期 II：δ=1, s=1, q=8 |",
        "|---:|---|---|",
    ]
    by_key = {r["spec"]["key"]: r["actual"] for r in all_results}
    for i in range(8):
        cells = []
        for key in ["period4", "period8"]:
            a = by_key[key][i]
            sr, sc = a["source_block"]
            ar, ac = a["actual_target_block"]
            dr, dc = a["predicted_delta_block"]
            cells.append(f"`({sr},{sc}) ⇒ ({ar},{ac})`，位移 `(+{dr},+{dc})`，误差 0")
        lines.append(f"| {i+1} | {cells[0]} | {cells[1]} |")
    (OUT / "two_period_coordinates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    compare_lines = [
        "# 预判坐标 vs 实际比对坐标",
        "",
        "坐标为块坐标 `(row,column)`；误差为实测目标减预判目标。",
        "",
        "| 周期 | # | 源块 | 预判目标 | 实测目标 | 误差 `(dr,dc)` | 切比雪夫误差 | 超过一格 |",
        "|---|---:|---|---|---|---|---:|:---:|",
    ]
    for res in all_results:
        q = res["spec"]["q"]
        label = res["spec"]["key"]
        for a in res["actual"]:
            sr, sc = a["source_block"]
            pr, pc = a["predicted_target_block"]
            ar, ac = a["actual_target_block"]
            edr, edc = a["delta_actual_minus_predicted_block"]
            compare_lines.append(
                f"| {label} (q={q}) | {a['i']} | `({sr},{sc})` | `({pr},{pc})` | "
                f"`({ar},{ac})` | `({edr:+d},{edc:+d})` | "
                f"{a['position_error_chebyshev_blocks']} | {'是' if a['error_gt_one_block'] else '否'} |"
            )
    (OUT / "pred_vs_actual.md").write_text("\n".join(compare_lines) + "\n", encoding="utf-8")

    # Candidate sets in a compact machine-readable CSV.
    with (OUT / "candidate_offsets.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["period", "q", "dr_block", "dc_block"])
        for res in all_results:
            for dr, dc in res["offsets"]:
                w.writerow([res["spec"]["key"], res["spec"]["q"], dr, dc])

    # Every actual exact candidate comparison.
    with (OUT / "all_candidate_matches.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["period", "q", "source_r", "source_c", "target_r", "target_c", "dr", "dc", "max_abs_diff"])
        for res in all_results:
            q = res["spec"]["q"]
            for sr, sc, tr, tc, dr, dc, diff in res["scan"]["match_rows"]:
                w.writerow([res["spec"]["key"], q, sr, sc, tr, tc, dr, dc, f"{diff:.3g}"])

    # All valid canonical nearest down-right pairs.
    with (OUT / "canonical_comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["period", "q", "source_r", "source_c", "predicted_r", "predicted_c",
                   "actual_r", "actual_c", "chebyshev_error_blocks", "exact_float64_equal"])
        for res in all_results:
            q = res["spec"]["q"]
            for row in res["canonical"]:
                sr, sc = row["source_block"]
                pr, pc = row["predicted_target_block"]
                ar, ac = row["actual_target_block"] or ["", ""]
                w.writerow([res["spec"]["key"], q, sr, sc, pr, pc, ar, ac,
                            row["position_error_chebyshev_blocks"], row["exact_float64_equal"]])

    # Exact block equivalence classes from the independent hash grouping.
    with (OUT / "block_classes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["period", "class_id", "class_size", "representative_r", "representative_c", "block_r", "block_c"])
        for res in all_results:
            class_points = sorted(res["groups"].values(), key=lambda pts: min(pts))
            for cid, pts in enumerate(class_points):
                rep = min(pts)
                for r, c in sorted(pts):
                    w.writerow([res["spec"]["key"], cid, len(pts), rep[0], rep[1], r, c])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_results = []
    for spec in SPECS:
        a = load_height(spec)
        blocks = block_view(a)
        components, offsets = candidate_offsets(spec["q"])
        predicted = predict_probe_coordinates(spec["q"])

        t0 = time.perf_counter()
        groups, brute_offsets, ordered_pairs = exact_ground_truth(blocks)
        brute_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        scan = candidate_scan(blocks, spec["q"], offsets)
        candidate_seconds = time.perf_counter() - t0

        match_set = {(r, c, tr, tc) for r, c, tr, tc, *_ in scan["match_rows"]}
        actual = actual_probe_coordinates(predicted, match_set)

        canonical = []
        q = spec["q"]
        for r in range(G - q):
            for c in range(G - q):
                pr, pc = r + q, c + q
                ok = (r, c, pr, pc) in match_set
                canonical.append(
                    {
                        "source_block": [r, c],
                        "predicted_target_block": [pr, pc],
                        "actual_target_block": [pr, pc] if ok else None,
                        "position_error_chebyshev_blocks": 0 if ok else None,
                        "exact_float64_equal": ok,
                    }
                )

        image_path = render_annotation(a, spec, actual)

        all_results.append(
            {
                "spec": spec,
                "components": components,
                "offsets": offsets,
                "predicted": predicted,
                "actual": actual,
                "canonical": canonical,
                "groups": groups,
                "brute_offsets": brute_offsets,
                "ordered_pairs": ordered_pairs,
                "scan": scan,
                "brute_seconds": brute_seconds,
                "candidate_seconds": candidate_seconds,
                "image_path": image_path,
            }
        )

    write_outputs(all_results)

    for res in all_results:
        sp = res["spec"]
        print(
            sp["key"],
            "components=", res["components"],
            "candidates=", len(res["offsets"]),
            "comparisons=", res["scan"]["valid_comparisons"],
            "matches=", res["scan"]["exact_matches"],
            "classes=", len(res["groups"]),
            "brute_offsets=", len(res["brute_offsets"]),
            "candidate_identity=", res["brute_offsets"] == set(res["offsets"]),
            "maxdiff=", res["scan"]["max_abs_diff"],
        )


if __name__ == "__main__":
    main()
