#!/usr/bin/env python3
"""把 8 带与 16 带的逐带结果排成两栏 Markdown 表，写入 REPORT.md 片段。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "artifacts" / "tables"


def load(name: str) -> list[dict]:
    with open(T / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_pct(v: str) -> str:
    if v in ("", None):
        return "—"
    x = float(v)
    s = f"{100 * x:+.2f}%"
    return f"**{s} ⚠**" if abs(x) > 0.10 else s


def block(rows: list[dict], k: int) -> list[str]:
    head = (
        "| 带 | 高度区间 | 预判面积 ∫Λdh | 实测面积(格点) | 面积偏差 | "
        "平均等值线总长 L̄ | 预判宽度 A/L̄ | 实测宽度 | 宽度偏差 | 四邻域连通域数 | 最大域 |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    out = [head]
    for r in rows:
        interval = f"[{float(r['h_lo']):.4f},{float(r['h_hi']):.4f})"
        out.append(
            f"| {r['band']} | {interval} | {r['pred_area']} | {r['raster_area']} | "
            f"{fmt_pct(r['area_rel_err'])} | {r['mean_contour_L']} | {r['pred_width']} | "
            f"{r['meas_width']} | {fmt_pct(r['width_rel_err'])} | {r['components4']} | "
            f"{r['largest_component']} |"
        )
    return out


def main() -> None:
    r8 = load("bands_8.csv")
    r16 = load("bands_16.csv")
    lines = []
    lines.append("## 附录 B：逐带比对数据（两栏）\n")
    lines.append("面积单位为格点（每格点面积=1），宽度单位为格距；面积按四邻域连通统计连通域。\n")
    lines.append("### B.1 八带分层\n")
    lines += block(r8, 8)
    lines.append("\n### B.2 十六带分层\n")
    lines += block(r16, 16)
    lines.append("")
    out = "\n".join(lines)
    dest = T / "band_tables.md"
    dest.write_text(out, encoding="utf-8")
    print(f"wrote {dest} ({len(r8)}+{len(r16)} rows)")


if __name__ == "__main__":
    sys.exit(main())
