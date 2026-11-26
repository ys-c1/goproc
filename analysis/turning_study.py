#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contour turning-point scaling study for the goproc terrain.

Definitions (pre-registered, documented in artifacts/turning_report.md):
  * Fine curvature-extrema count K: arc-length resample pitch 1 fine px,
    Gaussian sigma 1.5 px, signed curvature extrema with angular prominence
    >= 3 deg (sensitivity at 1/8 deg also reported).
  * Turning point T: contour simplified by Douglas-Peucker at tolerance
    0.5 px of the image being measured; a retained vertex whose deflection
    between incident segments exceeds 30 deg.  A windowed-chord detector is
    reported as sensitivity check.
  * Scaling m: m x m block mean (box anti-alias) then marching squares on
    the thumbnail; scaled sample step = m fine px = 1 thumbnail px.

Outputs into artifacts/: metrics.json, thumbnail_s{2,4}.png and overlays.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.measure import approximate_polygon, find_contours

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/ys/feedback/uidreczz28eFwoz4iUa"
ART = os.path.join(ROOT, "artifacts")
PGM = os.path.join(ART, "terrain_s7_o5_1024.pgm")

LEVELS = np.round(np.arange(0.20, 0.81, 0.10), 2)
SYNTH_LEVELS = (0.30, 0.40, 0.50, 0.60, 0.70)
SCALES = (2, 4)
SWEEP = (1, 2, 3, 4, 8)
ANGLE_THR = 30.0


# ---------------------------------------------------------------- pixels
def read_pgm(path):
    raw = open(path, "rb").read()
    # Locate the 4 header tokens, tolerating the pre-fix goproc header whose
    # separators were the literal two characters '\' 'n' instead of newlines.
    pos = 0
    toks = []
    while len(toks) < 4:
        while pos < len(raw) and raw[pos:pos + 1].isspace():
            pos += 1
        if raw[pos:pos + 1] == b"#":
            while raw[pos:pos + 1] not in (b"\n", b"\r"):
                pos += 1
            continue
        if raw[pos:pos + 2] == b"\\n":  # literal backslash-n separator
            pos += 2
            continue
        start = pos
        while pos < len(raw):
            if raw[pos:pos + 1].isspace() or raw[pos:pos + 2] == b"\\n":
                break
            pos += 1
        toks.append(raw[start:pos])
        if raw[pos:pos + 2] == b"\\n":
            pos += 2
    # payload starts after the single whitespace following maxval token
    while pos < len(raw) and (raw[pos:pos + 1].isspace() or raw[pos:pos + 2] == b"\\n"):
        pos += 1
    w, h = int(toks[1]), int(toks[2])
    img = np.frombuffer(raw[pos:], dtype=np.uint8).reshape(h, w)
    return img.astype(float) / float(toks[3])


def downscale_blockmean(h, m):
    n = h.shape[0] // m * m
    a = h[:n, :n]
    return a.reshape(n // m, m, n // m, m).mean(axis=(1, 3))


# ------------------------------------------------------------- geometry
def _closed(c):
    return len(c) > 2 and np.linalg.norm(c[0] - c[-1]) < 2.0


def resample_arc(c, step=1.0, closed=None):
    p = np.asarray(c, float)
    if closed is None:
        closed = _closed(p)
    if closed and np.linalg.norm(p[0] - p[-1]) > 1e-9:
        p = np.vstack([p, p[0]])
    seg = np.diff(p, axis=0)
    sl = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0], np.cumsum(sl)])
    L = s[-1]
    if L < step * 6:
        return None, closed
    if closed:
        n = max(6, int(round(L / step)))
        tt = np.arange(n) * (L / n)
        out = np.column_stack(
            [np.interp(tt, s, p[:, j], period=L) for j in range(2)]
        )
        return out, True
    tt = np.arange(0, L - step * 0.5, step)
    if len(tt) < 6:
        return None, False
    out = np.column_stack([np.interp(tt, s, p[:, j]) for j in range(2)])
    return out, False


def curvature_extrema(c, pitch=1.0, sigma=1.5, prom_deg=3.0, points=False):
    rp, closed = resample_arc(c, pitch)
    if rp is None:
        return (0, np.zeros((0, 2))) if points else 0
    mode = "wrap" if closed else "nearest"
    x = gaussian_filter1d(rp[:, 1], sigma, mode=mode)
    y = gaussian_filter1d(rp[:, 0], sigma, mode=mode)
    dx = np.gradient(x, pitch)
    dy = np.gradient(y, pitch)
    ddx = np.gradient(dx, pitch)
    ddy = np.gradient(dy, pitch)
    kappa = (dx * ddy - dy * ddx) / (dx * dx + dy * dy + 1e-12) ** 1.5
    turn_deg = np.degrees(kappa * pitch)
    pk, _ = find_peaks(turn_deg, prominence=prom_deg)
    tr, _ = find_peaks(-turn_deg, prominence=prom_deg)
    idx = np.concatenate([pk, tr]).astype(int)
    if points:
        return len(idx), rp[idx]
    return len(idx)


def rdp_turns(c, eps=0.5, thr=ANGLE_THR, points=False):
    p = np.asarray(c, float)
    closed = _closed(p)
    q = approximate_polygon(p, tolerance=eps)
    if closed and len(q) > 1 and np.linalg.norm(q[0] - q[-1]) < 1e-6:
        q = q[:-1]
    n = len(q)
    if n < 3:
        return (0, np.zeros((0, 2))) if points else 0
    if closed:
        prev = q[(np.arange(n) - 1) % n]
        nxt = q[(np.arange(n) + 1) % n]
    else:
        prev = np.vstack([q[0] - (q[1] - q[0]), q[:-1]])
        nxt = np.vstack([q[1:], q[-1] + (q[-1] - q[-2])])
    v1, v2 = q - prev, nxt - q
    a1 = np.arctan2(v1[:, 0], v1[:, 1])
    a2 = np.arctan2(v2[:, 0], v2[:, 1])
    ang = np.degrees(np.abs(np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1))))
    if not closed:
        ang[0] = ang[-1] = 0.0
    sel = ang > thr
    if points:
        return int(sel.sum()), q[sel]
    return int(sel.sum())


def chord_turns(c, sigma=1.0, w=1, thr=ANGLE_THR):
    rp, closed = resample_arc(c, 1.0)
    if rp is None:
        return 0
    mode = "wrap" if closed else "nearest"
    x = gaussian_filter1d(rp[:, 1], sigma, mode=mode)
    y = gaussian_filter1d(rp[:, 0], sigma, mode=mode)
    n = len(x)
    ii = np.arange(n)
    if closed:
        im, ip = (ii - w) % n, (ii + w) % n
    else:
        im, ip = np.clip(ii - w, 0, n - 1), np.clip(ii + w, 0, n - 1)
    a1 = np.arctan2(y - y[im], x - x[im])
    a2 = np.arctan2(y[ip] - y, x[ip] - x)
    d = np.degrees(np.abs(np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1))))
    if not closed:
        d[: w + 1] = d[-w - 1 :] = 0
    # greedy NMS, minimum separation w samples
    cand = np.where(d > thr)[0]
    kept = []
    for i in cand[np.argsort(-d[cand])]:
        ok = True
        for j in kept:
            dist = abs(int(i) - j)
            if closed:
                dist = min(dist, n - dist)
            if dist < max(1, w):
                ok = False
                break
        if ok:
            kept.append(int(i))
    return len(kept)


def count_all(img, levels, fn, min_raw=8):
    total = 0
    per_level = []
    for lv in levels:
        t = 0
        for c in find_contours(img, lv):
            if len(c) < min_raw:
                continue
            t += fn(c)
        per_level.append(t)
        total += t
    return total, per_level


# ------------------------------------------------------- synthetic C1
def triangle_phase(N, d, slope, x0):
    xs = np.arange(N)
    u = (xs - x0) / d
    j = np.floor(u).astype(int)
    t = u - j
    vj = slope * d / 2 * np.where(j % 2 == 0, 1.0, -1.0)
    return vj + (-vj - vj) * t


def synth_terrain(s, N=1024, slope_c=1.2):
    d = s
    f = triangle_phase(N, d, slope=slope_c * s, x0=(s - 1) / 2.0)
    z = np.arange(N)[:, None] - f[None, :]
    return (z - z.min()) / (z.max() - z.min())


# ------------------------------------------------------------- figures
def save_thumbnail_png(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def overlay_panel(img, levels, title, path, detector="rdp", crop=None, dots=True):
    """Two-panel archive: full map+contours and zoom with turn markers."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
    turns_all = []
    for ax, do_crop in ((axes[0], False), (axes[1], True)):
        ax.imshow(img, cmap="gray", origin="upper")
        nturn = 0
        for lv in levels:
            for c in find_contours(img, lv):
                if len(c) < 4:
                    continue
                ax.plot(c[:, 1], c[:, 0], lw=0.7, color="#2b7bff", alpha=0.9)
                if detector == "rdp":
                    t, pts = rdp_turns(c, points=True)
                else:
                    t = chord_turns(c)
                    pts = None
                nturn += t
                if dots and pts is not None and len(pts):
                    ms = 4.0 if do_crop else 1.4
                    ax.plot(pts[:, 1], pts[:, 0], "r.", ms=ms, mew=0,
                            alpha=0.8 if do_crop else 0.6)
        if do_crop and crop is not None:
            r0, r1, c0, c1 = crop
            ax.set_xlim(c0, c1)
            ax.set_ylim(r1, r0)
            ax.set_title("zoom with >30° turn points", fontsize=11)
        else:
            ax.set_title("full thumbnail + contours + turns", fontsize=11)
        ax.set_aspect("equal")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fine_extrema_figure(h, Kpts, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
    axes[0].imshow(h, cmap="gray")
    for lv in LEVELS:
        for c in find_contours(h, lv):
            axes[0].plot(c[:, 1], c[:, 0], lw=0.35, color="#2b7bff", alpha=0.7)
    axes[0].set_title("original 1024 px heightmap + 7 contours")
    axes[0].set_aspect("equal")
    axes[1].imshow(h, cmap="gray")
    for lv in (0.4, 0.5, 0.6):
        for c in find_contours(h, lv):
            rp, closed = resample_arc(c)
            if rp is None:
                continue
            k, pts = curvature_extrema(c, points=True)
            if len(pts):
                axes[1].plot(pts[:, 1], pts[:, 0], "r.", ms=4)
            axes[1].plot(c[:, 1], c[:, 0], lw=0.8, color="#2b7bff")
    axes[1].set_xlim(120, 360)
    axes[1].set_ylim(360, 120)
    axes[1].set_title("zoom: curvature extrema (red), prominence>=3°")
    axes[1].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def synth_figure(s, z, coarse, path):
    lvls = list(SYNTH_LEVELS)
    # crop in coarse pixels
    half = 46
    cx = coarse.shape[1] // 2
    crop = (coarse.shape[0] // 2 - half, coarse.shape[0] // 2 + half,
            cx - half, cx + half)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
    axes[0].imshow(coarse, cmap="gray", origin="upper")
    axes[0].set_title(f"s={s} thumbnail (N/s={coarse.shape[0]}) + zigzag contours")
    for c_ in [c for lv in lvls for c in find_contours(coarse, lv)]:
        axes[0].plot(c_[:, 1], c_[:, 0], lw=0.7, color="#00e08a")
    r0, r1, c0, c1 = crop
    axes[0].set_xlim(c0, c1)
    axes[0].set_ylim(r1, r0)
    axes[1].imshow(coarse, cmap="gray", origin="upper")
    n = 0
    for lv in lvls:
        for c_ in find_contours(coarse, lv):
            t, pts = rdp_turns(c_, eps=0.5, points=True)
            n += t
            axes[1].plot(c_[:, 1], c_[:, 0], lw=1.0, color="#00e08a")
            if len(pts):
                axes[1].plot(pts[:, 1], pts[:, 0], "r.", ms=9, mew=0.8,
                            mfc="none", mec="red")
    axes[1].set_xlim(c0, c1)
    axes[1].set_ylim(r1, r0)
    axes[1].set_title(f"zoom: one extremum / sample step, recovered turns")
    fig.suptitle(f"Equal-spacing construction at scale m={s} "
                 f"(spacing d={s} fine px = 1 thumbnail px)", fontsize=13)
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    pgm = sys.argv[1] if len(sys.argv) > 1 else PGM
    h = read_pgm(pgm)
    print("loaded", pgm, h.shape)
    metrics = {"pgm": pgm, "shape": list(h.shape), "levels": list(map(float, LEVELS))}

    # ---- K on fine terrain, sensitivity over prominence
    K_main, k_levels = count_all(h, LEVELS, lambda c: curvature_extrema(c))
    metrics["K_prom3"] = K_main
    metrics["K_per_level_prom3"] = k_levels
    metrics["K_sensitivity"] = {}
    for prom in (1.0, 3.0, 8.0):
        k, _ = count_all(h, LEVELS, lambda c, p=prom: curvature_extrema(c, prom_deg=p))
        metrics["K_sensitivity"][f"prom_{prom:g}deg"] = k
    print("K", metrics["K_sensitivity"])

    # fine extrema figure (collect points from 3 middle levels)
    pts_all = []
    for lv in (0.4, 0.5, 0.6):
        for c in find_contours(h, lv):
            if len(c) >= 8:
                _, pts = curvature_extrema(c, points=True)
                if len(pts):
                    pts_all.append(pts)
    Kpts = np.vstack(pts_all) if pts_all else np.zeros((0, 2))
    fine_extrema_figure(h, Kpts, os.path.join(ART, "overlay_real_fine.png"))

    # ---- T across scales
    metrics["real"] = {}
    for m in SWEEP:
        img = h if m == 1 else downscale_blockmean(h, m)
        Trdp, rdp_levels = count_all(img, LEVELS, lambda c: rdp_turns(c, eps=0.5))
        Trdp1, _ = count_all(img, LEVELS, lambda c: rdp_turns(c, eps=1.0))
        Tchord, _ = count_all(img, LEVELS, lambda c: chord_turns(c))
        b2 = K_main / m**2
        metrics["real"][str(m)] = {
            "T_rdp_eps0.5": Trdp,
            "T_rdp_eps1.0": Trdp1,
            "T_chord": Tchord,
            "rdp_per_level": rdp_levels,
            "ratio_rdp_over_K": Trdp / K_main,
            "ratio_rdp_over_K_over_m2": Trdp / b2,
            "ratio_rdp_over_K_over_m": Trdp / (K_main / m),
            "ratio_chord_over_K_over_m2": Tchord / b2,
        }
        print(f"m={m}: RDP.5={Trdp} RDP1={Trdp1} chord={Tchord} "
              f"T/(K/m2) rdp={Trdp/b2:.3f} chord={Tchord/b2:.3f}")

    # ---- archives for the two primary groups
    crops = {2: (120, 320, 120, 320), 4: (30, 80, 30, 80)}
    for m in SCALES:
        img = downscale_blockmean(h, m)
        save_thumbnail_png(img, os.path.join(ART, f"thumbnail_s{m}.png"))
        overlay_panel(
            img, LEVELS,
            f"goproc seed=7 octaves=5, m={m}x block-mean thumbnail "
            f"({img.shape[0]}x{img.shape[1]})",
            os.path.join(ART, f"overlay_real_s{m}.png"),
            crop=crops[m],
        )

    # ---- prescribed equal-spacing construction C1
    metrics["construction_equal_spaced"] = {}
    for m in SCALES:
        z = synth_terrain(m)
        coarse = downscale_blockmean(z, m)
        Kd, Td = 0, 0
        for lv in SYNTH_LEVELS:
            for c in find_contours(z, lv):
                if len(c) >= 8:
                    Kd += curvature_extrema(c)
            for c in find_contours(coarse, lv):
                if len(c) >= 4:
                    Td += rdp_turns(c, eps=0.5)
        corners_per_line = z.shape[0] // m
        metrics["construction_equal_spaced"][str(m)] = {
            "K_measured": Kd,
            "T_measured": Td,
            "T_over_K": Td / Kd,
            "T_over_K_over_m2": Td / (Kd / m**2),
            "designed_corners_per_line_approx": corners_per_line,
        }
        print(f"C1 m={m}: K={Kd} T={Td} T/K={Td/Kd:.4f} "
              f"T/(K/m2)={Td/(Kd/m**2):.3f}")
        synth_figure(m, z, coarse,
                     os.path.join(ART, f"overlay_construction_s{m}.png"))

    with open(os.path.join(ART, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("wrote", os.path.join(ART, "metrics.json"))


if __name__ == "__main__":
    main()
