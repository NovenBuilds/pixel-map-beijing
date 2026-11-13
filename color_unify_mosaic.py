#!/usr/bin/env python3
"""Whole-mosaic color grade toward Wangfujing / Chang'an cool-grey daylight.

Unlike color_unify.py / color_unify_strong.py (per-tile LAB mean shifts),
this grades from a stitched overview:

  1. Soft-protect glazed yellow roofs, red walls, water, and trees.
  2. Estimate a low-frequency ambient chroma field from mid/low-chroma
     gradeable pixels across the whole mosaic, then pull that field
     toward the Wangfujing low-chroma anchor (seamless across tiles).
  3. Residual beige/cream killer on open courtyards and paving.
  4. Cool and slightly deepen already-dark shadows.
  5. Compress leftover bright neutrals.

The global field is computed at 1/32 resolution from every styled tile
(including anchors, which act as the palette reference). The same
per-pixel function is then applied tile-by-tile so peak memory stays
low. Anchor tiles are never written. Outputs go exclusively to
tiles/staging/color_J_global/. Never writes tiles/styled or
tiles/mosaic/overview_*.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/beijing-pilot")
from color_unify import rgb_to_lab, lab_to_rgb

ROOT = Path("/workspace/beijing-pilot")
STY = ROOT / "tiles" / "styled"
OUT = ROOT / "tiles" / "staging" / "color_J_global"

RMIN, RMAX, CMIN, CMAX = -2, 3, -2, 4
TILE = 1024
N_C = CMAX - CMIN + 1
N_R = RMAX - RMIN + 1
MOS_W, MOS_H = N_C * TILE, N_R * TILE

# Palette authority — content is never rewritten.
ANCHORS = {
    (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 4),
    (2, 2), (2, 3), (2, 4),
    (3, 1), (3, 2), (3, 3), (3, 4),
    (-1, 2),
}
PALACE = {
    (-2, -1), (-2, 0), (-2, 1),
    (-1, -2), (-1, -1), (-1, 0), (-1, 1),
    (0, -2), (0, -1), (0, 0), (0, 1),
    (1, -2), (1, -1), (1, 0),
    (2, 0),
}

# Wangfujing mid/low-chroma ambient (measured on current styled anchors).
# Not the old "beige ground" mask, which was biased warm (B≈7.6).
TARGET_A = 2.2
TARGET_B = 2.7
# Packed-earth / courtyard after the cream is knocked back.
EARTH_L = 50.0
EARTH_A = 2.4
EARTH_B = 3.4

LF_STRENGTH = 0.92
LF_FACTOR = 32
LF_BLUR = 2.2
BEIGE_STRENGTH = 0.88
EARTH_STRENGTH = 0.70
SHADOW_COOL = 0.75
SHADOW_DEEPEN = 3.2
HIGHLIGHT_COMPRESS = 0.58
ROOF_HIGHLIGHT_TRIM = 0.18


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smooth_range(x, a0, a1, b0, b1):
    """1 between a1..b0, ramps 0 at a0 and b1."""
    return smoothstep(a0, a1, x) * (1.0 - smoothstep(b0, b1, x))


def region_of(R, C):
    if (R, C) in ANCHORS:
        return "wangfujing"
    if (R, C) in PALACE:
        return "palace"
    return "transition"


def tile_id(R, C):
    return f"t1024_r{R}_c{C}"


def existing_tiles():
    out = []
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            p = STY / f"{tile_id(R, C)}.jpg"
            if p.exists():
                out.append((R, C, p))
    return out


def load_tile_float(path):
    im = Image.open(path).convert("RGB")
    if im.size != (TILE, TILE):
        im = im.resize((TILE, TILE), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float32)


def water_mask(lab, rgb):
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    R, G, Bl = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return ((Bl > 80) & (Bl > R + 15) & (Bl > G + 5) & ((Bl - R) > 20)) | (
        (B < -8) & (A < 0) & (L < 70)
    )


def tree_mask(lab):
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.sqrt(A * A + B * B)
    return (A < -8) & (C > 12) & (L > 15) & (L < 75)


def soft_protections(lab, rgb):
    """Soft 0..1 weights for landmark materials. High = protect (do not grade)."""
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.sqrt(A * A + B * B)
    hue = np.degrees(np.arctan2(B, A))  # 0=red, 90=yellow

    water = water_mask(lab, rgb).astype(np.float32)
    trees = tree_mask(lab).astype(np.float32)

    # Glazed yellow / gold roofs, including darker eaves.
    # Chroma is the main split: sunlit gold is typically C>=32, cream
    # courtyards are C≈16-28 at the same yellow hue (and often low A).
    # The previous C~16 threshold was tagging packed earth as roof, which
    # is why color_unify_strong never recast the Wu Men plaza.
    roof_hue = smooth_range(hue, 52.0, 62.0, 96.0, 112.0)
    # Mid/high L (sunlit): need C>~30. Dark eaves: slightly lower, still >26.
    c_thr = 30.0 - 4.0 * np.clip((36.0 - L) / 20.0, 0.0, 1.0)
    roof_c = smoothstep(c_thr, c_thr + 8.0, C)
    roof = (
        roof_hue * roof_c
        * smoothstep(12.0, 20.0, L)
        * smoothstep(32.0, 24.0, A)   # drop orange-red into the wall mask
        * smoothstep(-8.0, -2.0, A)   # drop green-yellow foliage
    )

    # Red plaster walls, including shaded stretches.
    red_hue = smooth_range(hue, 8.0, 18.0, 50.0, 58.0)
    red_c = smoothstep(14.0, 22.0, C) * smoothstep(8.0, 16.0, A)
    walls = red_hue * red_c * smoothstep(8.0, 16.0, L) * (1.0 - smoothstep(80.0, 90.0, L))

    return {
        "water": water,
        "trees": trees,
        "roof": np.clip(roof, 0.0, 1.0),
        "walls": np.clip(walls, 0.0, 1.0),
        "chroma": C,
        "hue": hue,
    }


def grade_weight(prot, is_anchor):
    if is_anchor:
        return np.zeros(prot["chroma"].shape, dtype=np.float32)
    protect = np.maximum.reduce([
        prot["water"],
        prot["trees"] * 0.92,
        prot["roof"],
        prot["walls"],
    ])
    return (1.0 - np.clip(protect, 0.0, 1.0)).astype(np.float32)


def _gauss_blur(arr, sigma):
    """Separable Gaussian on a small 2D float32 field."""
    if sigma <= 0:
        return arr
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    pad = np.pad(arr, ((0, 0), (radius, radius)), mode="reflect")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, pad)
    pad2 = np.pad(tmp, ((radius, radius), (0, 0)), mode="reflect")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, pad2)
    return out.astype(np.float32)


def inpaint_nan(mean):
    filled = mean.copy()
    for _ in range(8):
        nan = ~np.isfinite(filled)
        if not nan.any():
            break
        acc = np.zeros_like(filled)
        cnt = np.zeros_like(filled)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            shifted = np.roll(np.roll(filled, dy, 0), dx, 1)
            ok = np.isfinite(shifted)
            acc = np.where(ok, acc + np.nan_to_num(shifted, nan=0.0), acc)
            cnt = np.where(ok, cnt + 1.0, cnt)
        fill = np.divide(acc, np.maximum(cnt, 1e-6))
        filled = np.where(nan & (cnt > 0), fill, filled)
    return np.where(np.isfinite(filled), filled, 0.0).astype(np.float32)


def box_reduce(val, weight, factor):
    """Weighted mean of a tile into (tile/factor)^2 cells."""
    h, w = val.shape
    nh, nw = h // factor, w // factor
    v = val[: nh * factor, : nw * factor].reshape(nh, factor, nw, factor)
    wt = weight[: nh * factor, : nw * factor].reshape(nh, factor, nw, factor)
    sw = wt.sum(axis=(1, 3))
    sv = (v * wt).sum(axis=(1, 3))
    return sv, sw


def build_lf_fields(tiles):
    """Global 1/32 ambient A/B/L from mid/low-chroma pixels, including anchors."""
    fh, fw = MOS_H // LF_FACTOR, MOS_W // LF_FACTOR
    sum_A = np.zeros((fh, fw), dtype=np.float64)
    sum_B = np.zeros((fh, fw), dtype=np.float64)
    sum_L = np.zeros((fh, fw), dtype=np.float64)
    sum_W = np.zeros((fh, fw), dtype=np.float64)

    for R, C, p in tiles:
        rgb = load_tile_float(p)
        lab = rgb_to_lab(rgb)
        prot = soft_protections(lab, rgb)
        # Include anchors in the estimate (they are the reference).
        protect = np.maximum.reduce([
            prot["water"], prot["trees"] * 0.5, prot["roof"], prot["walls"],
        ])
        w = ((1.0 - np.clip(protect, 0.0, 1.0)) * (prot["chroma"] < 34.0)).astype(np.float32)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
        sA, sw = box_reduce(A, w, LF_FACTOR)
        sB, _ = box_reduce(B, w, LF_FACTOR)
        sL, _ = box_reduce(L, w, LF_FACTOR)
        y0 = (R - RMIN) * (TILE // LF_FACTOR)
        x0 = (C - CMIN) * (TILE // LF_FACTOR)
        hh, ww = sA.shape
        sum_A[y0:y0 + hh, x0:x0 + ww] += sA
        sum_B[y0:y0 + hh, x0:x0 + ww] += sB
        sum_L[y0:y0 + hh, x0:x0 + ww] += sL
        sum_W[y0:y0 + hh, x0:x0 + ww] += sw
        print(f"  lf sample {tile_id(R, C)}")

    valid = sum_W > 8.0
    loc_A = np.divide(sum_A, np.maximum(sum_W, 1e-6), dtype=np.float64)
    loc_B = np.divide(sum_B, np.maximum(sum_W, 1e-6), dtype=np.float64)
    loc_L = np.divide(sum_L, np.maximum(sum_W, 1e-6), dtype=np.float64)
    loc_A = inpaint_nan(np.where(valid, loc_A, np.nan).astype(np.float32))
    loc_B = inpaint_nan(np.where(valid, loc_B, np.nan).astype(np.float32))
    loc_L = inpaint_nan(np.where(valid, loc_L, np.nan).astype(np.float32))
    loc_A = _gauss_blur(loc_A, LF_BLUR)
    loc_B = _gauss_blur(loc_B, LF_BLUR)
    loc_L = _gauss_blur(loc_L, LF_BLUR)
    return loc_A, loc_B, loc_L


def sample_lf(field, y0, x0, th=TILE, tw=TILE, factor=LF_FACTOR):
    """Bilinear-sample a tile crop from the downsampled field, with 1-cell margin."""
    pad = np.pad(field, 1, mode="edge")
    cy0 = y0 // factor
    cx0 = x0 // factor
    ncy, ncx = th // factor, tw // factor
    crop = pad[cy0:cy0 + ncy + 2, cx0:cx0 + ncx + 2]
    src_y0 = (cy0 - 1) * factor
    src_x0 = (cx0 - 1) * factor
    src_h = crop.shape[0] * factor
    src_w = crop.shape[1] * factor
    im = Image.fromarray(crop.astype(np.float32), mode="F").resize(
        (src_w, src_h), Image.Resampling.BILINEAR
    )
    arr = np.asarray(im, dtype=np.float32)
    yy, xx = y0 - src_y0, x0 - src_x0
    return arr[yy:yy + th, xx:xx + tw]


def apply_grade_tile(rgb, loc_A, loc_B, loc_L):
    """Apply the same per-pixel grade used on the stitched overview."""
    lab = rgb_to_lab(rgb)
    L = lab[..., 0].copy()
    A = lab[..., 1].copy()
    B = lab[..., 2].copy()
    prot = soft_protections(lab, rgb)
    C = prot["chroma"]
    hue = prot["hue"]
    grade = grade_weight(prot, is_anchor=False)

    # 1) Low-frequency-guided warmth compression.
    # Pull each pixel's OWN excess A/B toward the Wangfujing target.
    # The global field only boosts strength in warm regions — we never
    # subtract a regional mean (that over-cools already-grey pixels when
    # warmth bleeds across the blur from a neighbouring palace tile).
    warm_region = smoothstep(4.0, 10.0, loc_B)
    str_ab = LF_STRENGTH * (0.82 + 0.18 * warm_region)
    excess_B = np.clip(B - TARGET_B, 0.0, None)
    excess_A = np.clip(A - TARGET_A, 0.0, None)
    B = B - grade * str_ab * excess_B
    A = A - grade * str_ab * 0.70 * excess_A
    bright_block = smoothstep(EARTH_L + 2.0, EARTH_L + 10.0, loc_L)
    L = L - grade * bright_block * 0.45 * np.clip(L - EARTH_L, 0.0, 16.0)

    C2 = np.sqrt(A * A + B * B)

    # 2) Residual beige / cream killer.
    beige = (
        grade
        * smoothstep(3.0, 7.0, B)
        * (1.0 - smoothstep(30.0, 38.0, C2))
        * smoothstep(38.0, 46.0, L)
        * smooth_range(A, -8.0, -4.0, 20.0, 26.0)
        * (1.0 - prot["roof"])
        * (1.0 - prot["walls"])
        * (1.0 - prot["trees"])
        * (1.0 - prot["water"])
    )
    beige = np.clip(beige, 0.0, 1.0) * BEIGE_STRENGTH
    B = B - beige * (B - EARTH_B)
    A = A - beige * 0.65 * (A - EARTH_A)
    L = L - beige * 0.60 * np.clip(L - EARTH_L, 0.0, 22.0)

    # 3) Mid-chroma tan earth (not a roof).
    earth = (
        grade
        * (1.0 - prot["roof"])
        * (1.0 - prot["walls"])
        * (1.0 - prot["trees"])
        * (1.0 - prot["water"])
        * smooth_range(hue, 50.0, 60.0, 95.0, 110.0)
        * smoothstep(14.0, 20.0, C2)
        * (1.0 - smoothstep(32.0, 40.0, C2))
        * smoothstep(28.0, 36.0, L)
        * (1.0 - smoothstep(78.0, 88.0, L))
    )
    earth = np.clip(earth, 0.0, 1.0) * EARTH_STRENGTH
    B = B - earth * np.clip(B - 4.0, 0.0, None)
    A = A - earth * 0.40 * np.clip(A - EARTH_A, 0.0, None)

    # 4) Cool and deepen existing shadows.
    sh = grade * (1.0 - smoothstep(26.0, 38.0, L)) * (1.0 - prot["water"])
    B = B - sh * SHADOW_COOL * np.clip(B - 1.2, 0.0, 22.0)
    A = A - sh * 0.55 * np.clip(A - 1.0, 0.0, 16.0)
    # Only deepen lifted/muddy shadows; leave already-deep west tiles alone.
    L = L - sh * SHADOW_DEEPEN * np.clip(1.0 - L / 38.0, 0.0, 1.0) * smoothstep(16.0, 24.0, L)

    # 5) Compress leftover bright neutrals.
    hi = grade * smoothstep(58.0, 70.0, L) * (1.0 - prot["roof"])
    L = L - hi * HIGHLIGHT_COMPRESS * (L - 56.0)
    cool_hi = hi * smoothstep(0.0, 8.0, B) * (C2 < 22.0)
    B = B - cool_hi * 2.2
    A = A - cool_hi * 0.8

    # Very bright glazed roofs: trim toward gold, never desaturate to clay/white.
    roof_hi = prot["roof"] * smoothstep(78.0, 88.0, L)
    L = L - roof_hi * ROOF_HIGHLIGHT_TRIM * (L - 74.0)

    L = np.clip(L, 1.0, 100.0)
    out = lab_to_rgb(np.stack([L, A, B], axis=-1)).astype(np.float32)
    water_hard = prot["water"] > 0.5
    out = np.where(water_hard[..., None], rgb, out)
    return out, prot, grade


def write_overviews(tiles):
    """Assemble previews at target size from tiles (no full-res mosaic in RAM)."""
    def thumb_size(target):
        scale = target / max(MOS_W, MOS_H)
        return scale, int(round(TILE * scale)), int(round(MOS_W * scale)), int(round(MOS_H * scale))

    def compose(scale, ts, pw, ph, use_graded):
        mos = Image.new("RGB", (pw, ph), (30, 30, 30))
        status = {}
        for R in range(RMIN, RMAX + 1):
            for C in range(CMIN, CMAX + 1):
                styled = STY / f"{tile_id(R, C)}.jpg"
                staged = OUT / f"{tile_id(R, C)}.jpg"
                if use_graded and (R, C) not in ANCHORS and staged.exists():
                    p, st = staged, "graded"
                elif styled.exists():
                    p, st = styled, ("anchor" if (R, C) in ANCHORS else "styled")
                else:
                    status[(R, C)] = "empty"
                    continue
                im = Image.open(p).convert("RGB")
                im = im.resize((ts, ts), Image.Resampling.LANCZOS)
                mos.paste(im, (int(round((C - CMIN) * TILE * scale)),
                               int(round((R - RMIN) * TILE * scale))))
                status[(R, C)] = st
        return mos, status

    scale, ts, pw, ph = thumb_size(2560)
    after, status = compose(scale, ts, pw, ph, True)
    before, _ = compose(scale, ts, pw, ph, False)

    prev_path = OUT / "overview_2560.jpg"
    after.save(prev_path, "JPEG", quality=90)

    grid = after.copy()
    d = ImageDraw.Draw(grid)
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            x0 = int(round((C - CMIN) * TILE * scale))
            y0 = int(round((R - RMIN) * TILE * scale))
            x1 = int(round((C - CMIN + 1) * TILE * scale)) - 1
            y1 = int(round((R - RMIN + 1) * TILE * scale)) - 1
            st = status[(R, C)]
            color = {
                "graded": (40, 200, 80),
                "anchor": (40, 180, 255),
                "styled": (40, 200, 80),
                "empty": (180, 60, 60),
            }[st]
            d.rectangle([x0, y0, x1, y1], outline=color, width=2)
            d.text((x0 + 4, y0 + 4), f"R{R}C{C} {st[:3]}", fill=color)
    grid_path = OUT / "overview_2560_grid.jpg"
    grid.save(grid_path, "JPEG", quality=90)

    pair = Image.new("RGB", (before.size[0] * 2 + 8, before.size[1] + 28), (18, 18, 18))
    d = ImageDraw.Draw(pair)
    d.text((8, 6), "BEFORE  (current styled)", fill=(200, 200, 200))
    d.text((before.size[0] + 16, 6), "AFTER  (staging color_J_global, anchors unchanged)", fill=(200, 200, 200))
    pair.paste(before, (0, 28))
    pair.paste(after, (before.size[0] + 8, 28))
    pair_path = OUT / "overview_2560_before_after.jpg"
    pair.save(pair_path, "JPEG", quality=90)

    # Palace / transition close-up: c-2..c1, r-2..r2
    def compose_region(use_graded, c0=-2, c1=1, r0=-2, r1=2, long_side=1800):
        cols, rows = c1 - c0 + 1, r1 - r0 + 1
        scale = long_side / (max(cols, rows) * TILE)
        ts = max(1, int(round(TILE * scale)))
        mos = Image.new("RGB", (ts * cols, ts * rows), (30, 30, 30))
        for R in range(r0, r1 + 1):
            for C in range(c0, c1 + 1):
                styled = STY / f"{tile_id(R, C)}.jpg"
                staged = OUT / f"{tile_id(R, C)}.jpg"
                if use_graded and (R, C) not in ANCHORS and staged.exists():
                    p = staged
                elif styled.exists():
                    p = styled
                else:
                    continue
                im = Image.open(p).convert("RGB").resize((ts, ts), Image.Resampling.LANCZOS)
                mos.paste(im, ((C - c0) * ts, (R - r0) * ts))
        return mos

    bc, ac = compose_region(False), compose_region(True)
    close = Image.new("RGB", (bc.size[0] * 2 + 8, bc.size[1] + 28), (18, 18, 18))
    d = ImageDraw.Draw(close)
    d.text((8, 6), "BEFORE  palace / transition", fill=(200, 200, 200))
    d.text((bc.size[0] + 16, 6), "AFTER  palace / transition", fill=(200, 200, 200))
    close.paste(bc, (0, 28))
    close.paste(ac, (bc.size[0] + 8, 28))
    close_path = OUT / "palace_close_before_after.jpg"
    close.save(close_path, "JPEG", quality=90)

    return prev_path, grid_path, pair_path, close_path


def write_mask_preview(tiles, loc_A, loc_B, loc_L):
    """Low-res mask legend + LF B field. Recomputed per tile at 1/8."""
    factor = 8
    ts = TILE // factor
    vis = Image.new("RGB", (N_C * ts, N_R * ts), (20, 20, 20))
    field = Image.fromarray(
        np.clip((loc_B - 0.0) / 12.0 * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).resize((N_C * ts, N_R * ts), Image.Resampling.BILINEAR)
    field_rgb = Image.merge("RGB", (
        field.point(lambda v: min(255, int(v * 1.2))),
        field.point(lambda v: int(v * 0.4)),
        field.point(lambda v: int(40 + (255 - v) * 0.3)),
    ))
    for R, C, p in tiles:
        rgb = load_tile_float(p)
        lab = rgb_to_lab(rgb)
        prot = soft_protections(lab, rgb)
        is_anchor = (R, C) in ANCHORS
        grade = grade_weight(prot, is_anchor)
        small = np.zeros((ts, ts, 3), dtype=np.float32)
        def down(a):
            return a.reshape(ts, factor, ts, factor).mean(axis=(1, 3))
        g = down(grade)
        small[..., 0] = 40 + 80 * g
        small[..., 1] = 40 + 80 * g
        small[..., 2] = 40 + 80 * g
        small[..., 0] += 200 * down(prot["roof"])
        small[..., 1] += 160 * down(prot["roof"])
        small[..., 0] += 200 * down(prot["walls"])
        small[..., 2] += 180 * down(prot["water"])
        small[..., 1] += 200 * down(prot["trees"])
        im = Image.fromarray(np.clip(small, 0, 255).astype(np.uint8))
        vis.paste(im, ((C - CMIN) * ts, (R - RMIN) * ts))
    scale = 1600 / max(vis.size)
    vis = vis.resize((int(vis.size[0] * scale), int(vis.size[1] * scale)), Image.Resampling.NEAREST)
    field_rgb = field_rgb.resize(vis.size, Image.Resampling.NEAREST)
    sheet = Image.new("RGB", (vis.size[0] * 2 + 8, vis.size[1] + 24), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    d.text((8, 4), "masks  yellow=roof  red=wall  blue=water  green=tree  grey=gradeable  (anchors blacked)", fill=(200, 200, 200))
    d.text((vis.size[0] + 16, 4), "LF ambient B  (warmer = redder)", fill=(200, 200, 200))
    sheet.paste(vis, (0, 24))
    sheet.paste(field_rgb, (vis.size[0] + 8, 24))
    path = OUT / "masks_preview.jpg"
    sheet.save(path, "JPEG", quality=85)
    return path


def tile_delta(tiles):
    rows = []
    for R, C, p in tiles:
        before = load_tile_float(p)
        staged = OUT / f"{tile_id(R, C)}.jpg"
        after = load_tile_float(staged) if staged.exists() else before
        lab_b = rgb_to_lab(before)
        lab_a = rgb_to_lab(after)
        dL = float(lab_a[..., 0].mean() - lab_b[..., 0].mean())
        dA = float(lab_a[..., 1].mean() - lab_b[..., 1].mean())
        dB = float(lab_a[..., 2].mean() - lab_b[..., 2].mean())
        mae = float(np.mean(np.abs(after - before)))
        rows.append({
            "id": tile_id(R, C),
            "region": region_of(R, C),
            "anchor": (R, C) in ANCHORS,
            "mae": round(mae, 3),
            "dL": round(dL, 3),
            "dA": round(dA, 3),
            "dB": round(dB, 3),
            "B_before": round(float(lab_b[..., 2].mean()), 3),
            "B_after": round(float(lab_a[..., 2].mean()), 3),
        })
    rows.sort(key=lambda r: -r["mae"])
    return rows


def region_from_rows(rows, region, after=True):
    # fallback not used; computed in tile_delta loop instead if needed
    return None


def region_lab_from_disk(tiles, use_graded):
    out = {}
    for reg in ("palace", "wangfujing", "transition"):
        Ls = As = Bs = 0.0
        n = 0
        for R, C, p in tiles:
            if region_of(R, C) != reg:
                continue
            path = OUT / f"{tile_id(R, C)}.jpg" if (use_graded and (R, C) not in ANCHORS) else p
            if not Path(path).exists():
                path = p
            lab = rgb_to_lab(load_tile_float(path))
            Ls += float(lab[..., 0].mean()) * lab[..., 0].size
            As += float(lab[..., 1].mean()) * lab[..., 0].size
            Bs += float(lab[..., 2].mean()) * lab[..., 0].size
            n += lab[..., 0].size
        out[reg] = None if not n else {
            "L_mean": round(Ls / n, 3), "A_mean": round(As / n, 3),
            "B_mean": round(Bs / n, 3), "n": n,
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = existing_tiles()
    print(f"building global LF field from {len(tiles)} tiles...")
    loc_A, loc_B, loc_L = build_lf_fields(tiles)
    # Quick field range check
    print(f"  LF A {loc_A.min():.2f}..{loc_A.max():.2f} mean={loc_A.mean():.2f}")
    print(f"  LF B {loc_B.min():.2f}..{loc_B.max():.2f} mean={loc_B.mean():.2f}")
    print(f"  LF L {loc_L.min():.2f}..{loc_L.max():.2f} mean={loc_L.mean():.2f}")
    np.savez_compressed(OUT / "lf_field.npz", A=loc_A, B=loc_B, L=loc_L)

    print("grading non-anchor tiles...")
    written = []
    for R, C, p in tiles:
        if (R, C) in ANCHORS:
            print(f"  skip anchor {tile_id(R, C)}")
            continue
        rgb = load_tile_float(p)
        y0, x0 = (R - RMIN) * TILE, (C - CMIN) * TILE
        sA = sample_lf(loc_A, y0, x0)
        sB = sample_lf(loc_B, y0, x0)
        sL = sample_lf(loc_L, y0, x0)
        out, _, _ = apply_grade_tile(rgb, sA, sB, sL)
        path = OUT / f"{tile_id(R, C)}.jpg"
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
            path, "JPEG", quality=93, subsampling=0
        )
        written.append(tile_id(R, C))
        print(f"  wrote {path.name}")

    print("writing previews...")
    paths = write_overviews(tiles)
    mask_path = write_mask_preview(tiles, loc_A, loc_B, loc_L)
    print("  ", paths)
    print("  ", mask_path)

    print("measuring deltas...")
    deltas = tile_delta(tiles)
    before_regions = region_lab_from_disk(tiles, False)
    after_regions = region_lab_from_disk(tiles, True)

    stats_path = OUT / "stats.json"
    report = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    report["after"] = {
        "script": "color_unify_mosaic.py",
        "targets": {
            "lowc_A": TARGET_A, "lowc_B": TARGET_B,
            "earth_L": EARTH_L, "earth_A": EARTH_A, "earth_B": EARTH_B,
        },
        "strengths": {
            "LF_STRENGTH": LF_STRENGTH, "BEIGE_STRENGTH": BEIGE_STRENGTH,
            "EARTH_STRENGTH": EARTH_STRENGTH, "SHADOW_COOL": SHADOW_COOL,
            "SHADOW_DEEPEN": SHADOW_DEEPEN, "HIGHLIGHT_COMPRESS": HIGHLIGHT_COMPRESS,
        },
        "written_tiles": written,
        "n_written": len(written),
        "n_anchors_unchanged": len(ANCHORS),
        "regions_before_all": before_regions,
        "regions_after_all": after_regions,
        "tile_deltas": deltas,
        "largest_changes": [d for d in deltas if not d["anchor"]][:10],
    }
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("updated", stats_path)
    print("\nLargest changes:")
    for d in report["after"]["largest_changes"]:
        print(f"  {d['id']:22s} mae={d['mae']:5.2f}  B {d['B_before']:5.1f}->{d['B_after']:5.1f}  dL={d['dL']:+.2f}")
    print("\nRegion all-pixel LAB:")
    for reg in ("palace", "wangfujing", "transition"):
        b, a = before_regions[reg], after_regions[reg]
        print(f"  {reg:11s}  L {b['L_mean']:.1f}->{a['L_mean']:.1f}  A {b['A_mean']:.1f}->{a['A_mean']:.1f}  B {b['B_mean']:.1f}->{a['B_mean']:.1f}")


if __name__ == "__main__":
    main()
