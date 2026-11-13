#!/usr/bin/env python3
"""Narrow-band distance-weighted seam fixes for the axis expansion.

- Applies the one successful GenerateImage infill (Jingshan | Shenwu moat).
- Applies a narrow cosine overlap-blend on other new/new and new/old junctions
  so razor cuts soften without inventing a second landmark.
- Already-good old 5x4 internal seams are not touched, except a very narrow
  fade on the west edge of r1_c0 (new Wumen join). r0_c0|r0_c1 is left alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/workspace/beijing-pilot")
STYLED = ROOT / "tiles" / "styled"
INF = ROOT / "tiles" / "mosaic" / "seams_axis" / "inpaints"
TILE = 1024

# tiles we may write (new 12 + r0_c0 + narrow r1_c0 west)
TOUCHABLE = {
    ( -2, -1), (-2, 0), (-2, 1),
    ( -1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
    (  0, -2), ( 0, -1), ( 0, 0),
    (  1, -2), ( 1, -1),
}
# old tiles that may receive only a tiny fade (px)
OLD_FADE = {(1, 0): 48}  # r1_c0 west edge only


def load(r, c):
    p = STYLED / f"t1024_r{r}_c{c}.jpg"
    im = Image.open(p).convert("RGB")
    if im.size != (TILE, TILE):
        im = im.resize((TILE, TILE), Image.Resampling.LANCZOS)
    return np.asarray(im).astype(np.float32)


def save(r, c, arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(STYLED / f"t1024_r{r}_c{c}.jpg", "JPEG", quality=93, subsampling=0)
    print(f"  wrote t1024_r{r}_c{c}.jpg")


def cosine_weight(d, band):
    """d=0 at the seam, d=band at the interior edge. weight=1 at seam, 0 at edge."""
    t = np.clip(d / max(band, 1), 0, 1)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def overlap_vertical(left, right, band, right_band=None):
    """Blend left's right edge with right's left edge. Returns (left, right)."""
    rb = band if right_band is None else right_band
    L, R = left.copy(), right.copy()
    # left tile: columns TILE-band .. TILE-1, d = TILE-1-x
    for x in range(TILE - band, TILE):
        d = TILE - 1 - x
        w = cosine_weight(d, band) * 0.7  # cap so unique interiors stay
        src_x = min(TILE - 1, x - (TILE - band) )  # 0..band-1 on the right tile
        # actually the matching column on the right tile is (x - (TILE-band)) wait:
        # left col TILE-1 <-> right col 0 (the seam)
        src_x = (x - (TILE - 1))  # 0 at seam, negative inside left... 
        src_x = TILE - 1 - x  # 0 at seam -> right col 0; at band -> right col band
        L[:, x] = (1 - w) * L[:, x] + w * R[:, src_x]
    for x in range(0, rb):
        d = x
        w = cosine_weight(d, rb) * 0.7
        src_x = TILE - 1 - x  # left tile's right edge
        R[:, x] = (1 - w) * R[:, x] + w * L[:, src_x]
    return L, R


def overlap_horizontal(top, bot, band, bot_band=None):
    bb = band if bot_band is None else bot_band
    T, B = top.copy(), bot.copy()
    for y in range(TILE - band, TILE):
        d = TILE - 1 - y
        w = cosine_weight(d, band) * 0.7
        src_y = TILE - 1 - y
        T[y, :] = (1 - w) * T[y, :] + w * B[src_y, :]
    for y in range(0, bb):
        d = y
        w = cosine_weight(d, bb) * 0.7
        src_y = TILE - 1 - y
        B[y, :] = (1 - w) * B[y, :] + w * T[src_y, :]
    return T, B


def apply_h_infill(top, bot, infill_path, band=240):
    """Infill is a 1024x1024 canvas with the seam band centered (y=512)."""
    inf = np.asarray(Image.open(infill_path).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)).astype(np.float32)
    # detect content rows (not near-black padding)
    lum = inf.mean(axis=2)
    row_mean = lum.mean(axis=1)
    content = np.where(row_mean > 18)[0]
    if len(content) < 80:
        print("  infill content not found, skip")
        return top, bot
    # use the center 2*band around 512 if it looks right, else detected mid
    y0, y1 = int(content[0]), int(content[-1]) + 1
    mid = (y0 + y1) // 2
    # prefer 512 if it's inside content
    if y0 < 512 < y1:
        mid = 512
    T, B = top.copy(), bot.copy()
    # top tile bottom band
    for i, y in enumerate(range(TILE - band, TILE)):
        d = TILE - 1 - y
        w = cosine_weight(d, band)
        iy = mid - (TILE - y)  # mid-1, mid-2, ... at interior; mid at seam? 
        # at y=TILE-1 (seam) we want infill row mid-1 or mid
        # at y=TILE-band we want infill row mid-band
        iy = mid - (TILE - y)
        if iy < 0 or iy >= 1024:
            continue
        T[y, :] = (1 - w) * T[y, :] + w * inf[iy, :]
    for i, y in enumerate(range(0, band)):
        d = y
        w = cosine_weight(d, band)
        iy = mid + y
        if iy < 0 or iy >= 1024:
            continue
        B[y, :] = (1 - w) * B[y, :] + w * inf[iy, :]
    return T, B


def main():
    tiles = {}
    def get(r, c):
        if (r, c) not in tiles:
            tiles[(r, c)] = load(r, c)
        return tiles[(r, c)]

    # 1) successful Shenwu/Jingshan moat infill
    inf = INF / "infill_h_r-2--1_c0_GOOD.jpg"
    if inf.exists():
        print("applying Jingshan|Shenwu moat infill")
        t, b = apply_h_infill(get(-2, 0), get(-1, 0), inf, band=220)
        tiles[(-2, 0)], tiles[(-1, 0)] = t, b
    else:
        print("missing good infill, skip")

    # 2) narrow overlap blends on remaining new junctions
    v_pairs = [
        # r, c0, c1, band_left, band_right
        (0, -1, 0, 90, 90),      # halls | r0_c0 terrace (almost aligned)
        (0, -2, -1, 70, 70),     # west moat | halls
        (1, -2, -1, 70, 70),     # SW | Wumen
        (-1, -2, -1, 60, 60),
        (-1, -1, 0, 70, 70),
        (-1, 0, 1, 70, 70),
        (-2, -1, 0, 80, 80),
        (-2, 0, 1, 70, 70),
        (1, -1, 0, 100, 40),     # Wumen | old r1_c0 (protect old)
        (-1, 1, 2, 60, 0),       # NE | commercial (don't touch r-1_c2 much? both new)
    ]
    # fix last: both new, allow both sides
    v_pairs[-1] = (-1, 1, 2, 60, 60)

    h_pairs = [
        (-1, 0, 0, 80, 80),      # shenwu south | r0_c0 (content mismatch, just kill razor)
        (-1, 0, -1, 70, 70),
        (-1, 0, 1, 70, 70),
        (-1, 0, 2, 60, 0),       # don't touch old r0_c2 — WAIT r-1_c2 | r0_c2, r0_c2 is OLD
        (-2, -1, -1, 80, 80),
        (-2, -1, 1, 70, 70),
        (0, 1, -1, 80, 80),      # halls | wumen
        (0, 1, -2, 70, 70),
    ]
    h_pairs[3] = (-1, 0, 2, 70, 48)  # new r-1_c2 vs old r0_c2: tiny fade on old

    print("overlap vertical")
    for r, c0, c1, bL, bR in v_pairs:
        if bR == 0 and (r, c1) not in TOUCHABLE and (r, c1) not in OLD_FADE:
            bR = 0
        L, R = overlap_vertical(get(r, c0), get(r, c1), bL, right_band=bR)
        tiles[(r, c0)], tiles[(r, c1)] = L, R
        print(f"  v r{r} c{c0}|{c1} bands {bL}/{bR}")

    print("overlap horizontal")
    for r0, r1, c, bT, bB in h_pairs:
        T, B = overlap_horizontal(get(r0, c), get(r1, c), bT, bot_band=bB)
        tiles[(r0, c)], tiles[(r1, c)] = T, B
        print(f"  h r{r0}|{r1} c{c} bands {bT}/{bB}")

    # write only touchable + old-fade tiles
    written = []
    for (r, c), arr in tiles.items():
        if (r, c) in TOUCHABLE or (r, c) in OLD_FADE:
            save(r, c, arr)
            written.append(f"t1024_r{r}_c{c}")
        else:
            print(f"  skipped write r{r}_c{c} (protected old tile)")
    print("written", written)


if __name__ == "__main__":
    main()
