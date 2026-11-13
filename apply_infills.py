#!/usr/bin/env python3
"""Paste seam-centered 1024 infills onto the 5x4 mosaic, feathered, then split tiles back."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/workspace/beijing-pilot")
STYLED = ROOT / "tiles" / "styled"
INF = STYLED / "infills"
TILE = 1024
N_C, N_R = 5, 4
FEATHER = 96

INFILLS = [
    # name, dest top-left on mosaic (the 1024 is centered on the seam)
    ("seam_v_r0_c0-1.jpg", (0 * TILE + 512, 0 * TILE)),
    ("seam_v_r1_c0-1.jpg", (0 * TILE + 512, 1 * TILE)),
    ("seam_v_r2_c0-1.jpg", (0 * TILE + 512, 2 * TILE)),
    ("seam_h_r0-1_c0.jpg", (0 * TILE, 0 * TILE + 512)),
    ("seam_h_r0-1_c1.jpg", (1 * TILE, 0 * TILE + 512)),
    ("seam_h_r1-2_c0.jpg", (0 * TILE, 1 * TILE + 512)),
]


def load_tile(R, C):
    p = STYLED / f"t1024_r{R}_c{C}.jpg"
    im = Image.open(p).convert("RGB")
    if im.size != (TILE, TILE):
        im = im.resize((TILE, TILE), Image.Resampling.LANCZOS)
    return im


def feather_alpha(w, h, feather):
    a = np.ones((h, w), np.float32)
    if feather <= 0:
        return a
    ramp = np.linspace(1.0 / feather, 1.0, feather, dtype=np.float32)
    a[:feather, :] *= ramp[:, None]
    a[-feather:, :] *= ramp[::-1, None]
    a[:, :feather] *= ramp[None, :]
    a[:, -feather:] *= ramp[None, ::-1]
    return a


def main():
    W, H = N_C * TILE, N_R * TILE
    mosaic = Image.new("RGB", (W, H))
    for R in range(N_R):
        for C in range(N_C):
            mosaic.paste(load_tile(R, C), (C * TILE, R * TILE))
    base = np.asarray(mosaic).astype(np.float32)
    acc = base.copy()

    for name, (x, y) in INFILLS:
        path = INF / name
        if not path.exists():
            raise SystemExit(f"missing {path}")
        inf = Image.open(path).convert("RGB")
        if inf.size != (TILE, TILE):
            inf = inf.resize((TILE, TILE), Image.Resampling.LANCZOS)
        arr = np.asarray(inf).astype(np.float32)
        alpha = feather_alpha(TILE, TILE, FEATHER)[..., None]
        # clip to mosaic bounds
        x0, y0 = x, y
        x1, y1 = x + TILE, y + TILE
        sx0 = sy0 = 0
        sx1 = sy1 = TILE
        if x0 < 0:
            sx0 = -x0
            x0 = 0
        if y0 < 0:
            sy0 = -y0
            y0 = 0
        if x1 > W:
            sx1 -= x1 - W
            x1 = W
        if y1 > H:
            sy1 -= y1 - H
            y1 = H
        sl = arr[sy0:sy1, sx0:sx1]
        al = alpha[sy0:sy1, sx0:sx1]
        region = acc[y0:y1, x0:x1]
        acc[y0:y1, x0:x1] = region * (1.0 - al) + sl * al
        print("applied", name, "at", (x, y))

    out = Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))
    # write tiles back
    touched = set()
    for R in range(N_R):
        for C in range(N_C):
            # a tile is touched if any infill overlaps it
            tx0, ty0 = C * TILE, R * TILE
            hit = False
            for name, (x, y) in INFILLS:
                if x < tx0 + TILE and x + TILE > tx0 and y < ty0 + TILE and y + TILE > ty0:
                    hit = True
                    break
            if hit:
                crop = out.crop((tx0, ty0, tx0 + TILE, ty0 + TILE))
                dest = STYLED / f"t1024_r{R}_c{C}.jpg"
                crop.save(dest, "JPEG", quality=95, subsampling=0)
                touched.add(f"t1024_r{R}_c{C}")
                print("wrote", dest)
    print("touched", sorted(touched))
    # also save pre-export mosaic snapshot
    snap = ROOT / "tiles" / "mosaic" / "overview_after_infill_raw.jpg"
    snap.parent.mkdir(parents=True, exist_ok=True)
    out.save(snap, "JPEG", quality=90)
    print("snap", snap)


if __name__ == "__main__":
    main()
