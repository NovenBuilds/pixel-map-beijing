#!/usr/bin/env python3
"""Re-apply moat seam infills with distance-to-seam weights and a narrow band.

Restores moat-column tiles from v1_pre_seamfix first, so baked-in 96px
ghosts are discarded. Vertical infills only win near the vertical seam
(so unique halls in tile interiors are not replaced by a repeated
Donghuamen vista). Horizontal infills win near the horizontal seam
(preserving the already-good east-west moat joins).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/workspace/beijing-pilot")
STYLED = ROOT / "tiles" / "styled"
V1 = STYLED / "v1_pre_seamfix"
INF = STYLED / "infills"
TILE = 1024
N_C, N_R = 5, 4

# Restore only tiles that were overwritten by the first infill pass
RESTORE = [
    "t1024_r0_c0.jpg", "t1024_r0_c1.jpg",
    "t1024_r1_c0.jpg", "t1024_r1_c1.jpg",
    "t1024_r2_c0.jpg", "t1024_r2_c1.jpg",
]

# band = half-width in px around the seam where the infill is allowed to win
# fade = extra ramp outside the solid core
INFILLS = [
    # name, dest_xy, kind, seam_coord (x for v, y for h), band, fade
    ("seam_v_r0_c0-1.jpg", (512, 0), "v", 1024, 140, 12),
    ("seam_v_r1_c0-1.jpg", (512, 1024), "v", 1024, 140, 12),
    ("seam_v_r2_c0-1.jpg", (512, 2048), "v", 1024, 140, 12),
    ("seam_h_r0-1_c0.jpg", (0, 512), "h", 1024, 270, 16),
    ("seam_h_r0-1_c1.jpg", (1024, 512), "h", 1024, 270, 16),
    ("seam_h_r1-2_c0.jpg", (0, 1536), "h", 2048, 270, 16),
]


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(im).astype(np.float32)


def main():
    for name in RESTORE:
        src, dst = V1 / name, STYLED / name
        if not src.exists():
            raise SystemExit(f"missing v1 {src}")
        shutil.copy2(src, dst)
        print("restored", name)

    W, H = N_C * TILE, N_R * TILE
    mosaic = np.zeros((H, W, 3), np.float32)
    for R in range(N_R):
        for C in range(N_C):
            arr = load_rgb(STYLED / f"t1024_r{R}_c{C}.jpg", (TILE, TILE))
            mosaic[R * TILE:(R + 1) * TILE, C * TILE:(C + 1) * TILE] = arr

    # per-pixel: collect best two (weight, color) from infills
    # We'll accumulate weighted sums
    weight_sum = np.zeros((H, W, 1), np.float32)
    color_sum = np.zeros((H, W, 3), np.float32)
    # For competition: store closest v and h weights separately then resolve
    # Simpler: iterate infills and keep a list? Memory heavy.
    # Two-pass: compute for each pixel the min distance v and h, then write.

    # Stack candidates as arrays of weight and color, max 6, then normalize
    # Only update pixels an infill covers.
    # Use "winner takes most": we'll store current best_d and second, then combine.

    best_d = np.full((H, W), 1e9, np.float32)
    best_c = mosaic.copy()
    best_w = np.zeros((H, W), np.float32)
    sec_d = np.full((H, W), 1e9, np.float32)
    sec_c = np.zeros((H, W, 3), np.float32)
    sec_w = np.zeros((H, W), np.float32)

    def consider(y0, y1, x0, x1, d, w, col):
        bd = best_d[y0:y1, x0:x1]
        bw = best_w[y0:y1, x0:x1]
        bc = best_c[y0:y1, x0:x1]
        sd = sec_d[y0:y1, x0:x1]
        sw = sec_w[y0:y1, x0:x1]
        sc = sec_c[y0:y1, x0:x1]
        better = d < bd
        move = better & (bw > 0)
        # demote best -> second
        sd[:] = np.where(move, bd, sd)
        sw[:] = np.where(move, bw, sw)
        sc[:] = np.where(move[..., None], bc, sc)
        bd[:] = np.where(better, d, bd)
        bw[:] = np.where(better, w, bw)
        bc[:] = np.where(better[..., None], col, bc)
        mid = (~better) & (d < sd)
        sd[:] = np.where(mid, d, sd)
        sw[:] = np.where(mid, w, sw)
        sc[:] = np.where(mid[..., None], col, sc)

    for name, (x, y), kind, seam, band, fade in INFILLS:
        path = INF / name
        inf = load_rgb(path, (TILE, TILE))
        # clip
        x0, y0 = x, y
        x1, y1 = x + TILE, y + TILE
        sx0 = sy0 = 0
        sx1 = sy1 = TILE
        if x0 < 0:
            sx0, x0 = -x0, 0
        if y0 < 0:
            sy0, y0 = -y0, 0
        if x1 > W:
            sx1 -= x1 - W
            x1 = W
        if y1 > H:
            sy1 -= y1 - H
            y1 = H
        yy, xx = np.ogrid[y0:y1, x0:x1]
        if kind == "v":
            dist = np.abs(xx - seam).astype(np.float32)
        else:
            dist = np.abs(yy - seam).astype(np.float32)
        # weight 1 inside band, ramp to 0 over fade
        w = np.clip((band + fade - dist) / max(fade, 1), 0.0, 1.0)
        col = inf[sy0:sy1, sx0:sx1]
        # ignore zero-weight
        w = w.astype(np.float32)
        consider(y0, y1, x0, x1, dist + (1.0 - w) * 1e6, w, col)
        print("loaded", name, "kind", kind, "band", band)

    # Combine best and second when both close; then blend into mosaic by best_w
    # If second is close (within 24px), mix
    tau = 24.0
    has2 = (sec_w > 0) & (np.abs(best_d - sec_d) < tau * 2)
    # softmax-like
    k1 = np.exp(-np.minimum(best_d, 500) / tau)
    k2 = np.exp(-np.minimum(sec_d, 500) / tau)
    s = k1 + k2
    mix = (best_c * k1[..., None] + sec_c * k2[..., None]) / s[..., None]
    chosen = best_c.copy()
    chosen[has2] = mix[has2]
    # apply weight against original mosaic (best_w already 0..1)
    # if second exists, use max weight
    wfin = best_w.copy()
    wfin[has2] = np.maximum(best_w[has2], sec_w[has2])
    out = mosaic * (1.0 - wfin[..., None]) + chosen * wfin[..., None]
    im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    touched = []
    for R in range(N_R):
        for C in range(N_C):
            name = f"t1024_r{R}_c{C}"
            if name + ".jpg" not in RESTORE:
                continue
            crop = im.crop((C * TILE, R * TILE, (C + 1) * TILE, (R + 1) * TILE))
            crop.save(STYLED / f"{name}.jpg", "JPEG", quality=95, subsampling=0)
            touched.append(name)
            print("wrote", name)
    print("touched", touched)


if __name__ == "__main__":
    main()
