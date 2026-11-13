#!/usr/bin/env python3
"""Apply the new r1 vertical-moat infill (narrow distance-to-seam band)
while protecting the already-good horizontal moat joins, then unify canal
water color along the west moat column.
Does NOT restore v1 tiles. Does not touch r3_c2–c4.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/workspace/beijing-pilot")
STYLED = ROOT / "tiles" / "styled"
INF = STYLED / "infills"
TILE = 1024
N_C, N_R = 5, 4
W, H = N_C * TILE, N_R * TILE


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(im).astype(np.float32)


def save_rgb(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
        path, "JPEG", quality=95, subsampling=0
    )


def build_mosaic():
    mosaic = np.zeros((H, W, 3), np.float32)
    for R in range(N_R):
        for C in range(N_C):
            mosaic[R * TILE:(R + 1) * TILE, C * TILE:(C + 1) * TILE] = load_rgb(
                STYLED / f"t1024_r{R}_c{C}.jpg", (TILE, TILE)
            )
    return mosaic


def water_mask(rgb, loose=False):
    """Blue canal pixels: hue in blue, mid/high value, some saturation."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx > 1, (mx - mn) / mx, 0)
    # hue: blue when b is max and b clearly above r
    is_blue = (b > g) & (b > r) & (b - r > (18 if not loose else 10))
    if loose:
        return is_blue & (sat > 0.12) & (mx > 50) & (mx < 230)
    return is_blue & (sat > 0.18) & (mx > 60) & (mx < 210) & (b > 90)


def apply_v_infill(mosaic, name, dest_xy, seam_x, band, fade, protect_h_ys, protect_h_band):
    """Paste a vertical infill; weight is 0 near protected horizontal seams."""
    x, y = dest_xy
    inf = load_rgb(INF / name, (TILE, TILE))
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
    dist = np.abs(xx - seam_x).astype(np.float32)
    w = np.clip((band + fade - dist) / max(fade, 1), 0.0, 1.0)
    # protect horizontal seams: ramp to 0 inside protect_h_band of each y
    prot = np.zeros_like(w)
    for hy in protect_h_ys:
        dhy = np.abs(yy - hy).astype(np.float32)
        # full protect inside protect_h_band-40, fade 40
        p = np.clip((protect_h_band - dhy) / 40.0, 0.0, 1.0)
        prot = np.maximum(prot, p)
    w = w * (1.0 - prot)
    col = inf[sy0:sy1, sx0:sx1]
    sl = mosaic[y0:y1, x0:x1]
    mosaic[y0:y1, x0:x1] = sl * (1 - w[..., None]) + col * w[..., None]
    print(f"applied {name} band={band} fade={fade} maxw={w.max():.2f} meanw={w.mean():.3f}")
    return w


def unify_water(mosaic, target_rgb, corridor_x, half_width, also_h=None):
    """Shift water pixels in a vertical corridor toward target_rgb, softly."""
    r, g, b = mosaic[..., 0], mosaic[..., 1], mosaic[..., 2]
    mask = water_mask(mosaic, loose=True)
    xx = np.arange(W)[None, :]
    in_corr = np.abs(xx - corridor_x) <= half_width
    use = mask & in_corr
    if also_h:
        # also unify horizontal canal strips (west palace), y bands
        yy = np.arange(H)[:, None]
        for y0, y1, x1 in also_h:
            use |= mask & (yy >= y0) & (yy < y1) & (np.arange(W)[None, :] < x1)
    n = int(use.sum())
    if n < 100:
        print("unify_water: too few pixels", n)
        return
    # Match mean of current water in the tight core to target
    core = mask & (np.abs(xx - corridor_x) <= 80)
    if core.sum() > 50:
        cur_mean = mosaic[core].mean(axis=0)
    else:
        cur_mean = mosaic[use].mean(axis=0)
    delta = np.asarray(target_rgb, np.float32) - cur_mean
    # Apply a slightly stronger correction on water, with a 40px fade at corridor edge
    fade = np.clip((half_width - np.abs(xx - corridor_x)) / 40.0, 0.0, 1.0)
    strength = 0.85 * fade  # leave a little natural variation
    w = np.where(use, strength, 0).astype(np.float32)
    mosaic += w[..., None] * delta[None, None, :]
    np.clip(mosaic, 0, 255, out=mosaic)
    print(f"unify_water n={n} cur_mean={cur_mean} target={target_rgb} delta={delta}")


def mild_c0_earth_harmony(mosaic):
    """Gently pull packed-earth / grass in column c0 toward a shared tan-green."""
    # restrict to c0 interiors, avoid the moat corridor (x>900) and r3 which is now urban
    x0, x1 = 0, 900
    y0, y1 = 0, 3 * TILE  # r0-r2 only
    sl = mosaic[y0:y1, x0:x1]
    r, g, b = sl[..., 0], sl[..., 1], sl[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx > 1, (mx - mn) / mx, 0)
    # earth: tan, mid value, low-mid sat, r~g or r>g, not blue, not strong red walls, not yellow roofs
    tan = (r > 110) & (r < 210) & (g > 90) & (g < 200) & (b > 60) & (b < 170)
    tan &= (r >= g - 8) & (r > b + 10) & (sat < 0.45) & (sat > 0.08)
    tan &= ~((r > 160) & (g < 80))  # not red walls
    tan &= ~((r > 180) & (g > 140) & (b < 80))  # not yellow roofs
    # target: mid tan-green sampled from a mid region of r1_c0 courtyard
    # Use median of tan pixels in r1_c0 upper courtyard as target
    sample = tan.copy()
    # r1 is y 1024-2048 in full mosaic, in this slice y 1024-2048, but slice starts at 0
    # slice y0=0 so r1 is [1024:2048]
    samp = sl[1100:1700, 80:700]
    sr, sg, sb = samp[..., 0], samp[..., 1], samp[..., 2]
    smx = np.maximum(np.maximum(sr, sg), sb)
    smn = np.minimum(np.minimum(sr, sg), sb)
    ssat = np.where(smx > 1, (smx - smn) / smx, 0)
    stan = (sr > 110) & (sr < 210) & (sg > 90) & (sb > 60) & (sr >= sg - 8) & (sr > sb + 10) & (ssat < 0.45) & (ssat > 0.08)
    if stan.sum() < 200:
        print("earth harmony: no sample")
        return
    target = samp[stan].mean(axis=0)
    cur = sl[tan].mean(axis=0) if tan.sum() else target
    delta = (target - cur) * 0.45  # mild
    sl[tan] = sl[tan] + delta
    np.clip(sl, 0, 255, out=sl)
    mosaic[y0:y1, x0:x1] = sl
    print(f"earth harmony n={int(tan.sum())} target={target} delta={delta}")


def write_tiles(mosaic, touched):
    for R, C in touched:
        crop = mosaic[R * TILE:(R + 1) * TILE, C * TILE:(C + 1) * TILE]
        save_rgb(crop, STYLED / f"t1024_r{R}_c{C}.jpg")
        print("wrote", f"t1024_r{R}_c{C}.jpg")


def main():
    mosaic = build_mosaic()

    # Sample target water from the already-good horizontal moat close-up if present,
    # else from the new r1 infill canal center.
    h_ref = ROOT / "tiles" / "mosaic" / "seams" / "seam_h_r0-1_c1.jpg"
    if h_ref.exists():
        href = load_rgb(h_ref)
        hm = water_mask(href)
        target = href[hm].mean(axis=0) if hm.sum() > 50 else np.array([50, 110, 160], np.float32)
    else:
        target = np.array([54, 112, 165], np.float32)
    print("target water", target)

    # Apply new r1 v-infill. dest (512, 1024) covers mosaic x 512-1536, y 1024-2048.
    # Protect the good horizontal seam at y=1024 (r0-r1, especially c1) and y=2048 (r1-r2 c0).
    apply_v_infill(
        mosaic,
        "seam_v_r1_c0-1.jpg",
        dest_xy=(512, 1024),
        seam_x=1024,
        band=200,   # cover full river + both banks
        fade=28,    # fade in textured buildings, not on water
        protect_h_ys=(1024, 2048),
        protect_h_band=260,  # keep the 270px good h-join intact
    )

    # Unify water along the whole vertical moat (all rows), and the west horizontal canals
    # Horizontal moats in c0 roughly sit in several y bands — detect via mask later.
    # also_h: (y0,y1,x_max) strips
    also_h = [
        (0, 3 * TILE, 1024),  # any water in the west 1024 px of r0-r2 (horizontal palace moats)
    ]
    unify_water(mosaic, target, corridor_x=1024, half_width=280, also_h=also_h)

    mild_c0_earth_harmony(mosaic)

    # Write only tiles that the moat/color pass may have changed
    touched = [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0), (2, 1),
        # r3_c0/c1 were just installed; water unify might touch west water if any
        (3, 0), (3, 1),
    ]
    write_tiles(mosaic, touched)


if __name__ == "__main__":
    main()
