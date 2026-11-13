#!/usr/bin/env python3
"""E typological redraws: accept/composite/warp, smear fix, color unify,
narrow halls infill, narrow seams. Does not start F/C."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/beijing-pilot")
from color_unify import REF_IDS, apply_tile, sample_stats

ROOT = Path("/workspace/beijing-pilot")
STY = ROOT / "tiles" / "styled"
CAND = STY / "candidates"
BAK = STY / "pre_redraw_E"
INF_DIR = ROOT / "tiles" / "mosaic" / "seams_axis" / "inpaints"
OUT = ROOT / "tiles" / "mosaic" / "qa_color_E"
TILE = 1024

INFILL_SRC = Path(
    "/cursor/stores/user/grok-bot/agents/0dca2891-24ca-4c88-8740-c3a04a2e72ed/assets/"
    "30ec4b3eecc0b4003ad9a612d0f7482d265a2027dd3d34dc94907dcaafceb6e9.jpg"
)


def load(p):
    im = Image.open(p).convert("RGB")
    if im.size != (TILE, TILE):
        im = im.resize((TILE, TILE), Image.Resampling.LANCZOS)
    return np.asarray(im).astype(np.float32)


def save_tile(r, c, arr):
    p = STY / f"t1024_r{r}_c{c}.jpg"
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
        p, "JPEG", quality=93, subsampling=0
    )
    print("wrote", p.name)


def save_jpg(arr, p, q=92):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
        p, "JPEG", quality=q, subsampling=0
    )


def water_mask(arr):
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    return (B > 80) & (B > R + 15) & (B > G + 5) & ((B - R) > 20)


def moat_edges(arr, degree=1):
    w = water_mask(arr)
    L = np.full(TILE, np.nan)
    Rt = np.full(TILE, np.nan)
    for y in range(TILE):
        xs = np.where(w[y])[0]
        if len(xs) < 6:
            continue
        runs = []
        s = prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
            else:
                runs.append((s, prev))
                s = prev = x
        runs.append((s, prev))
        s, e = max(runs, key=lambda t: t[1] - t[0])
        if e - s + 1 >= 6:
            L[y], Rt[y] = s, e

    def fill(a):
        n = np.isnan(a)
        if n.all():
            return a
        idx = np.arange(len(a))
        a = a.copy()
        a[n] = np.interp(idx[n], idx[~n], a[~n])
        coef = np.polyfit(idx, a, degree)
        return np.polyval(coef, idx)

    return fill(L), fill(Rt)


def warp_moat(src, L_src, R_src, L_dst, R_dst):
    xs = np.arange(TILE, dtype=np.float32)
    src_x = np.zeros((TILE, TILE), np.float32)
    for y in range(TILE):
        Ls, Rs, Ld, Rd = L_src[y], R_src[y], L_dst[y], R_dst[y]
        if not np.isfinite([Ls, Rs, Ld, Rd]).all() or Rs <= Ls + 4 or Rd <= Ld + 4:
            src_x[y] = xs
            continue
        src_x[y] = np.interp(xs, [0, Ld, Rd, TILE - 1], [0, Ls, Rs, TILE - 1])
    src_x = np.clip(src_x, 0, TILE - 1)
    x0 = np.floor(src_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, TILE - 1)
    t = (src_x - x0)[..., None]
    yidx = np.arange(TILE)[:, None]
    return (1 - t) * src[yidx, x0] + t * src[yidx, x1]


def cosine_weight(d, band):
    t = np.clip(d / max(band, 1), 0, 1)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def overlap_vertical(left, right, band, cap=0.5):
    L, R = left.copy(), right.copy()
    for x in range(TILE - band, TILE):
        d = TILE - 1 - x
        w = cosine_weight(d, band) * cap
        src_x = TILE - 1 - x
        L[:, x] = (1 - w) * L[:, x] + w * R[:, src_x]
    for x in range(0, band):
        d = x
        w = cosine_weight(d, band) * cap
        src_x = TILE - 1 - x
        R[:, x] = (1 - w) * R[:, x] + w * L[:, src_x]
    return L, R


def overlap_horizontal(top, bot, band, cap=0.5):
    T, B = top.copy(), bot.copy()
    for y in range(TILE - band, TILE):
        d = TILE - 1 - y
        w = cosine_weight(d, band) * cap
        src_y = TILE - 1 - y
        T[y, :] = (1 - w) * T[y, :] + w * B[src_y, :]
    for y in range(0, band):
        d = y
        w = cosine_weight(d, band) * cap
        src_y = TILE - 1 - y
        B[y, :] = (1 - w) * B[y, :] + w * T[src_y, :]
    return T, B


def apply_v_fullbleed(left, right, inf, band=72, cap=0.65):
    L, R = left.copy(), right.copy()
    for x in range(TILE - band, TILE):
        d = TILE - 1 - x
        w = cosine_weight(d, band) * cap
        ix = x - 512  # left x=1023 -> inf 511; x=1024-band -> 512-band
        if 0 <= ix < TILE:
            L[:, x] = (1 - w) * L[:, x] + w * inf[:, ix]
    for x in range(0, band):
        d = x
        w = cosine_weight(d, band) * cap
        ix = 512 + x
        if 0 <= ix < TILE:
            R[:, x] = (1 - w) * R[:, x] + w * inf[:, ix]
    return L, R


def backup(ids):
    BAK.mkdir(parents=True, exist_ok=True)
    for tid in ids:
        src = STY / f"{tid}.jpg"
        dst = BAK / f"{tid}.jpg"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print("backup", tid)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    INF_DIR.mkdir(parents=True, exist_ok=True)

    ids = [
        "t1024_r-1_c-2",
        "t1024_r0_c-2",
        "t1024_r-2_c-1",
        "t1024_r0_c-1",
        "t1024_r0_c0",
        "t1024_r-1_c-1",
        "t1024_r1_c-2",
    ]
    backup(ids)

    ref_g = sample_stats(REF_IDS, "ground")
    ref_r = sample_stats(REF_IDS, "roof")
    ref_all = sample_stats(REF_IDS, "all")

    # ----- r0_c-2: warp v3 moat to current edges, then color unify -----
    cur_c2 = load(STY / "t1024_r0_c-2.jpg")
    v3 = load(CAND / "t1024_r0_c-2_v3_styled.jpg")
    Ls, Rs = moat_edges(v3, degree=1)
    Ld, Rd = moat_edges(cur_c2, degree=1)
    print(
        "warp targets top L/R",
        f"{Ld[0]:.1f}/{Rd[0]:.1f}",
        "bot",
        f"{Ld[-1]:.1f}/{Rd[-1]:.1f}",
        "src top",
        f"{Ls[0]:.1f}/{Rs[0]:.1f}",
    )
    warped = warp_moat(v3, Ls, Rs, Ld, Rd)
    r0c2 = apply_tile(warped, ref_g, ref_r, ref_all)
    save_tile(0, -2, r0c2)
    save_jpg(r0c2, CAND / "t1024_r0_c-2_v3_warped_cu.jpg")

    # ----- r-1_c-2: hutongs from v4 left, current right, then color unify -----
    cur_n = load(STY / "t1024_r-1_c-2.jpg")
    v4 = load(CAND / "t1024_r-1_c-2_v4_hutongs.jpg")
    w = np.zeros((TILE, TILE, 1), np.float32)
    for x in range(TILE):
        if x <= 500:
            w[:, x, 0] = 0.0
        elif x >= 620:
            w[:, x, 0] = 1.0
        else:
            t = (x - 500) / (620 - 500)
            w[:, x, 0] = 0.5 * (1 - np.cos(np.pi * t))
    comp = (1 - w) * v4 + w * cur_n
    r_1c2 = apply_tile(comp, ref_g, ref_r, ref_all)
    save_tile(-1, -2, r_1c2)
    save_jpg(r_1c2, CAND / "t1024_r-1_c-2_v4_comp_cu.jpg")

    # ----- r-2_c-1: restore clean bottom from color-matched pre_seam_b -----
    # already color-unified; do not run apply_tile again
    from color_unify import lab_to_rgb, rgb_to_lab

    cur_nw = load(STY / "t1024_r-2_c-1.jpg")
    pre = load(STY / "pre_seam_b/t1024_r-2_c-1.jpg")
    lab_s = rgb_to_lab(pre)
    lab_r = rgb_to_lab(cur_nw)
    for c in range(3):
        mu_s, sd_s = lab_s[..., c].mean(), lab_s[..., c].std() + 1e-6
        mu_r, sd_r = lab_r[..., c].mean(), lab_r[..., c].std() + 1e-6
        lab_s[..., c] = (lab_s[..., c] - mu_s) * (sd_r / sd_s) + mu_r
    pre_m = lab_to_rgb(lab_s)
    yw = np.zeros((TILE, 1, 1), np.float32)
    for y in range(TILE):
        if y <= 800:
            yw[y] = 0
        elif y >= 920:
            yw[y] = 1
        else:
            t = (y - 800) / (920 - 800)
            yw[y] = 0.5 * (1 - np.cos(np.pi * t))
    r_2c1 = (1 - yw) * cur_nw + yw * pre_m
    save_tile(-2, -1, r_2c1)
    save_jpg(r_2c1, CAND / "t1024_r-2_c-1_smearfix.jpg")

    # ----- halls infill on r0_c-1 | r0_c0 -----
    inf_dest = CAND / "infill_v_r0_c-1-0_fullbleed_E.jpg"
    if INFILL_SRC.exists():
        shutil.copy2(INFILL_SRC, inf_dest)
        shutil.copy2(INFILL_SRC, INF_DIR / "infill_v_r0_c-1-0_fullbleed_E.jpg")
    inf = load(inf_dest)
    left = load(STY / "t1024_r0_c-1.jpg")
    right = load(STY / "t1024_r0_c0.jpg")
    # before crop for QA
    before = np.concatenate([left[:, TILE - 80 :], right[:, :80]], axis=1)
    left2, right2 = apply_v_fullbleed(left, right, inf, band=72, cap=0.65)
    after = np.concatenate([left2[:, TILE - 80 :], right2[:, :80]], axis=1)
    save_jpg(np.concatenate([before, after], axis=1), OUT / "infill_apply_before_after.jpg")
    save_tile(0, -1, left2)
    save_tile(0, 0, right2)

    # ----- narrow seams on rewritten neighbors -----
    tiles = {
        (0, -2): load(STY / "t1024_r0_c-2.jpg"),
        (-1, -2): load(STY / "t1024_r-1_c-2.jpg"),
        (-2, -1): load(STY / "t1024_r-2_c-1.jpg"),
        (0, -1): load(STY / "t1024_r0_c-1.jpg"),
        (0, 0): load(STY / "t1024_r0_c0.jpg"),
        (-1, -1): load(STY / "t1024_r-1_c-1.jpg"),
        (1, -2): load(STY / "t1024_r1_c-2.jpg"),
        (-1, 0): load(STY / "t1024_r-1_c0.jpg") if (STY / "t1024_r-1_c0.jpg").exists() else None,
    }

    def get(rc):
        return tiles[rc]

    def put(rc, arr):
        tiles[rc] = arr

    # vertical
    for r, c0, c1, band in [
        (-1, -2, -1, 36),
        (0, -2, -1, 36),
        # halls already got infill; very light extra fade only
        (0, -1, 0, 24),
    ]:
        L, R = overlap_vertical(get((r, c0)), get((r, c1)), band, cap=0.5)
        put((r, c0), L)
        put((r, c1), R)
        print(f"  seam v r{r} c{c0}|{c1} band {band}")

    # horizontal
    for r0, r1, c, band in [
        (-1, 0, -2, 36),
        (0, 1, -2, 36),
        (-2, -1, -1, 32),
    ]:
        T, B = overlap_horizontal(get((r0, c)), get((r1, c)), band, cap=0.5)
        put((r0, c), T)
        put((r1, c), B)
        print(f"  seam h r{r0}|{r1} c{c} band {band}")

    # write all touched
    # r-1_c0 was only loaded, not seamed — skip if None or unchanged
    write_set = {
        (0, -2),
        (-1, -2),
        (-2, -1),
        (0, -1),
        (0, 0),
        (-1, -1),
        (1, -2),
    }
    for rc in write_set:
        r, c = rc
        save_tile(r, c, tiles[rc])

    print("apply_E_redraws done")


if __name__ == "__main__":
    main()
