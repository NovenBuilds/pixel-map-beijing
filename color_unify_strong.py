#!/usr/bin/env python3
"""Strong palette / lighting alignment toward old Wangfujing + Chang'an tiles.

Not a mild LAB nudge: aggressively recast beige courtyards to grey packed earth,
mute over-bright glazed-yellow roofs, compress highlights, and deepen shadows.
Protects water and keeps palace walls red (does not grey them).
Default is --preview (writes qa only). Pass --apply to write styled tiles.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/beijing-pilot")
from color_unify import (
    REF_IDS,
    rgb_to_lab,
    lab_to_rgb,
    ground_mask,
    roof_mask,
    sample_stats,
    load,
)

STY = Path("/workspace/beijing-pilot/tiles/styled")
BAK = STY / "pre_color_G"
OUT = Path("/workspace/beijing-pilot/tiles/mosaic/qa_color_G")
OUT.mkdir(parents=True, exist_ok=True)

# Wangfujing / St Joseph / Chang'an — style authority, never written
REF_IDS_G = [
    "t1024_r0_c2", "t1024_r0_c3", "t1024_r0_c4",
    "t1024_r1_c2", "t1024_r1_c3", "t1024_r1_c4",
    "t1024_r2_c2", "t1024_r2_c3", "t1024_r2_c4",
    "t1024_r3_c1", "t1024_r3_c2", "t1024_r3_c3", "t1024_r3_c4",
    "t1024_r-1_c2",
]

# Warm / bright / beige tiles to pull
TARGET_IDS = [
    "t1024_r-2_c-1", "t1024_r-2_c0", "t1024_r-2_c1",
    "t1024_r-1_c-2", "t1024_r-1_c-1", "t1024_r-1_c0", "t1024_r-1_c1",
    "t1024_r0_c-1", "t1024_r0_c0",
    "t1024_r1_c-2", "t1024_r1_c-1", "t1024_r1_c0",
    "t1024_r2_c0",
]
# already close to baseline; only a light ground/roof pass if still warm
LIGHT_IDS = ["t1024_r0_c-2", "t1024_r0_c1", "t1024_r1_c1", "t1024_r3_c0"]


def water_mask(lab, rgb):
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    R, G, Bl = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return ((Bl > 80) & (Bl > R + 15) & (Bl > G + 5) & ((Bl - R) > 20)) | (
        (B < -8) & (A < 0) & (L < 70)
    )


def red_wall_mask(lab, rm, wm):
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.sqrt(A * A + B * B)
    # red plaster: high A, moderate B, not a yellow roof, not water
    return (A > 18) & (A > B * 0.35) & (C > 20) & (L > 20) & (L < 75) & ~rm & ~wm


def tree_mask(lab):
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.sqrt(A * A + B * B)
    return (A < -8) & (C > 12) & (L > 15) & (L < 75)


def apply_strong(rgb, ref_g, ref_r, ref_all, strength=1.0):
    """strength=1 full G pass; ~0.4 = light pass for already-dark tiles.

    v2: ground is pulled to a cooler/darker packed-earth (L~50, B~5.2),
    below the sunlit Wangfujing ground mean, so open courtyards read grey
    rather than cream at overview scale.
    """
    lab = rgb_to_lab(rgb)
    L, A, B = lab[..., 0].copy(), lab[..., 1].copy(), lab[..., 2].copy()
    C = np.sqrt(A * A + B * B)
    wm = water_mask(lab, rgb)
    rm = roof_mask(lab)
    gm = ground_mask(lab, rgb)
    walls = red_wall_mask(lab, rm, wm)
    trees = tree_mask(lab)
    extra = (
        (L > 40) & (L < 96) & (B > 3) & (C < 34) & (A > -7) & (A < 22)
        & ~wm & ~rm & ~walls & ~trees
    )
    gm = gm | extra

    # darker/cooler than mean sunlit pavement
    gL_t, gA_t, gB_t = 50.0, 2.8, 5.2
    if ref_g is not None and gm.sum() > 200:
        blend = 0.95 * strength
        L[gm] = L[gm] + blend * (gL_t - L[gm].mean())
        A[gm] = A[gm] + blend * (gA_t - A[gm].mean())
        B[gm] = B[gm] + blend * (gB_t - B[gm].mean())
        B[gm] = B[gm] - 0.35 * strength * np.clip(B[gm] - gB_t - 0.8, 0, None)
        lo = gm & (L < L[gm].mean())
        L[lo] = L[lo] - 3.5 * strength

    hi = (L > 60) & ~wm
    L[hi] = L[hi] - 0.62 * strength * (L[hi] - 54)
    body = ~wm & ~rm
    if ref_all is not None and body.sum() > 500 and L[body].mean() > ref_all["L_mean"] + 1:
        shift = (L[body].mean() - ref_all["L_mean"]) * (0.62 * strength)
        L[body & (L > 28)] = L[body & (L > 28)] - shift
    sh = (L < 32) & ~wm & (L > 4)
    L[sh] = L[sh] - (4.2 * strength) * (1.0 - L[sh] / 32.0)

    marble = (L > 66) & (C < 20) & ~wm & ~rm & ~trees
    if marble.sum() > 80:
        L[marble] = L[marble] - 8 * strength
        B[marble] = B[marble] - 4 * strength
        A[marble] = A[marble] - 1.5 * strength

    if ref_r is not None and rm.sum() > 200:
        src_c = np.sqrt(A[rm] ** 2 + B[rm] ** 2).mean()
        ref_c = (ref_r["A_mean"] ** 2 + ref_r["B_mean"] ** 2) ** 0.5
        if src_c > ref_c:
            scale = (0.65 * ref_c + 0.35 * src_c) / src_c
            scale = 1 - strength * (1 - scale)
            A[rm] = A[rm] * scale
            B[rm] = B[rm] * scale
        if L[rm].mean() > ref_r["L_mean"]:
            L[rm] = L[rm] - 0.75 * strength * (L[rm].mean() - ref_r["L_mean"])

    if walls.sum() > 200:
        L[walls] = L[walls] - 2.5 * strength
        Cw = np.sqrt(A[walls] ** 2 + B[walls] ** 2)
        too = Cw > 55
        if too.any():
            s = np.ones_like(Cw)
            s[too] = 55.0 / (Cw[too] + 1e-6)
            s = 1 - 0.4 * strength * (1 - s)
            A[walls] = A[walls] * s
            B[walls] = B[walls] * s

    lush = trees & (L > 46)
    if lush.sum() > 100:
        L[lush] = L[lush] - 5 * strength
        A[lush] = A[lush] * (1 - 0.15 * strength)
        B[lush] = B[lush] - 1.5 * strength

    L = np.clip(L, 1, 100)
    out = lab_to_rgb(np.stack([L, A, B], axis=-1))
    out = np.where(wm[..., None], rgb, out)
    return out



def stitch_preview(ids, suffix, size=2560):
    """Fast mosaic of current styled (or provided dict of arrays). ids unused if arrs is None — always from disk."""
    RMIN, RMAX, CMIN, CMAX = -2, 3, -2, 4
    TILE = 1024
    W, H = (CMAX - CMIN + 1) * TILE, (RMAX - RMIN + 1) * TILE
    mosaic = Image.new("RGB", (W, H), (30, 30, 30))
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            p = STY / f"t1024_r{R}_c{C}.jpg"
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB")
            if im.size != (TILE, TILE):
                im = im.resize((TILE, TILE), Image.Resampling.LANCZOS)
            mosaic.paste(im, ((C - CMIN) * TILE, (R - RMIN) * TILE))
    scale = size / max(W, H)
    prev = mosaic.resize((int(W * scale), int(H * scale)), Image.Resampling.LANCZOS)
    path = OUT / f"overview_{size}{suffix}.jpg"
    prev.save(path, "JPEG", quality=90)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--preview", action="store_true", default=True)
    args = ap.parse_args()

    print("sampling refs...")
    ref_g = sample_stats(REF_IDS_G, "ground")
    ref_r = sample_stats(REF_IDS_G, "roof")
    ref_all = sample_stats(REF_IDS_G, "all")
    print("REF ground", ref_g)
    print("REF roof", ref_r)
    print("REF all", ref_all)
    report = {"ref_ground": ref_g, "ref_roof": ref_r, "ref_all": ref_all, "tiles": []}

    # apply in memory first, contact sheet
    thumbs = []
    outputs = {}
    jobs = [(tid, 1.0) for tid in TARGET_IDS] + [(tid, 0.40) for tid in LIGHT_IDS]
    for tid, st in jobs:
        p = STY / f"{tid}.jpg"
        if not p.exists():
            continue
        rgb = load(tid)
        out = apply_strong(rgb, ref_g, ref_r, ref_all, strength=st)
        outputs[tid] = out
        b = Image.fromarray(rgb.astype(np.uint8)).resize((256, 256), Image.Resampling.LANCZOS)
        a = Image.fromarray(out.astype(np.uint8)).resize((256, 256), Image.Resampling.LANCZOS)
        pair = Image.new("RGB", (512, 256))
        pair.paste(b, (0, 0))
        pair.paste(a, (256, 0))
        thumbs.append((tid, st, pair))
        # after stats
        lab_o = rgb_to_lab(out)
        lab_i = rgb_to_lab(rgb)
        report["tiles"].append({
            "id": tid, "strength": st,
            "L_before": float(lab_i[..., 0].mean()), "L_after": float(lab_o[..., 0].mean()),
            "B_before": float(lab_i[..., 2].mean()), "B_after": float(lab_o[..., 2].mean()),
        })
        print(f"{tid:22s} st={st:.2f} L {lab_i[...,0].mean():.1f}->{lab_o[...,0].mean():.1f}  B {lab_i[...,2].mean():.1f}->{lab_o[...,2].mean():.1f}")

    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (512 * cols, 256 * rows + 8), (20, 20, 20))
    for i, (_, _, im) in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 512, (i // cols) * 256))
    sheet.save(OUT / "before_after_strong.jpg", "JPEG", quality=90)

    # preview mosaic: write outputs to a temp overlay stitch without touching styled
    RMIN, RMAX, CMIN, CMAX = -2, 3, -2, 4
    TILE = 1024
    W, H = (CMAX - CMIN + 1) * TILE, (RMAX - RMIN + 1) * TILE
    mosaic = Image.new("RGB", (W, H), (30, 30, 30))
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            tid = f"t1024_r{R}_c{C}"
            p = STY / f"{tid}.jpg"
            if tid in outputs:
                im = Image.fromarray(outputs[tid].astype(np.uint8))
            elif p.exists():
                im = Image.open(p).convert("RGB")
                if im.size != (TILE, TILE):
                    im = im.resize((TILE, TILE), Image.Resampling.LANCZOS)
            else:
                continue
            if im.size != (TILE, TILE):
                im = im.resize((TILE, TILE), Image.Resampling.LANCZOS)
            mosaic.paste(im, ((C - CMIN) * TILE, (R - RMIN) * TILE))
    scale = 2560 / max(W, H)
    prev = mosaic.resize((int(W * scale), int(H * scale)), Image.Resampling.LANCZOS)
    prev.save(OUT / "overview_2560_strong_preview.jpg", "JPEG", quality=90)
    print("wrote previews", OUT)

    if args.apply:
        BAK.mkdir(exist_ok=True)
        for tid, arr in outputs.items():
            src = STY / f"{tid}.jpg"
            if not (BAK / f"{tid}.jpg").exists():
                shutil.copy2(src, BAK / f"{tid}.jpg")
            Image.fromarray(arr.astype(np.uint8)).save(src, "JPEG", quality=93, subsampling=0)
            print("wrote", tid)
        print("applied to styled, backups in", BAK)

    (OUT / "stats.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
