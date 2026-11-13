#!/usr/bin/env python3
"""LAB statistical alignment of new bright/orange tiles toward old Wangfujing style.

Does NOT blindly Reinhard-transfer the whole tile (that would grey out yellow
palace roofs). Instead:
  - slightly lowers overall L on bright tiles
  - shifts low-chroma beige/tan ground toward the reference packed-earth grey
  - gently reduces chroma of very saturated yellow roofs
  - slightly darkens and cools mid values
Protected old tiles are never written.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

STY = Path("/workspace/beijing-pilot/tiles/styled")
OUT_PREV = Path("/workspace/beijing-pilot/tiles/mosaic/qa_color_E")
BAK = STY / "pre_color_E"
OUT_PREV.mkdir(parents=True, exist_ok=True)
BAK.mkdir(exist_ok=True)

REF_IDS = [
    "t1024_r0_c2", "t1024_r0_c3", "t1024_r1_c2", "t1024_r1_c3",
    "t1024_r2_c2", "t1024_r0_c4",
]
# tiles to color-match (new 12 + r0_c0 west-fill style)
NEW_IDS = [
    "t1024_r-2_c-1", "t1024_r-2_c0", "t1024_r-2_c1",
    "t1024_r-1_c-2", "t1024_r-1_c-1", "t1024_r-1_c0", "t1024_r-1_c1", "t1024_r-1_c2",
    "t1024_r0_c-2", "t1024_r0_c-1", "t1024_r0_c0",
    "t1024_r1_c-2", "t1024_r1_c-1",
]


def rgb_to_lab(arr):
    # arr float 0-255, return Lab
    s = arr / 255.0
    a = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = a @ M.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6/29
    def f(t):
        return np.where(t > delta**3, np.cbrt(t), t / (3*delta**2) + 4/29)
    fx, fy, fz = f(xyz[...,0]), f(xyz[...,1]), f(xyz[...,2])
    L = 116*fy - 16
    A = 500*(fx-fy)
    B = 200*(fy-fz)
    return np.stack([L,A,B], axis=-1)


def lab_to_rgb(lab):
    L,A,B = lab[...,0], lab[...,1], lab[...,2]
    fy = (L+16)/116
    fx = fy + A/500
    fz = fy - B/200
    delta=6/29
    def finv(t):
        return np.where(t > delta, t**3, 3*delta**2*(t-4/29))
    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1) * np.array([0.95047,1.0,1.08883])
    Minv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    lin = xyz @ Minv.T
    lin = np.clip(lin, 0, None)
    s = np.where(lin <= 0.0031308, 12.92*lin, 1.055*np.power(np.clip(lin,0,None), 1/2.4)-0.055)
    return np.clip(s*255, 0, 255)


def load(tid):
    im=Image.open(STY/f"{tid}.jpg").convert("RGB")
    return np.asarray(im).astype(np.float32)


def ground_mask(lab, rgb):
    """Beige / tan packed-earth and pale paving, not water, not yellow roofs, not green trees."""
    L,A,B = lab[...,0], lab[...,1], lab[...,2]
    C = np.sqrt(A*A+B*B)
    # beige: mid-high L, moderate yellow (B>0), low-mid chroma
    beige = (L > 45) & (L < 92) & (C < 28) & (B > -2) & (A > -6) & (A < 18)
    # also very pale dirt
    pale = (L > 70) & (C < 16) & (B > -4)
    # exclude strong water blue (B negative, A slightly negative)
    water = (B < -8) & (A < 0) & (L < 70)
    # exclude saturated yellow roofs (high C and high B)
    roof = (C > 32) & (B > 20) & (L > 40)
    # exclude green trees (A negative, B positive-ish, mid L)
    trees = (A < -8) & (C > 12)
    return (beige | pale) & ~water & ~roof & ~trees


def roof_mask(lab):
    L,A,B = lab[...,0], lab[...,1], lab[...,2]
    C = np.sqrt(A*A+B*B)
    return (C > 28) & (B > 16) & (A > -5) & (L > 35) & (L < 90)


def sample_stats(ids, kind="ground"):
    Ls, As, Bs = [], [], []
    for tid in ids:
        p = STY/f"{tid}.jpg"
        if not p.exists():
            continue
        rgb = load(tid)
        lab = rgb_to_lab(rgb)
        m = ground_mask(lab, rgb) if kind=="ground" else (roof_mask(lab) if kind=="roof" else np.ones(lab.shape[:2], bool))
        if m.sum() < 500:
            continue
        Ls.append(lab[...,0][m]); As.append(lab[...,1][m]); Bs.append(lab[...,2][m])
    if not Ls:
        return None
    L=np.concatenate(Ls); A=np.concatenate(As); B=np.concatenate(Bs)
    return {
        "L_mean": float(L.mean()), "L_std": float(L.std()),
        "A_mean": float(A.mean()), "A_std": float(A.std()),
        "B_mean": float(B.mean()), "B_std": float(B.std()),
        "n": int(L.size),
    }


def apply_tile(rgb, ref_g, ref_r, ref_all):
    lab = rgb_to_lab(rgb)
    L,A,B = lab[...,0].copy(), lab[...,1].copy(), lab[...,2].copy()
    C = np.sqrt(A*A+B*B)
    gm = ground_mask(lab, rgb)
    rm = roof_mask(lab)

    # 1) overall slight darken on bright tiles (if mean L is above ref overall)
    src_L_mean = L.mean()
    if ref_all and src_L_mean > ref_all["L_mean"] + 4:
        shift = (src_L_mean - ref_all["L_mean"]) * 0.45
        L = L - shift
        # don't crush shadows too much
        L = np.clip(L, 3, 100)

    # 2) ground: move beige toward ref ground mean, tighten a bit
    if ref_g is not None and gm.sum() > 200:
        # blend means
        blend = 0.65
        L[gm] = L[gm] + blend * (ref_g["L_mean"] - L[gm].mean())
        A[gm] = A[gm] + blend * (ref_g["A_mean"] - A[gm].mean())
        B[gm] = B[gm] + blend * (ref_g["B_mean"] - B[gm].mean())
        # slight std squeeze toward ref
        if L[gm].std() > 1:
            L[gm] = (L[gm]-L[gm].mean()) * (0.55*ref_g["L_std"]/L[gm].std() + 0.45) + L[gm].mean()

    # 3) yellow roofs: gently lower L and chroma if oversaturated vs ref roofs
    if ref_r is not None and rm.sum() > 200:
        src_c = np.sqrt(A[rm]**2+B[rm]**2).mean()
        ref_c = (ref_r["A_mean"]**2 + ref_r["B_mean"]**2)**0.5
        if src_c > ref_c + 2:
            scale = (0.55*ref_c + 0.45*src_c) / src_c
            A[rm] = A[rm] * scale
            B[rm] = B[rm] * scale
        if L[rm].mean() > ref_r["L_mean"] + 3:
            L[rm] = L[rm] - 0.4*(L[rm].mean()-ref_r["L_mean"])

    # 4) mild cool/darken of remaining mid-high beige not caught by mask
    leftover = (L > 62) & (C < 22) & (B > 0) & ~gm & ~rm
    if leftover.sum() > 100 and ref_g:
        L[leftover] = L[leftover] - 4
        B[leftover] = B[leftover] - 2

    out = lab_to_rgb(np.stack([np.clip(L,0,100), A, B], axis=-1))
    return out


def main():
    print("sampling reference stats...")
    ref_g = sample_stats(REF_IDS, "ground")
    ref_r = sample_stats(REF_IDS, "roof")
    ref_all = sample_stats(REF_IDS, "all")
    src_g = sample_stats(NEW_IDS, "ground")
    src_r = sample_stats(NEW_IDS, "roof")
    print("REF ground", ref_g)
    print("SRC ground", src_g)
    print("REF roof", ref_r)
    print("SRC roof", src_r)
    print("REF all", ref_all)
    report = {"ref_ground": ref_g, "src_ground": src_g, "ref_roof": ref_r, "src_roof": src_r, "ref_all": ref_all, "tiles": []}

    # preview contact sheet: before/after for first 6
    thumbs = []
    for tid in NEW_IDS:
        p = STY/f"{tid}.jpg"
        if not p.exists():
            continue
        rgb = load(tid)
        out = apply_tile(rgb, ref_g, ref_r, ref_all)
        # backup then write
        import shutil
        if not (BAK/f"{tid}.jpg").exists():
            shutil.copy2(p, BAK/f"{tid}.jpg")
        Image.fromarray(out.astype(np.uint8)).save(p, "JPEG", quality=93, subsampling=0)
        # side by side preview
        b = Image.fromarray(rgb.astype(np.uint8)).resize((256,256), Image.Resampling.LANCZOS)
        a = Image.fromarray(out.astype(np.uint8)).resize((256,256), Image.Resampling.LANCZOS)
        pair = Image.new("RGB",(512,256))
        pair.paste(b,(0,0)); pair.paste(a,(256,0))
        thumbs.append((tid, pair))
        print("updated", tid)
        report["tiles"].append(tid)

    # contact
    cols=3
    rows=(len(thumbs)+cols-1)//cols
    sheet=Image.new("RGB",(512*cols, 256*rows),(20,20,20))
    for i,(_,im) in enumerate(thumbs):
        sheet.paste(im, ((i%cols)*512, (i//cols)*256))
    sheet.save(OUT_PREV/"before_after_sheet.jpg","JPEG",quality=90)
    (OUT_PREV/"stats.json").write_text(json.dumps(report, indent=2))
    print("wrote preview", OUT_PREV/"before_after_sheet.jpg")


if __name__ == "__main__":
    main()
