#!/usr/bin/env python3
"""One-shot global whitebox for Forbidden City + Jingshan + west half of the old pilot.

Same UTM origin / azimuth -15 / elevation -45 / 0.29296875 m per pixel as grid_axis.json.
The moat and palace wall are drawn once; tiles are slices of this single image.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/workspace/beijing-pilot")
sys.path.insert(0, str(ROOT))
import expand_axis as ea
import render_color_whitebox as rb
import build_tile_grid as tg

MPP = 300.0 / 1024.0  # 0.29296875, locked
RMIN, RMAX = -2, 3
CMIN, CMAX = -2, 4
TILE = 1024
N_C = CMAX - CMIN + 1
N_R = RMAX - RMIN + 1
W, H = N_C * TILE, N_R * TILE

OUT = ROOT / "tiles" / "whiteboxes" / "global"
SLICES = OUT / "slices"
OUT.mkdir(parents=True, exist_ok=True)
SLICES.mkdir(parents=True, exist_ok=True)


def mosaic_rot_bounds():
    x0 = ea.MINX + CMIN * ea.TILE_WX
    x1 = ea.MINX + (CMAX + 1) * ea.TILE_WX
    y1 = ea.MAXY - RMIN * ea.TILE_WY
    y0 = ea.MAXY - (RMAX + 1) * ea.TILE_WY
    return x0, x1, y0, y1


class GlobalProjector:
    """Same isometric as rb.Projector, but canvas is the full mosaic and mpp is locked."""

    def __init__(self):
        x0, x1, y0, y1 = mosaic_rot_bounds()
        self.cx = (x0 + x1) / 2.0
        self.cy = (y0 + y1) / 2.0
        self.lon0, self.lat0 = ea.rot_to_lonlat(self.cx, self.cy)
        self.az = math.radians(rb.AZIMUTH_DEG)
        self.elev = math.radians(rb.ELEVATION_DEG)
        self.mpp = MPP
        self.to_utm = tg.TO_UTM
        # EN origin at the SAME UTM origin as the grid (pilot SW), not the camera.
        # We work in grid EN / rot, then place rot relative to mosaic center.
        self.cos_a, self.sin_a = math.cos(self.az), math.sin(self.az)
        self.s = -math.sin(self.elev)
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def en(self, lon, lat):
        e, n = tg.lonlat_to_en(lon, lat)
        return e, n

    def rotate(self, e, n):
        return e * self.cos_a - n * self.sin_a, e * self.sin_a + n * self.cos_a

    def project_xyz(self, e, n, z=0.0):
        # rot relative to grid origin, then to mosaic center
        rx, ry = self.rotate(e, n)
        # screen: x increases with rot_x, y increases downward as rot_y decreases
        sx = rx - self.x0
        # in per-tile projector: sy = s*y + s*z, py = H/2 - sy/mpp
        # at mosaic: py = 0 at north (rot_y = y1)
        # tile R=RMIN, top of mosaic = rot_y = y1
        # py = (y1 - (ry + z)) / mpp?  Wait, elevation stretch.
        # Per-tile: sy = s * y + s * z, with s = -sin(elev) = sin(45)=0.707...
        # elev is -45, sin(-45)<0, s = -sin(elev) = -(-0.707)=0.707
        # py = H/2 - sy/mpp = H/2 - s*(y+z)/mpp
        # At tile center (cy_tile), py=H/2 when z=0, y=0 in tile-local EN rot.
        # Tile-local y = ry - cy_tile.
        # Global: we want py=0 at the top (ry=y1, z=0):
        # For consistency with tile slices:
        # tile (R,C) center is at rot (cx_t, cy_t)
        # local y = ry - cy_t
        # local py = 512 - s*(ry - cy_t + z)/mpp
        # global py = (R-RMIN)*1024 + local py
        # cy_t = MAXY - (2*R+1)*SY = MAXY - (R+0.5)*TILE_WY
        # (R-RMIN)*1024 + 512 - s*(ry - cy_t + z)/mpp
        # 1024 = TILE_WY * s / mpp ? 
        # TILE_WY = SY*2, SY = (512 * mpp) / sin(abs(el)) = 512 * mpp / s
        # TILE_WY = 1024 * mpp / s
        # so 1024 = TILE_WY * s / mpp. Yes!
        # cy_t = y1 - (R-RMIN+0.5)*TILE_WY   because y1 = MAXY - RMIN*TILE_WY
        # (R-RMIN)*1024 + 512 = (R-RMIN)*TILE_WY*s/mpp + (TILE_WY*s/mpp)/2
        #                 = (y1 - cy_t) * s / mpp
        # global py = (y1 - cy_t)*s/mpp - s*(ry - cy_t + z)/mpp
        #           = s*(y1 - ry - z)/mpp
        # global px = (C-CMIN)*1024 + 512 + (rx - cx_t)/mpp
        # cx_t = x0 + (C-CMIN+0.5)*TILE_WX, TILE_WX = 300 = 1024*mpp
        # (C-CMIN)*1024 + 512 = (cx_t - x0)/mpp
        # global px = (cx_t - x0)/mpp + (rx - cx_t)/mpp = (rx - x0)/mpp
        px = (rx - self.x0) / self.mpp
        py = self.s * (self.y1 - ry - z) / self.mpp
        # depth for painter's algorithm: same as per-tile (s*y - s*z) in local rot y
        # use global ry so order is consistent
        depth = self.s * ry - self.s * z
        return px, py, depth

    def shade_side(self, p0, p1):
        return rb.Projector.shade_side(self, p0, p1)


def render_global(buildings, water, roads):
    proj = GlobalProjector()
    # Bind so rb.build_faces / add_window_grid / draw_poly use our projector methods
    # build_faces uses proj.project_xyz, proj.en, proj.shade_side, typ colors
    print(f"global canvas {W}x{H}  camera lon/lat={proj.lon0:.6f},{proj.lat0:.6f}  rot center=({proj.cx:.1f},{proj.cy:.1f})")
    print(f"rot bounds x=[{proj.x0:.1f},{proj.x1:.1f}] y=[{proj.y0:.1f},{proj.y1:.1f}] mpp={proj.mpp}")

    img = Image.new("RGB", (W, H), (198, 176, 132))
    draw = ImageDraw.Draw(img, "RGBA")

    # water
    for w in water:
        coords = w["coords"]
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        pts = [(*proj.en(lon, lat), 0.0) for lon, lat in coords]
        rb.draw_poly(draw, proj, pts, (62, 118, 148), outline=(48, 96, 122))

    # roads
    for r in roads:
        hw = r["highway"]
        major = hw in ("trunk", "primary", "secondary")
        pedestrian = hw in ("pedestrian", "footway", "living_street")
        width = 7 if major else (6 if pedestrian else 3)
        color = (92, 90, 86) if major else ((168, 150, 126) if pedestrian else (110, 108, 104))
        xy = [proj.project_xyz(*proj.en(lon, lat), 0.0)[:2] for lon, lat in r["coords"]]
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=width, joint="curve")

    # buildings in padded mosaic lon/lat
    nearby = []
    for b in buildings:
        if 116.378 <= b["lon"] <= 116.415 and 39.898 <= b["lat"] <= 39.932:
            nearby.append(b)
    print(f"buildings drawn: {len(nearby)}")

    shadows = []
    for b in nearby:
        en = [proj.en(lon, lat) for lon, lat in b["coords"]]
        sp = rb.shadow_poly(en, b["height"], proj)
        if not sp:
            continue
        depths = [proj.project_xyz(*p)[2] for p in sp]
        shadows.append((min(depths) - 50, sp))
    shadows.sort(key=lambda x: x[0])
    for _, sp in shadows:
        xy = [proj.project_xyz(*p)[:2] for p in sp]
        if len(xy) >= 3:
            draw.polygon(xy, fill=(58, 50, 42, 110))

    faces = []
    for b in nearby:
        fs, en, h = rb.build_faces(b, proj)
        faces.extend(fs)
    faces.sort(key=lambda f: f["depth"])
    # Temporarily set rb size so window grids still work (they only check typ)
    old_w, old_h = rb.WIDTH_PX, rb.HEIGHT_PX
    rb.WIDTH_PX, rb.HEIGHT_PX = W, H
    try:
        for f in faces:
            outline = (40, 36, 32) if f["kind"] == "roof" else None
            rb.draw_poly(draw, proj, f["pts"], f["color"], outline=outline)
            if f["kind"] == "wall":
                rb.add_window_grid(img, proj, f)
    finally:
        rb.WIDTH_PX, rb.HEIGHT_PX = old_w, old_h

    # trees along moat, slightly denser so slices still show a line
    tree_draw = ImageDraw.Draw(img)
    rng = np.random.RandomState(42)
    for w in water:
        coords = w["coords"]
        step = max(1, len(coords) // 80)
        for i, (lon, lat) in enumerate(coords[::step]):
            e, n = proj.en(lon, lat)
            for sign in (8.0, -8.0):
                te, tn = e + sign, n + (rng.rand() - 0.5) * 6
                h = 7.0 + rng.rand() * 3
                p0 = proj.project_xyz(te, tn, 0)[:2]
                p1 = proj.project_xyz(te, tn, h * 0.4)[:2]
                crown = [proj.project_xyz(te + math.cos(a * math.pi / 3) * 2.2,
                                          tn + math.sin(a * math.pi / 3) * 2.2, h)[:2]
                         for a in range(6)]
                tree_draw.line([p0, p1], fill=(90, 70, 48), width=2)
                if len(crown) >= 3:
                    tree_draw.polygon(crown, fill=(46, 92, 58), outline=(32, 70, 44))

    raw = OUT / "global_whitebox.png"
    img.save(raw, "PNG")
    print("wrote", raw, img.size)

    # jpeg overview + labeled grid
    scale = 2560 / max(W, H)
    prev = img.resize((int(W * scale), int(H * scale)), Image.Resampling.LANCZOS).convert("RGB")
    prev.save(OUT / "global_whitebox_2560.jpg", "JPEG", quality=90)
    lab = prev.copy()
    d = ImageDraw.Draw(lab)
    sx, sy = prev.size[0] / W, prev.size[1] / H
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            x0, y0 = int((C - CMIN) * TILE * sx), int((R - RMIN) * TILE * sy)
            x1, y1 = int((C - CMIN + 1) * TILE * sx), int((R - RMIN + 1) * TILE * sy)
            d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(40, 90, 180), width=1)
            d.text((x0 + 4, y0 + 4), f"R{R}C{C}", fill=(20, 40, 80), font=font)
    lab.save(OUT / "global_whitebox_2560_grid.jpg", "JPEG", quality=90)

    # slices
    manifest = []
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            box = ((C - CMIN) * TILE, (R - RMIN) * TILE,
                   (C - CMIN + 1) * TILE, (R - RMIN + 1) * TILE)
            sl = img.crop(box)
            path = SLICES / f"t1024_r{R}_c{C}.png"
            sl.save(path, "PNG")
            manifest.append({"id": f"t1024_r{R}_c{C}", "R": R, "C": C, "path": str(path)})
    (OUT / "slices.json").write_text(json.dumps({"mpp": MPP, "W": W, "H": H,
                                                 "rows": list(range(RMIN, RMAX + 1)),
                                                 "cols": list(range(CMIN, CMAX + 1)),
                                                 "camera_center": {"lon": proj.lon0, "lat": proj.lat0,
                                                                   "rot_x": proj.cx, "rot_y": proj.cy},
                                                 "slices": manifest}, indent=2))
    print("wrote", len(manifest), "slices to", SLICES)
    return img, proj


def main():
    ea.update_calibration()
    ea.patch_zone()
    print("parsing OSM...")
    nodes, ways, rels = ea.parse_many(ea.OSM_PATHS)
    buildings, water, roads = ea.assemble_plus(nodes, ways, rels)
    render_global(buildings, water, roads)
    print("done")


if __name__ == "__main__":
    main()
