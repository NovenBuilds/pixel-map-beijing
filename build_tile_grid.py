#!/usr/bin/env python3
"""Plan 512 / 1024 isometric grid and render per-tile colored whiteboxes."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer

ROOT = Path("/workspace/beijing-pilot")
sys.path.insert(0, str(ROOT))
import render_color_whitebox as rb

WEST, SOUTH, EAST, NORTH = 116.39480, 39.90620, 116.40780, 39.91800
AZ = math.radians(rb.AZIMUTH_DEG)
EL = math.radians(rb.ELEVATION_DEG)
VIEW_H = rb.VIEW_HEIGHT_M
VIEWPORT = rb.WIDTH_PX
MPP = VIEW_H / VIEWPORT  # 0.29296875 m / px
CELL = 512
SX = CELL * MPP  # 150 m of screen / rotated-X
SY_ROT = (CELL * MPP) / math.sin(abs(EL))  # 212.132... m of rotated-Y
MARGIN_M = 40.0

LANDMARKS = {
    "donghuamen": (116.39580, 39.91370),
    "nanchizi": (116.39810, 39.91035),
    "wangfujing-north": (116.40540, 39.91490),
    "wangfujing-south": (116.40490, 39.90840),
    "st-joseph": (116.40590, 39.91585),  # 王府井天主堂 / 东堂, approx WGS84
    "changan-east": (116.40490, 39.90740),
}

TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
TO_LL = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)
OX, OY = TO_UTM.transform(WEST, SOUTH)

OUT_TILES = ROOT / "tiles"
WB_DIR = OUT_TILES / "whiteboxes"
OUT_TILES.mkdir(parents=True, exist_ok=True)
WB_DIR.mkdir(parents=True, exist_ok=True)


def lonlat_to_en(lon, lat):
    e, n = TO_UTM.transform(lon, lat)
    return e - OX, n - OY


def en_to_lonlat(e, n):
    lon, lat = TO_LL.transform(OX + e, OY + n)
    return lon, lat


def en_to_rot(e, n):
    dx = e * math.cos(AZ) - n * math.sin(AZ)
    dy = e * math.sin(AZ) + n * math.cos(AZ)
    return dx, dy


def rot_to_en(dx, dy):
    de = dx * math.cos(AZ) + dy * math.sin(AZ)
    dn = -dx * math.sin(AZ) + dy * math.cos(AZ)
    return de, dn


def lonlat_to_rot(lon, lat):
    return en_to_rot(*lonlat_to_en(lon, lat))


def rot_to_lonlat(dx, dy):
    return en_to_lonlat(*rot_to_en(dx, dy))


def plan_grid():
    corners = [(WEST, SOUTH), (EAST, SOUTH), (EAST, NORTH), (WEST, NORTH)]
    rots = [lonlat_to_rot(lon, lat) for lon, lat in corners]
    xs = [r[0] for r in rots]
    ys = [r[1] for r in rots]
    minx, maxx = min(xs) - MARGIN_M, max(xs) + MARGIN_M
    miny, maxy = min(ys) - MARGIN_M, max(ys) + MARGIN_M
    ncols = math.ceil((maxx - minx) / SX)
    nrows = math.ceil((maxy - miny) / SY_ROT)
    if ncols % 2:
        ncols += 1
    if nrows % 2:
        nrows += 1
    # keep origin at the unpadded min, but expand to even cell counts
    # (do not recenter; extra cells hang off the east/south edge)
    cells_512 = []
    for r in range(nrows):
        for c in range(ncols):
            cx = minx + (c + 0.5) * SX
            cy = maxy - (r + 0.5) * SY_ROT
            lon, lat = rot_to_lonlat(cx, cy)
            cells_512.append({
                "row": r, "col": c,
                "lon": lon, "lat": lat,
                "rot_x": cx, "rot_y": cy,
                "px": c * CELL, "py": r * CELL,
            })
    tiles_1024 = []
    nR, nC = nrows // 2, ncols // 2
    for R in range(nR):
        for C in range(nC):
            cx = minx + (2 * C + 1) * SX
            cy = maxy - (2 * R + 1) * SY_ROT
            lon, lat = rot_to_lonlat(cx, cy)
            quads = [(2 * R, 2 * C), (2 * R, 2 * C + 1), (2 * R + 1, 2 * C), (2 * R + 1, 2 * C + 1)]
            tiles_1024.append({
                "R": R, "C": C,
                "id": f"t1024_r{R}_c{C}",
                "lon": lon, "lat": lat,
                "rot_x": cx, "rot_y": cy,
                "px": C * 1024, "py": R * 1024,
                "quads": [{"row": q[0], "col": q[1]} for q in quads],
            })
    # landmark cell lookup
    landmark_cells = {}
    for name, (lon, lat) in LANDMARKS.items():
        rx, ry = lonlat_to_rot(lon, lat)
        c = int((rx - minx) // SX)
        r = int((maxy - ry) // SY_ROT)
        landmark_cells[name] = {"row": r, "col": c, "lon": lon, "lat": lat,
                                "in_grid": 0 <= r < nrows and 0 <= c < ncols}
    # priority: continuous band covering 东华门 → 南池子 → 王府井
    rows_needed, cols_needed = set(), set()
    for name in ("donghuamen", "nanchizi", "wangfujing-north", "wangfujing-south", "st-joseph"):
        info = landmark_cells[name]
        if info["in_grid"]:
            rows_needed.update(range(max(0, info["row"] - 1), min(nrows, info["row"] + 2)))
            cols_needed.update(range(max(0, info["col"] - 1), min(ncols, info["col"] + 2)))
    # force a continuous west-east, north-south corridor that links them
    if landmark_cells["donghuamen"]["in_grid"] and landmark_cells["wangfujing-north"]["in_grid"]:
        r0 = min(landmark_cells[n]["row"] for n in ("donghuamen", "nanchizi", "wangfujing-north", "wangfujing-south"))
        r1 = max(landmark_cells[n]["row"] for n in ("donghuamen", "nanchizi", "wangfujing-north", "wangfujing-south"))
        c0 = min(landmark_cells[n]["col"] for n in ("donghuamen", "nanchizi", "wangfujing-north", "wangfujing-south"))
        c1 = max(landmark_cells[n]["col"] for n in ("donghuamen", "nanchizi", "wangfujing-north", "wangfujing-south"))
        rows_needed.update(range(max(0, r0 - 1), min(nrows, r1 + 2)))
        cols_needed.update(range(max(0, c0 - 1), min(ncols, c1 + 2)))
    # snap priority to even 1024 alignment (include both cells of each 1024)
    pri_rows = sorted({r - (r % 2) + k for r in rows_needed for k in (0, 1) if 0 <= r - (r % 2) + k < nrows})
    pri_cols = sorted({c - (c % 2) + k for c in cols_needed for k in (0, 1) if 0 <= c - (c % 2) + k < ncols})
    pri_1024 = []
    for t in tiles_1024:
        if (2 * t["R"] in pri_rows or 2 * t["R"] + 1 in pri_rows) and (
            2 * t["C"] in pri_cols or 2 * t["C"] + 1 in pri_cols
        ):
            # only keep 1024 tiles fully inside the priority rectangle
            if 2 * t["R"] in pri_rows and 2 * t["R"] + 1 in pri_rows and 2 * t["C"] in pri_cols and 2 * t["C"] + 1 in pri_cols:
                pri_1024.append(t["id"])
    grid = {
        "crs": "WGS84 / UTM 50N origin at box SW",
        "box": {"west": WEST, "south": SOUTH, "east": EAST, "north": NORTH},
        "camera": {"azimuth_deg": rb.AZIMUTH_DEG, "elevation_deg": rb.ELEVATION_DEG,
                   "view_height_meters": VIEW_H, "viewport_px": VIEWPORT},
        "cell_512_px": CELL,
        "cell_screen_m": SX,
        "cell_rot_y_m": SY_ROT,
        "origin_rot": {"minx": minx, "maxy": maxy, "miny": miny, "maxx": maxx},
        "nrows_512": nrows,
        "ncols_512": ncols,
        "n_tiles_1024_rows": nR,
        "n_tiles_1024_cols": nC,
        "n_cells_512": nrows * ncols,
        "n_tiles_1024": nR * nC,
        "mosaic_px": [ncols * CELL, nrows * CELL],
        "landmarks": landmark_cells,
        "priority_512_rows": pri_rows,
        "priority_512_cols": pri_cols,
        "priority_1024_ids": pri_1024,
        "cells_512": cells_512,
        "tiles_1024": tiles_1024,
    }
    return grid


def draw_grid_overlay(grid, plan_path: Path, out_path: Path):
    if not plan_path.exists():
        return
    plan = Image.open(plan_path).convert("RGBA")
    # render_plan uses UTM with 80m pad; recompute the same transform
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    corners = [to_utm.transform(lon, lat) for lon, lat in ((WEST, SOUTH), (EAST, SOUTH), (EAST, NORTH), (WEST, NORTH))]
    min_e = min(c[0] for c in corners) - 80
    max_e = max(c[0] for c in corners) + 80
    min_n = min(c[1] for c in corners) - 80
    max_n = max(c[1] for c in corners) + 80
    W, H = plan.size
    scale = min(W / (max_e - min_e), H / (max_n - min_n))

    def xy_en(e_abs, n_abs):
        return (e_abs - min_e) * scale, H - (n_abs - min_n) * scale

    overlay = Image.new("RGBA", plan.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    pri = set(grid["priority_1024_ids"])
    for t in grid["tiles_1024"]:
        # four corners of the 1024 footprint in rot space (axis-aligned in camera)
        # width 2*SX in rot X, 2*SY_ROT in rot Y, centered on tile
        hw, hh = SX, SY_ROT
        corners_rot = [
            (t["rot_x"] - hw, t["rot_y"] + hh),
            (t["rot_x"] + hw, t["rot_y"] + hh),
            (t["rot_x"] + hw, t["rot_y"] - hh),
            (t["rot_x"] - hw, t["rot_y"] - hh),
        ]
        pts = []
        for dx, dy in corners_rot:
            e, n = rot_to_en(dx, dy)
            pts.append(xy_en(OX + e, OY + n))
        color = (40, 120, 220, 70) if t["id"] in pri else (20, 20, 20, 30)
        outline = (30, 90, 200, 200) if t["id"] in pri else (40, 40, 40, 120)
        d.polygon(pts, fill=color, outline=outline)
        cx, cy = xy_en(*TO_UTM.transform(t["lon"], t["lat"]))
        d.text((cx - 18, cy - 6), f"R{t['R']}C{t['C']}", fill=(20, 20, 20, 220))
    for name, info in grid["landmarks"].items():
        e, n = TO_UTM.transform(info["lon"], info["lat"])
        x, y = xy_en(e, n)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(200, 40, 40, 220))
        d.text((x + 6, y - 8), name, fill=(160, 20, 20, 255))
    out = Image.alpha_composite(plan.convert("RGBA"), overlay)
    out.convert("RGB").save(out_path, "JPEG", quality=90)
    print("wrote", out_path)


def render_whiteboxes(grid, only_ids=None):
    print("parsing OSM + assembling scene (once)...")
    nodes, ways, rels = rb.parse_osm(rb.OSM)
    buildings, water, roads = rb.assemble(nodes, ways, rels)
    print(f"buildings={len(buildings)} water={len(water)} roads={len(roads)}")
    written = []
    tiles = grid["tiles_1024"]
    if only_ids:
        tiles = [t for t in tiles if t["id"] in only_ids]
    for i, t in enumerate(tiles):
        out = WB_DIR / f"{t['id']}.png"
        view = {
            "id": t["id"],
            "title": f"tile {t['id']}",
            "lon": t["lon"],
            "lat": t["lat"],
        }
        rb.render_view(view, buildings, water, roads, out, labeled=False)
        t["whitebox"] = str(out)
        print(f"[{i+1}/{len(tiles)}] wrote {out}")
        written.append(str(out))
    return written


def main():
    grid = plan_grid()
    summary = {k: grid[k] for k in grid if k not in ("cells_512", "tiles_1024")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT_TILES / "grid.json").write_text(json.dumps(grid, ensure_ascii=False, indent=2))
    print("wrote", OUT_TILES / "grid.json")
    plan = ROOT / "renders_v2" / "00_plan_colored.png"
    draw_grid_overlay(grid, plan, OUT_TILES / "grid_overlay.jpg")
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    if only == ["--plan-only"]:
        return
    render_whiteboxes(grid, only_ids=set(only) if only else None)
    # rewrite grid with whitebox paths
    (OUT_TILES / "grid.json").write_text(json.dumps(grid, ensure_ascii=False, indent=2))
    print("done")


if __name__ == "__main__":
    main()
