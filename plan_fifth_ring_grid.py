#!/usr/bin/env python3
"""Planning grid for Beijing inside the 5th Ring Road.

Same UTM 50N origin, camera, and 0.29296875 m/px as grid_axis.json.
Writes tiles/grid_fifth_ring.json only. Does NOT expand the current mosaic window.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from pyproj import Geod, Transformer
from shapely.geometry import LineString, MultiLineString, Polygon, box, mapping
from shapely.ops import unary_union, transform as shp_transform

ROOT = Path("/workspace/beijing-pilot")
sys.path.insert(0, str(ROOT))
import expand_axis as ea

GEOM_PATH = Path("/tmp/bj-osm/overpass_5th_geom.json")
OUT = ROOT / "tiles" / "grid_fifth_ring.json"

MPP = 300.0 / 1024.0
TILE_WX, TILE_WY = ea.TILE_WX, ea.TILE_WY
MINX, MAXY = ea.MINX, ea.MAXY
TO_UTM = ea.TO_UTM
GEOD = Geod(ellps="WGS84")

# Current styled/whitebox mosaic window — do not expand this to thousands of tiles.
CUR = {"R_min": -2, "R_max": 3, "C_min": -2, "C_max": 4}

# Published figures (not a substitute for the OSM geometry).
PUBLISHED_AREA_KM2 = 667.0
PUBLISHED_LENGTH_KM = 98.6
PUBLISHED_SOURCES = [
    {
        "figure": "五环内面积 667 km²；五环路周长 98.6 km",
        "url": "https://www.bfzh.com.cn/NewsDetail/5846236.html",
        "note": "环路与环路之间面积/人口密度备忘，引用市统计口径常见数字",
    },
    {
        "figure": "五环以内 667 km²",
        "url": "https://datanews.caixin.com/interactive/2018/56ring/",
        "note": "财新数据新闻 2018「五环以外」",
    },
    {
        "figure": "五环路全长约 98 km，距市中心约 10 km（距离随方位变化）",
        "url": "https://zh.wikipedia.org/wiki/北京五环路",
        "note": "中文维基「北京五环路」",
    },
]


def load_ring_lines():
    data = json.loads(GEOM_PATH.read_text())
    els = data["elements"]
    assert els and els[0]["id"] == 295957
    rel = els[0]
    bounds = rel["bounds"]
    tags = rel.get("tags", {})
    lines_ll = []
    lines_utm = []
    pts_ll = []
    for m in rel.get("members", []):
        geom = m.get("geometry") or []
        if len(geom) < 2:
            continue
        ll = [(p["lon"], p["lat"]) for p in geom]
        utm = [TO_UTM.transform(lon, lat) for lon, lat in ll]
        lines_ll.append(LineString(ll))
        lines_utm.append(LineString(utm))
        pts_ll.extend(ll)
    return rel, bounds, tags, lines_ll, lines_utm, pts_ll


def extreme_points(pts_ll):
    west = min(pts_ll, key=lambda p: p[0])
    east = max(pts_ll, key=lambda p: p[0])
    south = min(pts_ll, key=lambda p: p[1])
    north = max(pts_ll, key=lambda p: p[1])
    return {
        "west": {"lon": west[0], "lat": west[1]},
        "east": {"lon": east[0], "lat": east[1]},
        "south": {"lon": south[0], "lat": south[1]},
        "north": {"lon": north[0], "lat": north[1]},
    }


def recover_interior(lines_utm, buffers_m=(40.0, 60.0, 80.0, 100.0, 120.0)):
    """Merge dual carriageways with a buffer and take the largest hole as the interior.

    The hole of buffer(B) is the interior eroded by B. Buffering the hole back by B
    approximates the area inside the road centreline / inner edge.
    """
    net = unary_union(lines_utm)
    attempts = []
    best = None
    for B in buffers_m:
        buf = net.buffer(B)
        polys = list(buf.geoms) if buf.geom_type == "MultiPolygon" else [buf]
        holes = []
        for p in polys:
            for interior in p.interiors:
                hole = Polygon(interior)
                holes.append(hole)
        if not holes:
            attempts.append({"buffer_m": B, "n_holes": 0})
            continue
        holes.sort(key=lambda h: h.area, reverse=True)
        raw = holes[0]
        recovered = raw.buffer(B)
        if recovered.geom_type == "MultiPolygon":
            recovered = max(recovered.geoms, key=lambda g: g.area)
        rec = {
            "buffer_m": B,
            "n_holes": len(holes),
            "raw_hole_area_km2": raw.area / 1e6,
            "recovered_area_km2": recovered.area / 1e6,
            "recovered": recovered,
        }
        attempts.append({k: rec[k] for k in rec if k != "recovered"})
        # Prefer the recovery closest to the published 667 km²
        err = abs(rec["recovered_area_km2"] - PUBLISHED_AREA_KM2)
        if best is None or err < best["err"]:
            best = dict(rec)
            best["err"] = err
    return best, attempts, net


def rot_of_lonlat(lon, lat):
    return ea.lonlat_to_rot(lon, lat)


def utm_to_rot_coords(xs, ys):
    """Vectorized UTM (EPSG:32650) -> rotated camera plane."""
    import numpy as np
    e = np.asarray(xs) - ea.OX
    n = np.asarray(ys) - ea.OY
    # same as tg.en_to_rot / azimuth -15
    az = ea.AZ
    rx = e * np.cos(az) - n * np.sin(az)
    ry = e * np.sin(az) + n * np.cos(az)
    return rx, ry


def utm_to_rot_poly(poly_utm):
    import shapely
    def _t(coords):
        # coords shape (N, 2) or (N, 3)
        rx, ry = utm_to_rot_coords(coords[:, 0], coords[:, 1])
        out = coords.copy()
        out[:, 0] = rx
        out[:, 1] = ry
        return out
    return shapely.transform(poly_utm, _t)


def tile_rc(rx, ry):
    C = math.floor((rx - MINX) / TILE_WX)
    R = math.floor((MAXY - ry) / TILE_WY)
    return R, C


def tile_bounds_rot(R, C):
    x0 = MINX + C * TILE_WX
    x1 = MINX + (C + 1) * TILE_WX
    y1 = MAXY - R * TILE_WY
    y0 = MAXY - (R + 1) * TILE_WY
    return x0, x1, y0, y1


def cover_polygon(poly_rot):
    minx, miny, maxx, maxy = poly_rot.bounds
    C0 = math.floor((minx - MINX) / TILE_WX)
    C1 = math.floor((maxx - MINX) / TILE_WX)
    R0 = math.floor((MAXY - maxy) / TILE_WY)
    R1 = math.floor((MAXY - miny) / TILE_WY)
    hits = []
    for R in range(R0, R1 + 1):
        for C in range(C0, C1 + 1):
            x0, x1, y0, y1 = tile_bounds_rot(R, C)
            if poly_rot.intersects(box(x0, y0, x1, y1)):
                hits.append((R, C))
    return R0, R1, C0, C1, hits


def bbox_area_km2(west, south, east, north):
    # geodesic rectangle approximation: mid-lat width * mid-lon height
    _, _, width_s = GEOD.inv(west, south, east, south)
    _, _, width_n = GEOD.inv(west, north, east, north)
    _, _, height_w = GEOD.inv(west, south, west, north)
    _, _, height_e = GEOD.inv(east, south, east, north)
    width = 0.5 * (width_s + width_n)
    height = 0.5 * (height_w + height_e)
    return (width * height) / 1e6, width / 1000.0, height / 1000.0


def main():
    if not GEOM_PATH.exists():
        raise SystemExit(f"missing {GEOM_PATH}; fetch OSM rel 295957 first")
    rel, bounds, tags, lines_ll, lines_utm, pts_ll = load_ring_lines()
    print(f"relation {rel['id']} {tags.get('name')} members-with-geom={len(lines_ll)} pts={len(pts_ll)}")
    print("OSM bounds", bounds)

    extremes = extreme_points(pts_ll)
    west, east = bounds["minlon"], bounds["maxlon"]
    south, north = bounds["minlat"], bounds["maxlat"]
    bbox_km2, bbox_w_km, bbox_h_km = bbox_area_km2(west, south, east, north)
    print(f"WGS84 bbox area ~{bbox_km2:.1f} km²  ({bbox_w_km:.2f} × {bbox_h_km:.2f} km)")

    best, attempts, net = recover_interior(lines_utm)
    if best is None:
        raise SystemExit(f"failed to recover an interior hole: {attempts}")
    print("buffer attempts", attempts)
    print(f"chosen buffer {best['buffer_m']} m  recovered interior {best['recovered_area_km2']:.2f} km²")

    interior_utm = best["recovered"]
    interior_rot = utm_to_rot_poly(interior_utm)
    # Also a rot AABB of the ring road points themselves (centreline envelope)
    rots = [rot_of_lonlat(lon, lat) for lon, lat in pts_ll]
    rx0, ry0 = min(r[0] for r in rots), min(r[1] for r in rots)
    rx1, ry1 = max(r[0] for r in rots), max(r[1] for r in rots)

    R0, R1, C0, C1, hits = cover_polygon(interior_rot)
    n_rect = (R1 - R0 + 1) * (C1 - C0 + 1)
    n_hit = len(hits)
    tile_km2 = (TILE_WX * TILE_WY) / 1e6
    print(f"interior-covering rectangular range R={R0}..{R1} C={C0}..{C1}")
    print(f"  n_rect={n_rect}  n_intersect={n_hit}")
    print(f"  rect rot-area={n_rect * tile_km2:.1f} km²  intersect rot-area={n_hit * tile_km2:.1f} km²")
    print(f"  tile {TILE_WX:.3f} m × {TILE_WY:.3f} m = {tile_km2:.6f} km²")

    # Compare with current 32/42 tile window
    cur_n = (CUR["R_max"] - CUR["R_min"] + 1) * (CUR["C_max"] - CUR["C_min"] + 1)
    print(f"current mosaic window {cur_n} tiles, {cur_n * tile_km2:.3f} km² rot-area")

    # Corners of the rectangular envelope in WGS84 (rotated mosaic AABB)
    x0, x1 = MINX + C0 * TILE_WX, MINX + (C1 + 1) * TILE_WX
    y1, y0 = MAXY - R0 * TILE_WY, MAXY - (R1 + 1) * TILE_WY
    corners = [ea.rot_to_lonlat(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]

    grid = {
        "purpose": (
            "Planning 1024-tile grid covering the interior of Beijing's 5th Ring Road "
            "(五环以内). Same origin and camera as the Forbidden City–Wangfujing pilot. "
            "This file must NOT be used to expand the current mosaic window to thousands of tiles."
        ),
        "do_not_expand_current_mosaic": True,
        "crs": "WGS84 / UTM 50N origin at box SW",
        "pilot_box_sw_origin": {"lon": 116.39480, "lat": 39.90620, "epsg": 32650},
        "origin_rot": {"minx": MINX, "maxy": MAXY},
        "camera": {
            "azimuth_deg": -15.0,
            "elevation_deg": -45.0,
            "view_height_meters": 300.0,
            "viewport_px": 1024,
        },
        "metres_per_pixel": MPP,
        "tile_1024": {
            "px": 1024,
            "rot_x_m": TILE_WX,
            "rot_y_m": TILE_WY,
            "rot_area_km2": tile_km2,
            "index_rule": (
                "C = floor((rot_x - origin_rot.minx) / rot_x_m); "
                "R = floor((origin_rot.maxy - rot_y) / rot_y_m); "
                "south and west are positive R / negative C from the original 5×4 pilot."
            ),
        },
        "fifth_ring": {
            "name": "五环路",
            "name:en": "5th Ring Road",
            "ref": "S50",
            "osm_relation": 295957,
            "osm_source": "Overpass API, relation 295957 out geom, WGS84, ODbL 1.0",
            "crs_note": "Coordinates are OSM WGS84 (EPSG:4326). Do not mix with GCJ-02 (Amap/Tencent) or BD-09.",
            "wgs84_bbox": {
                "west": bounds["minlon"],
                "south": bounds["minlat"],
                "east": bounds["maxlon"],
                "north": bounds["maxlat"],
                "width_km": bbox_w_km,
                "height_km": bbox_h_km,
                "axis_aligned_area_km2": bbox_km2,
                "note": "Bounding box of the ring ROAD itself, larger than the interior area.",
            },
            "extreme_points_wgs84": extremes,
            "published_interior_area_km2": PUBLISHED_AREA_KM2,
            "published_length_km": PUBLISHED_LENGTH_KM,
            "published_sources": PUBLISHED_SOURCES,
            "computed_interior": {
                "method": (
                    "Unary-union of OSM route centreline segments in UTM 50N, buffer to merge "
                    "dual carriageways, take the largest interior hole, then buffer the hole "
                    "back by the same distance to approximate the area inside the road."
                ),
                "chosen_buffer_m": best["buffer_m"],
                "raw_hole_area_km2": best["raw_hole_area_km2"],
                "area_km2": best["recovered_area_km2"],
                "buffer_attempts": attempts,
                "note": (
                    "This is an OSM-geometry estimate, not an official cadastral area. "
                    "It should be close to the commonly cited 667 km²."
                ),
            },
        },
        "grid": {
            "R_min": R0,
            "R_max": R1,
            "C_min": C0,
            "C_max": C1,
            "n_rows": R1 - R0 + 1,
            "n_cols": C1 - C0 + 1,
            "n_tiles_rectangular": n_rect,
            "rectangular_rot_area_km2": n_rect * tile_km2,
            "n_tiles_intersecting_interior": n_hit,
            "intersecting_rot_area_km2": n_hit * tile_km2,
            "ring_points_rot_aabb": {"minx": rx0, "miny": ry0, "maxx": rx1, "maxy": ry1},
            "rectangular_mosaic_wgs84_corners": [
                {"lon": lon, "lat": lat} for lon, lat in corners
            ],
            "note": (
                "The rectangular R/C range is the axis-aligned envelope in rotated camera space "
                "of tiles that intersect the recovered 5th-ring interior. Because the camera is "
                "yawed -15° and the ring is not a rectangle, n_tiles_rectangular × tile area is "
                "larger than 667 km²; n_tiles_intersecting_interior is the tighter planning count. "
                "Do not pre-allocate or render this many tiles into the current mosaic window."
            ),
        },
        "current_mosaic_window": {
            **CUR,
            "n_tiles": cur_n,
            "rot_area_km2": cur_n * tile_km2,
            "note": "R=-2..3, C=-2..4 (6×7=42). Leave this window unchanged by this planning file.",
        },
        "comparison_with_prior_estimate": {
            "prior_rough_count": "~9000 tiles for 667 km²",
            "prior_basis_note": (
                "An earlier scoping used ~2.3 km² / 32 finished tiles, which understates the "
                "rotated 1024-tile area (each tile is 300 × 424.264 m = 0.1273 km²). "
                "667 / 0.1273 ≈ 5240 tiles if tiles packed the interior with no slack; "
                "the intersecting count below is the geometry-based number, and the rectangular "
                "envelope is larger because of the 15° yaw and the ring's rounded-rectangle shape."
            ),
        },
    }
    OUT.write_text(json.dumps(grid, ensure_ascii=False, indent=2) + "\n")
    print("wrote", OUT)
    return grid


if __name__ == "__main__":
    main()
