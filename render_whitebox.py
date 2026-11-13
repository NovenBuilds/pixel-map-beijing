#!/usr/bin/env python3
"""Whitebox orthographic renders for the Forbidden City–Wangfujing pilot.

Data: OpenStreetMap extract (ODbL), local file osm_map.xml.
Camera matches isometric-nyc: azimuth -15°, elevation -45°, view_height 300 m
on a 1024 px viewport (see plan_tiles.py / whitebox.py).
Heights: tagged height, else building:levels * 3.0 m, else default 6.0 m
(clearly marked as assumed). Not photorealistic; no commercial tiles.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer, Geod

ROOT = Path("/workspace/beijing-pilot")
OSM = ROOT / "data" / "osm_map.xml"
OUT = ROOT / "renders"
OUT.mkdir(parents=True, exist_ok=True)

# Recommended pilot box (WGS84). Documented in REPORT.
PILOT = {
    "west": 116.39480,
    "south": 39.90620,
    "east": 116.40780,
    "north": 39.91800,
}

# Padded clip used to load geometry so facades can lean into the frame.
CLIP = {
    "west": 116.39450,
    "south": 39.90580,
    "east": 116.40850,
    "north": 39.92300,
}

AZIMUTH_DEG = -15.0
ELEVATION_DEG = -45.0
VIEW_HEIGHT_M = 300.0
WIDTH_PX = 1024
HEIGHT_PX = 1024
LEVEL_HEIGHT_M = 3.0
DEFAULT_HEIGHT_M = 6.0

# View centers (WGS84) for sample renders.
VIEWS = [
    {
        "id": "donghuamen",
        "title": "东华门 / 筒子河东段 / 故宫东墙",
        "lat": 39.91370,
        "lon": 116.39580,
    },
    {
        "id": "nanchizi",
        "title": "南池子大街 / 胡同与四合院肌理",
        "lat": 39.91035,
        "lon": 116.39810,
    },
    {
        "id": "wangfujing-north",
        "title": "王府井大街北段 / 金鱼胡同",
        "lat": 39.91490,
        "lon": 116.40540,
    },
    {
        "id": "wangfujing-south",
        "title": "王府井大街南段 / 北京饭店",
        "lat": 39.90840,
        "lon": 116.40490,
    },
]


def parse_osm(path: Path):
    nodes = {}
    ways = []
    rels = []
    for _ev, el in ET.iterparse(path, events=("end",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("lon")), float(el.get("lat")))
            el.clear()
        elif el.tag == "way":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            nds = [n.get("ref") for n in el.findall("nd")]
            ways.append({"id": el.get("id"), "tags": tags, "nds": nds})
            el.clear()
        elif el.tag == "relation":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            members = [
                {"type": m.get("type"), "ref": m.get("ref"), "role": m.get("role")}
                for m in el.findall("member")
            ]
            rels.append({"id": el.get("id"), "tags": tags, "members": members})
            el.clear()
    return nodes, ways, rels


def parse_height(tags: dict) -> tuple[float, str]:
    raw = tags.get("height") or tags.get("building:height")
    if raw:
        try:
            # strip units like "10 m"
            h = float(str(raw).lower().replace("m", "").strip())
            if h > 0:
                return h, "height"
        except ValueError:
            pass
    raw = tags.get("building:levels") or tags.get("levels")
    if raw:
        try:
            # handle "3;1" roughly
            levels = float(str(raw).split(";")[0].split(",")[0])
            if levels > 0:
                return levels * LEVEL_HEIGHT_M, "levels"
        except ValueError:
            pass
    return DEFAULT_HEIGHT_M, "default"


def way_coords(way, nodes):
    coords = [nodes[i] for i in way["nds"] if i in nodes]
    if len(coords) < 2:
        return None
    return coords


def assemble_features(nodes, ways, rels):
    # Collect way ids used as inner/outer of building multipolygons to avoid double-count
    # We'll still draw building ways; for relations, add outers not already a building way.
    features = []

    water_ways = []
    road_ways = []
    buildings = []

    building_way_ids = set()
    for w in ways:
        tags = w["tags"]
        coords = way_coords(w, nodes)
        if not coords:
            continue
        if "building" in tags or tags.get("building:part"):
            closed = coords[0] == coords[-1] and len(coords) >= 4
            if not closed:
                # still keep walls as lines extruded? skip unclosed scraps
                if tags.get("building") == "wall" and len(coords) >= 2:
                    pass
                else:
                    continue
            h, src = parse_height(tags)
            kind = "part" if tags.get("building:part") else (
                "wall" if tags.get("building") == "wall" or "城墙" in (tags.get("name") or "") else "building"
            )
            buildings.append(
                {
                    "id": f"way/{w['id']}",
                    "name": tags.get("name"),
                    "coords": coords,
                    "height": h,
                    "height_src": src,
                    "kind": kind,
                    "historic": tags.get("historic"),
                }
            )
            building_way_ids.add(w["id"])
        elif tags.get("natural") == "water" or tags.get("water") or (
            tags.get("landuse") in ("basin", "reservoir")
        ):
            if coords[0] == coords[-1] and len(coords) >= 4:
                water_ways.append({"id": f"way/{w['id']}", "name": tags.get("name"), "coords": coords})
        elif tags.get("highway") and tags.get("highway") not in ("proposed",):
            road_ways.append(
                {
                    "id": f"way/{w['id']}",
                    "name": tags.get("name"),
                    "coords": coords,
                    "highway": tags.get("highway"),
                }
            )

    # Add water from multipolygon relations
    way_index = {w["id"]: w for w in ways}
    for r in rels:
        tags = r["tags"]
        if tags.get("natural") == "water" or tags.get("water") or tags.get("landuse") == "basin":
            outers = []
            for m in r["members"]:
                if m["type"] == "way" and m["role"] in ("outer", "") and m["ref"] in way_index:
                    c = way_coords(way_index[m["ref"]], nodes)
                    if c and c[0] == c[-1] and len(c) >= 4:
                        outers.append(c)
            for c in outers:
                water_ways.append(
                    {"id": f"rel/{r['id']}", "name": tags.get("name"), "coords": c}
                )

    return buildings, water_ways, road_ways


class Projector:
    def __init__(self, lon0, lat0, azimuth_deg=AZIMUTH_DEG, elevation_deg=ELEVATION_DEG,
                 view_height_m=VIEW_HEIGHT_M, width_px=WIDTH_PX, height_px=HEIGHT_PX):
        self.lon0 = lon0
        self.lat0 = lat0
        self.az = math.radians(azimuth_deg)
        self.elev = math.radians(elevation_deg)
        self.view_height_m = view_height_m
        self.width_px = width_px
        self.height_px = height_px
        self.mpp = view_height_m / height_px
        # UTM 50N
        self.to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
        self.e0, self.n0 = self.to_utm.transform(lon0, lat0)
        self.cos_a = math.cos(self.az)
        self.sin_a = math.sin(self.az)
        # screen_y = cos(|elev|) * y_rot + sin(|elev|) * z   with elev negative
        # From derivation: screen_y = -sin(elev)*? wait we use:
        # camera forward / true_up for elev = -45:
        # screen_x = x_rot
        # screen_y = y_rot * (-sin(elev))? 
        # elev=-45, sin(elev)=-√2/2, cos(elev)=√2/2
        # screen_y = 0.707 * y_rot + 0.707 * z = -sin(elev)*y_rot + (-sin(elev))*z? 
        # -sin(-45)= -(-0.707)=0.707, and also *z 0.707 = -sin(elev)*z
        self.s = -math.sin(self.elev)  # 0.707 for -45
        self.c = math.cos(self.elev)   # 0.707

    def en(self, lon, lat):
        e, n = self.to_utm.transform(lon, lat)
        return e - self.e0, n - self.n0

    def rotate(self, e, n):
        x = e * self.cos_a - n * self.sin_a
        y = e * self.sin_a + n * self.cos_a
        return x, y

    def project_point(self, lon, lat, z=0.0):
        e, n = self.en(lon, lat)
        x, y = self.rotate(e, n)
        sx = x
        sy = self.s * y + self.s * z
        depth = self.s * y - self.s * z  # 0.707*(y-z)
        px = self.width_px / 2 + sx / self.mpp
        py = self.height_px / 2 - sy / self.mpp
        return px, py, depth

    def project_xyz(self, e, n, z=0.0):
        x, y = self.rotate(e, n)
        sx = x
        sy = self.s * y + self.s * z
        depth = self.s * y - self.s * z
        px = self.width_px / 2 + sx / self.mpp
        py = self.height_px / 2 - sy / self.mpp
        return px, py, depth


COLORS = {
    "ground": (232, 226, 214),
    "water": (120, 168, 196),
    "road_major": (90, 90, 90),
    "road_minor": (130, 126, 118),
    "bldg_height": (196, 150, 96),
    "bldg_height_side": (168, 122, 78),
    "bldg_levels": (186, 142, 102),
    "bldg_levels_side": (158, 116, 82),
    "bldg_default": (188, 186, 178),
    "bldg_default_side": (160, 158, 150),
    "wall": (140, 92, 70),
    "wall_side": (112, 72, 54),
    "part": (204, 160, 108),
    "part_side": (172, 130, 86),
    "stroke": (70, 64, 56),
    "caption_bg": (255, 255, 255, 210),
}


def face_color(kind, src, is_roof):
    if kind == "wall":
        return COLORS["wall"] if is_roof else COLORS["wall_side"]
    if kind == "part":
        return COLORS["part"] if is_roof else COLORS["part_side"]
    if src == "height":
        return COLORS["bldg_height"] if is_roof else COLORS["bldg_height_side"]
    if src == "levels":
        return COLORS["bldg_levels"] if is_roof else COLORS["bldg_levels_side"]
    return COLORS["bldg_default"] if is_roof else COLORS["bldg_default_side"]


def faces_from_building(bldg, proj: Projector):
    coords = bldg["coords"]
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    h = bldg["height"]
    # skip tiny
    if len(coords) < 4:
        return []
    base = [proj.en(lon, lat) for lon, lat in coords]
    roof = [(e, n, h) for e, n in base]
    ground = [(e, n, 0.0) for e, n in base]
    faces = []
    # walls
    for i in range(len(base) - 1):
        p0 = ground[i]
        p1 = ground[i + 1]
        p2 = (base[i + 1][0], base[i + 1][1], h)
        p3 = (base[i][0], base[i][1], h)
        # skip degenerate edges
        if p0[0] == p1[0] and p0[1] == p1[1]:
            continue
        pts = [p0, p1, p2, p3]
        depths = [proj.project_xyz(*p)[2] for p in pts]
        faces.append(
            {
                "pts": pts,
                "depth": sum(depths) / 4,
                "color": face_color(bldg["kind"], bldg["height_src"], False),
                "kind": "wall_face",
            }
        )
    # roof
    depths = [proj.project_xyz(*p)[2] for p in roof]
    faces.append(
        {
            "pts": roof,
            "depth": sum(depths) / len(depths),
            "color": face_color(bldg["kind"], bldg["height_src"], True),
            "kind": "roof",
        }
    )
    return faces


def draw_poly(draw: ImageDraw.ImageDraw, proj: Projector, pts_enz, fill, outline=None, width=1):
    xy = []
    for p in pts_enz:
        px, py, _ = proj.project_xyz(*p)
        xy.append((px, py))
    if len(xy) >= 3:
        draw.polygon(xy, fill=fill, outline=outline)
    elif len(xy) == 2:
        draw.line(xy, fill=outline or fill, width=width)


def render_view(view, buildings, water, roads, out_path: Path, extra_caption=None):
    proj = Projector(view["lon"], view["lat"])
    img = Image.new("RGB", (WIDTH_PX, HEIGHT_PX), COLORS["ground"])
    draw = ImageDraw.Draw(img, "RGBA")

    # ground rectangle slightly larger than view in EN
    # not strictly needed; background is already ground color

    # water
    for w in water:
        coords = w["coords"]
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        pts = [(*proj.en(lon, lat), 0.0) for lon, lat in coords]
        draw_poly(draw, proj, pts, COLORS["water"], outline=(90, 140, 170))

    # roads as polylines on ground
    for r in roads:
        hw = r["highway"]
        width = 3 if hw in ("trunk", "primary", "secondary", "pedestrian") else 1
        color = COLORS["road_major"] if width >= 3 else COLORS["road_minor"]
        xy = []
        for lon, lat in r["coords"]:
            px, py, _ = proj.project_point(lon, lat, 0.0)
            xy.append((px, py))
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=width, joint="curve")

    faces = []
    # only buildings that might appear in frame: center within ~500 m
    for b in buildings:
        lon, lat = b["coords"][0]
        e, n = proj.en(lon, lat)
        if abs(e) > 600 or abs(n) > 700:
            continue
        faces.extend(faces_from_building(b, proj))
    faces.sort(key=lambda f: f["depth"])  # far first
    for f in faces:
        outline = COLORS["stroke"] if f["kind"] == "roof" else None
        draw_poly(draw, proj, f["pts"], f["color"], outline=outline)

    # caption bar
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16, index=2)
        font_s = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 13, index=2)
    except Exception:
        font = ImageFont.load_default()
        font_s = font

    caption = (
        f"{view['title']}\n"
        f"center WGS84 {view['lat']:.5f}°N, {view['lon']:.5f}°E   "
        f"camera az {AZIMUTH_DEG}°  el {ELEVATION_DEG}°  view_height {VIEW_HEIGHT_M} m  {WIDTH_PX}px\n"
        f"whitebox from OSM footprints (ODbL). Color: ochre=tagged height, tan=levels×{LEVEL_HEIGHT_M:.0f}m, grey=default {DEFAULT_HEIGHT_M:.0f}m, brown=wall.\n"
        f"No photorealistic textures; geometry and texture are not aligned. © OpenStreetMap contributors."
    )
    if extra_caption:
        caption += "\n" + extra_caption
    # background
    draw.rectangle([0, HEIGHT_PX - 92, WIDTH_PX, HEIGHT_PX], fill=(30, 28, 26, 210))
    draw.multiline_text((12, HEIGHT_PX - 86), caption, font=font_s, fill=(245, 240, 232), spacing=3)
    img.save(out_path, "PNG")
    return out_path


def render_plan(buildings, water, roads, out_path: Path):
    """Top-down diagnostic of the pilot box (not isometric)."""
    west, south, east, north = PILOT["west"], PILOT["south"], PILOT["east"], PILOT["north"]
    # pad 80 m visually? keep exact box, add margin in image
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    e0, n0 = to_utm.transform(west, south)
    e1, n1 = to_utm.transform(east, south)
    e2, n2 = to_utm.transform(east, north)
    e3, n3 = to_utm.transform(west, north)
    min_e = min(e0, e1, e2, e3)
    max_e = max(e0, e1, e2, e3)
    min_n = min(n0, n1, n2, n3)
    max_n = max(n0, n1, n2, n3)
    # 10% margin
    mx = (max_e - min_e) * 0.06
    my = (max_n - min_n) * 0.06
    min_e -= mx
    max_e += mx
    min_n -= my
    max_n += my
    W, H = 1600, 1600
    sx = W / (max_e - min_e)
    sy = H / (max_n - min_n)
    scale = min(sx, sy)
    # letterbox
    def xy(lon, lat):
        e, n = to_utm.transform(lon, lat)
        px = (e - min_e) * scale
        py = H - (n - min_n) * scale
        return px, py

    img = Image.new("RGB", (W, H), (245, 242, 236))
    draw = ImageDraw.Draw(img)
    # box
    box = [xy(west, south), xy(east, south), xy(east, north), xy(west, north), xy(west, south)]
    draw.polygon(box, fill=(232, 226, 214), outline=(80, 40, 40))
    for w in water:
        pts = [xy(lon, lat) for lon, lat in w["coords"]]
        if len(pts) >= 3:
            draw.polygon(pts, fill=(120, 168, 196), outline=(90, 140, 170))
    for r in roads:
        pts = [xy(lon, lat) for lon, lat in r["coords"]]
        if len(pts) >= 2:
            hw = r["highway"]
            width = 2 if hw in ("trunk", "primary", "secondary", "pedestrian") else 1
            draw.line(pts, fill=(120, 116, 110) if width == 1 else (70, 70, 70), width=width)
    for b in buildings:
        pts = [xy(lon, lat) for lon, lat in b["coords"]]
        if len(pts) < 3:
            continue
        color = face_color(b["kind"], b["height_src"], True)
        draw.polygon(pts, fill=color, outline=(70, 64, 56))
    # view centers
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18, index=2)
        font_s = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 14, index=2)
    except Exception:
        font = ImageFont.load_default()
        font_s = font
    for v in VIEWS:
        px, py = xy(v["lon"], v["lat"])
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(180, 30, 30), outline=(80, 0, 0))
        draw.text((px + 10, py - 8), v["id"], font=font_s, fill=(120, 20, 20))
    # labels for box corners
    draw.line(box, fill=(160, 40, 40), width=3)
    cap = (
        "Pilot WGS84 box  E 116.39480–116.40780  N 39.90620–39.91800\n"
        "Plan view (not isometric). Ochre=OSM height, tan=levels, grey=default 6 m, brown=wall.\n"
        "Red dots: sample isometric view centers. © OpenStreetMap contributors (ODbL)."
    )
    draw.rectangle([0, H - 80, W, H], fill=(30, 28, 26))
    draw.multiline_text((16, H - 72), cap, font=font_s, fill=(245, 240, 232), spacing=3)
    img.save(out_path, "PNG")
    return out_path


def write_bounds():
    geod = Geod(ellps="WGS84")
    w, s, e, n = PILOT["west"], PILOT["south"], PILOT["east"], PILOT["north"]
    mid_lat = (s + n) / 2
    mid_lon = (w + e) / 2
    _, _, width_m = geod.inv(w, mid_lat, e, mid_lat)
    _, _, height_m = geod.inv(mid_lon, s, mid_lon, n)
    area_m2 = width_m * height_m  # rectangle approximation

    # camera-grid coverage
    az = math.radians(AZIMUTH_DEG)
    elev = math.radians(ELEVATION_DEG)
    # 512 px quadrant screen size
    q_screen = VIEW_HEIGHT_M * (512 / 1024)  # 150 m
    q_ground_x = q_screen
    q_ground_y = q_screen / abs(math.sin(elev))

    # extent of axis-aligned box in rotated frame
    # x_rot = E*cos(az) - N*sin(az); y_rot = E*sin(az) + N*cos(az)
    corners = [(0, 0), (width_m, 0), (0, height_m), (width_m, height_m)]
    xs, ys = [], []
    ca, sa = math.cos(az), math.sin(az)
    for E, N in corners:
        xs.append(E * ca - N * sa)
        ys.append(E * sa + N * ca)
    range_x = max(xs) - min(xs)
    range_y = max(ys) - min(ys)
    n_cols = range_x / q_ground_x
    n_rows = range_y / q_ground_y
    tiles_ceil = math.ceil(n_cols) * math.ceil(n_rows)
    # facade lean margin: assume max reasonably common height 80 m along view
    lean_m = 80.0 / abs(math.sin(elev))  # extra ground along y_rot? lean along view dir on ground = h / tan(|elev|) wait
    # At 45°, horizontal lean = height / tan(45) = height. Along y_rot.
    lean_ground = 80.0 / math.tan(abs(elev))
    n_rows_lean = (range_y + lean_ground) / q_ground_y
    tiles_lean = math.ceil(n_cols) * math.ceil(n_rows_lean)

    # extended north option
    n_ext = 39.92260
    _, _, height_ext = geod.inv(mid_lon, s, mid_lon, n_ext)
    corners_e = [(0, 0), (width_m, 0), (0, height_ext), (width_m, height_ext)]
    xs_e, ys_e = [], []
    for E, N in corners_e:
        xs_e.append(E * ca - N * sa)
        ys_e.append(E * sa + N * ca)
    range_x_e = max(xs_e) - min(xs_e)
    range_y_e = max(ys_e) - min(ys_e)
    tiles_ext = math.ceil(range_x_e / q_ground_x) * math.ceil(range_y_e / q_ground_y)

    payload = {
        "crs": "EPSG:4326",
        "type": "Polygon",
        "coordinates": [[
            [w, s], [e, s], [e, n], [w, n], [w, s]
        ]],
        "properties": {
            "name": "beijing-pilot-forbidden-city-to-wangfujing",
            "west": w, "south": s, "east": e, "north": n,
            "width_m_geod": round(width_m, 1),
            "height_m_geod": round(height_m, 1),
            "area_m2_rect": round(area_m2, 1),
            "camera": {
                "azimuth_deg": AZIMUTH_DEG,
                "elevation_deg": ELEVATION_DEG,
                "view_height_m": VIEW_HEIGHT_M,
                "render_px": 1024,
                "quadrant_px": 512,
            },
            "quadrant_512_ground_m": {
                "along_rotated_x": q_ground_x,
                "along_rotated_y": q_ground_y,
                "note": "screen 150 m square; ground Y stretched by 1/sin(|elev|)",
            },
            "unique_512_quadrants_estimate": {
                "rotated_range_m": [round(range_x, 1), round(range_y, 1)],
                "cols_rows_float": [round(n_cols, 2), round(n_rows, 2)],
                "ceil_product": tiles_ceil,
                "with_80m_facade_lean_ceil": tiles_lean,
                "extended_north_39_92260_ceil": tiles_ext,
            },
        },
    }
    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": payload["properties"],
                "geometry": {"type": "Polygon", "coordinates": payload["coordinates"]},
            }
        ],
    }
    path = ROOT / "data" / "pilot_bounds.geojson"
    path.write_text(json.dumps(gj, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path = ROOT / "data" / "pilot_metrics.json"
    stats_path.write_text(json.dumps(payload["properties"], ensure_ascii=False, indent=2), encoding="utf-8")
    return payload["properties"], path


def height_stats(buildings, box):
    w, s, e, n = box["west"], box["south"], box["east"], box["north"]
    inside = []
    for b in buildings:
        lons = [c[0] for c in b["coords"]]
        lats = [c[1] for c in b["coords"]]
        if max(lons) < w or min(lons) > e or max(lats) < s or min(lats) > n:
            continue
        inside.append(b)
    from collections import Counter
    src = Counter(b["height_src"] for b in inside)
    kinds = Counter(b["kind"] for b in inside)
    return {
        "n": len(inside),
        "height_src": dict(src),
        "kinds": dict(kinds),
        "with_real_or_levels": src.get("height", 0) + src.get("levels", 0),
        "pct_with_vertical": round(100 * (src.get("height", 0) + src.get("levels", 0)) / len(inside), 1) if inside else None,
    }


def main():
    print("parsing OSM...")
    nodes, ways, rels = parse_osm(OSM)
    buildings, water, roads = assemble_features(nodes, ways, rels)
    print(f"buildings={len(buildings)} water={len(water)} roads={len(roads)}")
    metrics, bounds_path = write_bounds()
    stats_all = height_stats(buildings, CLIP)
    stats_pilot = height_stats(buildings, PILOT)
    (ROOT / "data" / "osm_height_stats.json").write_text(
        json.dumps({"clip": stats_all, "pilot": stats_pilot, "level_height_m": LEVEL_HEIGHT_M, "default_height_m": DEFAULT_HEIGHT_M}, indent=2),
        encoding="utf-8",
    )
    print("pilot height stats", stats_pilot)
    print("metrics", json.dumps(metrics, indent=2))
    plan = render_plan(buildings, water, roads, OUT / "00_plan_pilot_box.png")
    print("wrote", plan)
    for v in VIEWS:
        p = OUT / f"{v['id']}.png"
        render_view(v, buildings, water, roads, p)
        print("wrote", p)
    print("done")


if __name__ == "__main__":
    main()
