#!/usr/bin/env python3
"""Colored massing / simple-roof whitebox for style-transfer input.

Not a final artwork. Calibrated control heights + typological defaults,
Overture-only footprints, palace/hutong/commercial palettes, hard shadows.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from shapely.geometry import Polygon, Point, shape, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT = Path("/workspace/beijing-pilot")
OSM = ROOT / "data" / "osm_map.xml"
CAL = json.loads((ROOT / "data" / "height_calibration.json").read_text())
OVERTURE = ROOT / "data" / "overture_buildings.geojson"
OUT = ROOT / "renders_v2"
OUT.mkdir(parents=True, exist_ok=True)

PILOT = dict(west=116.39480, south=39.90620, east=116.40780, north=39.91800)
AZIMUTH_DEG = -15.0
ELEVATION_DEG = -45.0
VIEW_HEIGHT_M = 300.0
WIDTH_PX = 1024
HEIGHT_PX = 1024

# Light in rotated camera frame: from upper-left, shadows go down-right.
LIGHT = np.array([-0.55, 0.35, 1.15], dtype=float)
LIGHT = LIGHT / np.linalg.norm(LIGHT)

VIEWS = [
    {"id": "donghuamen", "title": "东华门 / 筒子河东段 / 故宫东墙", "lat": 39.91370, "lon": 116.39580},
    {"id": "nanchizi", "title": "南池子大街 / 胡同四合院", "lat": 39.91035, "lon": 116.39810},
    {"id": "wangfujing-north", "title": "王府井大街北段 / 金鱼胡同", "lat": 39.91490, "lon": 116.40540},
    {"id": "wangfujing-south", "title": "王府井大街南段 / 北京饭店", "lat": 39.90840, "lon": 116.40490},
]


def parse_osm(path: Path):
    nodes, ways, rels = {}, [], []
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
            members = [{"type": m.get("type"), "ref": m.get("ref"), "role": m.get("role")} for m in el.findall("member")]
            rels.append({"id": el.get("id"), "tags": tags, "members": members})
            el.clear()
    return nodes, ways, rels


def way_coords(way, nodes):
    coords = [nodes[i] for i in way["nds"] if i in nodes]
    return coords if len(coords) >= 2 else None


def lonlat_poly(coords):
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    if len(coords) < 4:
        return None
    try:
        p = Polygon(coords)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area <= 0:
            return None
        return p
    except Exception:
        return None


PALACE_LONS = 116.39635  # east of moat / east wall
HUTONG_LONS = 116.40180  # west of Wangfujing commercial core


def classify(tags, lon, lat, kind_hint=None):
    name = tags.get("name") or ""
    btyp = tags.get("building") or ""
    if kind_hint:
        return kind_hint
    if "城墙" in name or btyp == "wall":
        return "palace_wall"
    if tags.get("historic") == "city_gate" or name in ("东华门", "西华门", "保泰门", "敛禧门"):
        return "palace_gate"
    if lon <= PALACE_LONS:
        if any(k in name for k in ("殿", "宫", "阁", "楼", "库", "堂", "馆", "门", "所")):
            return "palace_hall"
        return "palace_hall" if btyp != "roof" else "palace_hall"
    if "天主堂" in name or btyp == "church":
        return "church"
    if btyp in ("hotel",) or any(k in name for k in ("饭店", "酒店", "公寓")):
        return "hotel"
    if btyp in ("retail", "commercial") or any(k in name for k in ("购物", "大厦", "apm", "银泰", "中环", "中心", "商场", "书店")):
        return "commercial"
    if btyp in ("office", "hospital", "school") or any(k in name for k in ("部", "院", "医院", "学校", "小学")):
        return "office"
    if lon >= HUTONG_LONS:
        if btyp in ("house",) or "胡同" in name or "故居" in name:
            return "hutong"
        # default east of Nanheyan / Chenguang: mixed commercial if large, else hutong
        return "commercial_or_hutong"
    return "hutong"


def height_for(osm_id, tags, typ, area_m2):
    cal = CAL["by_osm_way"].get(str(osm_id))
    if cal:
        return float(cal["height_m"]), "calibrated", cal.get("uncertain", False)
    raw = tags.get("height") or tags.get("building:height")
    if raw:
        try:
            h = float(str(raw).lower().replace("m", "").strip())
            if h > 0:
                return h, "osm_height", False
        except ValueError:
            pass
    levels = None
    raw = tags.get("building:levels") or tags.get("levels")
    if raw:
        try:
            levels = float(str(raw).split(";")[0].split(",")[0])
        except ValueError:
            levels = None
    td = CAL["typology_defaults"]
    if typ in ("palace_wall",):
        return td["palace_wall_m"], "typology", False
    if typ == "palace_gate":
        return 23.0, "typology", True
    if typ == "palace_hall":
        h = 15.0 if area_m2 >= 400 else 12.0
        if levels:
            h = max(h, levels * 3.5)
        return h, "typology", True
    if typ == "hutong":
        if levels and levels >= 2:
            return 3.5 * levels + 1.8, "levels_hutong", True
        return td["hutong_1storey_ridge_m"], "typology", False
    if typ == "church":
        return 28.0, "typology", True
    # commercial / hotel / office
    mpf = {"hotel": td["hotel_m_per_floor"], "commercial": td["retail_m_per_floor"],
           "office": td["office_m_per_floor"], "commercial_or_hutong": td["default_m_per_floor"]}.get(typ, td["default_m_per_floor"])
    if levels and typ == "commercial_or_hutong" and levels <= 2:
        return 3.5 * levels + 1.6, "levels_hutong", True
    if levels:
        return levels * mpf, "levels", True
    # no levels
    if typ == "commercial_or_hutong":
        if area_m2 < 250:
            return td["hutong_1storey_ridge_m"], "typology", True
        typ = "commercial"
    if typ == "hotel":
        return 32.0 if area_m2 >= 2000 else 24.0, "typology", True
    if typ in ("commercial", "office"):
        if area_m2 >= 8000:
            return 30.0, "typology", True
        if area_m2 >= 3000:
            return 24.0, "typology", True
        if area_m2 >= 800:
            return 18.0, "typology", True
        return 12.0, "typology", True
    return 6.3, "typology", True


def has_pitched_roof(typ, h):
    return typ in ("hutong", "palace_hall", "palace_gate", "church") or (typ == "commercial_or_hutong" and h <= 10)


# palettes: roof, roof_dark, wall, wall_dark, wall_lit
PAL = {
    "palace_hall":  ((218, 176, 52), (176, 132, 28), (176, 46, 40), (128, 30, 28), (200, 72, 58)),
    "palace_gate":  ((224, 186, 58), (180, 136, 30), (168, 42, 38), (122, 28, 26), (196, 68, 56)),
    "palace_wall":  ((150, 142, 132), (118, 110, 102), (150, 72, 52), (112, 50, 40), (176, 92, 68)),
    "hutong":       ((118, 124, 126), (88, 94, 96), (176, 166, 152), (140, 130, 118), (198, 188, 174)),
    "commercial":   ((142, 144, 146), (110, 112, 114), (186, 176, 162), (148, 138, 126), (210, 202, 190)),
    "hotel":        ((168, 170, 174), (128, 130, 134), (198, 196, 190), (158, 156, 150), (222, 220, 214)),
    "office":       ((150, 156, 162), (116, 122, 128), (164, 172, 178), (128, 136, 142), (188, 196, 202)),
    "church":       ((132, 80, 70), (100, 56, 50), (210, 206, 198), (168, 164, 156), (230, 226, 218)),
    "commercial_or_hutong": ((122, 126, 128), (92, 96, 98), (180, 170, 156), (144, 134, 122), (200, 190, 176)),
    "overture_hutong": ((118, 124, 126), (88, 94, 96), (168, 160, 148), (132, 124, 114), (190, 182, 170)),
    "overture_commercial": ((146, 148, 150), (114, 116, 118), (176, 168, 158), (140, 132, 124), (200, 194, 186)),
}


def tint(typ, osm_id):
    rng = np.random.RandomState(int(str(osm_id)[-6:]) if str(osm_id).isdigit() else hash(str(osm_id)) % 100000)
    roof, rd, wall, wd, wl = PAL.get(typ, PAL["hutong"])
    j = rng.randint(-10, 11)
    def add(c, j):
        return tuple(max(0, min(255, v + j)) for v in c)
    # commercial variety: some glassier, some brickier
    if typ in ("commercial", "office", "hotel") and rng.rand() < 0.35:
        wall = add((120, 148, 162), j)
        wd = add((90, 112, 126), j)
        wl = add((150, 176, 188), j)
    elif typ in ("commercial", "hotel") and rng.rand() < 0.25:
        wall = add((196, 188, 176), j)
        wd = add((156, 148, 136), j)
        wl = add((220, 214, 204), j)
    else:
        wall, wd, wl = add(wall, j), add(wd, j), add(wl, j)
        roof, rd = add(roof, j // 2), add(rd, j // 2)
    return roof, rd, wall, wd, wl


class Projector:
    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = lon0, lat0
        self.az = math.radians(AZIMUTH_DEG)
        self.elev = math.radians(ELEVATION_DEG)
        self.mpp = VIEW_HEIGHT_M / HEIGHT_PX
        self.to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
        self.e0, self.n0 = self.to_utm.transform(lon0, lat0)
        self.cos_a, self.sin_a = math.cos(self.az), math.sin(self.az)
        self.s = -math.sin(self.elev)

    def en(self, lon, lat):
        e, n = self.to_utm.transform(lon, lat)
        return e - self.e0, n - self.n0

    def rotate(self, e, n):
        return e * self.cos_a - n * self.sin_a, e * self.sin_a + n * self.cos_a

    def project_xyz(self, e, n, z=0.0):
        x, y = self.rotate(e, n)
        sx, sy = x, self.s * y + self.s * z
        depth = self.s * y - self.s * z
        px = WIDTH_PX / 2 + sx / self.mpp
        py = HEIGHT_PX / 2 - sy / self.mpp
        return px, py, depth

    def shade_side(self, p0, p1):
        # outward-ish 2d normal of wall in rotated X/Y, compare to light
        e0, n0, _ = p0
        e1, n1, _ = p1
        x0, y0 = self.rotate(e0, n0)
        x1, y1 = self.rotate(e1, n1)
        # wall direction, normal pointing (dy, -dx) or vice versa
        dx, dy = x1 - x0, y1 - y0
        nx, ny = dy, -dx
        nrm = math.hypot(nx, ny) or 1
        nx, ny = nx / nrm, ny / nrm
        # light in rotated frame (LIGHT x,y are already in rot space)
        # We defined LIGHT in rotated coordinates.
        lit = nx * LIGHT[0] + ny * LIGHT[1]
        return lit  # positive = facing light


def footprint_area_m2(coords, proj_tmp=None):
    # local EN using first point
    t = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    e0, n0 = t.transform(coords[0][0], coords[0][1])
    pts = []
    for lon, lat in coords:
        e, n = t.transform(lon, lat)
        pts.append((e - e0, n - n0))
    poly = lonlat_poly(pts)
    return poly.area if poly else 0.0


def ridge_geometry(en_coords, eave_z, ridge_z):
    if en_coords[0] != en_coords[-1]:
        en_coords = en_coords + [en_coords[0]]
    try:
        poly = Polygon(en_coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        mrr = poly.minimum_rotated_rectangle
        if mrr.is_empty:
            return None
        xs, ys = mrr.exterior.coords[0], None
        corners = list(mrr.exterior.coords)[:-1]
        if len(corners) != 4:
            return None
        # edges
        d01 = math.hypot(corners[1][0]-corners[0][0], corners[1][1]-corners[0][1])
        d12 = math.hypot(corners[2][0]-corners[1][0], corners[2][1]-corners[1][1])
        if d01 >= d12:
            a, b, c, d = corners[0], corners[1], corners[2], corners[3]
        else:
            a, b, c, d = corners[1], corners[2], corners[3], corners[0]
        # long edges a-b and d-c; ridge midpoints
        r0 = ((a[0]+d[0])/2, (a[1]+d[1])/2, ridge_z)
        r1 = ((b[0]+c[0])/2, (b[1]+c[1])/2, ridge_z)
        eave_ab = [(a[0], a[1], eave_z), (b[0], b[1], eave_z)]
        eave_dc = [(d[0], d[1], eave_z), (c[0], c[1], eave_z)]
        plane1 = [eave_ab[0], eave_ab[1], r1, r0]
        plane2 = [eave_dc[0], eave_dc[1], r1, r0]
        return plane1, plane2, (r0, r1)
    except Exception:
        return None


def assemble(nodes, ways, rels):
    buildings = []
    water, roads = [], []
    osm_polys = []
    t = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

    def en_list(coords, origin=None):
        if origin is None:
            origin = t.transform(coords[0][0], coords[0][1])
        out = []
        for lon, lat in coords:
            e, n = t.transform(lon, lat)
            out.append((e, n))
        return out

    for w in ways:
        tags = w["tags"]
        coords = way_coords(w, nodes)
        if not coords:
            continue
        if "building" in tags or tags.get("building:part"):
            poly = lonlat_poly(coords)
            if not poly:
                continue
            lon, lat = poly.centroid.x, poly.centroid.y
            area = footprint_area_m2(coords)
            typ = classify(tags, lon, lat)
            if typ == "commercial_or_hutong" and area < 280:
                typ = "hutong"
            elif typ == "commercial_or_hutong" and area >= 800:
                typ = "commercial"
            h, src, unc = height_for(w["id"], tags, typ, area)
            buildings.append({
                "id": w["id"], "source": "osm", "name": tags.get("name"),
                "coords": coords, "poly": poly, "height": h, "height_src": src,
                "uncertain": unc, "typ": typ, "area": area, "lon": lon, "lat": lat,
                "part": bool(tags.get("building:part")),
            })
            osm_polys.append(poly)
        elif tags.get("natural") == "water" or tags.get("water") or tags.get("landuse") in ("basin", "reservoir"):
            if lonlat_poly(coords):
                water.append({"name": tags.get("name"), "coords": coords})
        elif tags.get("highway") and tags.get("highway") not in ("proposed",):
            roads.append({"coords": coords, "highway": tags.get("highway"), "name": tags.get("name")})

    way_index = {w["id"]: w for w in ways}
    for r in rels:
        tags = r["tags"]
        if tags.get("natural") == "water" or tags.get("water"):
            for m in r["members"]:
                if m["type"] == "way" and m["ref"] in way_index:
                    c = way_coords(way_index[m["ref"]], nodes)
                    if c and lonlat_poly(c):
                        water.append({"name": tags.get("name"), "coords": c})

    # Overture unique footprints
    added = 0
    if OVERTURE.exists():
        gj = json.loads(OVERTURE.read_text())
        valid_osm = [p for p in osm_polys if p is not None and not p.is_empty]
        tree = STRtree(valid_osm) if valid_osm else None
        for feat in gj["features"]:
            geom = feat.get("geometry")
            if not geom:
                continue
            try:
                g = shape(geom)
                if not g.is_valid:
                    g = g.buffer(0)
                if g.is_empty:
                    continue
                if g.geom_type == "MultiPolygon":
                    parts = list(g.geoms)
                else:
                    parts = [g]
            except Exception:
                continue
            props = feat.get("properties") or {}
            for part in parts:
                if part.is_empty or part.area <= 0:
                    continue
                c = part.centroid
                # skip far outside padded box
                if not (116.3940 <= c.x <= 116.4088 and 39.9055 <= c.y <= 39.9190):
                    continue
                overlap = 0.0
                if tree is not None:
                    cands = tree.query(part)
                    for ix in cands:
                        osm_p = valid_osm[int(ix)] if not hasattr(ix, "area") else ix
                        # STRtree.query in shapely 2 returns indices
                    # handle both
                    try:
                        for ix in np.atleast_1d(cands):
                            osm_p = valid_osm[int(ix)]
                            if part.intersects(osm_p):
                                overlap += part.intersection(osm_p).area
                    except Exception:
                        pass
                if part.area > 0 and overlap / part.area >= 0.25:
                    continue
                # unique or mostly unique
                coords = list(part.exterior.coords)
                area = footprint_area_m2(coords)
                typ = classify({}, c.x, c.y)
                if typ == "commercial_or_hutong":
                    typ = "hutong" if area < 400 else "commercial"
                # typ palette prefix
                vis_typ = "overture_hutong" if typ == "hutong" else "overture_commercial" if typ == "commercial" else typ
                htag = props.get("height")
                floors = props.get("num_floors")
                tags = {}
                if htag:
                    tags["height"] = str(htag)
                if floors:
                    tags["building:levels"] = str(floors)
                h, src, unc = height_for("ovt", tags, typ, area)
                buildings.append({
                    "id": f"ovt/{props.get('id') or added}", "source": "overture",
                    "name": props.get("name"), "coords": coords, "poly": part,
                    "height": h, "height_src": "overture_" + src, "uncertain": True,
                    "typ": vis_typ, "area": area, "lon": c.x, "lat": c.y, "part": False,
                })
                added += 1
    print(f"buildings={len(buildings)} (overture unique added≈{added}) water={len(water)} roads={len(roads)}")
    return buildings, water, roads


def shadow_poly(en_ring, height, proj: Projector):
    # LIGHT is in rotated space; convert to EN for extrusion on ground
    # rotated: x = e ca - n sa; y = e sa + n ca
    # inverse: e = x ca + y sa; n = -x sa + y ca
    lx, ly, lz = LIGHT
    # horizontal shadow vector in rotated space: -horizontal_light / lz * height
    if abs(lz) < 1e-6:
        return None
    sx_r = -lx / lz * height
    sy_r = -ly / lz * height
    ca, sa = proj.cos_a, proj.sin_a
    de = sx_r * ca + sy_r * sa
    dn = -sx_r * sa + sy_r * ca
    base = en_ring if en_ring[0] == en_ring[-1] else en_ring + [en_ring[0]]
    shift = [(e + de, n + dn) for e, n in base]
    # union-ish silhouette: base + shift (simple: concatenate as a lozenge along drop)
    # Use shapely
    try:
        a = Polygon(base)
        b = Polygon(shift)
        hull = unary_union([a, b, Polygon([base[0], base[-2] if base[0]==base[-1] else base[-1], shift[-2] if shift[0]==shift[-1] else shift[-1], shift[0]])])
        # better: Minkowski-like — convex hull of base+shift is OK for shadows
        from shapely.geometry import MultiPoint
        hull = MultiPoint(base + shift).convex_hull
        if hull.is_empty:
            return None
        return [(e, n, 0.0) for e, n in hull.exterior.coords]
    except Exception:
        return None


def build_faces(bldg, proj: Projector):
    coords = bldg["coords"]
    en = [proj.en(lon, lat) for lon, lat in coords]
    if en[0] != en[-1]:
        en = en + [en[0]]
    h = bldg["height"]
    typ = bldg["typ"]
    roof_c, roof_d, wall, wall_d, wall_l = tint(typ, bldg["id"])
    pitched = has_pitched_roof(typ, h)
    if typ == "palace_wall":
        eave_z = h
        ridge_z = h  # flat
        pitched = False
    elif pitched:
        if typ in ("palace_hall", "palace_gate"):
            eave_z = h * 0.72
            ridge_z = h
        elif typ == "church":
            eave_z = h * 0.7
            ridge_z = h
        else:
            eave_z = min(h * 0.70, h - 1.4)
            ridge_z = h
    else:
        eave_z = h
        ridge_z = h

    faces = []
    # walls up to eave
    for i in range(len(en) - 1):
        p0 = (en[i][0], en[i][1], 0.0)
        p1 = (en[i + 1][0], en[i + 1][1], 0.0)
        if p0[0] == p1[0] and p0[1] == p1[1]:
            continue
        p2 = (en[i + 1][0], en[i + 1][1], eave_z)
        p3 = (en[i][0], en[i][1], eave_z)
        lit = proj.shade_side(p0, p1)
        col = wall_l if lit > 0.15 else (wall if lit > -0.15 else wall_d)
        depths = [proj.project_xyz(*p)[2] for p in (p0, p1, p2, p3)]
        faces.append({"pts": (p0, p1, p2, p3), "depth": sum(depths) / 4, "color": col, "kind": "wall",
                      "eave": eave_z, "typ": typ})
    if pitched:
        rg = ridge_geometry(en, eave_z, ridge_z)
        if rg:
            plane1, plane2, _ridge = rg
            for pl in (plane1, plane2):
                depths = [proj.project_xyz(*p)[2] for p in pl]
                # darker roof plane facing away
                # crude: average screen exposure
                mid = tuple(sum(c[k] for c in pl) / 4 for k in range(3))
                # compare light to roof normal later; use two shades by order
                faces.append({"pts": pl, "depth": sum(depths) / 4, "color": roof_c, "kind": "roof", "typ": typ})
            # make second slightly darker
            faces[-1]["color"] = roof_d
        else:
            roof = [(e, n, eave_z) for e, n in en]
            depths = [proj.project_xyz(*p)[2] for p in roof]
            faces.append({"pts": roof, "depth": sum(depths) / len(depths), "color": roof_c, "kind": "roof", "typ": typ})
    else:
        roof = [(e, n, eave_z) for e, n in en]
        depths = [proj.project_xyz(*p)[2] for p in roof]
        col = roof_c if typ != "palace_wall" else (150, 142, 132)
        faces.append({"pts": roof, "depth": sum(depths) / len(depths), "color": col, "kind": "roof", "typ": typ})
    return faces, en, h


def draw_poly(draw, proj, pts, fill, outline=None):
    xy = [proj.project_xyz(*p)[:2] for p in pts]
    if len(xy) >= 3:
        draw.polygon(xy, fill=fill, outline=outline)


def add_window_grid(img, proj, face):
    if face["kind"] != "wall":
        return
    if face.get("typ") not in ("commercial", "hotel", "office", "overture_commercial"):
        return
    pts = face["pts"]
    eave = face.get("eave") or pts[2][2]
    if eave < 10:
        return
    # sample a grid on the wall quad
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = pts
    width = math.hypot(x1 - x0, y1 - y0)
    if width < 8 or eave < 10:
        return
    cols = max(2, int(width / 4.2))
    rows = max(2, int(eave / 3.6))
    cols = min(cols, 18)
    rows = min(rows, 22)
    draw = ImageDraw.Draw(img)
    for r in range(rows):
        for c in range(cols):
            u0, u1 = (c + 0.22) / cols, (c + 0.78) / cols
            v0, v1 = (r + 0.28) / rows, (r + 0.82) / rows
            def corner(u, v):
                # bottom edge 0-1, top edge 3-2
                bx, by = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
                tx, ty = x3 + (x2 - x3) * u, y3 + (y2 - y3) * u
                bz, tz = 0.0, eave
                return (bx + (tx - bx) * v, by + (ty - by) * v, bz + (tz - bz) * v)
            q = [corner(u0, v0), corner(u1, v0), corner(u1, v1), corner(u0, v1)]
            xy = [proj.project_xyz(*p)[:2] for p in q]
            # skip if tiny
            if max(p[0] for p in xy) < 0 or min(p[0] for p in xy) > WIDTH_PX:
                continue
            draw.polygon(xy, fill=(55, 62, 70))


def zone_ground_color(lon, lat):
    if lon <= PALACE_LONS:
        return (198, 176, 132)  # packed earth / palace court
    if lon < HUTONG_LONS:
        return (206, 196, 178)
    return (186, 184, 178)


def render_view(view, buildings, water, roads, out_path: Path, labeled: bool = True):
    proj = Projector(view["lon"], view["lat"])
    # ground with slight zone tint via a coarse fill
    img = Image.new("RGB", (WIDTH_PX, HEIGHT_PX), (204, 196, 180))
    draw = ImageDraw.Draw(img, "RGBA")

    # paint ground quadrants
    # sample a grid in EN and project — too heavy. Instead fill base then overlay street color.

    # water
    for w in water:
        coords = w["coords"]
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        pts = [(*proj.en(lon, lat), 0.0) for lon, lat in coords]
        draw_poly(draw, proj, pts, (62, 118, 148), outline=(48, 96, 122))

    # roads
    for r in roads:
        hw = r["highway"]
        pedestrian = hw in ("pedestrian", "footway", "living_street")
        major = hw in ("trunk", "primary", "secondary")
        width = 7 if major else (6 if pedestrian else 3)
        color = (92, 90, 86) if major else ((168, 150, 126) if pedestrian else (110, 108, 104))
        xy = [proj.project_xyz(*proj.en(lon, lat), 0.0)[:2] for lon, lat in r["coords"]]
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=width, joint="curve")

    # select nearby buildings
    nearby = []
    for b in buildings:
        e, n = proj.en(b["lon"], b["lat"])
        if abs(e) < 650 and abs(n) < 750:
            nearby.append(b)

    # shadows first
    shadows = []
    for b in nearby:
        en = [proj.en(lon, lat) for lon, lat in b["coords"]]
        sp = shadow_poly(en, b["height"], proj)
        if not sp:
            continue
        depths = [proj.project_xyz(*p)[2] for p in sp]
        shadows.append((min(depths) - 50, sp))  # behind buildings slightly
    shadows.sort(key=lambda x: x[0])
    for _, sp in shadows:
        xy = [proj.project_xyz(*p)[:2] for p in sp]
        if len(xy) >= 3:
            draw.polygon(xy, fill=(58, 50, 42, 110))

    faces = []
    ens = []
    for b in nearby:
        fs, en, h = build_faces(b, proj)
        faces.extend(fs)
        ens.append((b, en, h))
    faces.sort(key=lambda f: f["depth"])
    # draw walls first pass without windows, then windows on wall faces, but painter's order:
    # I'll draw face then windows immediately
    px_img = img
    for f in faces:
        outline = (40, 36, 32) if f["kind"] == "roof" else None
        draw_poly(draw, proj, f["pts"], f["color"], outline=outline)
        if f["kind"] == "wall":
            add_window_grid(px_img, proj, f)

    # simple trees along moat: points offset from water rings
    tree_draw = ImageDraw.Draw(img)
    rng = np.random.RandomState(42)
    for w in water:
        coords = w["coords"]
        step = max(1, len(coords) // 40)
        for i, (lon, lat) in enumerate(coords[::step]):
            e, n = proj.en(lon, lat)
            for sign in (8.0, -8.0):
                # offset roughly east-west
                te, tn = e + sign, n + (rng.rand() - 0.5) * 6
                h = 7.0 + rng.rand() * 3
                # trunk + crown
                p0 = proj.project_xyz(te, tn, 0)[:2]
                p1 = proj.project_xyz(te, tn, h * 0.4)[:2]
                crown = []
                for a in range(6):
                    ang = a * math.pi / 3
                    crown.append(proj.project_xyz(te + math.cos(ang) * 2.2, tn + math.sin(ang) * 2.2, h)[:2])
                tree_draw.line([p0, p1], fill=(90, 70, 48), width=2)
                if len(crown) >= 3:
                    tree_draw.polygon(crown, fill=(46, 92, 58), outline=(32, 70, 44))

    img.save(out_path, "PNG")
    if labeled:
        lab = img.copy()
        d = ImageDraw.Draw(lab)
        try:
            font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 14, index=2)
        except Exception:
            font = ImageFont.load_default()
        d.rectangle([0, HEIGHT_PX - 36, WIDTH_PX, HEIGHT_PX], fill=(28, 26, 24, 200))
        d.text((10, HEIGHT_PX - 28), f"{view['title']}  WGS84 {view['lat']:.5f}N {view['lon']:.5f}E  az-15 el-45  colored whitebox input", font=font, fill=(240, 234, 224))
        lab.save(out_path.with_name(out_path.stem + "_labeled.png"), "PNG")
    return out_path


def render_plan(buildings, water, roads, out_path: Path):
    west, south, east, north = PILOT["west"], PILOT["south"], PILOT["east"], PILOT["north"]
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    corners = [to_utm.transform(lon, lat) for lon, lat in ((west, south), (east, south), (east, north), (west, north))]
    min_e, max_e = min(c[0] for c in corners) - 80, max(c[0] for c in corners) + 80
    min_n, max_n = min(c[1] for c in corners) - 80, max(c[1] for c in corners) + 80
    W = H = 1600
    scale = min(W / (max_e - min_e), H / (max_n - min_n))

    def xy(lon, lat):
        e, n = to_utm.transform(lon, lat)
        return (e - min_e) * scale, H - (n - min_n) * scale

    img = Image.new("RGB", (W, H), (230, 224, 210))
    draw = ImageDraw.Draw(img)
    boxp = [xy(west, south), xy(east, south), xy(east, north), xy(west, north)]
    draw.polygon(boxp, fill=(214, 204, 186), outline=(140, 40, 40))
    for w in water:
        pts = [xy(lon, lat) for lon, lat in w["coords"]]
        if len(pts) >= 3:
            draw.polygon(pts, fill=(62, 118, 148), outline=(48, 96, 122))
    for r in roads:
        pts = [xy(lon, lat) for lon, lat in r["coords"]]
        if len(pts) >= 2:
            draw.line(pts, fill=(90, 88, 84), width=2 if r["highway"] in ("trunk", "primary", "secondary", "pedestrian") else 1)
    for b in buildings:
        pts = [xy(lon, lat) for lon, lat in b["coords"]]
        if len(pts) < 3:
            continue
        col = PAL.get(b["typ"], PAL["hutong"])[2]
        if b["source"] == "overture":
            col = PAL.get(b["typ"], PAL["overture_hutong"])[2]
        draw.polygon(pts, fill=col, outline=(50, 46, 42))
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16, index=2)
    except Exception:
        font = ImageFont.load_default()
    for v in VIEWS:
        px, py = xy(v["lon"], v["lat"])
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(200, 30, 30))
        draw.text((px + 10, py - 8), v["id"], font=font, fill=(140, 20, 20))
    draw.line(boxp + [boxp[0]], fill=(160, 40, 40), width=3)
    img.save(out_path, "PNG")
    return out_path


def main():
    print("parsing OSM...")
    nodes, ways, rels = parse_osm(OSM)
    buildings, water, roads = assemble(nodes, ways, rels)
    # stats
    from collections import Counter
    c = Counter((b["typ"], b["height_src"]) for b in buildings)
    print("typ/src", c.most_common(20))
    cal_n = sum(1 for b in buildings if b["height_src"] == "calibrated")
    print("calibrated buildings drawn:", cal_n)
    ovt = sum(1 for b in buildings if b["source"] == "overture")
    print("overture unique:", ovt)
    render_plan(buildings, water, roads, OUT / "00_plan_colored.png")
    for v in VIEWS:
        p = OUT / f"{v['id']}.png"
        render_view(v, buildings, water, roads, p)
        print("wrote", p)
    print("done")


if __name__ == "__main__":
    main()
