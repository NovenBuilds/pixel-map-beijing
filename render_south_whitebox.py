#!/usr/bin/env python3
"""One-shot global whitebox: 午门以南 to 前门 (Duanmen–Tiananmen–Square–Qianmen).

Same UTM 50N origin, origin_rot, azimuth -15, elevation -45, view_height 300 m,
0.29296875 m/px as tiles/grid_axis.json. Moat / wall / landmarks are drawn once;
1024 tiles are slices of this single image.

Writes only tiles/whiteboxes/global_south/ (and updates data/height_calibration.json
plus tiles/grid_axis.json after backup). Does not touch tiles/styled/ or
tiles/mosaic/overview_*.jpg or tiles/whiteboxes/global/.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, Point, box as sbox
from shapely.ops import unary_union, linemerge, polygonize

ROOT = Path("/workspace/beijing-pilot")
sys.path.insert(0, str(ROOT))
import expand_axis as ea
import render_color_whitebox as rb
import build_tile_grid as tg

MPP = 300.0 / 1024.0  # 0.29296875, locked
# Whitebox window: docks to existing r3 south edge, covers Wumen south to Qianmen.
RMIN, RMAX = 1, 5
CMIN, CMAX = -3, 0
TILE = 1024
W, H = (CMAX - CMIN + 1) * TILE, (RMAX - RMIN + 1) * TILE

OUT = ROOT / "tiles" / "whiteboxes" / "global_south"
SLICES = OUT / "slices"
GRID_PATH = ROOT / "tiles" / "grid_axis.json"
CAL_PATH = ROOT / "data" / "height_calibration.json"

# New tiles actually required for landmarks + continuity (not every cell of the rectangle).
NEW_TILES = [
    (2, -3), (2, -2), (2, -1),
    (3, -3), (3, -2), (3, -1),
    (4, -2), (4, -1), (4, 0),
    (5, -2),
]

# Existing tiles included in the whitebox for seam alignment / already-present landmarks.
SEAM_EXISTING = [
    (1, -2), (1, -1), (1, 0),
    (2, 0),
    (3, 0),
]

OSM_PATHS = [
    ROOT / "data" / "osm_map.xml",
    ROOT / "data" / "osm_palace.xml",
    ROOT / "data" / "osm_axis_south.xml",
    ROOT / "data" / "osm_tiananmen_south.xml",
]

# Official / published 通高, keyed by OSM way (and duplicated on relation ids below).
SOUTH_HEIGHTS = {
    # 天安门：城楼主屋面。故宫博物院《紫禁城的图像学意义》与 1970 年重建记述均为通高 34.7 m。
    "637830242": {
        "height_m": 34.7,
        "name": "天安门（城楼主屋面，通高）",
        "source": "故宫博物院文章《紫禁城的图像学意义》：天安门高 34.7 米。人民网/中国政府网转述新华社：1970 年重建后实际通高 34.7 米。",
        "evidence": [
            "https://www.dpm.org.cn/Uploads/File/pdf/21/37/8d/21378dd0b3131359d9cafc62a7a3369f.pdf",
            "http://dangshi.people.com.cn/n/2013/0514/c85037-21479598.html",
            "http://www.gov.cn/test/2009-09/24/content_1425400_5.htm",
        ],
    },
    # 端门主屋面
    "637950096": {
        "height_m": 34.37,
        "name": "端门（城楼主屋面，通高）",
        "source": "故宫博物院《紫禁城的图像学意义》：端门高 34.37 米（同文列天安门 34.7 米、午门 35.6 米）。",
        "evidence": [
            "https://www.dpm.org.cn/Uploads/File/pdf/21/37/8d/21378dd0b3131359d9cafc62a7a3369f.pdf",
        ],
    },
    # 正阳门城楼（整座通高，OSM 无精细 part）
    "25109960": {
        "height_m": 43.65,
        "name": "正阳门城楼（城台+城楼通高）",
        "source": "北京市文物局（北京中轴线遗产保护中心·正阳门）：城楼通高 43.65 米。2005 年大修前实测，自地平至正脊上皮。",
        "evidence": [
            "https://wwj.beijing.gov.cn/bjww/362771/362772/545898/index.html",
            "https://xinwen.bjd.com.cn/content/s66e164e9e4b0c25b287b4ad0.html",
        ],
    },
    "439956436": {
        "height_m": 35.37,
        "name": "正阳门箭楼（通高）",
        "source": "北京市文物局（北京中轴线遗产保护中心·正阳门）：箭楼通高 35.37 米。",
        "evidence": [
            "https://wwj.beijing.gov.cn/bjww/362771/362772/545898/index.html",
        ],
    },
    "637834383": {
        "height_m": 37.94,
        "name": "人民英雄纪念碑（碑身通高）",
        "source": "北京市人民政府天安门地区管理委员会：通高 37.94 米。国家文物局、全国人大网同数。OSM 该 way 已有 height=37.94。",
        "evidence": [
            "https://tamgw.beijing.gov.cn/diqufuwu/zjtam/201912/t20191219_1321036.html",
            "http://www.ncha.gov.cn/art/2019/9/25/art_2247_156814.html",
        ],
    },
    # 纪念堂：把偏高的 OSM part 收到官方通高 33.6 m
    "637908758": {
        "height_m": 33.6,
        "name": "毛主席纪念堂（主体通高）",
        "source": "中新网（设计过程记述）与东城区政府介绍：主体建筑通高 33.6 米，平面 105.5×105.5 米。",
        "evidence": [
            "https://www.chinanews.com.cn/cul/2013/11-24/5539815.shtml",
            "https://www.bjdch.gov.cn/mldc/bglj/hswh/202304/t20230408_2993346.html",
        ],
    },
    "637908757": {
        "height_m": 33.6,
        "name": "毛主席纪念堂（檐口体量，按通高收）",
        "source": "同纪念堂通高 33.6 m；OSM height=35 略高，改为官方通高。",
        "evidence": [
            "https://www.chinanews.com.cn/cul/2013/11-24/5539815.shtml",
        ],
    },
    "637908759": {
        "height_m": 33.6,
        "name": "毛主席纪念堂（上层体量，按通高收）",
        "source": "同纪念堂通高 33.6 m；OSM height=35 略高，改为官方通高。",
        "evidence": [
            "https://www.chinanews.com.cn/cul/2013/11-24/5539815.shtml",
        ],
    },
    # 大会堂最高部位
    "637929804": {
        "height_m": 46.5,
        "name": "人民大会堂（最高部位）",
        "source": "中国政府网文化专题、visitbeijing 与常见建筑记述：南北长 336 米、东西宽 206 米，高 46.5 米（最高部位）。",
        "evidence": [
            "http://www.ncha.gov.cn/art/2017/11/6/art_2035_144839.html",
            "https://s.visitbeijing.com.cn/attraction/101940",
        ],
    },
    "637929779": {
        "height_m": 46.5,
        "name": "人民大会堂（最高部位，北/中段）",
        "source": "同大会堂最高部位 46.5 m；OSM height=45，改为公开通高。",
        "evidence": ["https://s.visitbeijing.com.cn/attraction/101940"],
    },
    "637929787": {
        "height_m": 46.5,
        "name": "人民大会堂（最高部位，南/中段）",
        "source": "同大会堂最高部位 46.5 m；OSM height=45，改为公开通高。",
        "evidence": ["https://s.visitbeijing.com.cn/attraction/101940"],
    },
    # 国博：光明日报 2011-03-29 记改扩建后最高处 42.5 m（26/34/42 三层）。OSM 48/46/43 偏高，封顶。
    "637939207": {
        "height_m": 42.5,
        "name": "中国国家博物馆（主体最高处）",
        "source": "《光明日报》2011年3月29日《新国博“新”在哪儿（上）》：建筑主体外观形成 26、34、42 米左右三个层次，最高处达到 42.5 米。",
        "evidence": [
            "https://epaper.gmw.cn/gmrb/html/2011-03/29/nw.D110000gmrb_20110329_1-09.htm",
        ],
    },
    "637939216": {
        "height_m": 42.5,
        "name": "中国国家博物馆（高层体量，按最高处收）",
        "source": "同国博最高处 42.5 m；OSM height=46，改为公开数。",
        "evidence": [
            "https://epaper.gmw.cn/gmrb/html/2011-03/29/nw.D110000gmrb_20110329_1-09.htm",
        ],
    },
}

# Additional part ids that OSM tagged taller than the published cap (国博 48 m parts).
MUSEUM_CAP_IDS = {
    "637939215",  # height 48
}
# We will also cap any remaining 国博 part > 42.5 after assemble, by name.

# Relation ids (so assemble_plus calibration lookup hits them too)
SOUTH_HEIGHTS_REL = {
    "8847697": {**SOUTH_HEIGHTS["637830242"], "name": "天安门（multipolygon，通高 34.7 m）"},
    "8847698": {**SOUTH_HEIGHTS["637830242"], "name": "天安门（building relation，通高 34.7 m）"},
    "8848564": {**SOUTH_HEIGHTS["637950096"], "name": "端门（building relation，通高 34.37 m）"},
    "8847722": {**SOUTH_HEIGHTS["637834383"], "name": "人民英雄纪念碑（multipolygon）"},
    "8847723": {**SOUTH_HEIGHTS["637834383"], "name": "人民英雄纪念碑（building relation）"},
    "8848142": {**SOUTH_HEIGHTS["637908758"], "name": "毛主席纪念堂（multipolygon，通高 33.6 m）"},
    "8848144": {**SOUTH_HEIGHTS["637908758"], "name": "毛主席纪念堂（building relation，通高 33.6 m）"},
    "8848388": {**SOUTH_HEIGHTS["637929804"], "name": "人民大会堂（multipolygon，最高 46.5 m）"},
    "8848390": {**SOUTH_HEIGHTS["637929804"], "name": "人民大会堂（building relation，最高 46.5 m）"},
    "8607825": {**SOUTH_HEIGHTS["637939207"], "name": "中国国家博物馆（multipolygon，最高 42.5 m）"},
    "8848468": {**SOUTH_HEIGHTS["637939207"], "name": "中国国家博物馆（building relation，最高 42.5 m）"},
}

# Parent / untagged outers that would paint a second solid over detailed parts.
DROP_OUTLINES = {
    "637830319", "637830343",          # 天安门 untagged double-outer
    "25097188",                        # 端门 gatehouse parent (keep parts)
    "637929798", "637929778",          # 大会堂 outer / inner
    "637939189", "637939181",          # 国博 two outers
    # 纪念堂 untagged outline fragments
    "637895268", "637895176", "637895269", "637895212", "637895274",
    "637895262", "637895265", "637895241", "637895282", "637895257",
    "637895287", "637895297",
    # 纪念碑 multipolygon outers (keep shaft/base parts 637834382/383)
    "637467890", "637880215", "637880217", "637880210", "637467889",
    "637880212", "637880220", "637880209", "637467891", "637880207",
    "637880221", "637880206", "637880198", "637880196", "637880224",
    "637880213",
}

LANDMARK_CENTROIDS = {
    "午门": (116.390998, 39.912234),
    "端门": (116.391153, 39.909026),
    "天安门城楼": (116.391297, 39.907369),
    "外金水桥": (116.391237, 39.906780),
    "人民英雄纪念碑": (116.391429, 39.903193),
    "毛主席纪念堂": (116.391544, 39.901110),
    "人民大会堂": (116.387128, 39.903278),
    "国家博物馆": (116.395661, 39.903710),
    "正阳门城楼": (116.391526, 39.899185),
    "正阳门箭楼": (116.391638, 39.897966),
    "前门大街北段": (116.39174, 39.89650),
    "长安街西段": (116.38750, 39.90610),
}

# Footprints used to assign a primary tile (largest overlap) and to warn about seams.
LANDMARK_BBOXES = {
    "午门": (116.389885, 39.911621, 116.392115, 39.912618),
    "端门": (116.390486, 39.908812, 116.391876, 39.909193),
    "天安门城楼": (116.390568, 39.907148, 116.391962, 39.907529),
    "外金水桥": (116.390908, 39.906591, 116.391577, 39.906962),
    "人民英雄纪念碑": (116.391088, 39.902888, 116.391762, 39.903506),
    "毛主席纪念堂": (116.390912, 39.900567, 116.392178, 39.901662),
    "人民大会堂": (116.386034, 39.901929, 116.388577, 39.904992),
    "国家博物馆": (116.394207, 39.902206, 116.396680, 39.905217),
    "正阳门城楼": (116.390973, 39.899073, 116.392262, 39.899295),
    "正阳门箭楼": (116.391300, 39.897804, 116.392017, 39.898140),
    "天安门广场": (116.389653, 39.898888, 116.393277, 39.905831),
    "长安街西段": (116.3845, 39.9058, 116.3906, 39.9072),
    "前门大街北段": (116.3913, 39.8945, 116.3922, 39.8978),
}

GATE_GREY = {"正阳门", "正阳门箭楼"}
GATE_YELLOW = {"天安门", "端门", "午门"}
NAME_MONUMENT = {"人民英雄纪念碑"}
NAME_MEMORIAL = {"毛主席纪念堂"}
NAME_HALL = {"人民大会堂"}
NAME_MUSEUM = {"中国国家博物馆", "国家博物馆"}


def update_calibration():
    cal = json.loads(CAL_PATH.read_text())
    n_new = 0
    for k, v in {**SOUTH_HEIGHTS, **SOUTH_HEIGHTS_REL}.items():
        cal["by_osm_way"][k] = v
        n_new += 1
    # cap leftover museum 48 m parts
    for k in MUSEUM_CAP_IDS:
        cal["by_osm_way"][k] = {
            "height_m": 42.5,
            "name": "中国国家博物馆（高层体量，按最高处 42.5 m 收）",
            "source": "同国博最高处 42.5 m（《光明日报》2011-03-29）。",
            "evidence": [
                "https://epaper.gmw.cn/gmrb/html/2011-03/29/nw.D110000gmrb_20110329_1-09.htm",
            ],
        }
        n_new += 1
    CAL_PATH.write_text(json.dumps(cal, ensure_ascii=False, indent=2) + "\n")
    rb.CAL = cal
    print("updated", CAL_PATH, "south-axis entries written:", n_new)


# Bound to expand_axis.classify as it was at import time (before we patch ea.classify).
_ORIG_EA_CLASSIFY = ea.classify


def classify_south(tags, lon, lat, kind_hint=None):
    name = tags.get("name") or ""
    if kind_hint:
        return kind_hint
    if name in GATE_GREY:
        return "city_gate_grey"
    if name in GATE_YELLOW or (
        tags.get("historic") == "city_gate" and name in GATE_YELLOW
    ):
        return "palace_gate"
    if name in NAME_MONUMENT or (
        tags.get("man_made") == "obelisk" and "英雄" in name
    ):
        return "monument"
    if name in NAME_MEMORIAL:
        return "memorial_hall"
    if name in NAME_HALL:
        return "civic_hall"
    if name in NAME_MUSEUM:
        return "museum"
    if tags.get("man_made") == "bridge" and "金水桥" in name:
        return "stone_bridge"
    # South of Chang'an / Tiananmen: do not default everything west of
    # PALACE_LONS to palace_hall (that was meant for the Forbidden City).
    # Shift lon just east of the palace cutoff so hutong/commercial logic applies.
    if lat is not None and lat < 39.9073:
        return _ORIG_EA_CLASSIFY(tags, max(lon, 116.39640), lat, kind_hint)
    return _ORIG_EA_CLASSIFY(tags, lon, lat, kind_hint)


def patch_styles():
    rb.PAL["city_gate_grey"] = (
        (96, 108, 96), (70, 80, 70), (168, 52, 48), (122, 34, 32), (196, 72, 64)
    )  # grey tile + green-grey ridge, red walls (正阳门 灰筒瓦绿琉璃剪边)
    rb.PAL["monument"] = (
        (206, 202, 194), (168, 164, 156), (200, 196, 188), (150, 146, 138), (220, 216, 208)
    )
    rb.PAL["memorial_hall"] = (
        (204, 176, 72), (164, 136, 48), (186, 166, 132), (148, 128, 100), (210, 194, 160)
    )  # granite walls, golden eaves
    rb.PAL["civic_hall"] = (
        (186, 168, 84), (150, 132, 60), (210, 196, 162), (168, 154, 124), (226, 214, 182)
    )
    rb.PAL["museum"] = (
        (176, 162, 86), (140, 126, 62), (200, 190, 172), (158, 148, 132), (220, 212, 196)
    )
    rb.PAL["stone_bridge"] = (
        (214, 210, 204), (180, 176, 170), (210, 206, 200), (168, 164, 158), (226, 222, 216)
    )
    orig_pitched = rb.has_pitched_roof

    def has_pitched(typ, h):
        if typ in ("city_gate_grey",):
            return True
        if typ in ("monument", "memorial_hall", "civic_hall", "museum", "stone_bridge"):
            return False
        return orig_pitched(typ, h)

    rb.has_pitched_roof = has_pitched

    orig_windows = rb.add_window_grid

    def add_window_grid(img, proj, face):
        typ = face.get("typ")
        if typ in ("civic_hall", "museum"):
            face = dict(face)
            face["typ"] = "office"
        return orig_windows(img, proj, face)

    rb.add_window_grid = add_window_grid


def tile_poly(R, C):
    x0 = ea.MINX + C * ea.TILE_WX
    x1 = ea.MINX + (C + 1) * ea.TILE_WX
    y1 = ea.MAXY - R * ea.TILE_WY
    y0 = ea.MAXY - (R + 1) * ea.TILE_WY
    corners = [tg.rot_to_lonlat(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return Polygon(corners)


def tile_of_lonlat(lon, lat):
    rx, ry = tg.lonlat_to_rot(lon, lat)
    C = math.floor((rx - ea.MINX) / ea.TILE_WX)
    R = math.floor((ea.MAXY - ry) / ea.TILE_WY)
    fx = (rx - ea.MINX) / ea.TILE_WX - C
    fy = (ea.MAXY - ry) / ea.TILE_WY - R
    return R, C, fx, fy


def mosaic_rot_bounds():
    x0 = ea.MINX + CMIN * ea.TILE_WX
    x1 = ea.MINX + (CMAX + 1) * ea.TILE_WX
    y1 = ea.MAXY - RMIN * ea.TILE_WY
    y0 = ea.MAXY - (RMAX + 1) * ea.TILE_WY
    return x0, x1, y0, y1


class GlobalProjector:
    """Same isometric as render_global_whitebox.GlobalProjector, south window."""

    def __init__(self):
        x0, x1, y0, y1 = mosaic_rot_bounds()
        self.cx = (x0 + x1) / 2.0
        self.cy = (y0 + y1) / 2.0
        self.lon0, self.lat0 = ea.rot_to_lonlat(self.cx, self.cy)
        self.az = math.radians(rb.AZIMUTH_DEG)
        self.elev = math.radians(rb.ELEVATION_DEG)
        self.mpp = MPP
        self.to_utm = tg.TO_UTM
        self.cos_a, self.sin_a = math.cos(self.az), math.sin(self.az)
        self.s = -math.sin(self.elev)
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def en(self, lon, lat):
        e, n = tg.lonlat_to_en(lon, lat)
        return e, n

    def rotate(self, e, n):
        return e * self.cos_a - n * self.sin_a, e * self.sin_a + n * self.cos_a

    def project_xyz(self, e, n, z=0.0):
        rx, ry = self.rotate(e, n)
        px = (rx - self.x0) / self.mpp
        py = self.s * (self.y1 - ry - z) / self.mpp
        depth = self.s * ry - self.s * z
        return px, py, depth

    def shade_side(self, p0, p1):
        return rb.Projector.shade_side(self, p0, p1)


def collect_named_areas(nodes, ways, rels):
    """Plaza / water polygons we want to paint as ground or water."""
    way_index = {w["id"]: w for w in ways}
    areas = {"square": [], "jinshui": [], "bridges": []}

    def way_coords(w):
        return rb.way_coords(w, nodes)

    for w in ways:
        tags = w["tags"]
        name = tags.get("name") or ""
        coords = way_coords(w)
        if not coords:
            continue
        if name == "外金水河" or (
            tags.get("waterway") in ("drain", "canal", "river") and "金水" in name and "内" not in name
        ):
            areas["jinshui"].append(coords)
        if tags.get("man_made") == "bridge" and name == "外金水桥":
            areas["bridges"].append({"id": w["id"], "coords": coords, "tags": tags})

    for r in rels:
        tags = r["tags"]
        name = tags.get("name") or ""
        if name != "天安门广场":
            continue
        rings = []
        for m in r["members"]:
            if m["type"] != "way" or m["ref"] not in way_index:
                continue
            c = way_coords(way_index[m["ref"]])
            if c:
                rings.append(c)
        areas["square"].append(rings)
    return areas


def add_bridges_and_water(buildings, water, areas):
    # 外金水河 may be a closed polygon or a fat centerline
    for coords in areas["jinshui"]:
        poly = rb.lonlat_poly(coords)
        if poly:
            water.append({"name": "外金水河", "coords": coords})
        else:
            # treat as a thin band later via water stroke
            water.append({"name": "外金水河", "coords": coords, "as_line": True})

    have = {str(b["id"]) for b in buildings}
    for br in areas["bridges"]:
        if str(br["id"]) in have:
            # relabel if already ingested
            for b in buildings:
                if str(b["id"]) == str(br["id"]):
                    b["typ"] = "stone_bridge"
                    b["height"] = min(float(b["height"] or 3.5), 4.0)
                    b["name"] = b.get("name") or "外金水桥"
            continue
        poly = rb.lonlat_poly(br["coords"])
        if not poly:
            continue
        lon, lat = poly.centroid.x, poly.centroid.y
        buildings.append({
            "id": br["id"], "source": "osm", "name": "外金水桥",
            "coords": br["coords"], "poly": poly, "height": 3.6,
            "height_src": "typology_bridge", "uncertain": True,
            "typ": "stone_bridge", "area": rb.footprint_area_m2(br["coords"]),
            "lon": lon, "lat": lat, "part": False,
        })
        have.add(str(br["id"]))
    return buildings, water


def fix_south_massing(buildings):
    out = []
    dropped = Counter()
    for b in buildings:
        bid = str(b["id"])
        name = b.get("name") or ""
        if bid in DROP_OUTLINES:
            dropped["outline"] += 1
            continue
        # cap museum leftovers
        if name in NAME_MUSEUM and b["height"] > 42.5:
            b["height"] = 42.5
            b["height_src"] = "calibrated_cap"
        if name in NAME_MEMORIAL and b["height"] > 33.6 and (b.get("area") or 0) >= 200:
            # keep 24 m colonnade posts; cap the main volumes
            b["height"] = 33.6
            b["height_src"] = "calibrated_cap"
        if name in NAME_HALL and b["height"] > 46.5:
            b["height"] = 46.5
            b["height_src"] = "calibrated_cap"
        # Force typ by footprint first (building:part usually has no name),
        # then by exact / substring name.
        typ_set = False
        try:
            pt = Point(b["lon"], b["lat"])
        except Exception:
            pt = None
        if pt is not None:
            def in_bb(key):
                bb = LANDMARK_BBOXES[key]
                return (bb[0] <= pt.x <= bb[2]) and (bb[1] <= pt.y <= bb[3])
            if in_bb("天安门城楼") or in_bb("端门") or in_bb("午门"):
                b["typ"] = "palace_gate"; typ_set = True
            elif in_bb("正阳门城楼") or in_bb("正阳门箭楼"):
                b["typ"] = "city_gate_grey"; typ_set = True
            elif in_bb("人民英雄纪念碑"):
                b["typ"] = "monument"; typ_set = True
            elif in_bb("毛主席纪念堂"):
                b["typ"] = "memorial_hall"; typ_set = True
            elif in_bb("人民大会堂"):
                b["typ"] = "civic_hall"; typ_set = True
            elif in_bb("国家博物馆"):
                b["typ"] = "museum"; typ_set = True
        if not typ_set:
            if name in GATE_GREY or name.startswith("正阳门"):
                b["typ"] = "city_gate_grey"
            elif any(name == k or name.startswith(k) for k in GATE_YELLOW):
                b["typ"] = "palace_gate"
            elif name.startswith("人民英雄纪念碑") or name in NAME_MONUMENT:
                b["typ"] = "monument"
            elif name.startswith("毛主席纪念堂"):
                b["typ"] = "memorial_hall"
            elif name.startswith("人民大会堂"):
                b["typ"] = "civic_hall"
            elif name.startswith("中国国家博物馆") or name.startswith("国家博物馆"):
                b["typ"] = "museum"
        out.append(b)
    print(f"fix_south_massing: dropped {dict(dropped)} kept {len(out)}")
    return out


def apply_named_calibration(buildings):
    """Apply official heights to already-ingested parts by id (rb.CAL may have been updated)."""
    cal = rb.CAL["by_osm_way"]
    n = 0
    for b in buildings:
        rec = cal.get(str(b["id"]))
        if rec:
            b["height"] = float(rec["height_m"])
            b["height_src"] = "calibrated"
            if rec.get("name") and not b.get("name"):
                b["name"] = rec["name"]
            n += 1
    print("applied calibrated heights to", n, "buildings")
    return buildings


def backup_and_expand_grid():
    if not GRID_PATH.exists():
        raise SystemExit(f"missing {GRID_PATH}")
    # already backed up by the caller; make a second stamp if none in this minute
    grid = json.loads(GRID_PATH.read_text())
    existing = {t["id"]: t for t in grid["tiles_1024"]}
    new_recs = []
    for R, C in NEW_TILES:
        rec = ea.tile_record(R, C)
        if rec["id"] in existing:
            # do not alter origin/scale of an already-present tile
            print("skip existing origin", rec["id"])
            new_recs.append(existing[rec["id"]])  # still list it as a south-expansion tile
            continue
        existing[rec["id"]] = rec
        new_recs.append(rec)
    tiles = list(existing.values())
    # stable-ish order: by R, C
    tiles.sort(key=lambda t: (t["R"], t["C"]))
    grid["tiles_1024"] = tiles

    rows = sorted({t["R"] for t in tiles})
    cols = sorted({t["C"] for t in tiles})
    grid["south_expansion"] = {
        "step": 2,
        "new_ids": [f"t1024_r{R}_c{C}" for R, C in NEW_TILES],
        "new_tiles": [{"R": R, "C": C, "id": f"t1024_r{R}_c{C}"} for R, C in NEW_TILES],
        "whitebox_window": {
            "rows": list(range(RMIN, RMAX + 1)),
            "cols": list(range(CMIN, CMAX + 1)),
            "mosaic_px": [W, H],
            "mosaic_origin_tile": {"R": RMIN, "C": CMIN},
            "note": "rectangular render window; only NEW_TILES were appended to tiles_1024",
        },
        "rows_all": rows,
        "cols_all": cols,
        "n_tiles_total": len(tiles),
        "n_new": len(NEW_TILES),
        "wgs84_whitebox_bbox": ea.mosaic_extent(list(range(RMIN, RMAX + 1)), list(range(CMIN, CMAX + 1))),
        "grid_continuity": "same UTM origin, azimuth -15, elevation -45, 0.29296875 m/px, origin_rot unchanged; existing tile records not modified",
        "does_not_modify": ["origin_rot", "camera", "box", "existing tiles_1024 records"],
    }
    GRID_PATH.write_text(json.dumps(grid, ensure_ascii=False, indent=2) + "\n")
    print("wrote", GRID_PATH, "new tiles", [t["id"] for t in new_recs])
    return grid, new_recs


def project_ring(proj, coords, z=0.0):
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    return [proj.project_xyz(*proj.en(lon, lat), z)[:2] for lon, lat in coords]


def render_global(buildings, water, roads, areas):
    proj = GlobalProjector()
    print(
        f"south canvas {W}x{H}  camera lon/lat={proj.lon0:.6f},{proj.lat0:.6f}  "
        f"rot center=({proj.cx:.1f},{proj.cy:.1f})"
    )
    print(f"rot bounds x=[{proj.x0:.1f},{proj.x1:.1f}] y=[{proj.y0:.1f},{proj.y1:.1f}] mpp={proj.mpp}")

    # Default urban grey-earth (not cream palace fill for the whole south window)
    img = Image.new("RGB", (W, H), (168, 164, 156))
    draw = ImageDraw.Draw(img, "RGBA")

    # Duanmen–Wumen corridor: slightly warmer packed earth, still grey-leaning
    # Paint a rectangle in lon/lat between the gates.
    corridor = [
        (116.3896, 39.9076),
        (116.3928, 39.9076),
        (116.3928, 39.9130),
        (116.3896, 39.9130),
    ]
    xy = project_ring(proj, corridor)
    if len(xy) >= 3:
        draw.polygon(xy, fill=(166, 158, 148))

    # Tiananmen Square — grey granite plaza
    for rings in areas["square"]:
        # try to build a polygon from member ways
        polys = []
        for ring in rings:
            p = rb.lonlat_poly(ring)
            if p:
                polys.append(p)
        if not polys:
            # fallback: bbox
            bb = LANDMARK_BBOXES["天安门广场"]
            p = sbox(*bb)
            polys = [p]
        try:
            u = unary_union(polys)
            geoms = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
        except Exception:
            geoms = polys
        for g in geoms:
            if g.is_empty:
                continue
            coords = list(g.exterior.coords)
            xy = project_ring(proj, coords)
            if len(xy) >= 3:
                draw.polygon(xy, fill=(150, 148, 144))

    # water (筒子河 only appears at the north edge if present; 外金水河 is the east-west canal)
    for w in water:
        coords = w["coords"]
        if w.get("as_line") or (coords[0] != coords[-1] and len(coords) < 4):
            xy = [proj.project_xyz(*proj.en(lon, lat), 0.0)[:2] for lon, lat in coords]
            if len(xy) >= 2:
                draw.line(xy, fill=(62, 118, 148), width=10, joint="curve")
            continue
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        pts = [(*proj.en(lon, lat), 0.0) for lon, lat in coords]
        rb.draw_poly(draw, proj, pts, (62, 118, 148), outline=(48, 96, 122))

    # roads — Chang'an (trunk) slightly thicker so the west section reads
    for r in roads:
        hw = r["highway"]
        name = r.get("name") or ""
        major = hw in ("trunk", "primary", "secondary")
        pedestrian = hw in ("pedestrian", "footway", "living_street")
        width = 7 if major else (6 if pedestrian else 3)
        if "长安" in name and hw == "trunk":
            width = 14
        if name == "前门大街":
            width = max(width, 8)
        color = (92, 90, 86) if major else ((168, 150, 126) if pedestrian else (110, 108, 104))
        if "长安" in name:
            color = (86, 84, 80)
        xy = [proj.project_xyz(*proj.en(lon, lat), 0.0)[:2] for lon, lat in r["coords"]]
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=width, joint="curve")

    nearby = []
    for b in buildings:
        if 116.378 <= b["lon"] <= 116.405 and 39.892 <= b["lat"] <= 39.916:
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

    raw = OUT / "global_south_whitebox.png"
    img.save(raw, "PNG")
    print("wrote", raw, img.size)
    return img, proj


def write_previews_and_slices(img, proj, landmark_report):
    scale = 2560 / max(W, H)
    prev = img.resize((int(W * scale), int(H * scale)), Image.Resampling.LANCZOS).convert("RGB")
    prev.save(OUT / "global_south_whitebox_2560.jpg", "JPEG", quality=90)

    lab = prev.copy()
    d = ImageDraw.Draw(lab)
    sx, sy = prev.size[0] / W, prev.size[1] / H
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_s = font

    new_set = {(R, C) for R, C in NEW_TILES}
    exist_set = {(R, C) for R, C in SEAM_EXISTING}
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            x0, y0 = int((C - CMIN) * TILE * sx), int((R - RMIN) * TILE * sy)
            x1, y1 = int((C - CMIN + 1) * TILE * sx), int((R - RMIN + 1) * TILE * sy)
            if (R, C) in new_set:
                outline = (20, 140, 60)
            elif (R, C) in exist_set:
                outline = (40, 90, 180)
            else:
                outline = (140, 80, 40)  # context-only slice, not added to grid
            d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=outline, width=2)
            tag = "NEW" if (R, C) in new_set else ("SEAM" if (R, C) in exist_set else "ctx")
            d.text((x0 + 4, y0 + 4), f"R{R}C{C} {tag}", fill=(20, 40, 80), font=font)

    # landmark dots
    for name, (lon, lat) in LANDMARK_CENTROIDS.items():
        px, py, _ = proj.project_xyz(*proj.en(lon, lat), 0.0)
        x, y = int(px * sx), int(py * sy)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(200, 30, 30))
        d.text((x + 6, y - 7), name, fill=(120, 20, 20), font=font_s)

    lab.save(OUT / "global_south_whitebox_2560_grid.jpg", "JPEG", quality=90)
    print("wrote previews")

    SLICES.mkdir(parents=True, exist_ok=True)
    manifest = []
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            box = (
                (C - CMIN) * TILE, (R - RMIN) * TILE,
                (C - CMIN + 1) * TILE, (R - RMIN + 1) * TILE,
            )
            sl = img.crop(box)
            path = SLICES / f"t1024_r{R}_c{C}.png"
            sl.save(path, "PNG")
            kind = (
                "new" if (R, C) in new_set
                else ("seam_existing" if (R, C) in exist_set else "context_only")
            )
            rec = ea.tile_record(R, C)
            manifest.append({
                "id": rec["id"], "R": R, "C": C,
                "lon": rec["lon"], "lat": rec["lat"],
                "rot_x": rec["rot_x"], "rot_y": rec["rot_y"],
                "kind": kind,
                "in_grid_axis": (R, C) in new_set or (R, C) in exist_set or kind == "seam_existing",
                "path": str(path),
            })
    payload = {
        "mpp": MPP,
        "W": W, "H": H,
        "rows": list(range(RMIN, RMAX + 1)),
        "cols": list(range(CMIN, CMAX + 1)),
        "camera_center": {"lon": proj.lon0, "lat": proj.lat0, "rot_x": proj.cx, "rot_y": proj.cy},
        "grid_continuity": "same UTM origin / origin_rot / azimuth -15 / elevation -45 / 0.29296875 m/px",
        "landmarks": landmark_report,
        "slices": manifest,
    }
    (OUT / "slices.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("wrote", len(manifest), "slices to", SLICES)
    return manifest


def landmark_tile_report():
    report = {}
    # precompute tile polygons for the window plus a small halo
    polys = {}
    for R in range(RMIN - 1, RMAX + 2):
        for C in range(CMIN - 1, CMAX + 2):
            polys[(R, C)] = tile_poly(R, C)
    new_set = {(R, C) for R, C in NEW_TILES}
    for name, bb in LANDMARK_BBOXES.items():
        p = sbox(*bb)
        hits = []
        for (R, C), poly in polys.items():
            if poly.intersects(p):
                inter = poly.intersection(p)
                frac = (inter.area / p.area) if p.area else 0
                if frac > 0.001:
                    hits.append({"R": R, "C": C, "overlap": round(frac, 3),
                                 "new": (R, C) in new_set})
        hits.sort(key=lambda h: -h["overlap"])
        primary = hits[0] if hits else None
        cen = LANDMARK_CENTROIDS.get(name)
        cen_tile = None
        if cen:
            R, C, fx, fy = tile_of_lonlat(*cen)
            cen_tile = {"R": R, "C": C, "fx": round(fx, 3), "fy": round(fy, 3),
                        "lon": cen[0], "lat": cen[1]}
        report[name] = {
            "centroid": cen_tile,
            "primary_tile": primary,
            "tiles_touched": hits,
            "crosses_seam": len(hits) > 1,
            "bbox_wgs84": {"west": bb[0], "south": bb[1], "east": bb[2], "north": bb[3]},
        }
    return report


def write_notes(grid, new_recs, manifest, report, osm_notes):
    lines = []
    lines.append("# 中轴线南段白盒（午门以南–前门）")
    lines.append("")
    lines.append("给后续并行上色用的格网 / 地标 / 接缝清单。白盒是**一次**按现有相机与尺度渲出的整幅图再切片，")
    lines.append("不是每格自造尺度，也不会再画第二圈筒子河或宫墙。")
    lines.append("")
    lines.append("输出目录：`tiles/whiteboxes/global_south/`。未覆盖 `tiles/whiteboxes/global/`、")
    lines.append("`tiles/styled/`、`tiles/mosaic/overview_*.jpg`。")
    lines.append("")
    lines.append("## 相机与尺度（与现有格完全一致）")
    lines.append("")
    lines.append("- CRS：WGS84 / UTM 50N，原点为试点 box SW（116.39480, 39.90620）")
    lines.append("- `origin_rot` 未改：minx=-40.0, maxy=1302.7542106015187")
    lines.append("- azimuth -15°，elevation -45°，view_height_meters=300，0.29296875 m/px")
    lines.append(f"- 白盒窗口：R={RMIN}..{RMAX}，C={CMIN}..{CMAX}，画布 {W}×{H} px")
    lines.append("")
    lines.append("## 新格清单（已写入 grid_axis.json，未改已有格的 origin/尺度）")
    lines.append("")
    lines.append("| R | C | id | 中心 WGS84 | 角色 |")
    lines.append("|--:|--:|----|------------|------|")
    role = {
        (2, -3): "长安街西段（更西），大会堂北缘一带",
        (2, -2): "长安街西段、外金水河西段、广场西北",
        (2, -1): "端门、天安门城楼、外金水桥（五座）",
        (3, -3): "人民大会堂（主落点）",
        (3, -2): "人民英雄纪念碑（主落点）、广场中部、大会堂东翼",
        (3, -1): "广场东部、国博西翼、纪念碑东缘",
        (4, -2): "毛主席纪念堂、正阳门城楼、箭楼北主体",
        (4, -1): "国博西南、广场东南",
        (4, 0): "国博东南（与已上色 r3_c0 南接）",
        (5, -2): "正阳门箭楼南缘、前门大街北段",
    }
    for R, C in NEW_TILES:
        rec = ea.tile_record(R, C)
        lines.append(
            f"| {R} | {C} | `{rec['id']}` | {rec['lon']:.6f}, {rec['lat']:.6f} | {role.get((R,C), '')} |"
        )
    lines.append("")
    lines.append("说明：R2C-2/R2C-1/R3C-2/R3C-1 落在既有概念窗口 R=-2..3、C=-2..4 的空格里，这次才写入 `tiles_1024`。")
    lines.append("R3C-3、R4*、R5C-2 是向西、向南的最小扩展，没有按矩形补到五环。")
    lines.append("")
    lines.append("## 白盒矩形里未写入 grid 的切片（仅供对齐/上下文）")
    lines.append("")
    lines.append("- 已上色接缝：`t1024_r1_c-2`、`t1024_r1_c-1`（午门）、`t1024_r1_c0`、`t1024_r2_c0`、`t1024_r3_c0`（国博东半，主落点）")
    lines.append("- 上下文-only（切片有，但未加入 `tiles_1024`）：R1C-3、R4C-3、R5C-3、R5C-1、R5C0")
    lines.append("")
    lines.append("## 地标落点（每个地标只应出现一次；跨缝的按 footprint 切，不要复制）")
    lines.append("")
    lines.append("| 地标 | 主格（面积最大） | 质心格 | 是否跨缝 | 备注 |")
    lines.append("|------|------------------|--------|----------|------|")
    notes_extra = {
        "午门": "已在现用 r1_c-1；白盒北缘对齐用。凹字形墩台+楼，不要在南段再画一座。",
        "端门": "城台+城楼，形制近天安门，不要画成午门。",
        "天安门城楼": "城台+五洞城楼（重檐歇山），不要画成午门的凹字形。",
        "外金水桥": "五座东西并排的石桥，横跨外金水河；不是南北一串，也不是内金水桥。",
        "人民英雄纪念碑": "广场中央石碑，通高 37.94 m，略高于天安门。东缘极细跨进 r3_c-1。",
        "毛主席纪念堂": "纪念碑与正阳门之间的方形大厅，花岗岩柱廊+金黄重檐，通高 33.6 m。",
        "人民大会堂": "广场西侧，山字形体量，黄花岗岩石墙。东翼跨进 r3_c-2，主楼在 r3_c-3。",
        "国家博物馆": "广场东侧，与大会堂相对。主落点在已上色 r3_c0，西翼在 r3_c-1，南缘进入 r4_c-1 / r4_c0。",
        "正阳门城楼": "在北（纪念堂以南），灰筒瓦绿剪边，通高 43.65 m。全图只此一座。",
        "正阳门箭楼": "在南，主落点仍在 r4_c-2，南缘跨进 r5_c-2。全图只此一座，不要和城楼对调。",
        "天安门广场": "本身跨多格（r3_c-2 / r3_c-1 / r4_c-2 为主），是场地不是建筑。",
        "长安街西段": "西长安街，天安门以西的主干道，主在 r2_c-2，更西在 r2_c-3。",
        "前门大街北段": "箭楼以南的步行街北段，整段在 r5_c-2。",
    }
    for name, info in report.items():
        pri = info["primary_tile"]
        cen = info["centroid"]
        pri_s = f"R{pri['R']}C{pri['C']} ({pri['overlap']:.0%})" if pri else "—"
        cen_s = f"R{cen['R']}C{cen['C']}" if cen else "—"
        seam = "是" if info["crosses_seam"] else "否"
        lines.append(f"| {name} | {pri_s} | {cen_s} | {seam} | {notes_extra.get(name, '')} |")
    lines.append("")
    lines.append("约束核对（白盒里就应当能辨认）：")
    lines.append("")
    lines.append("- 天安门 = 城台 + 五洞城楼，不是午门。")
    lines.append("- 外金水桥 = 五座并排石桥（OSM `man_made=bridge` ways 637833012–016）。")
    lines.append("- 正阳门城楼在北、箭楼在南，各只一座。")
    lines.append("- 大会堂在广场西、国博在东。")
    lines.append("")
    lines.append("## 高度来源（已写入 data/height_calibration.json）")
    lines.append("")
    lines.append("| 地标 | 采用通高 | 来源 |")
    lines.append("|------|----------|------|")
    lines.append("| 天安门 | 34.7 m | 故宫博物院《紫禁城的图像学意义》；人民网/中国政府网述 1970 年重建后实际通高 34.7 m |")
    lines.append("| 端门 | 34.37 m | 故宫博物院《紫禁城的图像学意义》 |")
    lines.append("| 正阳门城楼 | 43.65 m | 北京市文物局（中轴线遗产保护中心·正阳门） |")
    lines.append("| 正阳门箭楼 | 35.37 m | 同上 |")
    lines.append("| 人民英雄纪念碑 | 37.94 m | 天安门地区管委会；国家文物局。OSM 已有同数 |")
    lines.append("| 毛主席纪念堂 | 33.6 m | 中新网设计记述；东城区政府介绍 |")
    lines.append("| 人民大会堂 | 46.5 m（最高部位） | 国家文物局专题、visitbeijing |")
    lines.append("| 国家博物馆 | 42.5 m（最高处） | 《光明日报》2011-03-29 改扩建报道（26/34/42 三层，最高 42.5）。OSM part 有 43–48 m，已按此封顶 |")
    lines.append("")
    lines.append("城台/墩台、柱廊等 building:part 仍用 OSM 分段高度（如天安门城台 part 约 21 m、端门城台 part 约 13 m），")
    lines.append("只把主屋面/最高体积收到公开通高。正阳门城楼 OSM 没有拆 part，按整座通高 43.65 m 一块体量。")
    lines.append("")
    lines.append("## OSM 缺口与补抽")
    lines.append("")
    for para in osm_notes:
        lines.append(para)
        lines.append("")
    lines.append("## 与现有 r3 的接缝")
    lines.append("")
    lines.append("- 现用已上色最南一行是 `r3_c0..c4`。本窗口东缘落到 `r3_c0`（国博东半已在该格）和 `r2_c0`。")
    lines.append("- 新格 `r3_c-1` 紧贴 `r3_c0` 西侧；`r4_c0` 紧贴 `r3_c0` 南侧。上色时这两条缝必须对齐国博南、西轮廓，不要把国博复制一份。")
    lines.append("- `r2_c-1`（天安门）北接已有 `r1_c-1`（午门）。端门在 `r2_c-1` 北部，午门在 `r1_c-1` 南部，中间是端门–午门广场，不要在接缝上再画一道宫墙或第二圈河。")
    lines.append("- 外金水河是东西向窄河，在天安门正南、长安街北，不要画成筒子河环。")
    lines.append("- 生成 `r4_c-2` 时城楼与箭楼必须同时在正确位置；箭楼南缘会切进 `r5_c-2`，两格是同一座箭楼的切片，不是两座。")
    lines.append("- 纪念碑几乎全在 `r3_c-2`，不要在 `r3_c-1` 再画一座。")
    lines.append("")
    lines.append("## 文件")
    lines.append("")
    lines.append("- `global_south_whitebox.png` — 整幅彩色白盒")
    lines.append("- `global_south_whitebox_2560.jpg` / `global_south_whitebox_2560_grid.jpg` — 总览与格网标注（绿=新格，蓝=已有接缝格，棕=未入 grid 的上下文切片；红点为地标质心）")
    lines.append("- `slices/t1024_r{R}_c{C}.png` 与 `slices.json`")
    lines.append("- 网格备份：`tiles/grid_axis.json.bak_*`")
    lines.append("")
    path = OUT / "NOTES.md"
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SLICES.mkdir(parents=True, exist_ok=True)

    update_calibration()
    ea.patch_zone()
    patch_styles()

    print("expanding grid_axis.json (existing records unchanged)...")
    grid, new_recs = backup_and_expand_grid()

    print("parsing OSM (existing extracts + osm_tiananmen_south.xml)...")
    orig_classify = ea.classify
    # classify_south already calls ea.classify as fallback; hook rb/ea
    rb.classify = classify_south
    ea.classify = classify_south
    try:
        nodes, ways, rels = ea.parse_many(OSM_PATHS)
        buildings, water, roads = ea.assemble_plus(nodes, ways, rels)
    finally:
        rb.classify = orig_classify
        ea.classify = orig_classify

    areas = collect_named_areas(nodes, ways, rels)
    buildings, water = add_bridges_and_water(buildings, water, areas)
    buildings = fix_south_massing(buildings)
    buildings = apply_named_calibration(buildings)

    # typ counts in the window
    win = [b for b in buildings if 116.378 <= b["lon"] <= 116.405 and 39.892 <= b["lat"] <= 39.916]
    print("window typ", Counter(b["typ"] for b in win).most_common(20))
    named = [b for b in win if b.get("name") and any(
        k in b["name"] for k in ("天安门", "端门", "正阳", "箭楼", "纪念碑", "纪念堂", "大会堂", "博物馆", "金水")
    )]
    print("named south landmarks", len(named))
    for b in sorted(named, key=lambda x: -(x["lat"] or 0))[:40]:
        print(f"  {str(b.get('name')):16s} id={b['id']} typ={b['typ']:16s} h={b['height']:.1f} {b['height_src']} {b['lon']:.6f},{b['lat']:.6f}")

    n_bridges = sum(1 for b in buildings if b.get("name") == "外金水桥")
    print("外金水桥 massing count:", n_bridges)

    report = landmark_tile_report()

    osm_notes = [
        "- 既有 `osm_axis_south.xml` bounds 为 39.8955–39.9115 N，116.3845–116.3975 E。"
        "大会堂西缘约 116.3860、国博东缘约 116.3967，都还在该盒内；"
        "但白盒格的旋转足迹超出该盒：西到约 116.3815（R2/R3 C=-3）、南到约 39.8932（R5C-2）、"
        "东到约 116.4016（R4C0 / 已有 r3_c0）。",
        "- 已用 OSM Overpass（WGS84，未混用 GCJ-02，未爬商业瓦片）按白盒窗口外扩 80 m 补抽，"
        "另存 `data/osm_tiananmen_south.xml`（未覆盖已有 xml）。"
        "查询只取 way + type 为 multipolygon/building/collection/site 的 relation 及其成员，"
        "避免把行政区边界递归成全国节点。",
        "- 补抽 bounds：39.892436–39.915173 N，116.380533–116.402526 E。"
        "前门大街几何可延伸到约 39.890，北段（箭楼以南约 200–300 m）已落在 R5C-2。"
        "若后续再往南或再往西，需要再补抽。",
        "- 西长安街在既有/补抽 OSM 里都有 `highway=trunk` 分段，可直接作为道路画出来。",
    ]

    print("rendering south whitebox...")
    img, proj = render_global(buildings, water, roads, areas)
    manifest = write_previews_and_slices(img, proj, report)
    write_notes(grid, new_recs, manifest, report, osm_notes)
    print("done")


if __name__ == "__main__":
    main()
