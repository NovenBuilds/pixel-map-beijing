#!/usr/bin/env python3
"""Global whitebox for the palace south edge + Tiananmen Square + Qianmen.

Same UTM origin / azimuth -15 / elevation -45 / 0.29296875 m per pixel.
The moat and walls are taken from OSM once on a single canvas; tiles are slices.
Writes tiles/whiteboxes/global_south/ only. Does not touch:
  - tiles/styled/ finished tiles
  - tiles/whiteboxes/global/global_whitebox.png
  - viewer logic
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

MPP = 300.0 / 1024.0
# Palace south (午门, R=1) through 前门大街 (R=6); 大会堂 west (C=-3) through 国博 east (C=0)
RMIN, RMAX = 1, 6
CMIN, CMAX = -3, 0
TILE = 1024
N_C = CMAX - CMIN + 1
N_R = RMAX - RMIN + 1
W, H = N_C * TILE, N_R * TILE

OUT = ROOT / "tiles" / "whiteboxes" / "global_south"
SLICES = OUT / "slices"
OUT.mkdir(parents=True, exist_ok=True)
SLICES.mkdir(parents=True, exist_ok=True)

# Current mosaic window (for annotating overlap vs new tiles)
CUR_R = range(-2, 4)
CUR_C = range(-2, 5)

OSM_EXTRA = [
    ROOT / "data" / "osm_square_east.xml",
    ROOT / "data" / "osm_qianmen_south.xml",
]

CAL_PATH = ROOT / "data" / "height_calibration.json"

# Official / heritage-bureau heights. Uncertain=True when sources disagree or part mapping is typological.
SOUTH_HEIGHTS = {
    "637830242": {
        "height_m": 34.7,
        "name": "天安门城楼（通高，含城台）",
        "source": "1970 年按原规制重建后通高 34.7 m（原高 33.87 m）。北京市文旅 visitbeijing：城台+城楼总高 34.7 m。故宫博物院刊载文章亦记天安门高 34.7 m。OSM 该 part 原 height=35，按文献改为 34.7。",
        "evidence": [
            "https://www.visitbeijing.com.cn/article/47Qre4hsjWF",
            "https://www.dpm.org.cn/Uploads/File/pdf/21/37/8d/21378dd0b3131359d9cafc62a7a3369f.pdf",
        ],
    },
    "637950096": {
        "height_m": 34.37,
        "name": "端门（通高）",
        "source": "故宫博物院刊载《紫禁城的图像学意义》：端门高 34.37 m（同文：天安门 34.7 m、午门 35.6 m）。OSM 该 part 原 height=35。",
        "evidence": [
            "https://www.dpm.org.cn/Uploads/File/pdf/21/37/8d/21378dd0b3131359d9cafc62a7a3369f.pdf",
        ],
    },
    "637834383": {
        "height_m": 37.94,
        "name": "人民英雄纪念碑（通高）",
        "source": "通高 37.94 m。OSM building:part way 637834383 已标注 height=37.94，与通行官方数字一致。",
        "evidence": [
            "OSM way 637834383 height=37.94",
            "https://zh.wikipedia.org/wiki/人民英雄纪念碑",
        ],
    },
    "637908759": {
        "height_m": 33.6,
        "name": "毛主席纪念堂（主体通高）",
        "source": "北京市东城区政府网站：主体建筑长宽各 105.5 m，高度 33.6 m（基座高 4 m，上立 17.5 m 廊柱，重檐琉璃屋顶）。OSM 该 part 原 height=35，按院方/区府记述改为 33.6。",
        "evidence": ["https://www.bjdch.gov.cn/mldc/bglj/hswh/202304/t20230408_2993346.html"],
    },
    "637908757": {
        "height_m": 33.6,
        "name": "毛主席纪念堂（主体通高）",
        "source": "同 33.6 m。OSM 该 part 原 height=35。",
        "evidence": ["https://www.bjdch.gov.cn/mldc/bglj/hswh/202304/t20230408_2993346.html"],
    },
    "637908758": {
        "height_m": 33.6,
        "name": "毛主席纪念堂（主体通高）",
        "source": "同 33.6 m。OSM 该 part 原 height=37，略高于文献通高，改为 33.6。",
        "evidence": ["https://www.bjdch.gov.cn/mldc/bglj/hswh/202304/t20230408_2993346.html"],
    },
    "25109960": {
        "height_m": 43.65,
        "name": "正阳门城楼（通高）",
        "source": "北京市文物局、首都之窗：正阳门城楼通高 43.65 m。visitbeijing 另记整座城楼整体高度 42 m（城台 13.2 m、楼身 27.3 m）。本条取文物局数字。",
        "evidence": [
            "https://wwj.beijing.gov.cn/bjww/362771/362772/545898/index.html",
            "https://www.beijing.gov.cn/renwen/rwzyd/gdwh/zym/202107/t20210706_2430351.html",
            "https://www.visitbeijing.com.cn/article/47Qre4hsjWF",
        ],
        "uncertain": True,
    },
    "439956436": {
        "height_m": 35.37,
        "name": "正阳门箭楼（通高）",
        "source": "北京市文物局、首都之窗：箭楼通高 35.37 m。visitbeijing 另记连城台通高 38 m。本条取文物局数字。",
        "evidence": [
            "https://wwj.beijing.gov.cn/bjww/362771/362772/545898/index.html",
            "https://www.beijing.gov.cn/renwen/rwzyd/gdwh/zym/202107/t20210706_2430351.html",
            "https://www.visitbeijing.com.cn/article/47Qre4hsjWF",
        ],
        "uncertain": True,
    },
    "637929779": {
        "height_m": 46.5,
        "name": "人民大会堂（通高；此为较高体量部分）",
        "source": "光明日报 2011-03-29《新国博“新”在哪儿》：人民大会堂建筑面积 17.18 万 m²，高 46.5 m。建筑「两翼略低、中部稍高」，该 way 为较高体量之一。OSM 原 height=45。",
        "evidence": ["https://epaper.gmw.cn/gmrb/html/2011-03/29/nw.D110000gmrb_20110329_1-09.htm"],
        "uncertain": True,
    },
    "637929804": {
        "height_m": 46.5,
        "name": "人民大会堂（通高；此为较高体量部分）",
        "source": "同 46.5 m。OSM 原 height=45。部分与中部/两翼的精确对应仍标不确定。",
        "evidence": ["https://epaper.gmw.cn/gmrb/html/2011-03/29/nw.D110000gmrb_20110329_1-09.htm"],
        "uncertain": True,
    },
    "637929787": {
        "height_m": 46.5,
        "name": "人民大会堂（通高；此为较高体量部分）",
        "source": "同 46.5 m。OSM 原 height=45。",
        "evidence": ["https://epaper.gmw.cn/gmrb/html/2011-03/29/nw.D110000gmrb_20110329_1-09.htm"],
        "uncertain": True,
    },
}

# 国博：官方最高 42.5 m（26 / 34 / 42.5 三层）。将 OSM 高于 42.5 及接近最高层的 part 收到 42.5。
NMC_CAP_IDS = {
    "637939178", "637939169", "637939207",  # OSM 43
    "637939210", "637939214", "637939208", "637939209", "637939215",  # OSM 48
    "637939216",  # OSM 46
}
for _wid in NMC_CAP_IDS:
    SOUTH_HEIGHTS[_wid] = {
        "height_m": 42.5,
        "name": "中国国家博物馆（最高处 42.5 m）",
        "source": "中国国家博物馆简介（官网存档）及光明日报 2011-03-29：新馆主体外观高度约 26 / 34 / 42 m 三层，最高处 42.5 m。该 OSM part 原 height 为 43–48，已按官方最高处封顶为 42.5。",
        "evidence": [
            "https://web.archive.org/web/20201104104627/http://www.chnmuseum.cn/gbgk/gbjj/",
            "https://epaper.gmw.cn/gmrb/html/2011-03/29/nw.D110000gmrb_20110329_1-09.htm",
        ],
        "uncertain": True,
    }

# Filled courtyard / outline parents — keep building:parts only (same rule as 午门).
DROP_SOUTH = {
    "25097188",  # 端门 outline
    "637830319", "637830343",  # 天安门 outers
    "637939189", "637939181",  # 国博 outers
    "637929798",  # 大会堂 outer
    # 毛主席纪念堂 outers
    "637895268", "637895176", "637895269", "637895212", "637895274", "637895262",
    "637895265", "637895241", "637895282", "637895257", "637895287", "637895297",
    # 人民英雄纪念碑 star-platform outers (must not extrude to 37.94 m)
    "637467890", "637880215", "637880217", "637880210", "637467889", "637880212",
    "637880220", "637880209", "637467891", "637880207", "637880221", "637880206",
    "637880198", "637880196", "637880224", "637880213",
}

CITY_GATES = {"正阳门", "正阳门箭楼", "箭楼"}
CIVIC = {"人民大会堂", "中国国家博物馆"}
MEMORIAL = {"毛主席纪念堂"}
MONUMENT = {"人民英雄纪念碑"}

# Footprints used to flag required tiles (WGS84, from OSM).
FOOTPRINTS = {
    "午门": (116.389885, 39.911621, 116.392115, 39.912618),
    "端门": (116.390486, 39.908812, 116.391876, 39.909193),
    "天安门": (116.390568, 39.907148, 116.391962, 39.907529),
    "外金水桥": (116.390908, 39.906591, 116.391577, 39.906962),
    "长安街(轴附近)": (116.386034, 39.906126, 116.396680, 39.906945),
    "天安门广场开敞面": (116.388577, 39.899073, 116.394207, 39.906600),
    "人民英雄纪念碑": (116.391088, 39.902888, 116.391762, 39.903506),
    "人民大会堂": (116.386034, 39.901929, 116.388577, 39.904992),
    "中国国家博物馆": (116.394207, 39.902206, 116.396680, 39.905217),
    "毛主席纪念堂": (116.390912, 39.900567, 116.392178, 39.901662),
    "正阳门城楼": (116.390973, 39.899073, 116.392262, 39.899295),
    "正阳门箭楼": (116.391300, 39.897804, 116.392017, 39.898140),
    "前门大街(步行街±40m)": (116.391276, 39.890159, 116.392581, 39.897260),
}


def update_south_calibration():
    cal = json.loads(CAL_PATH.read_text())
    for k, v in SOUTH_HEIGHTS.items():
        cal["by_osm_way"][k] = v
    note = cal.get("notes") or ""
    extra = " Also includes Tiananmen Square / Qianmen axis heights (literature / heritage-bureau), with uncertain flags where sources disagree or part mapping is typological."
    if "Qianmen axis" not in note:
        cal["notes"] = (note.rstrip() + extra).strip()
    CAL_PATH.write_text(json.dumps(cal, ensure_ascii=False, indent=2) + "\n")
    rb.CAL = cal
    print("updated", CAL_PATH, "south entries", len(SOUTH_HEIGHTS))


def patch_styles():
    rb.PAL["city_gate"] = ((88, 108, 92), (64, 82, 70), (168, 56, 48), (120, 36, 32), (196, 80, 68))
    rb.PAL["civic_hall"] = ((196, 186, 164), (160, 150, 130), (210, 200, 178), (168, 158, 138), (228, 220, 200))
    rb.PAL["memorial_hall"] = ((196, 168, 72), (156, 128, 48), (210, 204, 196), (168, 162, 154), (230, 224, 216))
    rb.PAL["monument"] = ((200, 198, 192), (160, 158, 152), (186, 184, 178), (148, 146, 140), (220, 218, 214))
    orig_pitched = rb.has_pitched_roof

    def has_pitched(typ, h):
        if typ in ("city_gate", "memorial_hall"):
            return True
        if typ in ("civic_hall", "monument"):
            return False
        return orig_pitched(typ, h)

    rb.has_pitched_roof = has_pitched


def classify_south(tags, lon, lat, kind_hint=None):
    """Do not use rb.classify's PALACE_LONS<=116.39635 catch-all; that was for the
    east-wall sliver of the original pilot and would paint the whole square as palace."""
    name = tags.get("name") or ""
    btyp = tags.get("building") or ""
    if kind_hint:
        return kind_hint
    if name in ("午门", "端门", "天安门") or name in ("太和门", "神武门", "东华门", "西华门"):
        return "palace_gate"
    if name in CITY_GATES:
        return "city_gate"
    if name in CIVIC or name in ("中国国家博物馆", "国家博物馆"):
        return "civic_hall"
    if name in MEMORIAL:
        return "memorial_hall"
    if name in MONUMENT:
        return "monument"
    if "角楼" in name:
        return "palace_gate"
    if "城墙" in name or btyp == "wall":
        return "palace_wall"
    if name in ea.GATE_NAMES and lat >= 39.90715:
        return "palace_gate"
    if tags.get("historic") == "city_gate" or btyp == "gatehouse":
        return "city_gate" if lat < 39.90715 else "palace_gate"
    # Forbidden City + 端门/天安门院落 only (north of the Tiananmen south facade)
    if 116.3850 <= lon <= 116.3965 and 39.90715 <= lat <= 39.9225:
        if "城墙" in name or btyp == "wall":
            return "palace_wall"
        if tags.get("historic") == "city_gate" or btyp == "gatehouse" or name in ea.GATE_NAMES:
            return "palace_gate"
        return "palace_hall"
    if "天主堂" in name or btyp == "church":
        return "church"
    if btyp in ("hotel",) or any(k in name for k in ("饭店", "酒店", "公寓", "宾馆")):
        return "hotel"
    if btyp in ("retail", "commercial") or any(k in name for k in ("购物", "大厦", "商场", "书店", "商业")):
        return "commercial"
    if btyp in ("office", "hospital", "school") or any(k in name for k in ("部", "院", "医院", "学校", "小学", "博物馆")):
        return "office"
    return "hutong"


def collect_rel_member_ids(rels, rel_ids):
    ids = set()
    index = {str(r["id"]): r for r in rels}
    stack = list(rel_ids)
    seen = set()
    while stack:
        rid = str(stack.pop())
        if rid in seen or rid not in index:
            continue
        seen.add(rid)
        for m in index[rid]["members"]:
            if m["type"] == "way":
                ids.add(str(m["ref"]))
            elif m["type"] == "relation":
                stack.append(m["ref"])
    return ids


def patch_massing(rels):
    orig = ea.fix_massing
    nmc_ids = collect_rel_member_ids(rels, ["8607825", "8848468", "8848467", "8848466"])
    ghp_ids = collect_rel_member_ids(rels, ["8848388", "8848390", "8848389"])
    hall_ids = collect_rel_member_ids(rels, ["8848142", "8848144"])
    mon_ids = collect_rel_member_ids(rels, ["8847722", "8847723"])
    tian_ids = collect_rel_member_ids(rels, ["8847697", "8847698"])
    duan_ids = collect_rel_member_ids(rels, ["8848564"])
    print(f"rel members 国博={len(nmc_ids)} 大会堂={len(ghp_ids)} 纪念堂={len(hall_ids)} 纪念碑={len(mon_ids)} 天安门={len(tian_ids)} 端门={len(duan_ids)}")

    def fix(buildings):
        out = []
        dropped = 0
        for b in orig(buildings):
            bid = str(b["id"])
            if bid in DROP_SOUTH:
                dropped += 1
                continue
            name = b.get("name") or ""
            if bid in nmc_ids or name in CIVIC and "博物馆" in (name or "博物馆"):
                if bid in nmc_ids:
                    b["typ"] = "civic_hall"
                    b["name"] = b["name"] or "中国国家博物馆"
            if bid in ghp_ids:
                b["typ"] = "civic_hall"
                b["name"] = b["name"] or "人民大会堂"
            if bid in hall_ids:
                b["typ"] = "memorial_hall"
                b["name"] = b["name"] or "毛主席纪念堂"
            if bid in mon_ids:
                b["typ"] = "monument"
                b["name"] = b["name"] or "人民英雄纪念碑"
            if bid in tian_ids and bid != "637950169":
                # 天安门 parts; keep terrace logic below for 端门
                if bid != "637950169":
                    b["typ"] = "palace_gate"
                    b["name"] = b["name"] or "天安门"
            if bid in duan_ids and bid != "637950169":
                b["typ"] = "palace_gate"
                b["name"] = b["name"] or "端门"
            if name in CITY_GATES or bid in ("25109960", "439956436"):
                b["typ"] = "city_gate"
                if bid == "25109960":
                    b["name"] = "正阳门"
                elif bid == "439956436":
                    b["name"] = "正阳门箭楼"
            if bid == "637950169":
                b["typ"] = "palace_terrace"
                b["name"] = "端门城台"
            if bid in SOUTH_HEIGHTS and not b.get("name"):
                b["name"] = SOUTH_HEIGHTS[bid]["name"]
            out.append(b)
        print(f"south fix_massing: dropped {dropped} outline parents, kept {len(out)}")
        return out

    ea.fix_massing = fix


def mosaic_rot_bounds():
    x0 = ea.MINX + CMIN * ea.TILE_WX
    x1 = ea.MINX + (CMAX + 1) * ea.TILE_WX
    y1 = ea.MAXY - RMIN * ea.TILE_WY
    y0 = ea.MAXY - (RMAX + 1) * ea.TILE_WY
    return x0, x1, y0, y1


class GlobalProjector:
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
        return tg.lonlat_to_en(lon, lat)

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


def fill_lonlat_rect(draw, proj, west, south, east, north, color, nx=8, ny=8):
    """Fill a geographic rectangle by projecting a mesh of quads."""
    for i in range(nx):
        for j in range(ny):
            lon0 = west + (east - west) * i / nx
            lon1 = west + (east - west) * (i + 1) / nx
            lat0 = south + (north - south) * j / ny
            lat1 = south + (north - south) * (j + 1) / ny
            corners = [(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)]
            xy = [proj.project_xyz(*proj.en(lon, lat), 0.0)[:2] for lon, lat in corners]
            if len(xy) >= 3:
                draw.polygon(xy, fill=color)


def tiles_for_bbox(w, s, e, n, n_samp=11):
    tiles = set()
    for i in range(n_samp):
        for j in range(n_samp):
            lon = w + (e - w) * i / (n_samp - 1)
            lat = s + (n - s) * j / (n_samp - 1)
            rx, ry = ea.lonlat_to_rot(lon, lat)
            C = math.floor((rx - ea.MINX) / ea.TILE_WX)
            R = math.floor((ea.MAXY - ry) / ea.TILE_WY)
            tiles.add((R, C))
    return tiles


def required_new_tiles():
    all_tiles = set()
    by_name = {}
    for name, bb in FOOTPRINTS.items():
        ts = tiles_for_bbox(*bb)
        by_name[name] = sorted(ts)
        all_tiles.update(ts)
    new = sorted(t for t in all_tiles if t[0] not in CUR_R or t[1] not in CUR_C)
    overlap = sorted(t for t in all_tiles if t[0] in CUR_R and t[1] in CUR_C)
    return sorted(all_tiles), new, overlap, by_name


def render_global(buildings, water, roads, required_meta):
    proj = GlobalProjector()
    print(f"south canvas {W}x{H}  camera lon/lat={proj.lon0:.6f},{proj.lat0:.6f}  rot center=({proj.cx:.1f},{proj.cy:.1f})")
    print(f"rot bounds x=[{proj.x0:.1f},{proj.x1:.1f}] y=[{proj.y0:.1f},{proj.y1:.1f}] mpp={proj.mpp}")
    print(f"window R={RMIN}..{RMAX} C={CMIN}..{CMAX}")

    # Urban grey-tan base (not palace packed-earth) so the square is not cream.
    img = Image.new("RGB", (W, H), (186, 184, 178))
    draw = ImageDraw.Draw(img, "RGBA")

    # Palace south courts / 端门院: packed earth
    fill_lonlat_rect(draw, proj, 116.3888, 39.90753, 116.3933, 39.91262, (198, 176, 132), nx=6, ny=8)
    # Tiananmen Square pavement
    fill_lonlat_rect(draw, proj, 116.388577, 39.899073, 116.394207, 39.90660, (168, 168, 166), nx=8, ny=12)
    # Chang'an Street band
    fill_lonlat_rect(draw, proj, 116.3845, 39.90605, 116.3990, 39.90715, (96, 96, 94), nx=12, ny=2)

    for w in water:
        coords = w["coords"]
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        pts = [(*proj.en(lon, lat), 0.0) for lon, lat in coords]
        rb.draw_poly(draw, proj, pts, (62, 118, 148), outline=(48, 96, 122))

    for r in roads:
        hw = r["highway"]
        major = hw in ("trunk", "primary", "secondary")
        pedestrian = hw in ("pedestrian", "footway", "living_street")
        width = 7 if major else (6 if pedestrian else 3)
        color = (92, 90, 86) if major else ((168, 150, 126) if pedestrian else (110, 108, 104))
        xy = [proj.project_xyz(*proj.en(lon, lat), 0.0)[:2] for lon, lat in r["coords"]]
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=width, joint="curve")

    nearby = []
    for b in buildings:
        if 116.380 <= b["lon"] <= 116.410 and 39.887 <= b["lat"] <= 39.916:
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

    raw = OUT / "global_whitebox.png"
    img.save(raw, "PNG")
    print("wrote", raw, img.size)

    scale = 2560 / max(W, H)
    prev = img.resize((int(W * scale), int(H * scale)), Image.Resampling.LANCZOS).convert("RGB")
    prev.save(OUT / "global_whitebox_2560.jpg", "JPEG", quality=90)
    lab = prev.copy()
    d = ImageDraw.Draw(lab)
    sx, sy = prev.size[0] / W, prev.size[1] / H
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = font_s = ImageFont.load_default()
    required = set(tuple(t) for t in required_meta["new_tiles"])
    overlap = set(tuple(t) for t in required_meta["overlap_existing_window"])
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            x0, y0 = int((C - CMIN) * TILE * sx), int((R - RMIN) * TILE * sy)
            x1, y1 = int((C - CMIN + 1) * TILE * sx), int((R - RMIN + 1) * TILE * sy)
            if (R, C) in required:
                col, tag = (200, 40, 40), "NEW"
            elif (R, C) in overlap:
                col, tag = (40, 140, 70), "overlap"
            else:
                col, tag = (40, 90, 180), "ctx"
            d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=col, width=2)
            d.rectangle([x0 + 3, y0 + 3, x0 + 150, y0 + 36], fill=(20, 20, 20, 180))
            d.text((x0 + 6, y0 + 4), f"R{R}C{C} {tag}", fill=(240, 240, 230), font=font)
    lab.save(OUT / "global_whitebox_2560_grid.jpg", "JPEG", quality=90)

    manifest = []
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            box = ((C - CMIN) * TILE, (R - RMIN) * TILE,
                   (C - CMIN + 1) * TILE, (R - RMIN + 1) * TILE)
            sl = img.crop(box)
            path = SLICES / f"t1024_r{R}_c{C}.png"
            sl.save(path, "PNG")
            role = "new" if (R, C) in required else ("overlap_existing_window" if (R, C) in overlap else "window_context")
            lon, lat, cx, cy = ea.tile_center(R, C)
            manifest.append({
                "id": f"t1024_r{R}_c{C}", "R": R, "C": C,
                "role": role, "lon": lon, "lat": lat,
                "path": str(path),
            })
    slices_json = {
        "mpp": MPP, "W": W, "H": H,
        "rows": list(range(RMIN, RMAX + 1)),
        "cols": list(range(CMIN, CMAX + 1)),
        "camera_center": {"lon": proj.lon0, "lat": proj.lat0, "rot_x": proj.cx, "rot_y": proj.cy},
        "does_not_overwrite": [
            "tiles/whiteboxes/global/global_whitebox.png",
            "tiles/styled/",
        ],
        "required_new_tiles": [f"t1024_r{R}_c{C}" for R, C in required_meta["new_tiles"]],
        "overlap_existing_window": [f"t1024_r{R}_c{C}" for R, C in required_meta["overlap_existing_window"]],
        "footprints": required_meta["by_name"],
        "slices": manifest,
    }
    (OUT / "slices.json").write_text(json.dumps(slices_json, ensure_ascii=False, indent=2) + "\n")
    print("wrote", len(manifest), "slices to", SLICES)
    return img, proj


def main():
    update_south_calibration()
    ea.patch_zone()
    patch_styles()

    all_tiles, new_tiles, overlap, by_name = required_new_tiles()
    print("footprint tiles", all_tiles)
    print("NEW tiles (outside current mosaic window)", new_tiles)
    print("overlap with current window", overlap)
    for name, ts in by_name.items():
        print(f"  {name}: {ts}")
    required_meta = {
        "all_tiles": all_tiles,
        "new_tiles": new_tiles,
        "overlap_existing_window": overlap,
        "by_name": {k: [list(t) for t in v] for k, v in by_name.items()},
    }

    paths = list(ea.OSM_PATHS) + [p for p in OSM_EXTRA if p.exists()]
    print("parsing OSM...", [p.name for p in paths])
    nodes, ways, rels = ea.parse_many(paths)
    ea.classify = classify_south
    rb.classify = classify_south
    patch_massing(rels)
    buildings, water, roads = ea.assemble_plus(nodes, ways, rels)
    # named landmark sanity
    want = ("天安门", "端门", "正阳门", "箭楼", "人民大会堂", "国家博物馆", "纪念堂", "纪念碑", "午门")
    named = [b for b in buildings if b.get("name") and any(k in b["name"] for k in want)]
    from collections import Counter
    print("typ counts", Counter(b["typ"] for b in buildings).most_common())
    print("named south landmarks", len(named))
    for b in sorted(named, key=lambda x: -(x["lat"] or 0))[:60]:
        print(f"  {b['name']:18s} id={b['id']} typ={b['typ']:14s} h={b['height']:.1f} {b['height_src']} {b['lon']:.6f},{b['lat']:.6f}")

    render_global(buildings, water, roads, required_meta)
    print("done")


if __name__ == "__main__":
    main()
