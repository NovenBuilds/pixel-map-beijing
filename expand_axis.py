#!/usr/bin/env python3
"""Expand the Forbidden City–Wangfujing 1024 grid along the Beijing axis.

Keeps the original camera, projection, origin (UTM 50N at pilot SW), and
metres-per-pixel. New tiles use signed row/col indices (west = negative C,
north = negative R) so existing t1024_r{R}_c{C}.jpg files stay valid.
"""
from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

ROOT = Path("/workspace/beijing-pilot")
sys.path.insert(0, str(ROOT))
import render_color_whitebox as rb
_ORIG_CLASSIFY = rb.classify
import build_tile_grid as tg

# Same origin / camera as the 5x4 pilot
WEST, SOUTH = 116.39480, 39.90620
AZ = math.radians(rb.AZIMUTH_DEG)
SX, SY = tg.SX, tg.SY_ROT
TILE_WX, TILE_WY = 2 * SX, 2 * SY
MINX, MAXY = -40.0, 1302.7542106015187
TO_UTM = tg.TO_UTM
TO_LL = tg.TO_LL
OX, OY = tg.OX, tg.OY

OSM_PATHS = [
    ROOT / "data" / "osm_map.xml",
    ROOT / "data" / "osm_palace.xml",
    ROOT / "data" / "osm_axis_south.xml",
]

CAL_PATH = ROOT / "data" / "height_calibration.json"
OUT_WB = ROOT / "tiles" / "whiteboxes"
OUT_TILES = ROOT / "tiles"
REFS = ROOT / "tiles" / "refs"

# Step 1 tiles: palace axis + north moat + corner towers (not existing 5x4)
STEP1 = [
    (-2, -1), (-2, 0), (-2, 1),
    (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
    (0, -2), (0, -1),
    (1, -2), (1, -1),
]
# Existing tiles whose whitebox should be re-rendered as inpaint guides
GUIDE_EXISTING = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]

# Palace interior (WGS84), slightly inside the moat ring
PALACE_BOX = (116.3858, 39.91135, 116.3959, 39.92155)

GATE_NAMES = {
    "午门", "神武门", "东华门", "西华门", "太和门", "昭德门", "贞度门",
    "乾清门", "坤宁门", "熙和门", "协和门", "阙左门", "阙右门",
    "中左门", "中右门", "后左门", "后右门", "左翼门", "右翼门",
    "景运门", "隆宗门", "日精门", "月华门", "景和门", "隆福门",
    "端门", "天安门", "保泰门", "敛禧门", "宁寿门", "皇极门",
    "锡庆门", "贞顺门", "衍祺门", "养性门", "内左门", "内右门",
}
CORNER_WAYS = {
    "42740554": "西北角楼",
    "42740573": "东北角楼",
    "468870457": "西南角楼",
}

# Official / typological ridge-or-total heights for the *building footprint*
# (terrace / 墩台 modelled separately when parts exist).
AXIS_HEIGHTS = {
    "638156366": {"height_m": 12.0, "name": "午门（墩台轮廓；楼体见 building:part）",
                  "source": "故宫博物院：午门墩台高 12.00 m。通高楼体由 OSM building:part 的 height 表达（主楼部分达 37–45 m）。",
                  "evidence": ["https://www.dpm.org.cn/explore/building/236454.html"]},
    "638308745": {"height_m": 23.8, "name": "太和门",
                  "source": "通高 23.8 m（重檐歇山门座 + 汉白玉须弥座）。",
                  "evidence": ["https://zh.wikipedia.org/wiki/太和門"]},
    "638449347": {"height_m": 26.92, "name": "太和殿（殿身，不含三层台基）",
                  "source": "故宫博物院：高 26.92 m，连同台基通高 35.05 m；三层汉白玉台基高 8.13 m。此为殿身高度。",
                  "evidence": ["https://www.dpm.org.cn/explore/building/236465.html"]},
    "638473398": {"height_m": 21.4, "name": "保和殿（殿身，估计）",
                  "source": "常见记述保和殿高 29.50 m；三大殿共用 8.13 m 三层台基，29.50−8.13≈21.4 m 为殿身。低于太和殿殿身 26.92 m，符合实际层级。",
                  "evidence": ["typological, shared terrace 8.13 m (DPM 太和殿)"], "uncertain": True},
    "25097156": {"height_m": 20.0, "name": "中和殿",
                 "source": "OSM building:part way 638463064 height=20。方形小殿，低于太和殿、保和殿。",
                 "evidence": ["OSM way 638463064 height=20"], "uncertain": True},
    "638807674": {"height_m": 16.0, "name": "乾清门",
                  "source": "内廷正门，体量小于太和门，按门座类型估计约 16 m。",
                  "evidence": ["typological"], "uncertain": True},
    "638981317": {"height_m": 20.0, "name": "乾清宫",
                  "source": "OSM height=11 过低（疑为檐高或层数误标）。按内廷正殿黄琉璃重檐体量取脊高约 20 m。",
                  "evidence": ["typological, overriding OSM 11"], "uncertain": True},
    "638981229": {"height_m": 16.0, "name": "交泰殿",
                  "source": "后三宫中最小的方形殿，低于乾清宫、坤宁宫，估计约 16 m。",
                  "evidence": ["typological"], "uncertain": True},
    "639012193": {"height_m": 18.0, "name": "坤宁宫",
                  "source": "OSM height=9 过低。按内廷后殿体量取约 18 m。",
                  "evidence": ["typological, overriding OSM 9"], "uncertain": True},
    "638741722": {"height_m": 31.0, "name": "神武门（城台+城楼通高，估计）",
                  "source": "重檐庑殿顶、面阔五间城楼，形制低于午门、高于东华门。未找到院方通高，取 31 m。",
                  "evidence": ["typological"], "uncertain": True},
    "40343948": {"height_m": 23.0, "name": "西华门",
                 "source": "与东华门同为侧门，沿用东华门 23 m 估计。",
                 "evidence": ["same typology as 东华门 way 40343947"], "uncertain": True},
    "40795244": {"height_m": 17.0, "name": "钦安殿（御花园）",
                 "source": "御花园主殿，黄琉璃重檐，体量中等，估计约 17 m。",
                 "evidence": ["typological"], "uncertain": True},
    "42740554": {"height_m": 27.5, "name": "西北角楼",
                 "source": "故宫四角角楼通高 27.5 m（十字脊、三重檐）。按位置归到未命名 OSM way 42740554。",
                 "evidence": ["https://www.visitbeijing.com.cn/article/4MjQEx8JhA1"]},
    "42740573": {"height_m": 27.5, "name": "东北角楼",
                 "source": "同角楼 27.5 m，归到 way 42740573。",
                 "evidence": ["https://www.visitbeijing.com.cn/article/4MjQEx8JhA1"]},
    "468870457": {"height_m": 27.5, "name": "西南角楼（候选）",
                  "source": "角楼 27.5 m。way 468870457 距西南城角约 44 m，作为候选轮廓。",
                  "evidence": ["https://www.visitbeijing.com.cn/article/4MjQEx8JhA1"], "uncertain": True},
    "43512262": {"height_m": 18.0, "name": "武英殿",
                 "source": "外朝西路主殿，低于三大殿，估计约 18 m。",
                 "evidence": ["typological"], "uncertain": True},
    "43519304": {"height_m": 18.0, "name": "文华殿",
                 "source": "外朝东路主殿，与武英殿对称，估计约 18 m。",
                 "evidence": ["typological"], "uncertain": True},
    "26390235": {"height_m": 17.0, "name": "万春亭（景山）",
                 "source": "景山主峰亭，黄琉璃三重檐圆亭，不含山体，估计亭身约 17 m。",
                 "evidence": ["typological"], "uncertain": True},
}


def update_calibration():
    cal = json.loads(CAL_PATH.read_text())
    for k, v in AXIS_HEIGHTS.items():
        cal["by_osm_way"][k] = v
    CAL_PATH.write_text(json.dumps(cal, ensure_ascii=False, indent=2) + "\n")
    rb.CAL = cal
    print("updated", CAL_PATH, "axis entries", len(AXIS_HEIGHTS))


def parse_many(paths):
    nodes, ways, rels = {}, {}, {}
    for path in paths:
        if not path.exists():
            print("missing", path)
            continue
        n = 0
        for _ev, el in ET.iterparse(path, events=("end",)):
            if el.tag == "node":
                nodes[el.get("id")] = (float(el.get("lon")), float(el.get("lat")))
                el.clear(); n += 1
            elif el.tag == "way":
                tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
                nds = [nd.get("ref") for nd in el.findall("nd")]
                ways[el.get("id")] = {"id": el.get("id"), "tags": tags, "nds": nds}
                el.clear(); n += 1
            elif el.tag == "relation":
                tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
                members = [{"type": m.get("type"), "ref": m.get("ref"), "role": m.get("role")}
                           for m in el.findall("member")]
                rels[el.get("id")] = {"id": el.get("id"), "tags": tags, "members": members}
                el.clear(); n += 1
        print(f"parsed {path.name}: running totals nodes={len(nodes)} ways={len(ways)} rels={len(rels)}")
    return nodes, list(ways.values()), list(rels.values())


def in_palace(lon, lat):
    w, s, e, n = PALACE_BOX
    return w <= lon <= e and s <= lat <= n


def classify(tags, lon, lat, kind_hint=None):
    name = tags.get("name") or ""
    btyp = tags.get("building") or ""
    osm_id = tags.get("_osm_id") or ""
    if kind_hint:
        return kind_hint
    if osm_id in CORNER_WAYS or "角楼" in name:
        return "palace_gate"
    if "城墙" in name or btyp == "wall":
        return "palace_wall"
    if (tags.get("historic") == "city_gate" or btyp == "gatehouse" or name in GATE_NAMES):
        return "palace_gate"
    if in_palace(lon, lat):
        return "palace_hall"
    # fall back to original logic for the Wangfujing side
    return _ORIG_CLASSIFY(tags, lon, lat, kind_hint)



def fix_massing(buildings):
    """Keep halls/gates, turn giant terrace & wall polygons into stone, drop filled courtyards."""
    TERRACE_IDS = {
        "638449433", "638449448", "638449444", "638449443",  # 三大殿三层台基
        "638467894",  # 保和殿北侧相关大台
        "638981252",  # 后三宫台基
    }
    # 午门整座外轮廓 / 墩台大面会把午门广场填成一块黄屋顶，改为不画轮廓、只留 part
    DROP_IDS = {
        "638156366",   # 午门 filled courtyard outline
        "30802096",    # 中山公园 untagged enclosure (would become a giant yellow roof)
        "227109869",   # 劳动人民文化宫 untagged enclosure
        "25109939",    # 太庙 precinct outline
        "25109930",    # 社稷坛 precinct outline
    }
    PLATFORM_IDS = {"638157005"}  # 午门墩台 part，改石台
    rb.PAL["palace_terrace"] = ((210, 206, 198), (180, 176, 168), (200, 196, 188), (160, 156, 148), (220, 216, 208))
    orig_pitched = rb.has_pitched_roof
    def has_pitched(typ, h):
        if typ == "palace_terrace":
            return False
        return orig_pitched(typ, h)
    rb.has_pitched_roof = has_pitched
    out = []
    dropped = 0
    for b in buildings:
        bid = str(b["id"])
        name = b.get("name") or ""
        if bid in DROP_IDS:
            dropped += 1
            continue
        if "院墙" in name:
            b["typ"] = "palace_wall"
            if b["height_src"] != "calibrated" and b["height"] > 8:
                b["height"] = 4.5
                b["height_src"] = "typology_courtyard_wall"
        if bid in TERRACE_IDS or bid in PLATFORM_IDS:
            b["typ"] = "palace_terrace"
            if bid in PLATFORM_IDS:
                b["height"] = 12.0
                b["name"] = b["name"] or "午门墩台"
            elif bid == "638449433":
                b["height"] = 8.13
                b["name"] = b["name"] or "三大殿台基"
                b["height_src"] = "calibrated"
        elif b.get("part") and in_palace(b["lon"], b["lat"]) and (b.get("area") or 0) >= 2000 and b["height"] <= 9:
            b["typ"] = "palace_terrace"
        out.append(b)
    print(f"fix_massing: dropped {dropped} filled-courtyard parents, kept {len(out)}")
    return out


def assemble_plus(nodes, ways, rels):
    # Reuse rb.assemble for ways + water + roads + overture, but with patched classify
    orig_classify = rb.classify
    rb.classify = classify
    try:
        buildings, water, roads = rb.assemble(nodes, ways, rels)
    finally:
        rb.classify = orig_classify

    # Add building multipolygon outers that are not already ways-with-building
    have_ids = {str(b["id"]) for b in buildings}
    way_index = {w["id"]: w for w in ways}
    added = 0
    for r in rels:
        tags = r["tags"]
        is_bldg = bool(tags.get("building") or tags.get("historic") in
                       ("palace", "city_gate", "monument", "castle", "memorial", "tomb"))
        if tags.get("type") not in ("multipolygon", "building") and not is_bldg:
            continue
        if not (is_bldg or tags.get("building") or tags.get("name")):
            continue
        if tags.get("natural") == "water" or tags.get("water"):
            continue
        # Parks, landuse, admin boundaries, and named area outlines must not become roofs
        if tags.get("leisure") in ("park", "garden", "nature_reserve") or tags.get("landuse"):
            continue
        if tags.get("type") in ("boundary", "route"):
            continue
        if not tags.get("building") and not tags.get("building:part") and tags.get("historic") not in (
            "palace", "city_gate", "monument", "castle", "memorial", "tomb",
        ):
            # e.g. 中山公园 / 劳动人民文化宫 / 太庙 enclosure outers
            continue
        outers = [m for m in r["members"] if m["type"] == "way" and m["role"] in ("outer", "outline", "")]
        if not outers:
            outers = [m for m in r["members"] if m["type"] == "way" and m["role"] != "inner"]
        for m in outers:
            if m["ref"] in have_ids:
                continue
            if m["ref"] not in way_index:
                continue
            w = way_index[m["ref"]]
            # skip if the way itself was already ingested as a building
            if "building" in w["tags"] or w["tags"].get("building:part"):
                continue
            coords = rb.way_coords(w, nodes)
            if not coords:
                continue
            poly = rb.lonlat_poly(coords)
            if not poly:
                continue
            merged_tags = dict(w["tags"])
            merged_tags.update({k: v for k, v in tags.items() if k not in merged_tags})
            lon, lat = poly.centroid.x, poly.centroid.y
            area = rb.footprint_area_m2(coords)
            typ = classify(merged_tags, lon, lat)
            h, src, unc = rb.height_for(w["id"], merged_tags, typ, area)
            # also allow calibration by relation id
            cal = rb.CAL["by_osm_way"].get(str(r["id"])) or rb.CAL["by_osm_way"].get(str(w["id"]))
            if cal:
                h, src, unc = float(cal["height_m"]), "calibrated", cal.get("uncertain", False)
            buildings.append({
                "id": w["id"], "source": "osm", "name": merged_tags.get("name"),
                "coords": coords, "poly": poly, "height": h, "height_src": src,
                "uncertain": unc, "typ": typ, "area": area, "lon": lon, "lat": lat,
                "part": False,
            })
            have_ids.add(str(w["id"]))
            added += 1
    print(f"relation-building outers added={added}  total buildings={len(buildings)}")
    buildings = fix_massing(buildings)

    # Relabel corner towers by id / proximity
    corners = {
        "NE": (116.39539, 39.92119),
        "NW": (116.38620, 39.92119),
        "SW": (116.38620, 39.91162),
        "SE": (116.39539, 39.91162),
    }
    for b in buildings:
        bid = str(b["id"])
        if bid in CORNER_WAYS:
            b["typ"] = "palace_gate"
            b["name"] = b["name"] or CORNER_WAYS[bid]
            cal = rb.CAL["by_osm_way"].get(bid)
            if cal:
                b["height"] = float(cal["height_m"])
                b["height_src"] = "calibrated"
            continue
        for cn, (clon, clat) in corners.items():
            d = math.hypot((b["lon"] - clon) * 85300, (b["lat"] - clat) * 111000)
            if d < 35 and b["typ"] in ("palace_hall", "palace_gate") and not (b["name"] and "门" in (b["name"] or "") and "角楼" not in (b["name"] or "")):
                if (b.get("area") or 0) > 80 or bid in CORNER_WAYS:
                    b["typ"] = "palace_gate"
                    if not b["name"]:
                        b["name"] = {"NE": "东北角楼", "NW": "西北角楼", "SW": "西南角楼", "SE": "东南角楼"}[cn]
                    if b["height_src"] != "calibrated":
                        b["height"] = 27.5
                        b["height_src"] = "typology_corner"
    return buildings, water, roads


def lonlat_to_rot(lon, lat):
    return tg.lonlat_to_rot(lon, lat)


def rot_to_lonlat(dx, dy):
    return tg.rot_to_lonlat(dx, dy)


def tile_center(R, C):
    cx = MINX + (2 * C + 1) * SX
    cy = MAXY - (2 * R + 1) * SY
    lon, lat = rot_to_lonlat(cx, cy)
    return lon, lat, cx, cy


def tile_record(R, C):
    lon, lat, cx, cy = tile_center(R, C)
    return {
        "R": R, "C": C,
        "id": f"t1024_r{R}_c{C}",
        "lon": lon, "lat": lat,
        "rot_x": cx, "rot_y": cy,
        "px": C * 1024, "py": R * 1024,  # signed; mosaic will offset
    }


def mosaic_extent(rows, cols):
    x0 = MINX + min(cols) * TILE_WX
    x1 = MINX + (max(cols) + 1) * TILE_WX
    y1 = MAXY - min(rows) * TILE_WY
    y0 = MAXY - (max(rows) + 1) * TILE_WY
    corners = [rot_to_lonlat(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return {
        "west": min(lons), "south": min(lats), "east": max(lons), "north": max(lats),
        "note": "axis-aligned WGS84 bbox of the rotated mosaic corners (includes rotation slack)",
    }


def write_grid(new_tiles, existing_ids):
    old = json.loads((OUT_TILES / "grid.json").read_text())
    # Keep original tiles; append new
    by_id = {t["id"]: t for t in old["tiles_1024"]}
    for t in new_tiles:
        by_id[t["id"]] = t
    tiles = list(by_id.values())
    rows = sorted({t["R"] for t in tiles})
    cols = sorted({t["C"] for t in tiles})
    grid = dict(old)
    grid["tiles_1024"] = tiles
    grid["axis_expansion"] = {
        "step": 1,
        "new_ids": [t["id"] for t in new_tiles if t["id"] not in existing_ids],
        "rows": rows,
        "cols": cols,
        "n_tiles_total": len(tiles),
        "n_new": sum(1 for t in new_tiles if t["id"] not in existing_ids),
        "mosaic_origin_tile": {"R": min(rows), "C": min(cols)},
        "mosaic_px": [(max(cols) - min(cols) + 1) * 1024, (max(rows) - min(rows) + 1) * 1024],
        "wgs84_mosaic_bbox": mosaic_extent(rows, cols),
        "unchanged_pilot_box": old["box"],
        "grid_continuity": "same UTM origin, azimuth -15, elevation -45, 0.29296875 m/px, origin_rot unchanged",
    }
    path = OUT_TILES / "grid_axis.json"
    path.write_text(json.dumps(grid, ensure_ascii=False, indent=2))
    print("wrote", path)
    return grid


def patch_zone():
    orig = rb.zone_ground_color

    def zone(lon, lat):
        if in_palace(lon, lat) or (116.3850 <= lon <= 116.3965 and 39.9110 <= lat <= 39.9222):
            return (198, 176, 132)
        return orig(lon, lat)
    rb.zone_ground_color = zone


def render_tiles(buildings, water, roads, tiles, dest_dir: Path, suffix=""):
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, t in enumerate(tiles):
        out = dest_dir / f"{t['id']}{suffix}.png"
        view = {"id": t["id"], "title": f"tile {t['id']}", "lon": t["lon"], "lat": t["lat"]}
        rb.render_view(view, buildings, water, roads, out, labeled=False)
        print(f"[{i+1}/{len(tiles)}] wrote {out}")
        written.append(out)
    return written


def make_neighbor_sheet(R, C, styled_dir: Path, wb_path: Path, out_path: Path):
    """1024 sheet: this whitebox in the center-ish with 96px edge strips from existing neighbors."""
    canvas = Image.new("RGB", (1024, 1024), (30, 30, 30))
    wb = Image.open(wb_path).convert("RGB")
    if wb.size != (1024, 1024):
        wb = wb.resize((1024, 1024), Image.Resampling.LANCZOS)
    # start from whitebox
    canvas.paste(wb, (0, 0))
    d = ImageDraw.Draw(canvas)
    band = 96
    neighbors = [((R, C + 1), "E"), ((R, C - 1), "W"), ((R + 1, C), "S"), ((R - 1, C), "N")]
    used = []
    for (r, c), side in neighbors:
        p = styled_dir / f"t1024_r{r}_c{c}.jpg"
        if not p.exists():
            continue
        nb = Image.open(p).convert("RGB")
        if nb.size != (1024, 1024):
            nb = nb.resize((1024, 1024), Image.Resampling.LANCZOS)
        if side == "E":
            strip = nb.crop((0, 0, band, 1024))
            canvas.paste(strip, (1024 - band, 0))
            d.line([(1024 - band, 0), (1024 - band, 1023)], fill=(40, 200, 80), width=2)
        elif side == "W":
            strip = nb.crop((1024 - band, 0, 1024, 1024))
            canvas.paste(strip, (0, 0))
            d.line([(band, 0), (band, 1023)], fill=(40, 200, 80), width=2)
        elif side == "S":
            strip = nb.crop((0, 0, 1024, band))
            canvas.paste(strip, (0, 1024 - band))
            d.line([(0, 1024 - band), (1023, 1024 - band)], fill=(40, 200, 80), width=2)
        elif side == "N":
            strip = nb.crop((0, 1024 - band, 1024, 1024))
            canvas.paste(strip, (0, 0))
            d.line([(0, band), (1023, band)], fill=(40, 200, 80), width=2)
        used.append(f"{side}:r{r}c{c}")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    d.rectangle([8, 8, 700, 36], fill=(20, 20, 20))
    d.text((12, 12), f"{f't1024_r{R}_c{C}'} whitebox + neighbor strips {','.join(used) or 'none'}", fill=(240, 240, 220), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=92)
    return out_path, used


LANDMARK_BY_TILE = {
    "t1024_r0_c-1": "太和门（重檐歇山、汉白玉须弥座，通高约 23.8 m）在南；太和殿（面阔十一间、重檐庑殿黄琉璃、殿身 26.92 m）立在三层汉白玉台基（8.13 m，雕栏龙头）上，是全宫最高大的殿。中和殿在太和殿正北、稍小的方形单檐攒尖。三大殿必须在同一南北中轴上，只各画一次，左右对称的门庑和铜鼎、日晷。不要把太和殿画成普通小殿，也不要复制第二座太和殿。",
    "t1024_r0_c-2": "紫禁城西路：武英殿一组黄琉璃殿座、红墙院落，西城墙（灰顶马道、墙身约 9.9 m）和西段筒子河，可能见到西南向的城墙转角。西华门在偏南。不要把三大殿或午门再画进本格。中轴在本格东侧之外。",
    "t1024_r1_c-1": "午门（紫禁城正门，平面凹字形五凤楼）：高大红墙墩台（约 12 m）开三门，台上正中重檐庑殿黄琉璃大殿，两翼雁翅楼向南伸出，四隅重檐方亭。只画这一座午门。北面内金水河与五座内金水桥通向太和门方向。左右阙门。不要把午门画成普通城门楼或天安门。",
    "t1024_r1_c-2": "紫禁城西南角：西南角楼必须是十字脊、三重檐、多角黄琉璃的独特角楼（约 27.5 m），只画一座。西/南城墙和筒子河在此转角。西华门若入画则是侧门城楼（低于午门）。不要复制午门或三大殿。",
    "t1024_r-1_c0": "内廷中轴：乾清门、乾清宫、交泰殿、坤宁宫一线必须在同一南北轴上，黄琉璃、红墙、汉白玉台阶，体量小于太和殿。再往北是御花园（钦安殿、古树、亭座）和神武门（北门城楼，重檐庑殿，面阔五间，无午门那样的南伸雁翅）。神武门只画一座。北侧可见筒子河北段。不要把太和殿或第二座神武门画进来。",
    "t1024_r-1_c-1": "内廷西六宫一带：成组的黄琉璃小殿和红墙院落，西城墙与筒子河西北段。中轴在本格东侧（乾清宫一线不要整组复制到本格中央）。西北方向可看到角楼或北墙。",
    "t1024_r-1_c1": "内廷东六宫 / 宁寿宫一带：成组黄琉璃殿座和红墙，东北城墙和筒子河北段东段。东北角楼（十字脊三重檐，27.5 m）若落在本格则只画一座。不要把神武门或三大殿复制到这里。",
    "t1024_r-1_c-2": "筒子河西段北段、西城墙外侧河面和马道、对岸树列。宫墙内是西路院落边缘。水色与已有东段筒子河一致。不要在河上发明桥梁或把角楼画两座。",
    "t1024_r-1_c2": "筒子河北段东段继续向东，东北城墙外侧，河对岸树列。南边应能接上已完成的东华门以北格。不要再画一座王府井天主堂或重复的东北角楼。",
    "t1024_r-2_c0": "神武门外、筒子河北岸，再往北是景山南坡（土山、柏树）和山顶万春亭（黄琉璃圆亭）若入画则只画一座。河与北门在本格南侧。不要把故宫三大殿搬到景山上。",
    "t1024_r-2_c-1": "筒子河西北转角：西北角楼（十字脊三重檐黄琉璃，27.5 m）只画一座，城墙在此转直角，河面包围转角。北岸可有景山西麓树坡。不要复制第二座角楼或神武门。",
    "t1024_r-2_c1": "筒子河东北段：东北角楼若落在本格则只画一座（与南侧格分享时不要各画一座完整的）。城墙、河面、对岸树木。不要发明欧式建筑。",
}


def write_prompts(tiles):
    out = {
        "style_preamble": (
            "Seamless continuation of an isometric illustrated city map in the exact visual style of the isometric.nyc reference: "
            "warm painted/pixel-illustration look, hard sunlight from upper-left, crisp shadows down-right, "
            "highly detailed roofs and facades, 1024x1024, camera azimuth -15 degrees, elevation -45 degrees, same scale and horizon as the neighbor strips. "
            "Follow the colored massing / whitebox layout exactly: building footprints, heights, streets, and water must stay in the same place. "
            "This is Beijing, not New York and not a generic East-Asian theme park. "
            "Forbidden City architecture is historically accurate: red walls, yellow glazed-tile roofs (庑殿, 歇山, 攒尖 as appropriate), "
            "white marble terraces and railings, grey wall-walks on the crenellated palace walls (the wall top is grey paving, NOT a yellow roof), "
            "cyan-blue moat water matching already-finished tiles, packed-earth courtyards, cypress and scholar trees, no garbled shop signs, no Latin billboards, "
            "no European churches or plazas, no second copy of any unique landmark. Central axis runs south-north and is strictly symmetrical."
        ),
        "tiles": {},
    }
    for t in tiles:
        out["tiles"][t["id"]] = {
            "lon": t["lon"], "lat": t["lat"], "R": t["R"], "C": t["C"],
            "landmarks": LANDMARK_BY_TILE.get(t["id"], "Continue the Forbidden City / moat / hutong fabric accurately. Unique landmarks only once. Yellow glazed roofs and red walls inside the palace, grey wall-walk on the ramparts."),
        }
    path = ROOT / "tiles" / "logs" / "axis_step1_prompts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", path)
    return out


def main():
    update_calibration()
    patch_zone()
    print("parsing merged OSM...")
    nodes, ways, rels = parse_many(OSM_PATHS)
    # attach id onto tags for classify corner lookup? height_for already uses way id
    buildings, water, roads = assemble_plus(nodes, ways, rels)
    c = Counter((b["typ"], b["height_src"]) for b in buildings)
    print("typ/src", c.most_common(15))
    named = [b for b in buildings if b.get("name") and any(k in b["name"] for k in ("殿", "门", "宫", "角楼", "亭"))]
    print("named halls/gates", len(named))
    for b in sorted(named, key=lambda x: -(x["lat"] or 0))[:40]:
        print(f"  {b['name']:12s} id={b['id']} typ={b['typ']:12s} h={b['height']:.1f} {b['height_src']} {b['lon']:.6f},{b['lat']:.6f}")

    existing_ids = {f"t1024_r{R}_c{C}" for R in range(4) for C in range(5)}
    new_tiles = [tile_record(R, C) for R, C in STEP1]
    guide_tiles = [tile_record(R, C) for R, C in GUIDE_EXISTING]
    write_grid(new_tiles, existing_ids)
    write_prompts(new_tiles)

    print("rendering new whiteboxes...")
    render_tiles(buildings, water, roads, new_tiles, OUT_WB, suffix="")
    print("rendering existing-tile guide whiteboxes (do not overwrite originals)...")
    render_tiles(buildings, water, roads, guide_tiles, OUT_WB / "axis_guides", suffix="_axis")

    styled = ROOT / "tiles" / "styled"
    for t in new_tiles:
        sheet, used = make_neighbor_sheet(t["R"], t["C"], styled, OUT_WB / f"{t['id']}.png",
                                          REFS / "axis" / f"{t['id']}_neighbor_sheet.jpg")
        print("sheet", sheet, used)
    print("done")


if __name__ == "__main__":
    main()
