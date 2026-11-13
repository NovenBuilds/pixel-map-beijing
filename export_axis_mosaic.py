#!/usr/bin/env python3
"""Stitch the axis-expanded grid (signed R/C), write overviews, junction seams, DZI."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/workspace/beijing-pilot")
STYLED = ROOT / "tiles" / "styled"
WB = ROOT / "tiles" / "whiteboxes"
MOSAIC_DIR = ROOT / "tiles" / "mosaic"
SEAMS = MOSAIC_DIR / "seams_axis"
VIEWER = ROOT / "viewer"
DZI_NAME = "beijing-pilot"
TILE = 1024

# Step-1 mosaic window, same grid as the 5x4 pilot
RMIN, RMAX = -2, 3
CMIN, CMAX = -2, 4
N_R = RMAX - RMIN + 1
N_C = CMAX - CMIN + 1
W, H = N_C * TILE, N_R * TILE


def tile_path(R, C):
    return STYLED / f"t1024_r{R}_c{C}.jpg"


def load_tile(R, C):
    p = tile_path(R, C)
    if p.exists():
        im = Image.open(p).convert("RGB")
        if im.size != (TILE, TILE):
            im = im.resize((TILE, TILE), Image.Resampling.LANCZOS)
        return im, "styled"
    wbp = WB / f"t1024_r{R}_c{C}.png"
    if wbp.exists():
        im = Image.open(wbp).convert("RGB")
        if im.size != (TILE, TILE):
            im = im.resize((TILE, TILE), Image.Resampling.LANCZOS)
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for i in range(-TILE, TILE, 24):
            d.line([(i, 0), (i + TILE, TILE)], fill=(180, 40, 40, 70), width=2)
        d.rectangle([16, 16, 520, 72], fill=(20, 20, 20, 180))
        d.text((24, 32), f"MISSING styled  t1024_r{R}_c{C}", fill=(255, 220, 220, 255))
        return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB"), "missing"
    im = Image.new("RGB", (TILE, TILE), (36, 34, 32))
    d = ImageDraw.Draw(im)
    d.text((30, 30), f"empty r{R}c{C}", fill=(80, 80, 80))
    return im, "empty"


def mosaic_xy(R, C):
    return (C - CMIN) * TILE, (R - RMIN) * TILE


def stitch():
    MOSAIC_DIR.mkdir(parents=True, exist_ok=True)
    SEAMS.mkdir(parents=True, exist_ok=True)
    mosaic = Image.new("RGB", (W, H), (30, 30, 30))
    status = {}
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            im, st = load_tile(R, C)
            x, y = mosaic_xy(R, C)
            mosaic.paste(im, (x, y))
            status[f"t1024_r{R}_c{C}"] = st
    full = MOSAIC_DIR / "overview_full.jpg"
    mosaic.save(full, "JPEG", quality=92, subsampling=0)
    scale = 2560 / max(W, H)
    preview = mosaic.resize((int(W * scale), int(H * scale)), Image.Resampling.LANCZOS)
    prev_path = MOSAIC_DIR / "overview_2560.jpg"
    preview.save(prev_path, "JPEG", quality=90)
    scale2 = 3840 / max(W, H)
    mid = mosaic.resize((int(W * scale2), int(H * scale2)), Image.Resampling.LANCZOS)
    mid.save(MOSAIC_DIR / "overview_3840.jpg", "JPEG", quality=90)
    lab = preview.copy()
    d = ImageDraw.Draw(lab)
    sx, sy = preview.size[0] / W, preview.size[1] / H
    for R in range(RMIN, RMAX + 1):
        for C in range(CMIN, CMAX + 1):
            x0, y0 = mosaic_xy(R, C)
            x0, y0 = int(x0 * sx), int(y0 * sy)
            x1, y1 = int((x0 / sx + TILE) * sx), int((y0 / sy + TILE) * sy)
            st = status[f"t1024_r{R}_c{C}"]
            color = (40, 200, 80) if st == "styled" else ((220, 160, 40) if st == "missing" else (180, 60, 60))
            d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=2)
            d.text((x0 + 4, y0 + 4), f"R{R}C{C}", fill=color)
    grid_path = MOSAIC_DIR / "overview_2560_grid.jpg"
    lab.save(grid_path, "JPEG", quality=90)
    return mosaic, status, full, prev_path, grid_path


def save_crop(mosaic, box, path):
    # clamp
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    mosaic.crop((x0, y0, x1, y1)).save(path, "JPEG", quality=92)
    return str(path)


def seam_crops(mosaic, status):
    crops = []
    half = 256
    # Junctions between NEW axis tiles and the OLD 5x4, plus key internal axis seams
    vertical = [
        # old/new west join
        (0, -1, 0), (1, -1, 0), (0, -2, -1),
        # inner axis
        (-1, -1, 0), (-1, 0, 1), (-2, -1, 0), (-2, 0, 1),
        (-1, 1, 2),
        (1, -2, -1),
        (-1, -2, -1),
        (-2, -2, -1),
        (0, 0, 1),
    ]
    horizontal = [
        # old/new north join (adjacent rows only)
        (-1, 0, 0), (-1, 0, -1), (-1, 0, 1), (-1, 0, 2),
        # jingshan / inner court
        (-2, -1, -1), (-2, -1, 0), (-2, -1, 1),
        # halls / wumen / sw
        (0, 1, -1), (0, 1, -2),
    ]
    for R, C0, C1 in vertical:
        k0, k1 = f"t1024_r{R}_c{C0}", f"t1024_r{R}_c{C1}"
        if status.get(k0) != "styled" or status.get(k1) != "styled":
            continue
        x, y = mosaic_xy(R, C1)  # left edge of the right tile is the seam
        cx, cy = x, y + TILE // 2
        path = SEAMS / f"seam_v_r{R}_c{C0}-{C1}.jpg"
        p = save_crop(mosaic, (cx - half, cy - half, cx + half, cy + half), path)
        if p:
            crops.append(p)
    for R0, R1, C in horizontal:
        k0, k1 = f"t1024_r{R0}_c{C}", f"t1024_r{R1}_c{C}"
        if status.get(k0) != "styled" or status.get(k1) != "styled":
            continue
        x, y = mosaic_xy(R1, C)  # top edge of the lower tile is the seam
        cx, cy = x + TILE // 2, y
        path = SEAMS / f"seam_h_r{R0}-{R1}_c{C}.jpg"
        p = save_crop(mosaic, (cx - half, cy - half, cx + half, cy + half), path)
        if p:
            crops.append(p)
    return crops


def export_dzi(mosaic: Image.Image):
    tile_size, overlap, fmt = 256, 1, "jpg"
    max_dim = max(mosaic.size)
    max_level = math.ceil(math.log2(max_dim))
    out = VIEWER / f"{DZI_NAME}_files"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    dzi = VIEWER / f"{DZI_NAME}.dzi"
    dzi.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Image TileSize="{tile_size}" Overlap="{overlap}" Format="{fmt}" '
        'xmlns="http://schemas.microsoft.com/deepzoom/2008">\n'
        f'  <Size Width="{mosaic.size[0]}" Height="{mosaic.size[1]}"/>\n'
        "</Image>\n"
    )
    for level in range(max_level + 1):
        scale = 2 ** (level - max_level)
        lw = mosaic.size[0] if scale >= 1 else max(1, int(round(mosaic.size[0] * scale)))
        lh = mosaic.size[1] if scale >= 1 else max(1, int(round(mosaic.size[1] * scale)))
        level_im = mosaic if level == max_level else mosaic.resize((lw, lh), Image.Resampling.LANCZOS)
        cols = math.ceil(lw / tile_size)
        rows = math.ceil(lh / tile_size)
        ldir = out / str(level)
        ldir.mkdir(parents=True, exist_ok=True)
        for ty in range(rows):
            for tx in range(cols):
                left = max(0, tx * tile_size - overlap)
                top = max(0, ty * tile_size - overlap)
                right = min(lw, (tx + 1) * tile_size + overlap)
                bottom = min(lh, (ty + 1) * tile_size + overlap)
                level_im.crop((left, top, right, bottom)).save(ldir / f"{tx}_{ty}.{fmt}", "JPEG", quality=90)
    return dzi


def write_viewer():
    VIEWER.mkdir(parents=True, exist_ok=True)
    html = VIEWER / "index.html"
    html.write_text("""<!DOCTYPE html>
<html lang="zh-Hans">
<head>
  <meta charset="utf-8"/>
  <title>Beijing isometric — 中轴线扩展（故宫主体）</title>
  <style>
    html, body { margin: 0; height: 100%; background: #1a1816; color: #eee; font-family: sans-serif; }
    #osd { width: 100%; height: 100%; }
    #hint {
      position: absolute; left: 12px; bottom: 12px; z-index: 2;
      background: rgba(0,0,0,.6); padding: 8px 12px; font-size: 13px; border-radius: 6px;
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/openseadragon@4.1.1/build/openseadragon/openseadragon.min.js"></script>
</head>
<body>
  <div id="osd"></div>
  <div id="hint">故宫中轴扩展 · 滚轮缩放 · 拖拽平移 · 本地 Deep Zoom</div>
  <script>
    OpenSeadragon({
      id: "osd",
      prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1.1/build/openseadragon/images/",
      tileSources: "beijing-pilot.dzi",
      showNavigator: true,
      visibilityRatio: 0.8,
      minZoomImageRatio: 0.5,
      maxZoomPixelRatio: 3
    });
  </script>
</body>
</html>
""")
    return html


def main():
    mosaic, status, full, prev, grid = stitch()
    crops = seam_crops(mosaic, status)
    dzi = export_dzi(mosaic)
    html = write_viewer()
    missing = [k for k, v in status.items() if v != "styled"]
    done = [k for k, v in status.items() if v == "styled"]
    report = {
        "mosaic_px": [W, H],
        "rows": list(range(RMIN, RMAX + 1)),
        "cols": list(range(CMIN, CMAX + 1)),
        "overview_full": str(full),
        "overview_2560": str(prev),
        "overview_2560_grid": str(grid),
        "dzi": str(dzi),
        "viewer": str(html),
        "done_1024": sorted(done, key=lambda s: (s.split("_r")[-1])),
        "missing_or_empty": sorted(missing),
        "n_done": len(done),
        "n_cells": N_R * N_C,
        "seam_crops": crops,
    }
    (MOSAIC_DIR / "stitch_report_axis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
