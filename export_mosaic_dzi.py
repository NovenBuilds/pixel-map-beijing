#!/usr/bin/env python3
"""Stitch 1024 tiles, write seam close-ups, a Deep Zoom pyramid, and a local viewer."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/workspace/beijing-pilot")
GRID = json.loads((ROOT / "tiles" / "grid.json").read_text())
STYLED = ROOT / "tiles" / "styled"
WB = ROOT / "tiles" / "whiteboxes"
MOSAIC_DIR = ROOT / "tiles" / "mosaic"
SEAMS = MOSAIC_DIR / "seams"
VIEWER = ROOT / "viewer"
DZI_NAME = "beijing-pilot"
TILE = 1024
N_C = GRID["n_tiles_1024_cols"]
N_R = GRID["n_tiles_1024_rows"]
W, H = N_C * TILE, N_R * TILE


def load_tile(R, C):
    p = STYLED / f"t1024_r{R}_c{C}.jpg"
    if p.exists():
        im = Image.open(p).convert("RGB")
        if im.size != (TILE, TILE):
            im = im.resize((TILE, TILE), Image.Resampling.LANCZOS)
        return im, "styled"
    wbp = WB / f"t1024_r{R}_c{C}.png"
    if wbp.exists():
        im = Image.open(wbp).convert("RGB")
        # mark missing stylized tile with a thin hatch overlay and label
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for i in range(-TILE, TILE, 24):
            d.line([(i, 0), (i + TILE, TILE)], fill=(180, 40, 40, 70), width=2)
        d.rectangle([20, 20, 420, 80], fill=(20, 20, 20, 180))
        d.text((30, 36), f"MISSING styled  t1024_r{R}_c{C}", fill=(255, 220, 220, 255))
        im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        return im, "missing"
    im = Image.new("RGB", (TILE, TILE), (40, 40, 40))
    return im, "empty"


def stitch():
    MOSAIC_DIR.mkdir(parents=True, exist_ok=True)
    SEAMS.mkdir(parents=True, exist_ok=True)
    mosaic = Image.new("RGB", (W, H), (30, 30, 30))
    status = {}
    for R in range(N_R):
        for C in range(N_C):
            im, st = load_tile(R, C)
            mosaic.paste(im, (C * TILE, R * TILE))
            status[f"t1024_r{R}_c{C}"] = st
    full = MOSAIC_DIR / "overview_full.jpg"
    mosaic.save(full, "JPEG", quality=92, subsampling=0)
    # chat-viewable downscale (~2560 on the long side)
    scale = 2560 / max(W, H)
    preview = mosaic.resize((int(W * scale), int(H * scale)), Image.Resampling.LANCZOS)
    prev_path = MOSAIC_DIR / "overview_2560.jpg"
    preview.save(prev_path, "JPEG", quality=90)
    # also a slightly larger 3840 for inspection
    scale2 = 3840 / max(W, H)
    mid = mosaic.resize((int(W * scale2), int(H * scale2)), Image.Resampling.LANCZOS)
    mid_path = MOSAIC_DIR / "overview_3840.jpg"
    mid.save(mid_path, "JPEG", quality=90)
    # labeled grid overlay on the 2560 preview
    lab = preview.copy()
    d = ImageDraw.Draw(lab)
    sx, sy = preview.size[0] / W, preview.size[1] / H
    for R in range(N_R):
        for C in range(N_C):
            x0, y0 = int(C * TILE * sx), int(R * TILE * sy)
            x1, y1 = int((C + 1) * TILE * sx), int((R + 1) * TILE * sy)
            color = (40, 200, 80) if status[f"t1024_r{R}_c{C}"] == "styled" else (220, 60, 60)
            d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=2)
            d.text((x0 + 6, y0 + 6), f"R{R}C{C}", fill=color)
    lab.save(MOSAIC_DIR / "overview_2560_grid.jpg", "JPEG", quality=90)
    return mosaic, status, full, prev_path


def seam_crops(mosaic, status):
    # 512 px crops centered on interior seams (256 around the join)
    crops = []
    half = 256
    size = 512
    # vertical seams between (R,C) and (R,C+1)
    pairs_v = []
    pairs_h = []
    for R in range(N_R):
        for C in range(N_C - 1):
            if status[f"t1024_r{R}_c{C}"] == "styled" and status[f"t1024_r{R}_c{C+1}"] == "styled":
                pairs_v.append((R, C))
    for R in range(N_R - 1):
        for C in range(N_C):
            if status[f"t1024_r{R}_c{C}"] == "styled" and status[f"t1024_r{R+1}_c{C}"] == "styled":
                pairs_h.append((R, C))
    # pick representative + all landmark-adjacent
    prefer_v = {(1, 0), (1, 1), (1, 2), (1, 3), (2, 1), (3, 2), (0, 1), (2, 2)}
    prefer_h = {(0, 1), (1, 1), (1, 3), (2, 3), (1, 2), (0, 2), (2, 1), (2, 2)}
    selected_v = [p for p in pairs_v if p in prefer_v] or pairs_v[:6]
    selected_h = [p for p in pairs_h if p in prefer_h] or pairs_h[:6]
    for R, C in selected_v:
        cx = (C + 1) * TILE
        cy = R * TILE + TILE // 2
        box = (cx - half, cy - half, cx + half, cy + half)
        crop = mosaic.crop(box)
        path = SEAMS / f"seam_v_r{R}_c{C}-{C+1}.jpg"
        crop.save(path, "JPEG", quality=92)
        crops.append(str(path))
    for R, C in selected_h:
        cx = C * TILE + TILE // 2
        cy = (R + 1) * TILE
        box = (cx - half, cy - half, cx + half, cy + half)
        crop = mosaic.crop(box)
        path = SEAMS / f"seam_h_r{R}-{R+1}_c{C}.jpg"
        crop.save(path, "JPEG", quality=92)
        crops.append(str(path))
    return crops


def export_dzi(mosaic: Image.Image):
    """Pure-Pillow Deep Zoom (no libvips). TileSize 256, Overlap 1, JPEG."""
    tile_size = 256
    overlap = 1
    fmt = "jpg"
    max_dim = max(mosaic.size)
    max_level = math.ceil(math.log2(max_dim))
    out = VIEWER / f"{DZI_NAME}_files"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    # DZI descriptor
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
        lw = max(1, int(round(mosaic.size[0] * scale))) if scale < 1 else mosaic.size[0]
        lh = max(1, int(round(mosaic.size[1] * scale))) if scale < 1 else mosaic.size[1]
        if level == max_level:
            level_im = mosaic
        else:
            level_im = mosaic.resize((lw, lh), Image.Resampling.LANCZOS)
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
                tile = level_im.crop((left, top, right, bottom))
                tile.save(ldir / f"{tx}_{ty}.{fmt}", "JPEG", quality=90)
    return dzi


def write_viewer():
    VIEWER.mkdir(parents=True, exist_ok=True)
    html = VIEWER / "index.html"
    html.write_text("""<!DOCTYPE html>
<html lang="zh-Hans">
<head>
  <meta charset="utf-8"/>
  <title>Beijing isometric pilot — 故宫–王府井</title>
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
  <div id="hint">故宫–王府井 试点 · 滚轮缩放 · 拖拽平移 · 本地 Deep Zoom</div>
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
    readme = VIEWER / "README.md"
    readme.write_text(
        "# 故宫–王府井 isometric 试点观看器\n\n"
        "Deep Zoom tiles live next to this file (`beijing-pilot.dzi` + `beijing-pilot_files/`).\n\n"
        "Open locally (needed because browsers block `file://` tile loads):\n\n"
        "```bash\n"
        "cd /workspace/beijing-pilot/viewer\n"
        "python3 -m http.server 8765\n"
        "```\n\n"
        "Then visit http://127.0.0.1:8765/\n\n"
        "If you only want a still image, open:\n\n"
        "- `/workspace/beijing-pilot/tiles/mosaic/overview_2560.jpg` (chat-sized)\n"
        "- `/workspace/beijing-pilot/tiles/mosaic/overview_full.jpg` (5120×4096, full stitch)\n"
        "- `/workspace/beijing-pilot/tiles/mosaic/seams/` (seam close-ups)\n"
    )
    return html


def main():
    mosaic, status, full, prev = stitch()
    crops = seam_crops(mosaic, status)
    dzi = export_dzi(mosaic)
    html = write_viewer()
    missing = [k for k, v in status.items() if v != "styled"]
    done = [k for k, v in status.items() if v == "styled"]
    report = {
        "mosaic_px": [W, H],
        "overview_full": str(full),
        "overview_2560": str(prev),
        "dzi": str(dzi),
        "viewer": str(html),
        "done_1024": sorted(done),
        "missing_1024": sorted(missing),
        "n_done": len(done),
        "n_missing": len(missing),
        "n_cells_512_done": len(done) * 4,
        "seam_crops": crops,
    }
    (MOSAIC_DIR / "stitch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
