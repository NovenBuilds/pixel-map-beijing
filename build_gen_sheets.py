#!/usr/bin/env python3
"""Build caption-free reference sheets for GenerateImage (global context + neighbor strips)."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path("/workspace/beijing-pilot")
GRID = json.loads((ROOT / "tiles/grid_axis.json").read_text())
SLICES = ROOT / "tiles/whiteboxes/global/slices"
STYLED = ROOT / "tiles/styled"
GLOBAL = ROOT / "tiles/whiteboxes/global/global_whitebox.png"
OUT = ROOT / "tiles/refs/global_sheets"
OUT.mkdir(parents=True, exist_ok=True)

MOS_R0 = GRID["axis_expansion"]["mosaic_origin_tile"]["R"]
MOS_C0 = GRID["axis_expansion"]["mosaic_origin_tile"]["C"]
TILE = 1024
BAND = 120  # neighbor strip width, no labels

def mosaic_xy(r, c):
    return (c - MOS_C0) * TILE, (r - MOS_R0) * TILE

# Only use geometrically trusted styled tiles as neighbors.
# v1 expansion tiles were fragmented (repeated moat/wall rings) and must not be referenced.
STABLE = Path("/tmp/stable_tiles.json")
# default: original 5x4 (R>=0, C>=0) plus any later accepted v2 ids listed here
ACCEPTED = set()  # filled by argv or a sidecar
_sidecar = Path("/workspace/beijing-pilot/tiles/styled/accepted_v2.json")
if _sidecar.exists():
    import json as _json
    ACCEPTED |= set(_json.loads(_sidecar.read_text()))

def is_stable(r, c):
    tid = f"t1024_r{r}_c{c}"
    if tid in ACCEPTED:
        return True
    # original immutable 5x4 pilot
    if r >= 0 and c >= 0 and (STYLED / f"{tid}.jpg").exists():
        # but r0_c0 west is being rebuilt — still usable as scale/east-south reference
        return True
    return False

def styled_path(r, c):
    if not is_stable(r, c):
        return None
    p = STYLED / f"t1024_r{r}_c{c}.jpg"
    return p if p.exists() else None

def make_context(r, c, gimg):
    """Crop of the global whitebox around the tile, red outline, no text."""
    x, y = mosaic_xy(r, c)
    W, H = gimg.size
    pad = 900
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(W, x + TILE + pad)
    bot = min(H, y + TILE + pad)
    crop = gimg.crop((left, top, right, bot)).convert("RGB")
    # scale to ~1400 on long side
    scale = 1400 / max(crop.size)
    if scale < 1:
        crop = crop.resize((max(1, int(crop.size[0]*scale)), max(1, int(crop.size[1]*scale))), Image.Resampling.LANCZOS)
        s = scale
    else:
        s = 1.0
    d = ImageDraw.Draw(crop)
    rx = int((x - left) * s)
    ry = int((y - top) * s)
    rw = int(TILE * s)
    rh = int(TILE * s)
    for w in range(6):
        d.rectangle([rx-w, ry-w, rx+rw+w, ry+rh+w], outline=(220, 30, 30))
    out = OUT / f"t1024_r{r}_c{c}_context.jpg"
    crop.save(out, "JPEG", quality=90)
    return out

def make_neighbors(r, c):
    """Edge-strip collage of already-styled neighbors. No captions."""
    canvas = Image.new("RGB", (1024, 1024), (20, 20, 20))
    # start with a dark frame; paste available neighbors at their true edges
    # We'll also put a reduced interior so the model sees adjacency, but leave the
    # center nearly empty (dark) so it doesn't copy a neighbor into the tile.
    used = []
    for (dr, dc, box) in [
        (-1, 0, (0, 0, 1024, BAND)),             # north strip at top
        (1, 0, (0, 1024-BAND, 1024, 1024)),      # south
        (0, -1, (0, 0, BAND, 1024)),              # west
        (0, 1, (1024-BAND, 0, 1024, 1024)),       # east
    ]:
        p = styled_path(r+dr, c+dc)
        if not p:
            continue
        nb = Image.open(p).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        if dc == 0 and dr == -1:
            src = nb.crop((0, 1024-BAND, 1024, 1024))  # south edge of northern neighbor
        elif dc == 0 and dr == 1:
            src = nb.crop((0, 0, 1024, BAND))
        elif dr == 0 and dc == -1:
            src = nb.crop((1024-BAND, 0, 1024, 1024))
        else:
            src = nb.crop((0, 0, BAND, 1024))
        canvas.paste(src, box[:2])
        used.append((r+dr, c+dc))
    # If no neighbors, skip
    if not used:
        return None
    out = OUT / f"t1024_r{r}_c{c}_neighbors.jpg"
    canvas.save(out, "JPEG", quality=90)
    return out, used

def main():
    gimg = Image.open(GLOBAL)
    tiles = [
        (0, 0),  # west infill
        (0, -1), (0, -2),
        (1, -1), (1, -2),
        (-1, 0), (-1, -1), (-1, -2), (-1, 1), (-1, 2),
        (-2, 0), (-2, -1), (-2, 1),
    ]
    report = []
    for r, c in tiles:
        ctx = make_context(r, c, gimg)
        nb = make_neighbors(r, c)
        report.append({"r": r, "c": c, "context": str(ctx), "neighbors": None if not nb else str(nb[0]), "used": None if not nb else nb[1]})
        print(f"r{r}_c{c}: context ok, neighbors={None if not nb else nb[1]}")
    (OUT / "index.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
