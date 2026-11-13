# 故宫–王府井 isometric 试点观看器

Deep Zoom tiles live next to this file (`beijing-pilot.dzi` + `beijing-pilot_files/`).

Open locally (needed because browsers block `file://` tile loads):

```bash
cd /workspace/beijing-pilot/viewer
python3 -m http.server 8765
```

Then visit http://127.0.0.1:8765/

If you only want a still image, open:

- `/workspace/beijing-pilot/tiles/mosaic/overview_2560.jpg` (chat-sized)
- `/workspace/beijing-pilot/tiles/mosaic/overview_full.jpg` (5120×4096, full stitch)
- `/workspace/beijing-pilot/tiles/mosaic/seams/` (seam close-ups)
