# 故宫–王府井 isometric 试点观看器

Deep Zoom 瓦片需要放在本目录旁（`beijing-pilot.dzi` + `beijing-pilot_files/`）。这些文件不进 Git，请按仓库根目录 README 的说明，从 [GitHub Releases](https://github.com/NovenBuilds/pixel-map-beijing/releases) 下载并解压到仓库根目录。

本地启动（必须通过 HTTP，浏览器会拦截 `file://` 下的瓦片请求）：

```bash
cd viewer
python3 -m http.server 8765
```

然后打开 http://127.0.0.1:8765/

如果只想看静态总览，打开仓库里的 `examples/overview_2560.jpg`。
