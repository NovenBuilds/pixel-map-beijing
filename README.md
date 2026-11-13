# pixel-map-beijing

北京超大型等距像素地图。风格对标 [isometric.nyc](https://isometric.nyc/)，定位是积木城生成艺术，不是带搜索与标注的地图工具。当前仓库是故宫到王府井（并沿中轴线外扩中）的试点源码。

许可证为 MIT（见 `LICENSE`）。

## 仓库里有什么

- 源码：白盒渲染、格网、中轴线外扩、色板统一、拼接与 Deep Zoom 导出等 Python 脚本
- 配置与输入数据：`data/`（试点范围、高度校准、OSM / Overture 提取）、`tiles/grid.json` 与 `tiles/grid_axis.json`
- 文档：`REPORT.md`（试点过程记录）
- 示例：`examples/overview_2560.jpg`（缩小总览）、`styled/`（早期风格样张）、`renders/` 与 `renders_v2/`（白盒正交样图）
- 观看器页面：`viewer/index.html`

生成结果默认不进 Git：地图瓦片、拼接总览、白盒、DZI 金字塔、缓存和过程备份已写入 `.gitignore`。完整地图数据稍后通过 GitHub Releases 发布。

## 数据说明

- 建筑足迹主要来自 OpenStreetMap，遵循 [ODbL](https://www.openstreetmap.org/copyright)。若你基于本仓库数据制作衍生地图并公开发布，请按 ODbL 署名并提供同样许可。
- 部分轮廓来自 [Overture Maps](https://overturemaps.org/)。
- 坐标使用 WGS84，不要与 GCJ-02 混用。
- 试点盒（WGS84）：西 116.39480°E，南 39.90620°N，东 116.40780°E，北 39.91800°N。沿中轴线外扩后的拼接窗口见 `tiles/grid_axis.json`。

本仓库不包含第三方 `isometric-nyc` 源码，只在风格上参考该项目。

## 本地运行

依赖见 `requirements.txt`（Python 3，主要是 Pillow、numpy、shapely、pyproj）。

本地若已有生成好的瓦片（默认在 `tiles/styled/` 与 `tiles/mosaic/`，被 git 忽略）：

```bash
cd viewer
python3 -m http.server 8765
```

然后打开 http://127.0.0.1:8765/ 。不要用 `file://` 直接打开，浏览器会拦截瓦片请求。

没有本地瓦片时，可先看 `examples/overview_2560.jpg`。
