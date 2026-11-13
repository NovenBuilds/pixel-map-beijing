# pixel-map-beijing

北京超大型等距像素地图。风格对标 [isometric.nyc](https://isometric.nyc/)，定位是积木城生成艺术，不是带搜索与标注的地图工具。当前仓库是故宫到王府井（并沿中轴线外扩中）的试点源码。

许可证为 MIT（见 `LICENSE`）。

## 仓库里有什么

- 源码：白盒渲染、格网、中轴线外扩、色板统一、拼接与 Deep Zoom 导出等 Python 脚本
- 配置与输入数据：`data/`（试点范围、高度校准、OSM / Overture 提取）、`tiles/grid.json`、`tiles/grid_axis.json` 与 `tiles/grid_fifth_ring.json`
- 文档：`REPORT.md`（试点过程记录）
- 示例：`examples/overview_2560.jpg`（缩小总览）、`styled/`（早期风格样张）、`renders/` 与 `renders_v2/`（白盒正交样图）
- 观看器页面：`viewer/index.html`

生成结果默认不进 Git：地图瓦片、拼接总览、白盒、DZI 金字塔、缓存和过程备份已写入 `.gitignore`。完整地图数据通过 [GitHub Releases](https://github.com/NovenBuilds/pixel-map-beijing/releases) 分阶段发布。后续扩展会发布新的 Release，不会覆盖旧版本。

## 数据说明

- 建筑足迹主要来自 OpenStreetMap，遵循 [ODbL](https://www.openstreetmap.org/copyright)。若你基于本仓库数据制作衍生地图并公开发布，请按 ODbL 署名并提供同样许可。
- 部分轮廓来自 [Overture Maps](https://overturemaps.org/)。
- 坐标使用 WGS84，不要与 GCJ-02 混用。
- 试点盒（WGS84）：西 116.39480°E，南 39.90620°N，东 116.40780°E，北 39.91800°N。沿中轴线外扩后的拼接窗口见 `tiles/grid_axis.json`。

本仓库不包含第三方 `isometric-nyc` 源码，只在风格上参考该项目。

## 本地运行

只需 Python 3 标准库即可启动观看器。生成新瓦片才需要 `requirements.txt` 里的依赖（Pillow、numpy、shapely、pyproj）。

仓库本身不包含可缩放的 DZI 瓦片。请先克隆源码，再下载对应版本的 GitHub Release 数据包，解压到仓库根目录后启动本地 HTTP 服务。

### 1. 克隆仓库

```bash
git clone https://github.com/NovenBuilds/pixel-map-beijing.git
cd pixel-map-beijing
```

### 2. 下载并解压 Release 数据包

在仓库根目录下载预览版 `v0.1.0-preview` 的附件（也可换成更新的 Release，但不要覆盖已保存的旧包）：

```bash
# 需要已安装 GitHub CLI
gh release download v0.1.0-preview --repo NovenBuilds/pixel-map-beijing
```

也可以在浏览器打开 [Releases 页面](https://github.com/NovenBuilds/pixel-map-beijing/releases) ，下载 `beijing-pilot-dzi-v0.1.0-preview.zip` 和 `SHA256SUMS.txt`。

校验并解压到仓库根目录（会在 `viewer/` 下写入 `beijing-pilot.dzi` 与 `beijing-pilot_files/`）：

```bash
sha256sum -c SHA256SUMS.txt
unzip beijing-pilot-dzi-v0.1.0-preview.zip
```

版本号、对应 Git 提交、覆盖范围、文件体积和 SHA256 写在该 [Release 说明](https://github.com/NovenBuilds/pixel-map-beijing/releases/tag/v0.1.0-preview) 里，也可对照 `SHA256SUMS.txt`。

### 3. 启动观看器

```bash
cd viewer
python3 -m http.server 8765
```

然后在浏览器打开 http://127.0.0.1:8765/ 。不要用 `file://` 直接打开 HTML，浏览器会拦截瓦片请求。

本预览包覆盖当前直播拼接窗：行 R=−2…3、列 C=−2…4（7168×6144 像素），范围从景山到故宫、王府井，并已并入天安门。广场西侧与前门一带仍在草稿中，未进入本包。

没有下载瓦片时，可先看 `examples/overview_2560.jpg`。
