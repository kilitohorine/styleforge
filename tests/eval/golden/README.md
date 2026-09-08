# 黄金集说明

本目录是一期 **mock 评测** 的标注，不是作品集照片。

- `dialogues.json`：30 条路由标注（E1），对 `keyword_route` / fake LLM，不打真 API。
- `qa.json`：10 条 RAG 应命中文档（E4）。
- `negatives.json`：负例，3D/长视频不得创建付费 Job（G2）。
- 摄影源图 **不入库合成网图**。CI 用 OpenCV 几何色块当场绘制；你要用自有版权 JPG 替换时，把文件放到本目录 `photos/` 并在 `photos.json` 登记（当前未提交）。

`python scripts/eval_report.py` 写出 `data/eval/report.md`（`data/` gitignore）。
`pytest -m golden` 在 mock 下断言门禁与 E1/E2/E4。

E3 真 2D 出图、12 张版权人像/风光：**未跑**，报告必须写 skipped。
