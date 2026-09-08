# StyleForge

当前版本（一期）：**8 种摄影 Look（自烘焙 `.cube`）+ Style Pack RAG + image.2d 预算熔断 + Docker**。  
LangGraph：`route → perceive → execute → critique`（Look，0 元）；问答走 RAG；「改成水墨画」才走硅基流动。  
LUT 格式对齐 [CubeLUT](https://cubelut.cn/index.php) / Premiere。进度见 [`任务文档.md`](./任务文档.md)。

**本仓库不是 PhotoAgent 官方实现**。MCTS / GRPO 未做，开关默认关。

## 一天闭环能做什么

- 8 种摄影 Look：胶片暖调、电影青橙、港风夜景、黑白、复古褪色、黄金时刻、冷调青灰、电影哑光
- 多轮「再暗一点 / 再暖一点 / 少颗粒」；无风格口令时 `perceive` 按场景建议 Look
- `critique`：SSIM + 尺寸，最多再执行 1 次
- FastAPI：`/v1/health` `/v1/chat` `/v1/jobs` `/v1/styles` `/v1/rag/query` `/v1/budget`
- 15 个 Style Pack（8 摄影 + 7 艺术）可检索；艺术风格要出图需 Key
- Docker Compose：`api` + `ui`，数据目录 `STYLEFORGE_DATA_DIR`

## 明确不做（避免简历夸大）

- 未复现 PhotoAgent 的 MCTS / UGC Reward（GRPO）
- 未生成 7 种艺术风格作品集（无 Key 时 2D 失败关闭）
- 3D / 长视频仅 501 占位

## 环境（Windows）

Python **3.12**（`E:\python\python.exe`）。不要用 3.14。

```powershell
cd F:\item
E:\python\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## 启动

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

另开终端：`python app_ui.py` → http://127.0.0.1:7860  
OpenAPI：http://127.0.0.1:8000/docs

Docker（需已安装 Docker Desktop，先有 `.env`）：

```powershell
docker compose up --build
```

API `8000`，Gradio `7860`。容器内 UI 监听 `0.0.0.0`。

## 测试

```powershell
pytest -q
```

## LangGraph 节点（面试可画）

```
START → route → perceive → execute → critique → (失败则再 execute 一次) END
              ↘ 问答 / 无图 / 3D END
              ↘ image.2d execute END
```

摄影路径只调 `PhotoLookRenderer`；2D 只调 `Image2DRenderer`（厂商 URL 仅 `app/providers/`）。

## 简历建议写法（可直接改项目名/仓库链接）

中文：

> 独立完成修图 Agent StyleForge：用 LangGraph 编排 route→perceive→execute→critique；8 种摄影 Look 为自烘焙 Adobe `.cube` 3D LUT（OpenCV，零推理成本）；风格知识用 YAML Style Pack + Chroma RAG；2D 生图经硅基流动适配器，带日/项目预算熔断与失败关闭。参考 PhotoAgent 闭环，非论文官方复现。

English:

> Built StyleForge, a LangGraph photo-styling agent (route→perceive→execute→critique) with 8 self-baked .cube LUTs, Chroma style-pack RAG, and a budget-capped SiliconFlow image.2d adapter. Inspired by PhotoAgent; not an official reproduction.

不要写「开发了 PhotoAgent」「已支持 10 种风格出图」「已复现 MCTS/GRPO」。

## Citation

```bibtex
@article{yao2025photoagent,
  title   = {PhotoAgent: Agentic Photo Editing with Exploratory Visual Aesthetic Planning},
  author  = {Yao, Mingde and You, Zhiyuan and Tam, King-Man and Wang, Menglu and Xue, Tianfan},
  year    = {2025}
}
```

项目页：https://mdyao.github.io/PhotoAgent/
