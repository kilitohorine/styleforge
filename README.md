# StyleForge

当前版本（P1）：**Look LUT + Style Pack RAG + image.2d 硅基流动（预算熔断）**。  
上传照片 → LangGraph（route → execute）→ `.cube` 3D LUT；无图问答走 RAG；说「改成水墨画」才走 2D 生图。  
LUT 格式对齐 [CubeLUT](https://cubelut.cn/index.php) / Premiere，文件为仓库自烘焙。进度见 [`任务文档.md`](./任务文档.md)。

编排用 LangGraph，方便写进简历；修图思路参考 PhotoAgent 的感知→规划→执行闭环。  
**本仓库不是 PhotoAgent 官方实现**（官方代码尚未发布）。

## 一天闭环能做什么

- 三种 Look：`film_portra`、`cinematic_teal_orange`、`hk_night`（`.cube` LUT）
- 自然语言路由；多轮「再暗一点 / 再暖一点 / 少颗粒」
- FastAPI：`/v1/health` `/v1/chat` `/v1/jobs` `/v1/assets` `/v1/styles` `/v1/rag/query` `/v1/budget`
- 10 个 Style Pack YAML + Chroma 检索；「水彩和水墨差在哪」走 RAG citation，不出图
- `image.2d`：硅基流动（Kolors）；无 Key 失败关闭；超 `DAILY_BUDGET_CNY` / `PROJECT_BUDGET_CNY` 不发请求
- Gradio：原图、结果、原图|结果
- 费用：LUT 调色路径 **0 元**；2D 按次计费并记账

## 明确不做（避免简历夸大）

- 未复现 PhotoAgent 的完整 MCTS / UGC Reward（GRPO）
- 未做 7 种艺术风格作品集出图（接 Key 后可单张试 `改成水墨画`，受日预算熔断）
- 3D / 长视频返回能力保留为 reserved

## 环境（Windows）

Python **3.12**（本机已验证：`E:\python\python.exe`）。不要用 3.14 装这套依赖。

```powershell
cd F:\item
E:\python\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## 启动

终端 1 — API：

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开 http://127.0.0.1:8000/docs

终端 2 — 界面：

```powershell
.\.venv\Scripts\Activate.ps1
python app_ui.py
```

打开 http://127.0.0.1:7860 ，上传照片，输入「做成胶片暖调」。无图可问「水彩和水墨差在哪」。

首次风格检索会自动入库；也可手动：

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/v1/styles/ingest
curl.exe -s -X POST http://127.0.0.1:8000/v1/rag/query -H "Content-Type: application/json" -d "{\"query\":\"夜景霓虹\",\"k\":3}"
```

## 测试

```powershell
pytest -q
```

## LangGraph 节点（面试可画）

```
START → route → (有图且是 Look) execute → END
              ↘ (问答 / 无图) END
```

`execute` 对摄影 Look 只调用 `PhotoLookRenderer`（OpenCV）；对 `image.2d` 只调用 `Image2DRenderer`（厂商 URL 仅出现在 `app/providers/`）。

## 简历建议写法

> 参考 PhotoAgent（Yao et al.）的修图 Agent 闭环，使用 LangGraph 实现 route→execute 编排，摄影 Look 以 OpenCV 传统调色为执行器（零推理成本）。

不要写「开发了 PhotoAgent」或「复现了论文全部实验」。

## Citation

```bibtex
@article{yao2025photoagent,
  title   = {PhotoAgent: Agentic Photo Editing with Exploratory Visual Aesthetic Planning},
  author  = {Yao, Mingde and You, Zhiyuan and Tam, King-Man and Wang, Menglu and Xue, Tianfan},
  year    = {2025}
}
```

项目页：https://mdyao.github.io/PhotoAgent/
