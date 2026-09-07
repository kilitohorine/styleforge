# StyleForge

当前版本（P0）：**上传照片 → LangGraph（route → execute）→ `.cube` 3D LUT 调色 → 并排对比图**。  
多轮可说「再暗一点」（不丢风格）。LUT 格式对齐 [CubeLUT](https://cubelut.cn/index.php) / Premiere，文件为仓库自烘焙，不使用该站素材。  
进度见 [`任务文档.md`](./任务文档.md)。示意对比条见 `samples/`。

编排用 LangGraph，方便写进简历；修图思路参考 PhotoAgent 的感知→规划→执行闭环。  
**本仓库不是 PhotoAgent 官方实现**（官方代码尚未发布）。

## 一天闭环能做什么

- 三种 Look：`film_portra`、`cinematic_teal_orange`、`hk_night`（`.cube` LUT）
- 自然语言路由；多轮「再暗一点 / 再暖一点 / 少颗粒」
- FastAPI：`/v1/health` `/v1/chat` `/v1/chat/{thread_id}` `/v1/jobs` `/v1/assets`
- Gradio：原图、结果、原图|结果
- 费用：LUT 调色路径 **0 元**

## 明确不做（避免简历夸大）

- 未复现 PhotoAgent 的完整 MCTS / UGC Reward（GRPO）
- 未做水彩/吉卜力等生成式风格（预留 `image.2d`）
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

打开 http://127.0.0.1:7860 ，上传照片，输入「做成胶片暖调」。

## 测试

```powershell
pytest -q
```

## LangGraph 节点（面试可画）

```
START → route → (有图且是 Look) execute → END
              ↘ (问答 / 无图) END
```

`execute` 只调用 `PhotoLookRenderer`（OpenCV），不直连任何生图厂商。

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
