from __future__ import annotations

from typing import Literal, TypedDict

from app.renderers.photo_look import LOOKS
from app.settings import settings

LOOK_IDS = tuple(LOOKS.keys())


class AgentState(TypedDict, total=False):
    message: str
    asset_id: str | None
    intent: Literal["look", "qa", "unsupported"]
    style_id: str | None
    reply: str
    job_id: str | None
    citations: list[str]


def keyword_route(message: str, has_image: bool) -> tuple[str, str | None, str]:
    text = (message or "").strip().lower()
    if any(k in text for k in ("3d", "视频", "video", "mesh")):
        return "unsupported", None, "3D / 长视频是预留能力（HTTP 501），一天闭环只做摄影 Look。"

    ranked: list[tuple[int, str]] = []
    for style_id, meta in LOOKS.items():
        hits = sum(1 for kw in meta["keywords"] if kw.lower() in text or kw in (message or ""))
        if hits:
            ranked.append((hits, style_id))
    if ranked:
        ranked.sort(reverse=True)
        style_id = ranked[0][1]
        name = LOOKS[style_id]["name"]
        return "look", style_id, f"将使用 {name}（{style_id}）做摄影调色，不调用生图 API。"

    if has_image and text in ("", "修好", "美化", "增强", "好看一点"):
        return "look", "film_portra", "未指定风格，默认胶片暖调。"

    if any(k in text for k in ("水彩", "水墨", "赛博", "吉卜力", "像素", "油画")):
        return (
            "qa",
            None,
            "艺术风格化（文生图/图生图）在二期。一天闭环支持：胶片暖调、电影青橙、港风夜景。请上传照片并说明 Look。",
        )

    if has_image:
        return "look", "film_portra", "已上传照片，默认套用胶片暖调。可改口令：电影青橙 / 港风夜景。"

    return (
        "qa",
        None,
        "这是 StyleForge 一天闭环：上传照片后说「胶片暖调 / 电影青橙 / 港风夜景」。编排用 LangGraph；调色用 OpenCV（0 元）。思路参考 PhotoAgent 的感知→规划→执行闭环。",
    )


def _llm_route(message: str, has_image: bool) -> tuple[str, str | None, str] | None:
    if not settings.deepseek_api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field

        class Route(BaseModel):
            intent: Literal["look", "qa", "unsupported"]
            style_id: str | None = Field(default=None)
            reply: str

        llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
        )
        structured = llm.with_structured_output(Route)
        prompt = (
            "你是修图 Agent 路由器。只允许 look 风格: film_portra, cinematic_teal_orange, hk_night。\n"
            "用户有图则优先 look。3D/视频为 unsupported。\n"
            f"有图={has_image}\n用户: {message}"
        )
        out = structured.invoke(prompt)
        style = out.style_id if out.style_id in LOOK_IDS else ("film_portra" if out.intent == "look" else None)
        return out.intent, style, out.reply
    except Exception:
        return None


def route_node(state: AgentState) -> AgentState:
    has_image = bool(state.get("asset_id"))
    message = state.get("message") or ""
    parsed = _llm_route(message, has_image) or keyword_route(message, has_image)
    intent, style_id, reply = parsed
    return {
        **state,
        "intent": intent,
        "style_id": style_id,
        "reply": reply,
        "citations": [f"style:{style_id}"] if style_id else ["pack:photo_looks"],
    }


def after_route(state: AgentState) -> str:
    if state.get("intent") == "look" and state.get("asset_id") and state.get("style_id"):
        return "execute"
    return "end"
