from __future__ import annotations

from typing import Literal, TypedDict

from app.rag.retriever import compose_qa, query as rag_query
from app.renderers.image_2d import ART_STYLES, match_art_style, wants_generate
from app.renderers.photo_look import LOOKS, merge_params
from app.settings import settings

LOOK_IDS = tuple(LOOKS.keys())

PATCH_HINTS = (
    "再暗",
    "暗一点",
    "暗一些",
    "再亮",
    "亮一点",
    "少颗粒",
    "多颗粒",
    "再暖",
    "暖一点",
    "再冷",
    "对比强",
    "LUT淡",
    "lut淡",
    "强度低",
)


class AgentState(TypedDict, total=False):
    message: str
    asset_id: str | None
    thread_id: str | None
    intent: Literal["look", "qa", "unsupported", "image2d"]
    style_id: str | None
    reply: str
    job_id: str | None
    citations: list[str]
    params: dict
    is_patch: bool
    user_named_style: bool
    scene: str
    step: int
    need_retry: bool
    perceive: dict
    critique: dict


def parse_patch(message: str) -> dict | None:
    text = (message or "").strip()
    if not text:
        return None
    delta: dict[str, float] = {}
    if any(k in text for k in ("再暗", "暗一点", "暗一些", "低调一点")):
        delta["exposure"] = -0.08
    if any(k in text for k in ("再亮", "亮一点", "提亮")):
        delta["exposure"] = 0.08
    if "少颗粒" in text:
        delta["grain"] = -0.015
    if "多颗粒" in text:
        delta["grain"] = 0.015
    if any(k in text for k in ("再暖", "暖一点")):
        delta["warm"] = 0.06
    if any(k in text for k in ("再冷", "冷一点")):
        delta["warm"] = -0.06
    if any(k in text for k in ("LUT淡", "lut淡", "强度低", "淡一点")):
        delta["lut_strength"] = -0.15
    if not delta:
        if any(k in text for k in PATCH_HINTS):
            return {}
        return None
    return delta


def apply_param_delta(base: dict, delta: dict) -> dict:
    out = dict(base)
    if "exposure" in delta:
        out["exposure"] = float(out.get("exposure", 0.0)) + delta["exposure"]
    if "warm" in delta:
        out["warm"] = float(out.get("warm", 0.0)) + delta["warm"]
    if "grain" in delta:
        out["grain"] = max(0.0, float(out.get("grain", 0.0)) + delta["grain"])
    if "lut_strength" in delta:
        out["lut_strength"] = float(np_clip(out.get("lut_strength", 1.0) + delta["lut_strength"], 0.2, 1.0))
    return out


def np_clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def keyword_route(message: str, has_image: bool) -> tuple[str, str | None, str]:
    text = (message or "").strip().lower()
    if any(k in text for k in ("3d", "视频", "video", "mesh")):
        return "unsupported", None, "3D / 长视频是预留能力（HTTP 501），当前闭环只做摄影 Look / LUT。"

    ranked: list[tuple[int, str]] = []
    for style_id, meta in LOOKS.items():
        hits = sum(1 for kw in meta["keywords"] if kw.lower() in text or kw in (message or ""))
        if hits:
            ranked.append((hits, style_id))
    if ranked:
        ranked.sort(reverse=True)
        style_id = ranked[0][1]
        name = LOOKS[style_id]["name"]
        return "look", style_id, f"将使用 {name}（{style_id}，.cube LUT）做摄影调色，不调用生图 API。"

    if has_image and text in ("", "修好", "美化", "增强", "好看一点"):
        return "look", "film_portra", "未指定风格，默认胶片暖调（Portra LUT）。"

    art_id = match_art_style(message or "")
    if art_id and wants_generate(message or ""):
        name = ART_STYLES[art_id]["name"]
        return (
            "image2d",
            art_id,
            f"将使用 image.2d 风格 {name}（{art_id}）。未配 Key 或超预算则不会出网。",
        )
    if art_id or any(k in text for k in ("水彩", "水墨", "赛博", "吉卜力", "像素", "油画")):
        return (
            "qa",
            None,
            "艺术风格化走 RAG 问答；要出图请明确说「改成水墨画 / 生成水彩」并配置 SILICONFLOW_API_KEY。",
        )

    if has_image:
        return "look", "film_portra", "已上传照片，默认套用胶片暖调。可改口令：青橙 / 港风 / 黑白 / 复古 / 黄金时刻 / 冷调 / 哑光。"

    return (
        "qa",
        None,
        "StyleForge：上传照片后说「胶片暖调 / 电影青橙 / 港风夜景 / 黑白 / 复古 / 黄金时刻 / 冷调 / 哑光」。"
        "调色走 .cube LUT，多轮可说「再暗一点」。",
    )


def _llm_route(message: str, has_image: bool) -> tuple[str, str | None, str] | None:
    if not settings.deepseek_api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field

        class Route(BaseModel):
            intent: Literal["look", "qa", "unsupported", "image2d"]
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
            "你是修图 Agent 路由器。look 风格: film_portra, cinematic_teal_orange, hk_night, "
            "mono_bw, vintage_fade, golden_hour, cool_steel, matte_film。\n"
            "image2d 风格: watercolor, pastoral_anime, cyberpunk, ink_wash, oil_paint, pixel, flat_illustration。\n"
            "用户说「改成水墨/生成水彩」才用 image2d；问区别用 qa。有图且说胶片/调色/黑白/复古用 look。3D/视频为 unsupported。\n"
            f"有图={has_image}\n用户: {message}"
        )
        out = structured.invoke(prompt)
        if out.style_id in LOOK_IDS:
            style = out.style_id
        elif out.intent == "look":
            style = "film_portra"
        elif out.intent == "image2d" and out.style_id in ART_STYLES:
            style = out.style_id
        elif out.intent == "image2d":
            style = match_art_style(message)
        else:
            style = None
        return out.intent, style, out.reply
    except Exception:
        return None


def route_node(state: AgentState) -> AgentState:
    message = state.get("message") or ""
    has_image = bool(state.get("asset_id"))
    prev_style = state.get("style_id")
    prev_params = dict(state.get("params") or {})
    delta = parse_patch(message)
    style_hit = any(
        kw.lower() in message.lower() or kw in message
        for meta in LOOKS.values()
        for kw in meta["keywords"]
    )

    if delta is not None and prev_style and prev_style in LOOKS and not style_hit:
        params = apply_param_delta(merge_params(prev_style, prev_params), delta)
        name = LOOKS[prev_style]["name"]
        return {
            **state,
            "intent": "look" if has_image else "qa",
            "style_id": prev_style,
            "params": params,
            "is_patch": True,
            "user_named_style": True,
            "reply": f"保持 {name}（{prev_style}），只改参数 {delta}。",
            "citations": [f"style:{prev_style}", "lut:cube"],
        }

    parsed = None if delta is not None else _llm_route(message, has_image)
    parsed = parsed or keyword_route(message, has_image)
    intent, style_id, reply = parsed
    params = merge_params(style_id) if style_id and style_id in LOOKS else {}
    citations = [f"style:{style_id}", "lut:cube"] if style_id else ["pack:photo_looks"]
    if intent == "qa":
        hits = rag_query(message, k=3)
        citations = [f"pack:{h['style_id']}" for h in hits if h.get("style_id")]
        reply = compose_qa(message, hits)
    if intent == "image2d" and style_id:
        citations = [f"pack:{style_id}", "provider:siliconflow"]
    return {
        **state,
        "intent": intent,
        "style_id": style_id,
        "params": params,
        "is_patch": False,
        "user_named_style": bool(style_hit and intent == "look"),
        "reply": reply,
        "citations": citations,
    }


def after_route(state: AgentState) -> str:
    if state.get("intent") == "look" and state.get("asset_id") and state.get("style_id"):
        return "perceive"
    if state.get("intent") == "image2d" and state.get("style_id"):
        return "execute"
    return "end"
