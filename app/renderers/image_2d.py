"""image.2d renderer. Planner talks to this, not to SiliconFlow URLs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.providers import generate
from app.providers.errors import ProviderError
from app.rag import load_packs
from app.renderers.photo_look import encode_jpeg
from app.settings import settings

ART_STYLES: dict[str, dict] = {
    "watercolor": {"name": "水彩", "keywords": ("水彩", "watercolor")},
    "pastoral_anime": {"name": "田园动画风", "keywords": ("田园", "乡村动画")},
    "cyberpunk": {"name": "赛博朋克", "keywords": ("赛博", "cyberpunk", "赛博朋克")},
    "ink_wash": {"name": "水墨", "keywords": ("水墨", "墨韵", "ink")},
    "oil_paint": {"name": "油画", "keywords": ("油画", "厚涂")},
    "pixel": {"name": "像素", "keywords": ("像素", "pixel", "8bit", "16bit")},
    "flat_illustration": {"name": "扁平插画", "keywords": ("扁平", "插画色块")},
}

GEN_HINTS = ("改成", "生成", "画成", "图生图", "文生图", "做成水彩", "做成水墨", "做成油画", "做成像素")
ASK_HINTS = ("差在哪", "区别", "什么是", "怎么选", "介绍", "对比")


def match_art_style(message: str) -> str | None:
    text = message or ""
    ranked: list[tuple[int, str]] = []
    for style_id, meta in ART_STYLES.items():
        hits = sum(1 for kw in meta["keywords"] if kw.lower() in text.lower() or kw in text)
        if hits:
            ranked.append((hits, style_id))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][1]


def wants_generate(message: str) -> bool:
    text = message or ""
    if any(k in text for k in ASK_HINTS):
        return False
    return any(k in text for k in GEN_HINTS)


def _pack(style_id: str) -> dict:
    for pack in load_packs():
        if pack.get("style_id") == style_id:
            return pack
    return {}


def build_prompt(style_id: str, user_message: str) -> tuple[str, str]:
    pack = _pack(style_id)
    positive = (pack.get("positive") or ART_STYLES.get(style_id, {}).get("name") or style_id).strip()
    negative = (pack.get("negative") or "").strip()
    prompt = f"{positive}\nuser: {user_message}".strip()
    return prompt, negative


def _mock_jpeg(style_id: str) -> bytes:
    rng = abs(hash(style_id)) % 180 + 40
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[:, :] = (rng // 2, rng, 220 - rng)
    cv2.putText(img, style_id[:8], (4, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return encode_jpeg(img)


class Image2DRenderer:
    modality = "image.2d"

    def estimate_cost(self) -> float:
        from app.cost.table import estimate_cny

        return estimate_cny("image.2d", settings.image_2d_backend)

    def run(
        self,
        style_id: str,
        prompt: str,
        source: Path | None = None,
    ) -> tuple[bytes, dict]:
        if style_id not in ART_STYLES:
            raise ProviderError("UNKNOWN_STYLE", f"not an image.2d style: {style_id}")
        full_prompt, negative = build_prompt(style_id, prompt)
        params = {
            "backend": settings.image_2d_backend,
            "model": (
                "mock"
                if settings.image_2d_backend == "mock"
                else (
                    settings.dashscope_image_model
                    if settings.image_2d_backend in {"dashscope", "dashscope_wanxiang", "wanxiang"}
                    else settings.siliconflow_image_model
                )
            ),
            "prompt": full_prompt[:500],
        }
        if settings.image_2d_backend == "mock":
            return _mock_jpeg(style_id), params
        image_jpeg = None
        if source is not None:
            image_jpeg = Path(source).read_bytes()
        jpeg = generate(full_prompt, negative=negative, image_jpeg=image_jpeg)
        return jpeg, params
