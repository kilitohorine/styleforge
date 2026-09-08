"""DashScope / 万相 image API. Vendor URL lives only here."""

from __future__ import annotations

import httpx

from app.providers.errors import ProviderError
from app.settings import settings

generate_calls: list[dict] = []


def _first_url(payload: dict) -> str:
    output = payload.get("output") or payload
    for key in ("results", "choices", "images", "data"):
        items = output.get(key) or payload.get(key) or []
        if not items:
            continue
        item = items[0]
        if isinstance(item, str) and item.startswith("http"):
            return item
        if isinstance(item, dict):
            url = item.get("url") or item.get("image") or (item.get("message") or {}).get("content")
            if isinstance(url, str) and url.startswith("http"):
                return url
    raise ProviderError("PROVIDER_BAD_RESPONSE", "dashscope response missing image url")


def generate(
    prompt: str,
    *,
    negative: str = "",
    image_jpeg: bytes | None = None,
    timeout_s: float = 60.0,
) -> bytes:
    generate_calls.append({"prompt": prompt, "has_image": bool(image_jpeg)})
    key = (settings.dashscope_api_key or "").strip()
    if not key:
        raise ProviderError("PROVIDER_NOT_CONFIGURED", "DASHSCOPE_API_KEY is empty")
    url = settings.dashscope_base_url.rstrip("/") + settings.dashscope_image_path
    body: dict = {
        "model": settings.dashscope_image_model,
        "input": {"prompt": prompt},
        "parameters": {"size": "1024*1024", "n": 1},
    }
    if negative:
        body["input"]["negative_prompt"] = negative
    try:
        resp = httpx.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        raise ProviderError("PROVIDER_HTTP", str(exc), retryable=True) from exc
    if resp.status_code == 401:
        raise ProviderError("PROVIDER_AUTH", "dashscope unauthorized")
    if resp.status_code >= 400:
        raise ProviderError("PROVIDER_HTTP", f"dashscope HTTP {resp.status_code}: {resp.text[:300]}")
    image_url = _first_url(resp.json())
    try:
        img = httpx.get(image_url, timeout=timeout_s)
    except httpx.HTTPError as exc:
        raise ProviderError("PROVIDER_HTTP", f"download failed: {exc}", retryable=True) from exc
    if img.status_code >= 400 or not img.content:
        raise ProviderError("PROVIDER_HTTP", "empty generated image")
    return img.content
