"""SiliconFlow image API. This package is the only place vendor URLs may appear."""

from __future__ import annotations

import base64

import httpx

from app.settings import settings

# Tests patch this to assert the fuse blocked outbound calls.
generate_calls: list[dict] = []


class ProviderError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _image_field(raw: bytes) -> str:
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _first_url(payload: dict) -> str:
    for key in ("images", "data"):
        items = payload.get(key) or []
        if not items:
            continue
        item = items[0]
        if isinstance(item, str) and item.startswith("http"):
            return item
        if isinstance(item, dict):
            url = item.get("url") or item.get("image")
            if url:
                return str(url)
    raise ProviderError("PROVIDER_BAD_RESPONSE", "siliconflow response missing image url")


def generate(
    prompt: str,
    *,
    negative: str = "",
    image_jpeg: bytes | None = None,
    timeout_s: float = 60.0,
) -> bytes:
    generate_calls.append({"prompt": prompt, "has_image": bool(image_jpeg)})
    key = (settings.siliconflow_api_key or "").strip()
    if not key:
        raise ProviderError("PROVIDER_NOT_CONFIGURED", "SILICONFLOW_API_KEY is empty")
    url = settings.siliconflow_base_url.rstrip("/") + "/images/generations"
    body: dict = {
        "model": settings.siliconflow_image_model,
        "prompt": prompt,
        "image_size": "1024x1024",
        "batch_size": 1,
    }
    if negative:
        body["negative_prompt"] = negative
    if image_jpeg:
        body["image"] = _image_field(image_jpeg)
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
        raise ProviderError("PROVIDER_AUTH", "siliconflow unauthorized")
    if resp.status_code >= 400:
        raise ProviderError("PROVIDER_HTTP", f"siliconflow HTTP {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    image_url = _first_url(payload)
    try:
        img = httpx.get(image_url, timeout=timeout_s)
    except httpx.HTTPError as exc:
        raise ProviderError("PROVIDER_HTTP", f"download failed: {exc}", retryable=True) from exc
    if img.status_code >= 400 or not img.content:
        raise ProviderError("PROVIDER_HTTP", "empty generated image")
    return img.content
