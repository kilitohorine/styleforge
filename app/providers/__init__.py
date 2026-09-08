from app.providers.errors import ProviderError
from app.settings import settings

DASH_BACKENDS = {"dashscope", "dashscope_wanxiang", "wanxiang"}


def generate(prompt: str, **kwargs) -> bytes:
    """Dispatch 2D generation. Callers must not import vendor URLs."""
    backend = (settings.image_2d_backend or "").strip().lower()
    if backend in DASH_BACKENDS:
        from app.providers import dashscope

        return dashscope.generate(prompt, **kwargs)
    from app.providers import siliconflow

    return siliconflow.generate(prompt, **kwargs)


__all__ = ["ProviderError", "generate", "DASH_BACKENDS"]
