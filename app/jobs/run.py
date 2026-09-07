"""Create Look / image.2d jobs. Budget fuse runs before any paid HTTP."""

from __future__ import annotations

from app.cost.table import estimate_cny
from app.jobs.budget import BudgetExceeded, assert_can_spend, record_spend
from app.jobs.store import asset_path, dump_output, new_id, save_job
from app.providers.siliconflow import ProviderError
from app.renderers.image_2d import Image2DRenderer
from app.renderers.photo_look import LOOKS, PhotoLookRenderer, encode_jpeg
from app.schemas.job import ErrorBody, InputAsset, JobOut, JobTrace
from app.settings import settings


def failed_job(
    *,
    modality: str,
    style_id: str | None,
    code: str,
    message: str,
    estimated: float = 0.0,
    retryable: bool = False,
    source_asset_id: str | None = None,
) -> JobOut:
    job = JobOut(
        job_id=new_id("j"),
        status="failed",
        modality=modality,  # type: ignore[arg-type]
        estimated_cost_cny=estimated,
        actual_cost_cny=0.0,
        trace=JobTrace(style_id=style_id, source_asset_id=source_asset_id),
        error=ErrorBody(code=code, message=message, retryable=retryable),
    )
    save_job(job)
    return job


def run_photo_look(style_id: str, source_id: str, overrides: dict | None = None) -> JobOut:
    source = asset_path(source_id)
    renderer = PhotoLookRenderer()
    bgr, compare, params = renderer.run(source, style_id, overrides)
    out_id = dump_output(encode_jpeg(bgr), style_id, kind=f"look:{style_id}")
    cmp_id = dump_output(encode_jpeg(compare), style_id, kind="compare")
    job = JobOut(
        job_id=new_id("j"),
        status="succeeded",
        modality="image.photo_look",
        estimated_cost_cny=0.0,
        actual_cost_cny=0.0,
        outputs=[
            InputAsset(asset_id=out_id, role="result"),
            InputAsset(asset_id=cmp_id, role="compare"),
        ],
        trace=JobTrace(
            style_id=style_id,
            renderer="photo_look.cube_lut",
            params=params,
            actions=["perceive_intent", "plan_greedy_look", "execute_cube_lut"],
            source_asset_id=source_id,
            comparison_asset_id=cmp_id,
            lut=LOOKS[style_id]["lut"],
        ),
    )
    save_job(job)
    return job


def run_image_2d(
    style_id: str,
    prompt: str,
    source_id: str | None = None,
    budget_cny_max: float = 1.0,
) -> JobOut:
    backend = settings.image_2d_backend
    estimated = estimate_cny("image.2d", backend)
    try:
        assert_can_spend(estimated, budget_cny_max)
    except BudgetExceeded as exc:
        return failed_job(
            modality="image.2d",
            style_id=style_id,
            code=exc.code,
            message=str(exc),
            estimated=estimated,
            source_asset_id=source_id,
        )
    if backend != "mock" and not (settings.siliconflow_api_key or "").strip():
        return failed_job(
            modality="image.2d",
            style_id=style_id,
            code="PROVIDER_NOT_CONFIGURED",
            message="未配置 SILICONFLOW_API_KEY。摄影 Look 仍可用；2D 生图已熔断。",
            estimated=estimated,
            source_asset_id=source_id,
        )
    source = None
    if source_id:
        try:
            source = asset_path(source_id)
        except KeyError:
            return failed_job(
                modality="image.2d",
                style_id=style_id,
                code="ASSET_NOT_FOUND",
                message="source asset not found",
                estimated=estimated,
                source_asset_id=source_id,
            )
    renderer = Image2DRenderer()
    try:
        jpeg, params = renderer.run(style_id, prompt, source)
    except ProviderError as exc:
        return failed_job(
            modality="image.2d",
            style_id=style_id,
            code=exc.code,
            message=str(exc),
            estimated=estimated,
            retryable=exc.retryable,
            source_asset_id=source_id,
        )
    out_id = dump_output(jpeg, style_id, kind=f"2d:{style_id}")
    job = JobOut(
        job_id=new_id("j"),
        status="succeeded",
        modality="image.2d",
        estimated_cost_cny=estimated,
        actual_cost_cny=estimated,
        outputs=[InputAsset(asset_id=out_id, role="result")],
        trace=JobTrace(
            style_id=style_id,
            renderer=f"image_2d.{backend}",
            params=params,
            actions=["perceive_intent", "plan_greedy_2d", "execute_image_2d"],
            source_asset_id=source_id,
        ),
    )
    save_job(job)
    record_spend(job.job_id, estimated, "image.2d")
    return job
