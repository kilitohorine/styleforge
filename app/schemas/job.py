from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Modality = Literal["image.2d", "image.photo_look", "asset.3d", "video.clip", "video.long"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
LOOK_STYLE_IDS = (
    "film_portra",
    "cinematic_teal_orange",
    "hk_night",
)


class InputAsset(BaseModel):
    asset_id: str
    role: Literal["source", "reference", "mask", "result", "compare"] = "source"


class JobCreate(BaseModel):
    modality: Modality = "image.photo_look"
    style_id: str = "film_portra"
    prompt: str = ""
    input_assets: list[InputAsset] = Field(default_factory=list)
    budget_cny_max: float = 1.0


class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool = False


class JobTrace(BaseModel):
    style_id: str | None = None
    renderer: str | None = None
    params: dict = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)
    source_asset_id: str | None = None
    comparison_asset_id: str | None = None
    lut: str | None = None


class JobOut(BaseModel):
    job_id: str
    status: JobStatus
    modality: Modality
    estimated_cost_cny: float = 0.0
    actual_cost_cny: float = 0.0
    outputs: list[InputAsset] = Field(default_factory=list)
    trace: JobTrace = Field(default_factory=JobTrace)
    error: Optional[ErrorBody] = None


class ChatIn(BaseModel):
    thread_id: str | None = None
    message: str
    asset_ids: list[str] = Field(default_factory=list)


class ChatOut(BaseModel):
    thread_id: str
    reply: str
    job_id: str | None = None
    style_id: str | None = None
    citations: list[str] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)


class RagQueryIn(BaseModel):
    query: str
    k: int = 3


class RagHit(BaseModel):
    id: str = ""
    style_id: str | None = None
    name: str | None = None
    kind: str | None = None
    distance: float | None = None
    snippet: str = ""


class RagQueryOut(BaseModel):
    hits: list[RagHit] = Field(default_factory=list)
