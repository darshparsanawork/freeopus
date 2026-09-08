from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

CropMode = Literal["auto", "track", "split", "general"]


class CreateJobRequest(BaseModel):
    url: str


class MomentOut(BaseModel):
    id: str
    start: float
    end: float
    title: str
    reason: str
    score: float


class ClipOut(BaseModel):
    id: str
    moment_id: str
    status: str
    error: Optional[str] = None
    mode_used: Optional[str] = None
    title: str
    start: float
    end: float
    caption_formats: list[str] = []


class JobOut(BaseModel):
    id: str
    url: str
    stage: str
    progress: float
    message: str = ""
    error: Optional[str] = None
    expires_in_seconds: Optional[float] = None
    moments: list[MomentOut] = []
    clips: list[ClipOut] = []


class ProcessSelectionRequest(BaseModel):
    moment_ids: list[str]
    crop_mode: CropMode = "auto"
    subtitles_enabled: bool = True
    burn_in: bool = True
    caption_formats: list[str] = ["srt", "vtt"]


class SettingsIn(BaseModel):
    llm_provider: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    whisper_model_size: Optional[str] = None
    device: Optional[str] = None
    transcription_provider: Optional[str] = None
