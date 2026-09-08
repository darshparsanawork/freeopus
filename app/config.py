"""Local persisted settings (single-user, self-hosted app).

Keys are written to a JSON file on the host under DATA_DIR, never sent
anywhere except the provider they belong to. This app is meant to be
self-hosted by one person/team, not a multi-tenant SaaS.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = DATA_DIR / "config.json"

_lock = threading.Lock()


class Settings(BaseModel):
    llm_provider: str = "openrouter"  # "openrouter" | "gemini"
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "google/gemini-2.5-flash"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    whisper_model_size: str = "small"
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    transcription_provider: str = "local"  # "local" (faster-whisper) | "openrouter" (openai/whisper-1)


def _defaults_from_env() -> Settings:
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or None,
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
        llm_provider=os.environ.get("LLM_PROVIDER", "openrouter"),
    )


def load_settings() -> Settings:
    with _lock:
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text())
                return Settings(**data)
            except Exception:
                pass
        return _defaults_from_env()


def save_settings(settings: Settings) -> None:
    with _lock:
        CONFIG_PATH.write_text(settings.model_dump_json(indent=2))


def public_settings(settings: Settings) -> dict:
    """Settings payload safe to return to the frontend (no raw keys)."""
    return {
        "llm_provider": settings.llm_provider,
        "openrouter_model": settings.openrouter_model,
        "gemini_model": settings.gemini_model,
        "whisper_model_size": settings.whisper_model_size,
        "device": settings.device,
        "transcription_provider": settings.transcription_provider,
        "has_openrouter_key": bool(settings.openrouter_api_key),
        "has_gemini_key": bool(settings.gemini_api_key),
    }
