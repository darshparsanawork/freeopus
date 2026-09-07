from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import config
from ..models import SettingsIn
from ..pipeline import llm

router = APIRouter(prefix="/api", tags=["settings"])


class CookiesIn(BaseModel):
    cookies_txt: str


@router.get("/settings")
def get_settings():
    return config.public_settings(config.load_settings())


@router.post("/settings")
def update_settings(payload: SettingsIn):
    settings = config.load_settings()
    data = payload.model_dump(exclude_unset=True)

    # Empty-string keys mean "leave unchanged" (frontend never re-sends stored secrets).
    for field in ("openrouter_api_key", "gemini_api_key"):
        if field in data and not data[field]:
            data.pop(field)

    updated = settings.model_copy(update=data)
    config.save_settings(updated)
    return config.public_settings(updated)


@router.post("/youtube-cookies")
def set_youtube_cookies(payload: CookiesIn):
    """Saves a Netscape-format cookies.txt export so yt-dlp can authenticate
    as a logged-in user, which resolves YouTube's bot-check on most cloud IPs."""
    text = payload.cookies_txt.strip()
    if not text:
        raise HTTPException(400, "Paste your exported cookies.txt content")
    from ..pipeline.downloader import COOKIES_PATH

    COOKIES_PATH.write_text(text + "\n", encoding="utf-8")
    return {"saved": True}


@router.get("/youtube-cookies")
def has_youtube_cookies():
    from ..pipeline.downloader import COOKIES_PATH

    return {"has_cookies": COOKIES_PATH.exists()}


@router.get("/openrouter/models")
def openrouter_models():
    settings = config.load_settings()
    if not settings.openrouter_api_key:
        raise HTTPException(400, "Set an OpenRouter API key first.")
    try:
        return llm.list_openrouter_models(settings.openrouter_api_key)
    except llm.LLMError as exc:
        raise HTTPException(400, str(exc)) from exc
