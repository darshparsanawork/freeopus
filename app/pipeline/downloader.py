"""Video ingestion via yt-dlp.

yt-dlp handles YouTube plus hundreds of other sites, so a single code path
covers "any video URL" as requested. We always re-mux/transcode to a plain
mp4 (h264/aac) container so every downstream step (ffmpeg cropping,
whisper) can rely on a predictable format regardless of the source site.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import yt_dlp

from .. import config

COOKIES_PATH = config.DATA_DIR / "cookies.txt"

# YouTube increasingly serves "Sign in to confirm you're not a bot" to
# requests from datacenter/cloud IPs (common when self-hosting on a VPS or
# CI-like sandbox). Different internal player clients are throttled
# independently, so retrying with each one is the standard yt-dlp
# workaround and resolves the vast majority of cases without cookies.
_PLAYER_CLIENT_FALLBACKS = ["default", "android", "ios", "tv", "web_safari"]


class DownloadError(RuntimeError):
    pass


def _base_opts(out_dir: Path, progress_cb) -> dict:
    def hook(d: dict) -> None:
        if progress_cb is None:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            pct = (done / total * 100.0) if total else 0.0
            progress_cb(pct, "downloading")
        elif d.get("status") == "finished":
            progress_cb(100.0, "converting")

    opts = {
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "format": "bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "progress_hooks": [hook],
        "restrictfilenames": True,
        "retries": 5,
        "fragment_retries": 5,
        "postprocessors": [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ],
    }
    if COOKIES_PATH.exists():
        opts["cookiefile"] = str(COOKIES_PATH)
    return opts


def _is_bot_check_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "sign in to confirm" in msg or "not a bot" in msg


def download_video(url: str, out_dir: Path, progress_cb: Optional[Callable[[float, str], None]] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "source.mp4"

    last_exc: Optional[Exception] = None
    for client in _PLAYER_CLIENT_FALLBACKS:
        ydl_opts = _base_opts(out_dir, progress_cb)
        if client != "default":
            ydl_opts["extractor_args"] = {"youtube": {"player_client": [client]}}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
            last_exc = None
            break
        except yt_dlp.utils.DownloadError as exc:
            last_exc = exc
            for f in out_dir.glob("source.*"):
                f.unlink(missing_ok=True)
            if not _is_bot_check_error(exc):
                # Not a bot-check failure (e.g. bad URL, private video) -
                # retrying with a different client won't help.
                break
            continue

    if last_exc is not None:
        hint = ""
        if _is_bot_check_error(last_exc):
            hint = (
                " YouTube is asking for sign-in verification from this server's IP. "
                "This is common on cloud/VPS hosting. Fix: export cookies from a logged-in "
                "browser session (see README) and place them at "
                f"{COOKIES_PATH} to let downloads authenticate as you."
            )
        raise DownloadError(f"Could not download video: {last_exc}.{hint}") from last_exc

    # yt-dlp may have produced source.mp4 directly, or another extension
    # before the postprocessor converted it. Find whatever landed.
    candidates = sorted(out_dir.glob("source.*"))
    mp4_candidates = [c for c in candidates if c.suffix == ".mp4"]
    if mp4_candidates:
        final = mp4_candidates[0]
    elif candidates:
        final = candidates[0]
    else:
        raise DownloadError("Download completed but no output file was found")

    if final != target:
        final.rename(target)

    return target


def probe_title(url: str) -> str:
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get("title", url)
