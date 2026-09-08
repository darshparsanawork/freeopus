from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from .. import jobs as jobs_module
from ..models import CreateJobRequest, JobOut, ProcessSelectionRequest

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK_SIZE = 1024 * 1024  # 1MB read chunks - never buffers the whole file in RAM


def _stream_file_with_range(path: Path, request: Request, media_type: str) -> StreamingResponse:
    """Serves a local file with HTTP Range support, required for <video>
    seeking (jumping to a moment's timestamp) without downloading the
    whole file first. Reads in fixed-size chunks regardless of file size,
    so a multi-hour source video never gets loaded into memory at once."""
    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    start, end = 0, file_size - 1
    status_code = 200
    if range_header:
        match = _RANGE_RE.match(range_header)
        if match:
            status_code = 206
            if match.group(1):
                start = int(match.group(1))
            if match.group(2):
                end = int(match.group(2))
            else:
                end = file_size - 1

    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    length = end - start + 1

    def iterator():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return StreamingResponse(iterator(), status_code=status_code, media_type=media_type, headers=headers)


@router.get("", response_model=list[JobOut])
def list_jobs():
    return [job.to_out() for job in jobs_module.list_jobs()]


@router.post("", response_model=JobOut)
def create_job(payload: CreateJobRequest):
    if not payload.url.strip():
        raise HTTPException(400, "A video URL is required")
    job = jobs_module.create_job(payload.url.strip())
    return job.to_out()


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str):
    job = jobs_module.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_out()


@router.get("/{job_id}/source")
def get_source_video(job_id: str, request: Request):
    """Streams the downloaded source video with Range support, so the
    moment-selection screen can preview/seek to an exact timestamp without
    waiting for (or storing in memory) the entire file."""
    job = jobs_module.get_job(job_id)
    if not job or not job.source_path or not job.source_path.exists():
        raise HTTPException(404, "Source video not available")
    return _stream_file_with_range(job.source_path, request, "video/mp4")


@router.post("/{job_id}/process", response_model=JobOut)
def process_job(job_id: str, payload: ProcessSelectionRequest):
    job = jobs_module.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.stage != "awaiting_selection":
        raise HTTPException(400, f"Job is not ready for processing (stage={job.stage})")
    try:
        jobs_module.start_processing(
            job,
            payload.moment_ids,
            payload.crop_mode,
            payload.subtitles_enabled,
            payload.burn_in,
            payload.caption_formats,
            payload.caption_position,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return job.to_out()


@router.get("/{job_id}/clips/{clip_id}/download")
def download_clip(job_id: str, clip_id: str):
    job = jobs_module.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = job.dir / "clips" / f"{clip_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Clip not found or not finished yet")
    return FileResponse(path, media_type="video/mp4", filename=f"{clip_id}.mp4")


@router.get("/{job_id}/clips/{clip_id}/preview")
def preview_clip(job_id: str, clip_id: str, request: Request):
    """Range-enabled streaming for previewing a rendered clip inline
    (scrubbing) rather than only offering a full download."""
    job = jobs_module.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = job.dir / "clips" / f"{clip_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Clip not found or not finished yet")
    return _stream_file_with_range(path, request, "video/mp4")


@router.get("/{job_id}/clips/{clip_id}/captions/{fmt}")
def download_caption(job_id: str, clip_id: str, fmt: str):
    job = jobs_module.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = job.dir / "clips" / f"{clip_id}.{fmt}"
    if not path.exists():
        raise HTTPException(404, "Caption file not found")
    media_types = {"srt": "text/plain", "vtt": "text/vtt", "ass": "text/plain"}
    return FileResponse(path, media_type=media_types.get(fmt, "text/plain"), filename=f"{clip_id}.{fmt}")
