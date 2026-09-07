from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import jobs as jobs_module
from ..models import CreateJobRequest, JobOut, ProcessSelectionRequest

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


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


@router.post("/{job_id}/process", response_model=JobOut)
def process_job(job_id: str, payload: ProcessSelectionRequest):
    job = jobs_module.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.stage != "awaiting_selection":
        raise HTTPException(400, f"Job is not ready for processing (stage={job.stage})")
    try:
        jobs_module.start_processing(
            job, payload.moment_ids, payload.crop_mode, payload.subtitles_enabled, payload.burn_in, payload.caption_formats
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
