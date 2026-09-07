"""In-process job state machine.

Single-instance, in-memory job registry (fine for a self-hosted app run by
one person/team). Each job runs its pipeline stages in a background
thread so the API stays responsive for status polling.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .models import ClipOut, JobOut, MomentOut
from .pipeline import captions, cropper, downloader, ffmpeg_utils, llm, scenes, transcriber

_jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()


@dataclass
class ClipState:
    id: str
    moment_id: str
    title: str
    start: float
    end: float
    status: str = "pending"  # pending | rendering | captioning | done | error
    error: Optional[str] = None
    mode_used: Optional[str] = None
    caption_formats: list[str] = field(default_factory=list)


@dataclass
class Job:
    id: str
    url: str
    stage: str = "queued"
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    dir: Path = field(default_factory=Path)
    source_path: Optional[Path] = None
    transcript: Optional[transcriber.Transcript] = None
    scene_list: list[tuple[float, float]] = field(default_factory=list)
    moments: list[llm.Moment] = field(default_factory=list)
    clips: dict[str, ClipState] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set_stage(self, stage: str, progress: float = 0.0, message: str = "") -> None:
        with self.lock:
            self.stage = stage
            self.progress = progress
            self.message = message

    def to_out(self) -> JobOut:
        with self.lock:
            return JobOut(
                id=self.id,
                url=self.url,
                stage=self.stage,
                progress=self.progress,
                message=self.message,
                error=self.error,
                moments=[MomentOut(id=m.id, start=m.start, end=m.end, title=m.title, reason=m.reason, score=m.score) for m in self.moments],
                clips=[
                    ClipOut(
                        id=c.id,
                        moment_id=c.moment_id,
                        status=c.status,
                        error=c.error,
                        mode_used=c.mode_used,
                        title=c.title,
                        start=c.start,
                        end=c.end,
                        caption_formats=c.caption_formats,
                    )
                    for c in self.clips.values()
                ],
            )


def get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


def create_job(url: str) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job_dir = config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = Job(id=job_id, url=url, dir=job_dir)
    with _jobs_lock:
        _jobs[job_id] = job
    threading.Thread(target=_run_analysis, args=(job,), daemon=True).start()
    return job


def _run_analysis(job: Job) -> None:
    settings = config.load_settings()
    try:
        job.set_stage("downloading", 0, "Downloading video...")

        def dl_progress(pct: float, phase: str) -> None:
            job.set_stage("downloading", pct, f"Downloading... ({phase})")

        source_path = downloader.download_video(job.url, job.dir, dl_progress)
        job.source_path = source_path

        job.set_stage("transcribing", 0, "Transcribing audio...")

        def tr_progress(pct: float) -> None:
            job.set_stage("transcribing", pct, "Transcribing audio...")

        transcript = transcriber.transcribe(
            source_path, model_size=settings.whisper_model_size, device=settings.device, progress_cb=tr_progress
        )
        job.transcript = transcript

        job.set_stage("detecting_scenes", 50, "Detecting scene cuts...")
        job.scene_list = scenes.detect_scenes(source_path)

        job.set_stage("analyzing", 75, "Asking the AI to pick the best moments...")
        if settings.llm_provider == "gemini":
            moments = llm.pick_moments_gemini(settings.gemini_api_key, settings.gemini_model, transcript.segments, job.scene_list)
        else:
            moments = llm.pick_moments_openrouter(settings.openrouter_api_key, settings.openrouter_model, transcript.segments, job.scene_list)

        if not moments:
            raise RuntimeError("The AI did not return any candidate moments. Try a different model or a longer video.")

        job.moments = moments
        job.set_stage("awaiting_selection", 100, f"Found {len(moments)} candidate moments. Pick which ones to render.")
    except Exception as exc:  # noqa: BLE001
        job.error = str(exc)
        job.set_stage("error", 0, str(exc))
        traceback.print_exc()


def start_processing(job: Job, moment_ids: list[str], crop_mode: str, subtitles_enabled: bool, burn_in: bool, caption_formats: list[str]) -> None:
    selected = [m for m in job.moments if m.id in moment_ids]
    if not selected:
        raise ValueError("No matching moments selected")

    for m in selected:
        job.clips[m.id] = ClipState(id=f"clip_{m.id}", moment_id=m.id, title=m.title, start=m.start, end=m.end)

    job.set_stage("processing", 0, "Rendering clips...")
    threading.Thread(
        target=_run_processing, args=(job, selected, crop_mode, subtitles_enabled, burn_in, caption_formats), daemon=True
    ).start()


def _run_processing(job: Job, selected: list[llm.Moment], crop_mode: str, subtitles_enabled: bool, burn_in: bool, caption_formats: list[str]) -> None:
    use_gpu = ffmpeg_utils.gpu_encoder_available()
    clips_dir = job.dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    total = len(selected)

    for i, moment in enumerate(selected):
        state = job.clips[moment.id]
        try:
            state.status = "rendering"
            job.set_stage("processing", i / total * 100, f"Rendering clip {i+1}/{total}: {moment.title}")
            raw_out = clips_dir / f"{state.id}_raw.mp4"
            mode_used = cropper.render_clip(job.source_path, raw_out, moment.start, moment.end, mode=crop_mode, use_gpu=use_gpu)
            state.mode_used = mode_used

            final_out = clips_dir / f"{state.id}.mp4"
            if subtitles_enabled and job.transcript:
                state.status = "captioning"
                cap_paths = captions.write_captions(job.transcript.segments, moment.start, moment.end, clips_dir, state.id, caption_formats)
                state.caption_formats = list(cap_paths.keys())
                if burn_in and "ass" in cap_paths:
                    ffmpeg_utils.burn_subtitles(raw_out, cap_paths["ass"], final_out, use_gpu)
                    raw_out.unlink(missing_ok=True)
                else:
                    raw_out.rename(final_out)
            else:
                raw_out.rename(final_out)

            state.status = "done"
        except Exception as exc:  # noqa: BLE001
            state.status = "error"
            state.error = str(exc)
            traceback.print_exc()

    job.set_stage("done", 100, "All clips are ready.")
