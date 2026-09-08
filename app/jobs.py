"""In-process job state machine.

Single-instance, in-memory job registry (fine for a self-hosted app run by
one person/team). Each job runs its pipeline stages on a bounded worker
pool (not one thread per job) so a burst of requests can't pile up
unbounded CPU/RAM use — extra jobs simply queue instead of all running
at once. A background sweeper deletes job files (and drops the in-memory
Job object) once they're older than JOB_RETENTION_HOURS, so disk and RAM
usage from finished jobs don't grow forever.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .models import ClipOut, JobOut, MomentOut
from .pipeline import captions, cropper, downloader, ffmpeg_utils, llm, scenes, transcriber

_jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()

# Bounded worker pool: caps how many jobs actually download/transcribe/
# render at once. Extra jobs queue rather than competing for the same
# CPU/RAM all at once. Override with MAX_CONCURRENT_JOBS if you have more
# cores/RAM to spare.
_MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("MAX_CONCURRENT_JOBS", "1")))
_executor = ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_JOBS, thread_name_prefix="job-worker")

# How long a finished job's files (and in-memory state) stick around
# before automatic cleanup. Download your clips before this.
JOB_RETENTION_HOURS = float(os.environ.get("JOB_RETENTION_HOURS", "2"))
_SWEEP_INTERVAL_SECONDS = 600  # check every 10 minutes


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
    created_at: float = field(default_factory=time.time)

    def set_stage(self, stage: str, progress: float = 0.0, message: str = "") -> None:
        with self.lock:
            self.stage = stage
            self.progress = progress
            self.message = message

    def to_out(self) -> JobOut:
        with self.lock:
            expires_in = max(0.0, JOB_RETENTION_HOURS * 3600 - (time.time() - self.created_at))
            return JobOut(
                id=self.id,
                url=self.url,
                stage=self.stage,
                progress=self.progress,
                message=self.message,
                error=self.error,
                expires_in_seconds=expires_in,
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

    def touch_freed(self) -> None:
        """Drops large in-memory pipeline data once it's no longer needed
        (kept only long enough to render selected clips), independent of
        full job cleanup - cuts RAM use for jobs that finish but haven't
        hit the retention window yet."""
        with self.lock:
            self.transcript = None
            self.scene_list = []


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
    job.set_stage("queued", 0, "Queued — waiting for a free worker slot...")
    _executor.submit(_run_analysis, job)
    return job


def _run_analysis(job: Job) -> None:
    settings = config.load_settings()
    try:
        job.set_stage("downloading", 0, "Downloading video...")

        def dl_progress(pct: float, phase: str) -> None:
            job.set_stage("downloading", pct, f"Downloading... ({phase})")

        source_path = downloader.download_video(job.url, job.dir, dl_progress)
        job.source_path = source_path

        def tr_progress(pct: float) -> None:
            job.set_stage("transcribing", pct, "Transcribing audio...")

        if settings.transcription_provider == "openrouter":
            job.set_stage("transcribing", 0, "Transcribing audio via OpenRouter (openai/whisper-1)...")
            transcript = transcriber.transcribe_via_openrouter(
                source_path, settings.openrouter_api_key, progress_cb=tr_progress
            )
        else:
            job.set_stage("transcribing", 0, "Transcribing audio locally...")
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
    _executor.submit(_run_processing, job, selected, crop_mode, subtitles_enabled, burn_in, caption_formats)


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
    # Transcript/scene data already went into the caption files on disk -
    # no need to keep it resident in RAM for a job that's done rendering.
    job.touch_freed()


def _delete_job_files(job_dir: Path) -> None:
    shutil.rmtree(job_dir, ignore_errors=True)


def _sweep_expired_jobs() -> None:
    """Deletes job files (and drops in-memory Job objects) once they're
    older than JOB_RETENTION_HOURS. Also sweeps orphaned job directories
    left on disk from before a restart, using directory mtime, so cleanup
    stays effective even if the in-memory registry was reset."""
    cutoff_age = JOB_RETENTION_HOURS * 3600
    now = time.time()

    with _jobs_lock:
        expired_ids = [jid for jid, job in _jobs.items() if now - job.created_at > cutoff_age]
        expired_dirs = [_jobs[jid].dir for jid in expired_ids]
        for jid in expired_ids:
            del _jobs[jid]

    for job_dir in expired_dirs:
        _delete_job_files(job_dir)

    known_dirs = {job.dir for job in _jobs.values()}
    try:
        for entry in config.JOBS_DIR.iterdir():
            if not entry.is_dir() or entry in known_dirs:
                continue
            if now - entry.stat().st_mtime > cutoff_age:
                _delete_job_files(entry)
    except FileNotFoundError:
        pass


def _sweeper_loop() -> None:
    while True:
        try:
            _sweep_expired_jobs()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        time.sleep(_SWEEP_INTERVAL_SECONDS)


def start_background_sweeper() -> None:
    threading.Thread(target=_sweeper_loop, daemon=True).start()
