"""Transcription: either local faster-whisper, or OpenRouter's hosted
openai/whisper-1 (same OpenRouter key already used for moment-picking).

Local is free and private but uses this server's own CPU/RAM. OpenRouter
offloads the work entirely to OpenAI's cloud - faster, more accurate, no
local CPU/RAM cost, but requires internet and costs a small amount per
minute of audio (~$0.006/min at the time this was verified).

Device/compute-type auto-selects GPU when available and falls back to CPU
transparently for the local path, so the same code path works on a laptop
or a GPU box.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx
from faster_whisper import WhisperModel

_MODEL_CACHE: dict[tuple[str, str], WhisperModel] = {}


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    segments: list[Segment]

    def as_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)

    def words_between(self, start: float, end: float) -> list[Word]:
        out: list[Word] = []
        for seg in self.segments:
            for w in seg.words:
                if w.end >= start and w.start <= end:
                    out.append(w)
        return out

    def segments_between(self, start: float, end: float) -> list[Segment]:
        return [s for s in self.segments if s.end >= start and s.start <= end]


def _resolve_device(device_setting: str) -> tuple[str, str]:
    if device_setting == "cpu":
        return "cpu", "int8"
    if device_setting == "cuda":
        return "cuda", "float16"
    # auto: try cuda, fall back to cpu
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _get_model(model_size: str, device_setting: str) -> WhisperModel:
    device, compute_type = _resolve_device(device_setting)
    key = (model_size, device)
    if key not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception:
            # Hardware/driver mismatch -> hard fallback to CPU.
            _MODEL_CACHE[key] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _MODEL_CACHE[key]


def transcribe(
    video_path: Path,
    model_size: str = "small",
    device: str = "auto",
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Transcript:
    model = _get_model(model_size, device)
    segments_iter, info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        vad_filter=True,
    )

    duration = info.duration or 1.0
    segments: list[Segment] = []
    for seg in segments_iter:
        words = [Word(start=w.start, end=w.end, text=w.word) for w in (seg.words or [])]
        segments.append(Segment(start=seg.start, end=seg.end, text=seg.text, words=words))
        if progress_cb:
            progress_cb(min(99.0, seg.end / duration * 100.0))

    if progress_cb:
        progress_cb(100.0)

    return Transcript(language=info.language, segments=segments)


class TranscriptionError(RuntimeError):
    pass


OPENROUTER_TRANSCRIBE_URL = "https://openrouter.ai/api/v1/audio/transcriptions"

# Whisper's API caps uploads at 25MB. A 48kbps mono mp3 stays well under
# that for roughly up to ~65 minutes; longer audio gets split into chunks
# of this length and stitched back together with adjusted timestamps.
_MAX_CHUNK_SECONDS = 20 * 60


def _extract_audio(video_path: Path, out_path: Path, start: float = 0, duration: Optional[float] = None) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if start:
        cmd += ["-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _transcribe_chunk(audio_path: Path, api_key: str) -> dict:
    with open(audio_path, "rb") as f:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                OPENROUTER_TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (audio_path.name, f, "audio/mpeg")},
                data={
                    "model": "openai/whisper-1",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                },
            )
    if resp.status_code != 200:
        raise TranscriptionError(f"OpenRouter transcription error {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def transcribe_via_openrouter(
    video_path: Path,
    api_key: str,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Transcript:
    if not api_key:
        raise TranscriptionError("OpenRouter API key is not configured")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        full_audio = tmp_dir / "audio.mp3"
        _extract_audio(video_path, full_audio)
        total_duration = _probe_duration(full_audio)

        n_chunks = max(1, int(total_duration // _MAX_CHUNK_SECONDS) + 1)
        segments: list[Segment] = []
        language = "en"

        for i in range(n_chunks):
            chunk_start = i * _MAX_CHUNK_SECONDS
            if chunk_start >= total_duration:
                break
            if n_chunks == 1:
                chunk_path = full_audio
            else:
                chunk_path = tmp_dir / f"chunk_{i}.mp3"
                _extract_audio(video_path, chunk_path, start=chunk_start, duration=_MAX_CHUNK_SECONDS)

            data = _transcribe_chunk(chunk_path, api_key)
            language = data.get("language", language)
            words = [
                Word(start=w["start"] + chunk_start, end=w["end"] + chunk_start, text=w["word"])
                for w in data.get("words", [])
            ]
            if words:
                # whisper-1's verbose_json only gives one big text blob, not
                # per-sentence segments - group words into caption-sized
                # segments (~8s each) so downstream captioning behaves the
                # same as the local-whisper path.
                segments.extend(_group_words_into_segments(words))
            elif data.get("text"):
                segments.append(
                    Segment(start=chunk_start, end=chunk_start + total_duration, text=data["text"], words=[])
                )

            if progress_cb:
                progress_cb(min(99.0, (i + 1) / n_chunks * 100.0))

    if progress_cb:
        progress_cb(100.0)

    return Transcript(language=language, segments=segments)


def _group_words_into_segments(words: list[Word], max_segment_seconds: float = 8.0) -> list[Segment]:
    segments: list[Segment] = []
    current: list[Word] = []

    def flush():
        if current:
            text = " ".join(w.text.strip() for w in current)
            segments.append(Segment(start=current[0].start, end=current[-1].end, text=text, words=list(current)))

    for w in words:
        if current and (w.end - current[0].start > max_segment_seconds):
            flush()
            current = []
        current.append(w)
    flush()
    return segments
