"""Transcription via OpenRouter's hosted openai/whisper-1.

This is the only transcription path - no local model, no local CPU/RAM
cost, no multi-hundred-MB model download. It reuses the same OpenRouter
key already required for moment-picking, so there's nothing extra to
configure. Audio is extracted from the video once, then split into
chunks that are verified to stay under the API's per-request upload
limit before each chunk is sent, so long videos work the same as short
ones - no manual duration limit needed.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx


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


class TranscriptionError(RuntimeError):
    pass


OPENROUTER_TRANSCRIBE_URL = "https://openrouter.ai/api/v1/audio/transcriptions"

# The Whisper API hard-caps uploads at 25MB; we stay well clear of that so
# encoding overhead/rounding never accidentally tips a chunk over the edge.
_MAX_CHUNK_BYTES = 20 * 1024 * 1024
_AUDIO_BITRATE_KBPS = 48  # mono, 16kHz - plenty for speech, keeps files small


def _extract_full_audio(video_path: Path, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", f"{_AUDIO_BITRATE_KBPS}k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _slice_audio(src: Path, out_path: Path, start: float, duration: float) -> None:
    # Stream-copy (no re-encode) since src is already a compressed mp3 -
    # fast and lossless relative to the original extraction.
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ss", str(start), "-t", str(duration), "-c", "copy", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _split_audio_by_size(audio_path: Path, tmp_dir: Path, max_bytes: int = _MAX_CHUNK_BYTES) -> list[tuple[Path, float]]:
    """Splits audio_path into chunks that each stay under max_bytes,
    returning a list of (chunk_path, start_offset_seconds) with offsets
    relative to the start of audio_path."""
    size = audio_path.stat().st_size
    if size <= max_bytes:
        return [(audio_path, 0.0)]

    duration = _probe_duration(audio_path)
    n_chunks = math.ceil(size / max_bytes)
    chunk_duration = duration / n_chunks

    chunks: list[tuple[Path, float]] = []
    for i in range(n_chunks):
        start = i * chunk_duration
        if start >= duration:
            break
        chunk_path = tmp_dir / f"chunk_{uuid.uuid4().hex}.mp3"
        _slice_audio(audio_path, chunk_path, start, chunk_duration)

        # Safety net: verify the actual size, in case of an unusual source
        # where the bitrate assumption doesn't hold. Split further if needed
        # - sub-chunk offsets are relative to this chunk, so add `start` to
        # convert them back into audio_path's own timeline.
        if chunk_path.stat().st_size > max_bytes:
            for sub_path, sub_offset in _split_audio_by_size(chunk_path, tmp_dir, max_bytes):
                chunks.append((sub_path, start + sub_offset))
        else:
            chunks.append((chunk_path, start))

    return chunks


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


def transcribe(
    video_path: Path,
    api_key: str,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Transcript:
    if not api_key:
        raise TranscriptionError(
            "An OpenRouter API key is required for transcription. Add one in Settings."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        full_audio = tmp_dir / "audio.mp3"
        _extract_full_audio(video_path, full_audio)

        chunks = _split_audio_by_size(full_audio, tmp_dir)
        segments: list[Segment] = []
        language = "en"

        for i, (chunk_path, chunk_start) in enumerate(chunks):
            data = _transcribe_chunk(chunk_path, api_key)
            language = data.get("language", language)
            words = [
                Word(start=w["start"] + chunk_start, end=w["end"] + chunk_start, text=w["word"])
                for w in data.get("words", [])
            ]
            if words:
                # whisper-1's verbose_json only gives one big text blob, not
                # per-sentence segments - group words into caption-sized
                # segments so downstream captioning has natural cue points.
                segments.extend(_group_words_into_segments(words))
            elif data.get("text"):
                chunk_duration = data.get("duration") or _probe_duration(chunk_path)
                segments.append(
                    Segment(start=chunk_start, end=chunk_start + chunk_duration, text=data["text"], words=[])
                )

            if progress_cb:
                progress_cb(min(99.0, (i + 1) / len(chunks) * 100.0))

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
