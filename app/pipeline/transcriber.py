"""Transcription via faster-whisper with word-level timestamps.

Device/compute-type auto-selects GPU when available and falls back to CPU
transparently, so the same code path works on a laptop or a GPU box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

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
