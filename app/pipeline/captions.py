"""Caption/subtitle generation in multiple formats from whisper word timings.

- SRT / VTT: standard subtitle files, timestamps re-based to clip start.
- ASS: styled captions with per-word karaoke highlighting ("live captions"
  / moving-text look), burnable into the video via ffmpeg's `ass` filter.
"""
from __future__ import annotations

from pathlib import Path

from .transcriber import Segment, Word


def _fmt_srt_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_time(t: float) -> str:
    return _fmt_srt_time(t).replace(",", ".")


def _fmt_ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _rebase(segments: list[Segment], clip_start: float, clip_end: float) -> list[Segment]:
    out = []
    for seg in segments:
        if seg.end < clip_start or seg.start > clip_end:
            continue
        words = [
            Word(start=max(0.0, w.start - clip_start), end=min(clip_end, w.end) - clip_start, text=w.text)
            for w in seg.words
            if w.end >= clip_start and w.start <= clip_end
        ]
        out.append(
            Segment(
                start=max(0.0, seg.start - clip_start),
                end=min(clip_end, seg.end) - clip_start,
                text=seg.text,
                words=words,
            )
        )
    return out


def to_srt(segments: list[Segment], clip_start: float, clip_end: float, position: str = "bottom") -> str:
    rebased = _rebase(segments, clip_start, clip_end)
    lines = []
    for i, seg in enumerate(rebased, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: list[Segment], clip_start: float, clip_end: float, position: str = "bottom") -> str:
    rebased = _rebase(segments, clip_start, clip_end)
    lines = ["WEBVTT", ""]
    for seg in rebased:
        lines.append(f"{_fmt_vtt_time(seg.start)} --> {_fmt_vtt_time(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


# ASS numpad alignment: 2=bottom-center, 5=middle-center, 8=top-center.
# MarginV is measured from whichever edge the alignment anchors to (ignored
# for middle). These margins are tuned to sit clear of typical short-form
# platform UI overlap (profile/follow button up top, caption/engagement
# bar at the bottom) on a 1080x1920 frame, so captions never crowd the edge.
_ASS_POSITION = {
    "bottom": (2, 220),
    "middle": (5, 0),
    "top": (8, 140),
}


def _ass_header(position: str) -> str:
    alignment, margin_v = _ASS_POSITION.get(position, _ASS_POSITION["bottom"])
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Live,Arial Black,72,&H00FFFFFF,&H0000D7FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,4,2,{alignment},60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def to_ass_karaoke(segments: list[Segment], clip_start: float, clip_end: float, position: str = "bottom") -> str:
    """Word-by-word highlighted captions ("live caption" style)."""
    rebased = _rebase(segments, clip_start, clip_end)
    lines = [_ass_header(position)]
    for seg in rebased:
        if not seg.words:
            lines.append(
                f"Dialogue: 0,{_fmt_ass_time(seg.start)},{_fmt_ass_time(seg.end)},Live,,0,0,0,,{seg.text.strip()}"
            )
            continue
        karaoke_text = ""
        for w in seg.words:
            dur_cs = max(1, int(round((w.end - w.start) * 100)))
            karaoke_text += f"{{\\k{dur_cs}}}{w.text.strip()} "
        lines.append(
            f"Dialogue: 0,{_fmt_ass_time(seg.words[0].start)},{_fmt_ass_time(seg.words[-1].end)},Live,,0,0,0,,{karaoke_text.strip()}"
        )
    return "\n".join(lines)


FORMATS = {"srt": to_srt, "vtt": to_vtt, "ass": to_ass_karaoke}


def write_captions(
    segments: list[Segment],
    clip_start: float,
    clip_end: float,
    out_dir: Path,
    basename: str,
    formats: list[str],
    position: str = "bottom",
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in formats:
        if fmt not in FORMATS:
            continue
        content = FORMATS[fmt](segments, clip_start, clip_end, position)
        path = out_dir / f"{basename}.{fmt}"
        path.write_text(content, encoding="utf-8")
        paths[fmt] = path
    return paths
