"""Who is talking, from mouth movement plus audio energy.

A static split-screen (both speakers stacked, unconditionally, for the
whole clip) looks amateurish and wastes half the frame on someone sitting
silent. The fix used here follows the same approach as the OpenShorts
project this app is based on (see its `active_speaker.py`): full speaker
diarization is overkill for "which of these two known faces is moving its
mouth right now" - mouth movement plus an audio-energy gate answers that
cheaply, and a hysteresis hold on top stops the result from flapping on
every stray twitch or interjection.

Differences from the reference implementation: the hold threshold here is
tuned to ~4-5 seconds (vs. its ~1.2s) per an explicit product decision -
a turn has to be genuinely sustained before the frame commits to it,
which reads as more deliberate/edited rather than reactive.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

# Seconds per decision window. Short enough to catch a real exchange,
# long enough that one blurry/occluded frame cannot flip the answer.
WINDOW_SECONDS = 0.4

# A window counts as speech only if its audio RMS clears this fraction of
# the clip's loudest window. Silence and room tone sit far below it.
AUDIO_FLOOR = 0.18

# The winner's mouth must be this much more active than the other's,
# relative to their combined activity, before a window is attributed.
# Below it the window is "unclear" and does not vote.
MIN_MARGIN = 0.15

# Share of attributed windows each speaker needs before a two-shot counts
# as a genuine back-and-forth conversation (vs. one person dominating).
MIN_SHARE = 0.2

# How long someone must hold the floor before the frame commits to them.
MIN_HOLD_SECONDS = 4.5
MIN_HOLD_WINDOWS = max(1, round(MIN_HOLD_SECONDS / WINDOW_SECONDS))

ANALYSIS_WIDTH = 480


def mouth_region(box: tuple[float, float, float, float], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Pixel rect (x0, y0, x1, y1) around the mouth of a face box (x, y, w, h
    in the same pixel space as frame_w/frame_h)."""
    x, y, w, h = box
    cx = x + w / 2.0
    mouth_y = y + h * 0.72
    half_w = w * 0.36
    half_h = h * 0.22

    x0 = int(max(0, min(cx - half_w, frame_w - 1)))
    x1 = int(max(x0 + 1, min(cx + half_w, frame_w)))
    y0 = int(max(0, min(mouth_y - half_h, frame_h - 1)))
    y1 = int(max(y0 + 1, min(mouth_y + half_h, frame_h)))
    return x0, y0, x1, y1


def audio_envelope(video_path: Path, start_s: float, duration_s: float, window_s: float = WINDOW_SECONDS) -> list[float]:
    """Per-window RMS of the clip's audio, normalized to its own peak.

    Returns [] when there's no audio (or ffmpeg fails), which makes every
    window eligible rather than none - a silent source falls back to mouth
    movement alone instead of refusing to decide.
    """
    try:
        raw = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-ss", f"{start_s:.4f}", "-t", f"{duration_s:.4f}",
                "-i", str(video_path), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    if not raw:
        return []

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    per_window = max(1, int(8000 * window_s))
    n = len(samples) // per_window
    if n < 1:
        return []

    windows = samples[: n * per_window].reshape(n, per_window)
    rms = np.sqrt(np.mean(windows**2, axis=1))
    peak = rms.max()
    return (rms / peak).tolist() if peak > 0 else [0.0] * n


def normalise_activity(activity: list[list[float]]) -> list[list[float]]:
    """Rescales each speaker's mouth activity onto its own quiet-to-loud
    range, so whichever face happens to be better lit doesn't win every
    window just from raw contrast/lighting differences."""
    if not activity:
        return []
    columns = np.asarray(activity, dtype=float)
    if columns.ndim != 2 or columns.shape[1] < 2:
        return activity

    low = np.percentile(columns, 20, axis=0)
    high = np.percentile(columns, 80, axis=0)
    spread = np.where(high - low > 1e-6, high - low, 1.0)
    return np.clip((columns - low) / spread, 0.0, None).tolist()


def attribute_windows(activity: list[list[float]], loudness: list[float], min_margin: float = MIN_MARGIN) -> list[Optional[int]]:
    """Which speaker (0 or 1) owns each window, or None when unclear."""
    verdicts: list[Optional[int]] = []
    for i, pair in enumerate(activity):
        if loudness and i < len(loudness) and loudness[i] < AUDIO_FLOOR:
            verdicts.append(None)
            continue
        a, b = pair
        total = a + b
        if total <= 0:
            verdicts.append(None)
            continue
        margin = abs(a - b) / total
        verdicts.append(None if margin < min_margin else (0 if a > b else 1))
    return verdicts


def shares(verdicts: list[Optional[int]]) -> tuple[float, float]:
    counted = [v for v in verdicts if v is not None]
    if not counted:
        return 0.0, 0.0
    n = float(len(counted))
    return counted.count(0) / n, counted.count(1) / n


def is_conversation(verdicts: list[Optional[int]], min_share: float = MIN_SHARE) -> bool:
    """True when both speakers hold enough of the attributed windows to
    count as a genuine back-and-forth, rather than one person dominating
    while the other occasionally nods."""
    a, b = shares(verdicts)
    return min(a, b) >= min_share


def hold(verdicts: list[Optional[int]], min_windows: int = MIN_HOLD_WINDOWS) -> list[Optional[int]]:
    """Smooths per-window verdicts so the active speaker cannot flap: a new
    speaker must hold ``min_windows`` in a row before taking over, and
    unclear windows inherit whoever currently holds the floor."""
    out: list[Optional[int]] = []
    current: Optional[int] = None
    pending: Optional[int] = None
    run = 0
    for v in verdicts:
        if v is None or v == current:
            if v == current:
                pending, run = None, 0
            out.append(current)
            continue
        if v == pending:
            run += 1
        else:
            pending, run = v, 1
        if current is None or run >= min_windows:
            current, pending, run = v, None, 0
        out.append(current)

    if all(v is None for v in out):
        return out

    first = next(v for v in out if v is not None)
    return [first if v is None else v for v in out]


def decode_mouth_activity(
    video_path: Path, start_s: float, duration_s: float, boxes: list[tuple[float, float, float, float]], fps: float,
    window_s: float = WINDOW_SECONDS,
) -> list[list[float]]:
    """Mean absolute frame-to-frame change in each speaker's mouth region,
    one score per speaker per window. Decodes the clip once at a reduced
    width via a raw ffmpeg pipe (never loads the full-resolution video into
    memory), reading and discarding one frame at a time."""
    import cv2

    probe = cv2.VideoCapture(str(video_path))
    orig_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    if not orig_w or not orig_h:
        return []

    small_w = min(ANALYSIS_WIDTH, orig_w)
    small_w -= small_w % 2
    small_h = max(int(orig_h * small_w / orig_w), 2)
    small_h += small_h % 2
    if small_w <= 0 or small_h <= 0:
        return []
    scale = small_w / float(orig_w)
    frame_bytes = small_w * small_h * 3

    regions = [mouth_region((x * scale, y * scale, w * scale, h * scale), small_w, small_h) for (x, y, w, h) in boxes]

    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-ss", f"{start_s:.4f}", "-t", f"{duration_s:.4f}",
            "-i", str(video_path), "-vf", f"scale={small_w}:{small_h}", "-pix_fmt", "bgr24",
            "-f", "rawvideo", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    frames_per_window = max(1, round(fps * window_s))
    window_activity: list[list[float]] = []
    current_sums = [0.0, 0.0]
    current_count = 0
    prev_regions: list[Optional[np.ndarray]] = [None, None]

    assert proc.stdout is not None
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(small_h, small_w, 3)

            for i, (x0, y0, x1, y1) in enumerate(regions):
                crop = frame[y0:y1, x0:x1].astype(np.int16)
                if prev_regions[i] is not None and prev_regions[i].shape == crop.shape:
                    diff = float(np.mean(np.abs(crop - prev_regions[i])))
                    current_sums[i] += diff
                prev_regions[i] = crop

            current_count += 1
            if current_count >= frames_per_window:
                window_activity.append([s / current_count for s in current_sums])
                current_sums = [0.0, 0.0]
                current_count = 0
    finally:
        proc.stdout.close()
        proc.wait(timeout=30)

    if current_count > 0:
        window_activity.append([s / current_count for s in current_sums])

    return window_activity


def analyze_clip(
    video_path: Path,
    start_s: float,
    duration_s: float,
    boxes: list[tuple[float, float, float, float]],
    fps: float,
) -> dict:
    """Runs the full pipeline for one clip's two speaker boxes and returns
    {"is_conversation": bool, "held": list[int|None]} - `held` has one
    entry per WINDOW_SECONDS-sized window across the clip, naming which
    speaker (0/1, index into `boxes`) has the floor at that point."""
    activity = decode_mouth_activity(video_path, start_s, duration_s, boxes, fps)
    loudness = audio_envelope(video_path, start_s, duration_s)
    verdicts = attribute_windows(normalise_activity(activity), loudness)
    conversation = is_conversation(verdicts)
    held = hold(verdicts)
    return {"is_conversation": conversation, "held": held}
