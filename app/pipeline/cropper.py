"""Smart 9:16 reframing.

Computes a per-frame crop path from face detections, then renders the
clip with OpenCV (crop+resize per frame) while ffmpeg handles the final
encode + audio mux. Two detector backends are supported:

- mediapipe (preferred, better accuracy) if the optional dependency is
  installed and importable on this platform.
- OpenCV's bundled Haar cascade (always available) as a robust fallback.

Modes:
- "track": single dominant speaker, camera follows their face.
- "split": two speakers, each cropped around their face and stacked
  vertically to fill the 9:16 frame.
- "general": no reliable face found (or mode forced) -> centered crop
  over a blurred, full-frame background (classic "blurred bars" look).
- "auto": picks track/split/general automatically based on how many
  distinct faces are consistently detected.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import active_speaker

TARGET_W, TARGET_H = 1080, 1920

try:
    import mediapipe as mp

    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False


@dataclass
class FaceBox:
    x: float  # center x, normalized 0-1
    y: float  # center y, normalized 0-1
    w: float  # normalized width
    h: float  # normalized height


class FaceDetector:
    """Unified face detector: mediapipe if available, else Haar cascade."""

    def __init__(self) -> None:
        self.backend = "mediapipe" if _MP_AVAILABLE else "haar"
        if self.backend == "mediapipe":
            self._mp = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        h, w = frame_bgr.shape[:2]
        if self.backend == "mediapipe":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._mp.process(rgb)
            boxes = []
            if result.detections:
                for det in result.detections:
                    bb = det.location_data.relative_bounding_box
                    boxes.append(FaceBox(x=bb.xmin + bb.width / 2, y=bb.ymin + bb.height / 2, w=bb.width, h=bb.height))
            return boxes
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
            return [
                FaceBox(x=(x + fw / 2) / w, y=(y + fh / 2) / h, w=fw / w, h=fh / h) for (x, y, fw, fh) in faces
            ]


def _sample_faces(video_path: Path, start: float, end: float, sample_fps: float = 2.0) -> list[list[FaceBox]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = FaceDetector()
    step = max(1, int(round(fps / sample_fps)))
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    samples: list[list[FaceBox]] = []
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_frame) % step == 0:
            samples.append(detector.detect(frame))
        frame_idx += 1
    cap.release()
    return samples


def choose_mode(samples: list[list[FaceBox]]) -> str:
    counts = [len(s) for s in samples if s]
    if not counts:
        return "general"
    avg = sum(counts) / len(counts)
    if avg >= 1.5:
        return "split"
    return "track"


def _ema_path(centers: list[Optional[tuple[float, float]]], alpha: float = 0.15) -> list[tuple[float, float]]:
    path: list[tuple[float, float]] = []
    last = (0.5, 0.45)
    for c in centers:
        target = c if c is not None else last
        smoothed = (last[0] + alpha * (target[0] - last[0]), last[1] + alpha * (target[1] - last[1]))
        path.append(smoothed)
        last = smoothed
    return path


def _largest_face(faces: list[FaceBox]) -> Optional[FaceBox]:
    if not faces:
        return None
    return max(faces, key=lambda f: f.w * f.h)


def _two_largest(faces: list[FaceBox]) -> list[FaceBox]:
    return sorted(faces, key=lambda f: f.w * f.h, reverse=True)[:2]


def _median_seat_boxes(samples: list[list[FaceBox]]) -> Optional[tuple[FaceBox, FaceBox]]:
    """Two stable "seat" positions for a two-person scene, as the median
    left/right face box across every sample where both were detected.

    Mirrors the reference implementation's assumption for SPLIT scenes:
    the subjects of a two-shot are seated and the boxes hold, so one
    median position per seat for the whole clip is more robust than
    re-detecting (and potentially losing) a face every single frame.
    """
    lefts, rights = [], []
    for faces in samples:
        two = _two_largest(faces)
        if len(two) < 2:
            continue
        a, b = sorted(two, key=lambda f: f.x)
        lefts.append(a)
        rights.append(b)

    if len(lefts) < 3:  # not enough evidence of a stable two-shot
        return None

    def median_box(boxes: list[FaceBox]) -> FaceBox:
        return FaceBox(
            x=float(np.median([b.x for b in boxes])),
            y=float(np.median([b.y for b in boxes])),
            w=float(np.median([b.w for b in boxes])),
            h=float(np.median([b.h for b in boxes])),
        )

    return median_box(lefts), median_box(rights)


def _seat_segments(held: list[Optional[int]], min_windows: int) -> list[tuple[int, int, Optional[int]]]:
    """Groups a per-window `held` speaker sequence into contiguous runs,
    returning (start_window, end_window_exclusive, speaker_or_None).

    Every run *after* the first is already proof of a sustained turn: the
    hysteresis in `hold()` only lets a new speaker take over after holding
    the floor for `min_windows` in a row, so a transitioned-into run is
    valid at any length. Only the clip-opening run skips that hysteresis
    (the first speaker is accepted immediately, with no prior confirmation)
    and so is the one case checked against min_windows here, to avoid an
    unearned cutaway right at the start of the clip.
    """
    if not held:
        return []
    segments = []
    seg_start = 0
    current = held[0]
    for i in range(1, len(held) + 1):
        if i == len(held) or held[i] != current:
            length = i - seg_start
            is_opening_run = seg_start == 0
            speaker = current if (not is_opening_run or length >= min_windows) else None
            segments.append((seg_start, i, speaker))
            if i < len(held):
                seg_start = i
                current = held[i]
    return segments


def render_clip(
    source_path: Path,
    out_path: Path,
    start: float,
    end: float,
    mode: str = "auto",
    use_gpu: bool = False,
) -> str:
    """Renders a 1080x1920 clip from source_path[start:end] into out_path.

    Returns the mode actually used.
    """
    samples = _sample_faces(source_path, start, end)
    resolved_mode = choose_mode(samples) if mode == "auto" else mode

    cap = cv2.VideoCapture(str(source_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    n_frames = max(1, end_frame - start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Build a smoothed per-frame center path from the sparse face samples.
    sample_step = max(1, n_frames // max(1, len(samples)))
    raw_centers: list[Optional[tuple[float, float]]] = []
    for i in range(n_frames):
        sample_idx = min(len(samples) - 1, i // sample_step) if samples else 0
        faces = samples[sample_idx] if samples else []
        face = _largest_face(faces)
        raw_centers.append((face.x, face.y) if face else None)
    centers = _ema_path(raw_centers)

    raw_path = out_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (TARGET_W, TARGET_H))

    crop_h = src_h
    crop_w = int(crop_h * TARGET_W / TARGET_H)
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(crop_w * TARGET_H / TARGET_W)

    # For "split" clips, figure out (once, up front) whether this is a
    # genuine back-and-forth conversation and, if so, which stretches of
    # the clip have one person holding the floor long enough to earn a
    # full-frame cutaway instead of the stacked layout. Falls back to the
    # plain per-frame stacked split on any failure - this is a quality
    # refinement, never a reason to fail the whole render.
    per_frame_speaker_seat: list[Optional[int]] = [None] * n_frames
    seats: Optional[tuple[FaceBox, FaceBox]] = None
    if resolved_mode == "split":
        seats = _median_seat_boxes(samples)
        if seats is not None:
            try:
                boxes_px = [
                    (s.x * src_w - s.w * src_w / 2, s.y * src_h - s.h * src_h / 2, s.w * src_w, s.h * src_h)
                    for s in seats
                ]
                analysis = active_speaker.analyze_clip(source_path, start, end - start, boxes_px, fps)
                held = analysis["held"]
                if analysis["is_conversation"]:
                    segments = _seat_segments(held, active_speaker.MIN_HOLD_WINDOWS)
                    window_frames = max(1, round(fps * active_speaker.WINDOW_SECONDS))
                    for w_start, w_end, speaker in segments:
                        if speaker is None:
                            continue
                        f_start = w_start * window_frames
                        f_end = min(n_frames, w_end * window_frames)
                        for f in range(f_start, f_end):
                            per_frame_speaker_seat[f] = speaker
                elif held:
                    # Not a real back-and-forth - one person dominates the
                    # whole scene, so dedicate the entire clip to them
                    # instead of stacking a silent listener alongside them.
                    share0, share1 = active_speaker.shares(held)
                    dominant = 0 if share0 >= share1 else 1
                    per_frame_speaker_seat = [dominant] * n_frames
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                seats = None

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy = centers[i]

        if resolved_mode == "split":
            active_seat = per_frame_speaker_seat[i]
            if seats is not None and active_seat is not None:
                # A speaker has held the floor long enough - dedicate the
                # full frame to them, like a real editor punching in.
                seat = seats[active_seat]
                frame_out = _render_track(frame, seat.x, seat.y, crop_w, crop_h, src_w, src_h)
            elif seats is not None:
                frame_out = _render_split_seats(frame, seats, src_w, src_h)
            else:
                faces_here = samples[min(len(samples) - 1, i // sample_step)] if samples else []
                two = _two_largest(faces_here) if faces_here else []
                frame_out = _render_split(frame, two, src_w, src_h)
        elif resolved_mode == "general":
            frame_out = _render_blurred(frame, src_w, src_h)
        else:  # track
            frame_out = _render_track(frame, cx, cy, crop_w, crop_h, src_w, src_h)

        writer.write(frame_out)

    cap.release()
    writer.release()

    _mux_with_audio(source_path, raw_path, out_path, start, end, use_gpu)
    raw_path.unlink(missing_ok=True)
    return resolved_mode


def _render_track(frame, cx, cy, crop_w, crop_h, src_w, src_h):
    x = int(cx * src_w - crop_w / 2)
    y = int(cy * src_h - crop_h / 2)
    x = max(0, min(src_w - crop_w, x))
    y = max(0, min(src_h - crop_h, y))
    cropped = frame[y : y + crop_h, x : x + crop_w]
    return cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)


def _render_blurred(frame, src_w, src_h):
    bg = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=25)

    scale = TARGET_W / src_w
    fg_w = TARGET_W
    fg_h = int(src_h * scale)
    if fg_h > TARGET_H:
        scale = TARGET_H / src_h
        fg_h = TARGET_H
        fg_w = int(src_w * scale)
    fg = cv2.resize(frame, (fg_w, fg_h), interpolation=cv2.INTER_LANCZOS4)

    y_off = (TARGET_H - fg_h) // 2
    x_off = (TARGET_W - fg_w) // 2
    out = bg.copy()
    out[y_off : y_off + fg_h, x_off : x_off + fg_w] = fg
    return out


def _render_split(frame, faces: list[FaceBox], src_w, src_h):
    half_h = TARGET_H // 2
    if len(faces) < 2:
        # Not enough faces this frame: fall back to blurred layout so we
        # never crash on a transient miss.
        return _render_blurred(frame, src_w, src_h)

    faces = sorted(faces, key=lambda f: f.x)  # left speaker first
    panels = []
    crop_h = src_h
    crop_w = int(crop_h * TARGET_W / half_h)
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(crop_w * half_h / TARGET_W)

    for face in faces[:2]:
        cx, cy = face.x, face.y
        x = int(cx * src_w - crop_w / 2)
        y = int(cy * src_h - crop_h / 2)
        x = max(0, min(src_w - crop_w, x))
        y = max(0, min(src_h - crop_h, y))
        cropped = frame[y : y + crop_h, x : x + crop_w]
        panels.append(cv2.resize(cropped, (TARGET_W, half_h), interpolation=cv2.INTER_LANCZOS4))

    return np.vstack(panels)


def _render_split_seats(frame, seats: tuple[FaceBox, FaceBox], src_w, src_h):
    """Same layout as _render_split, but for the two fixed clip-wide seat
    positions rather than whatever faces were (or weren't) detected in
    this specific frame - robust to a momentary missed detection since the
    seat position doesn't depend on this frame at all."""
    half_h = TARGET_H // 2
    crop_h = src_h
    crop_w = int(crop_h * TARGET_W / half_h)
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(crop_w * half_h / TARGET_W)

    panels = []
    for seat in seats:
        x = int(seat.x * src_w - crop_w / 2)
        y = int(seat.y * src_h - crop_h / 2)
        x = max(0, min(src_w - crop_w, x))
        y = max(0, min(src_h - crop_h, y))
        cropped = frame[y : y + crop_h, x : x + crop_w]
        panels.append(cv2.resize(cropped, (TARGET_W, half_h), interpolation=cv2.INTER_LANCZOS4))

    return np.vstack(panels)


def _mux_with_audio(source_path: Path, raw_video_path: Path, out_path: Path, start: float, end: float, use_gpu: bool) -> None:
    import subprocess

    duration = max(0.1, end - start)
    codec = ["-c:v", "h264_nvenc"] if use_gpu else ["-c:v", "libx264", "-preset", "veryfast"]
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_video_path),
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        *codec,
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
