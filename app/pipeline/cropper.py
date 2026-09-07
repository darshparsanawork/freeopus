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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

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

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy = centers[i]

        if resolved_mode == "split":
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
