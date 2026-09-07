from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def gpu_encoder_available() -> bool:
    """Best-effort check for an nvenc-capable ffmpeg build + NVIDIA GPU."""
    if not shutil.which("ffmpeg"):
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10
        )
        if "h264_nvenc" not in result.stdout:
            return False
        nvsmi = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return nvsmi.returncode == 0
    except Exception:
        return False


def burn_subtitles(video_in: Path, subtitle_path: Path, video_out: Path, use_gpu: bool) -> None:
    """Burns an .ass subtitle file into the video."""
    codec = ["-c:v", "h264_nvenc"] if use_gpu else ["-c:v", "libx264", "-preset", "veryfast"]
    escaped = str(subtitle_path).replace("\\", "\\\\").replace(":", "\\:")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_in),
        "-vf",
        f"ass={escaped}",
        *codec,
        "-c:a",
        "copy",
        str(video_out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
