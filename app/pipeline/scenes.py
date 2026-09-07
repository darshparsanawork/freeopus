"""Scene-boundary detection via PySceneDetect.

Scene cuts are fed to the LLM alongside the transcript so moment picking
respects natural edit points instead of cutting mid-shot.
"""
from __future__ import annotations

from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video


def detect_scenes(video_path: Path, threshold: float = 27.0) -> list[tuple[float, float]]:
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        duration = video.duration.get_seconds() if video.duration else 0.0
        return [(0.0, duration)]
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]
