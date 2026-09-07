"""LLM-driven viral-moment picking, via OpenRouter (any model) or direct Gemini.

Both providers are given the same prompt: the transcript (with timestamps)
and the detected scene boundaries, and are asked to return a strict JSON
array of candidate clips. We parse defensively since LLMs occasionally
wrap JSON in prose or code fences.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


@dataclass
class Moment:
    id: str
    start: float
    end: float
    title: str
    reason: str
    score: float


class LLMError(RuntimeError):
    pass


def _build_prompt(transcript_text: str, scenes: list[tuple[float, float]], min_clips: int, max_clips: int) -> str:
    scene_str = ", ".join(f"[{s:.1f}-{e:.1f}]" for s, e in scenes[:200])
    return f"""You are an expert short-form video editor. Given a video transcript with
approximate word timings embedded as (t=SECONDS) markers, and the video's
scene-cut boundaries, pick the {min_clips} to {max_clips} most engaging,
self-contained moments that would work as viral vertical short-form clips
(15-90 seconds each). Prefer moments that align with scene boundaries.

Scene boundaries (seconds): {scene_str}

Transcript:
{transcript_text}

Respond with ONLY a JSON array (no prose, no markdown fences). Each item:
{{"start": <seconds float>, "end": <seconds float>, "title": "<short punchy title>",
  "reason": "<one sentence why this clip works>", "score": <0-100 virality score>}}
"""


def _extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise LLMError(f"Could not find a JSON array in the model response: {text[:200]}")
    return json.loads(match.group(0))


def _transcript_with_markers(segments) -> str:
    parts = []
    for seg in segments:
        parts.append(f"(t={seg.start:.1f}) {seg.text.strip()}")
    return "\n".join(parts)


def pick_moments_openrouter(
    api_key: str, model: str, segments, scenes: list[tuple[float, float]], min_clips: int = 3, max_clips: int = 8
) -> list[Moment]:
    if not api_key:
        raise LLMError("OpenRouter API key is not configured")
    prompt = _build_prompt(_transcript_with_markers(segments), scenes, min_clips, max_clips)
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
            },
        )
    if resp.status_code != 200:
        raise LLMError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected OpenRouter response: {data}") from exc
    return _to_moments(_extract_json_array(content))


def pick_moments_gemini(
    api_key: str, model: str, segments, scenes: list[tuple[float, float]], min_clips: int = 3, max_clips: int = 8
) -> list[Moment]:
    if not api_key:
        raise LLMError("Gemini API key is not configured")
    prompt = _build_prompt(_transcript_with_markers(segments), scenes, min_clips, max_clips)
    url = GEMINI_URL_TMPL.format(model=model, key=api_key)
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.4},
            },
        )
    if resp.status_code != 200:
        raise LLMError(f"Gemini error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected Gemini response: {data}") from exc
    return _to_moments(_extract_json_array(content))


def _to_moments(items: list[dict]) -> list[Moment]:
    moments = []
    for i, item in enumerate(items):
        try:
            moments.append(
                Moment(
                    id=f"m{i+1}",
                    start=float(item["start"]),
                    end=float(item["end"]),
                    title=str(item.get("title", f"Clip {i+1}"))[:120],
                    reason=str(item.get("reason", ""))[:300],
                    score=float(item.get("score", 50)),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    moments.sort(key=lambda m: m.score, reverse=True)
    return moments


def list_openrouter_models(api_key: str) -> list[dict]:
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(OPENROUTER_URL.replace("/chat/completions", "/models"), headers={"Authorization": f"Bearer {api_key}"})
    if resp.status_code != 200:
        raise LLMError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
    data = resp.json().get("data", [])
    return [{"id": m["id"], "name": m.get("name", m["id"])} for m in data]
