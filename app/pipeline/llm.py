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
(15-90 seconds each).

A clip is only acceptable if a viewer with zero other context can watch it
start to finish and fully understand it:
- The clip MUST start at the actual beginning of a thought - the start of a
  sentence, a setup, a question, or a new topic. Never start mid-sentence
  or mid-explanation.
- The clip MUST end at a natural conclusion - a punchline, an answer, a
  completed point, or a clear pause. Never cut off a sentence or a thought
  partway through.
- Set `start` and `end` to match an actual segment boundary from the
  transcript (a (t=...) timestamp, or immediately after a segment ends) -
  never an arbitrary time that would slice through the middle of a spoken
  line.
- Reject a moment if understanding it depends on something said before the
  clip starts or after it ends that isn't explained within the clip itself
  (e.g. "as I mentioned earlier...", answering a question the viewer never
  hears, referencing "that" or "this" without ever saying what it is).
- Prefer moments that also align with the scene-cut boundaries below, but
  the self-contained/complete-thought requirement above always wins if the
  two conflict.

Scene boundaries (seconds): {scene_str}

Transcript:
{transcript_text}

Respond with ONLY a JSON array (no prose, no markdown fences). Each item:
{{"start": <seconds float, at a segment boundary>, "end": <seconds float, at a segment boundary>,
  "title": "<short punchy title>",
  "reason": "<one sentence on why this clip works AND why it's fully self-contained>",
  "score": <0-100 virality score>}}
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


def _chat_completion(provider: str, api_key: str, model: str, prompt: str, temperature: float = 0.4) -> str:
    """Sends one prompt to whichever provider is configured and returns the
    raw text response. Shared by moment-picking and caption normalization
    so both providers only need to be wired up once."""
    if provider == "gemini":
        if not api_key:
            raise LLMError("Gemini API key is not configured")
        url = GEMINI_URL_TMPL.format(model=model, key=api_key)
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": temperature}})
        if resp.status_code != 200:
            raise LLMError(f"Gemini error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Gemini response: {data}") from exc

    if not api_key:
        raise LLMError("OpenRouter API key is not configured")
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature},
        )
    if resp.status_code != 200:
        raise LLMError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected OpenRouter response: {data}") from exc


def pick_moments_openrouter(
    api_key: str, model: str, segments, scenes: list[tuple[float, float]], min_clips: int = 3, max_clips: int = 8
) -> list[Moment]:
    prompt = _build_prompt(_transcript_with_markers(segments), scenes, min_clips, max_clips)
    content = _chat_completion("openrouter", api_key, model, prompt)
    return _to_moments(_extract_json_array(content))


def pick_moments_gemini(
    api_key: str, model: str, segments, scenes: list[tuple[float, float]], min_clips: int = 3, max_clips: int = 8
) -> list[Moment]:
    prompt = _build_prompt(_transcript_with_markers(segments), scenes, min_clips, max_clips)
    content = _chat_completion("gemini", api_key, model, prompt)
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


_CAPTION_BATCH_SIZE = 60


def _build_caption_prompt(lines: list[str]) -> str:
    numbered = "\n".join(f"{i}: {text}" for i, text in enumerate(lines))
    return f"""You are preparing on-screen captions for a short-form vertical video from
raw speech-to-text transcript lines. The speech-to-text engine sometimes
gets the language/script wrong - in particular, spoken Hindi is frequently
misrecognized and written out in Urdu (Perso-Arabic) script, because Hindi
and Urdu sound almost identical when spoken. Treat any Urdu-script or
Devanagari-script line as Hindi content.

For EACH numbered line below, decide what caption text should actually be
shown on screen, using these rules in order:
1. If the line is genuinely English, output it in clean English (fix obvious
   transcription typos, keep the meaning identical). Do not translate it.
2. If the line is Hindi in any script (Devanagari, Urdu/Perso-Arabic, or
   already Roman letters) or a Hindi-English code-mixed sentence, output it
   as "Hinglish": Hindi words written in plain Roman/Latin letters exactly
   the way Hindi speakers actually type when texting casually. Do not
   translate it to formal English and do not output Devanagari or Urdu
   script.
3. If the line is in some other language entirely (not English, not Hindi),
   translate it into natural, fluent English.
4. If you cannot confidently tell what the line actually says - it's too
   garbled, too short, nonsensical, or you are genuinely unsure of the
   language or content - output exactly the text SKIP for that line, so it
   can be left off screen instead of showing a wrong guess.

Respond with ONLY a JSON array, one object per line, same order, no prose,
no markdown fences:
[{{"i": <line number>, "text": "<caption text, or SKIP>"}}, ...]

Lines:
{numbered}
"""


def normalize_captions(provider: str, api_key: str, model: str, segments) -> list:
    """Rewrites transcript segments into on-screen-ready caption text per
    the language rules above (Hindi/Urdu-script -> Hinglish, other
    languages -> English, English passthrough, unclear -> dropped).

    Runs per-clip (not on the whole video) so it stays a handful of short
    lines per call - cheap, and captions end up scoped to exactly the
    content being captioned. Batches defensively for unusually long clips.
    Never raises: on any failure, returns the original segments unchanged
    so a captioning problem never fails the whole render.
    """
    from .transcriber import Segment, Word

    if not segments:
        return segments

    try:
        texts = [s.text.strip() for s in segments]
        results: dict[int, str] = {}
        for batch_start in range(0, len(texts), _CAPTION_BATCH_SIZE):
            batch = texts[batch_start : batch_start + _CAPTION_BATCH_SIZE]
            prompt = _build_caption_prompt(batch)
            content = _chat_completion(provider, api_key, model, prompt, temperature=0.2)
            for item in _extract_json_array(content):
                idx = int(item["i"]) + batch_start
                results[idx] = str(item.get("text", "")).strip()

        normalized: list = []
        for i, seg in enumerate(segments):
            new_text = results.get(i)
            if new_text is None or new_text.upper() == "SKIP" or not new_text:
                continue  # unclear / dropped - no caption for this stretch
            if new_text == seg.text.strip():
                normalized.append(seg)
                continue
            # Text changed (translated/transliterated): word-level timing
            # from the original language no longer lines up with the new
            # words, so spread the new words evenly across the segment's
            # original [start, end] - still gives a natural per-word
            # karaoke pace without needing a fresh forced-alignment pass.
            tokens = new_text.split()
            duration = max(0.05, seg.end - seg.start)
            per_word = duration / max(1, len(tokens))
            words = []
            t = seg.start
            for tok in tokens:
                words.append(Word(start=t, end=t + per_word, text=tok))
                t += per_word
            normalized.append(Segment(start=seg.start, end=seg.end, text=new_text, words=words))
        return normalized
    except Exception:  # noqa: BLE001
        return segments


def list_openrouter_models(api_key: str) -> list[dict]:
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(OPENROUTER_URL.replace("/chat/completions", "/models"), headers={"Authorization": f"Bearer {api_key}"})
    if resp.status_code != 200:
        raise LLMError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
    data = resp.json().get("data", [])
    return [{"id": m["id"], "name": m.get("name", m["id"])} for m in data]
