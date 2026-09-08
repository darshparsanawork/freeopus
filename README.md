# OpenShorts Lite

A drastically simpler, self-hosted take on [OpenShorts](https://github.com/mutonby/openshorts):
paste a video URL, get AI-picked viral moments reframed into 9:16 shorts with
captions — no Python environment to configure, no npm build, no GPU drivers
to fight with. One command, one URL, one dashboard.

## What it does

1. **Paste a URL** — YouTube, a public Google Drive share link, or anything else [yt-dlp](https://github.com/yt-dlp/yt-dlp) supports. Runs in the background: close the tab and it keeps going.
2. **Automatic transcription** — OpenRouter's `openai/whisper-1`, word-level timestamps, no local model or CPU cost. Long videos are automatically chunked to stay under the API's upload limit.
3. **AI moment picking** — an LLM (via OpenRouter, any model, or direct Gemini) reads the transcript + scene cuts and proposes 3-8 clip-worthy moments with a title, reason, and virality score.
4. **You review and pick** — preview each candidate moment in-browser at its exact timestamp before anything is rendered, then choose which ones to turn into clips.
5. **Smart 9:16 reframing** — face-tracking crop that follows a single speaker, or an active-speaker-aware layout for two people: a genuine back-and-forth gets a stacked split, but a turn held for ~4-5 seconds+ gets a full-frame cutaway to whoever's actually talking (see "How the smart cropping works" below). Falls back to a blurred-background layout when no face is reliably detected.
6. **Captions** — burned-in styled captions (word-by-word "live caption" highlight) at your choice of position (top/middle/bottom, previewed live before you generate) plus downloadable SRT/VTT/ASS files.
7. **Dashboard** — every video you've submitted shows up as a card with a live status badge (queued, downloading, needs your input, rendering, ready) and an auto-delete countdown, so nothing gets lost if you navigate away mid-job.
8. **Download** — grab the finished MP4 and caption files straight from the dashboard.

## Quickstart

Only requirement: [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows) or Docker Engine (Linux).

**macOS / Linux:**
```bash
./run.sh
```

**Windows (PowerShell):**
```powershell
.\run.ps1
```

Either script builds the image, starts the container, and prints the dashboard URL:

```
OpenShorts Lite is running at: http://localhost:8000
```

Open that URL, paste a video link, and go. Stop everything with `docker compose down`.

If you'd rather run it manually: `docker compose up -d --build`.

### GPU acceleration (optional, NVIDIA only)

If you have an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Video encoding (cropping/burning captions) will automatically use the GPU. Without a GPU, encoding falls back to libx264 software encoding - transcription itself always runs via OpenRouter regardless, so GPU/CPU only affects render speed, not transcription.

## Deploying to Railway

1. Push this repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo**, pick this repo. Railway detects the `Dockerfile` and `railway.json` automatically — no config needed.
3. Attach a **volume** mounted at `/data` (Railway dashboard → your service → Volumes) so downloaded videos, transcripts, and rendered clips survive redeploys.
4. Once deployed, open the generated `*.up.railway.app` URL — that's your dashboard.
5. Add your API keys either as Railway service variables (`OPENROUTER_API_KEY`, `GEMINI_API_KEY`) or directly in the in-app Settings panel after it's live.

## Setting up API keys

Click **⚙ Settings** in the dashboard:

- **OpenRouter API key is required** — paste one from [openrouter.ai/keys](https://openrouter.ai/keys). It's used for transcription (`openai/whisper-1`, ~$0.006/min of audio) unconditionally, plus moment-picking unless you switch that to Gemini below. Click **Load available models** to pick which model handles moment-picking (Gemini, Claude, GPT, Llama, etc. — anything OpenRouter exposes).
- **Gemini (optional, direct)**: paste a key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) to use Gemini directly for moment-picking instead of routing that step through OpenRouter too. Transcription still goes through OpenRouter either way.
- The Settings panel explains exactly what the moment-picking model is used for (only the transcript + scene list — never the video/audio itself) and what the original OpenShorts project uses for the same step, so you can judge tradeoffs before picking one instead of guessing from a bare model name.

Keys are stored locally in `/data/config.json` inside your own container/volume — they are never sent anywhere except the provider they belong to (OpenRouter or Google).

There's no local transcription model to configure (size, device, etc.) — everything transcription-related is handled by the one OpenRouter call, so there's nothing to tune and no local CPU/RAM/disk cost for it.

## Storage, cleanup, and resource limits

- **Everything lives under `/data`** (downloaded source video, transcript, rendered clips, captions, your saved settings) — this is what a Docker Compose volume or a Railway Volume should point at, so a redeploy or container restart doesn't lose in-flight or recent jobs.
- **Automatic 2-hour cleanup**: a background sweeper runs every 10 minutes and deletes a job's entire folder (video, clips, captions) plus its in-memory state once it's older than `JOB_RETENTION_HOURS` (default `2`). The dashboard shows a countdown on the results page so you know when to download. This also sweeps orphaned job folders left over from before a restart, using each folder's modified time, so cleanup keeps working even if the server restarted.
- **Bounded concurrency**: jobs run on a small worker pool (`MAX_CONCURRENT_JOBS`, default `1`) instead of one thread per request, so a burst of URLs queues up instead of all downloading/transcribing/rendering at once and starving the CPU/RAM available to any single job. Raise it if you're running on a host with more cores/RAM to spare.
- Finished jobs also drop their in-memory transcript/scene data as soon as their clips are rendered (it's already been written to the caption files by then), rather than waiting for the full retention window to free that RAM.
- **Idle CPU/RAM is near zero.** There's no local ML model resident in memory (transcription is a network call to OpenRouter), the worker pool's threads block on nothing when there's no job, and the video preview/download endpoints stream files in fixed 1MB chunks rather than loading them whole — so a multi-hour source video never sits in RAM. The only thing that runs on a timer regardless of activity is the cleanup sweep, once every 10 minutes.

### How YouTube downloads stay reliable

YouTube actively fights automated downloading in two layers, and this app handles both:

1. **Bot-check ("Sign in to confirm you're not a bot")** — triggered mostly from datacenter/cloud IPs (VPS hosting, Railway) and much less on a home connection. The app retries across several internal YouTube player clients automatically, which resolves most cases. If it still happens, add cookies from a logged-in browser session (below).
2. **PO tokens ("The page needs to be reloaded" / "No video formats found")** — YouTube now requires a per-request "proof of origin" token to actually serve video formats, enforced hardest against cloud IPs. The Docker image bundles [Deno](https://deno.com) plus the [bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider) plugin, which generates these tokens locally on every request — no external token server needed, and no action required from you. This also needs yt-dlp itself to stay current, which is why it's intentionally left unpinned in `requirements.txt` (rebuild the image periodically to pick up fixes for YouTube's frequent internal changes).

If downloads still fail with a bot-check error after all that, add cookies:

1. Install a "cookies.txt" export extension in a browser where you're logged into YouTube (e.g. *Get cookies.txt LOCALLY*).
2. Export cookies for youtube.com.
3. In the dashboard, open **⚙ Settings** → expand the bot-check section → paste the cookies file contents → **Save & validate cookies**.

The app immediately makes a real (download-free) request to YouTube with those cookies and tells you right there whether they actually work — no guessing until your next real download. Use **Re-check saved cookies** any time later to confirm they haven't expired.

### Google Drive links

Paste a `drive.google.com/file/d/...` share link the same way as any other URL. This works out of the box for **public** files (shared as "Anyone with the link"). If the file is private, the download fails with a clear message telling you to open its Share settings in Google Drive and set access to "Anyone with the link" — rather than a raw, confusing error.

## How the smart cropping works

- Faces are sampled a few times per second across each clip using [MediaPipe](https://developers.google.com/mediapipe) face detection when available, or OpenCV's built-in Haar cascade detector as an automatic fallback (MediaPipe wheels aren't available for every platform, so the app never hard-fails on this).
- **Auto** mode picks between three base layouts: `track` (one dominant face — camera smoothly follows it), `split` (two faces detected consistently), and `general` (no reliable face — centered crop over a blurred full-frame background).
- **Two-speaker scenes get an active-speaker-aware split**, not just a static stack. This follows the same approach as the [OpenShorts](https://github.com/mutonby/openshorts) project this app is based on (`active_speaker.py`): mouth-movement frame-differencing plus an audio-energy gate decide who's talking in each 0.4s window, and a hysteresis "hold" (tuned here to ~4-5 seconds, vs. the reference's ~1.2s) means a brief interjection is absorbed rather than triggering a switch.
  - If both people genuinely take turns, the clip defaults to the stacked split layout, but a turn held past the ~4-5s threshold gets a full-frame cutaway to whoever's actually talking — like a real edit punching in — before returning to the stack (or cutting to the other speaker) when the turn changes.
  - If one person dominates the whole scene (not a real back-and-forth), the split is skipped entirely and the clip tracks that person full-frame for its whole duration, instead of wasting half the frame on a silent listener.
- You can also force a specific mode per batch in the review screen.

## Architecture (for reference)

```
app/
  main.py              FastAPI app + static dashboard serving
  jobs.py              in-memory job state machine (background threads)
  config.py            local settings persistence (/data/config.json)
  models.py            request/response schemas
  pipeline/
    downloader.py      yt-dlp (YouTube, Google Drive, generic URLs)
    transcriber.py     OpenRouter openai/whisper-1, with size-based chunking
    scenes.py          PySceneDetect
    llm.py             OpenRouter + Gemini moment-picking
    cropper.py         face detection + smart 9:16 reframing
    active_speaker.py  mouth-movement + audio active-speaker detection for split scenes
    captions.py        SRT / VTT / ASS generation, position-aware
    ffmpeg_utils.py    subtitle burn-in, GPU encoder detection
  static/              vanilla HTML/CSS/JS dashboard (no build step)
```

Everything runs as a single container/process — no separate database, no
message broker, no frontend build pipeline. Job state lives in memory
(fine for a self-hosted, single-instance app); rendered files and settings
persist to disk under `/data` across restarts.

## Limitations vs. full OpenShorts

This intentionally leaves out OpenShorts' AI Shorts pipeline (synthetic
actors, ElevenLabs voice dubbing, fal.ai talking-head generation) and
YouTube Studio publishing — those require several more paid API keys and a
much heavier pipeline. This project focuses on doing the core
clip-generation flow (download → transcribe → AI moment picking → smart
crop → captions) simply and reliably. It's a single-user/single-instance
app: fine to self-host for yourself or a small team, not built as
multi-tenant SaaS.
