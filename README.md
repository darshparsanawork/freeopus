# OpenShorts Lite

A drastically simpler, self-hosted take on [OpenShorts](https://github.com/mutonby/openshorts):
paste a video URL, get AI-picked viral moments reframed into 9:16 shorts with
captions — no Python environment to configure, no npm build, no GPU drivers
to fight with. One command, one URL, one dashboard.

## What it does

1. **Paste a URL** — YouTube or anything [yt-dlp](https://github.com/yt-dlp/yt-dlp) supports.
2. **Automatic transcription** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper), word-level timestamps, runs on CPU or GPU.
3. **AI moment picking** — an LLM (via OpenRouter, any model, or direct Gemini) reads the transcript + scene cuts and proposes 3-8 clip-worthy moments with a title, reason, and virality score.
4. **You review and pick** — see every candidate before anything is rendered; choose which ones to turn into clips.
5. **Smart 9:16 reframing** — face-tracking crop that follows a single speaker, splits the frame for two speakers, or falls back to a blurred-background layout when no face is reliably detected.
6. **Captions** — burned-in styled captions (word-by-word "live caption" highlight) plus downloadable SRT/VTT/ASS files.
7. **Download** — grab the finished MP4 and caption files straight from the dashboard.

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

Whisper transcription and video encoding will automatically use the GPU. Without a GPU, everything still runs — just slower — entirely on CPU (Whisper `small`, libx264 software encoding).

## Deploying to Railway

1. Push this repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo**, pick this repo. Railway detects the `Dockerfile` and `railway.json` automatically — no config needed.
3. Attach a **volume** mounted at `/data` (Railway dashboard → your service → Volumes) so downloaded videos, transcripts, and rendered clips survive redeploys.
4. Once deployed, open the generated `*.up.railway.app` URL — that's your dashboard.
5. Add your API keys either as Railway service variables (`OPENROUTER_API_KEY`, `GEMINI_API_KEY`) or directly in the in-app Settings panel after it's live.

## Setting up API keys

Click **⚙ Settings** in the dashboard:

- **OpenRouter** (recommended): paste one API key from [openrouter.ai/keys](https://openrouter.ai/keys), click **Load available models**, and pick whichever model you want to use for moment-picking (Gemini, Claude, GPT, Llama, etc. — anything OpenRouter exposes).
- **Gemini (direct)**: alternatively, paste a key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and pick a Gemini model directly, without going through OpenRouter.
- The Settings panel explains exactly what this model is used for (only the transcript + scene list — never the video/audio itself — to pick clip-worthy moments) and what the original OpenShorts project uses for the same step, so you can judge tradeoffs before picking one instead of guessing from a bare model name.
- **Whisper model size**: `small` is the default and a good speed/accuracy balance on CPU. Use `tiny`/`base` for faster turnaround on long videos, or `medium` for higher accuracy if you have the compute.
- **Device**: leave on `Auto-detect` unless you need to force CPU or GPU.

Keys are stored locally in `/data/config.json` inside your own container/volume — they are never sent anywhere except the provider they belong to (OpenRouter or Google).

### How YouTube downloads stay reliable

YouTube actively fights automated downloading in two layers, and this app handles both:

1. **Bot-check ("Sign in to confirm you're not a bot")** — triggered mostly from datacenter/cloud IPs (VPS hosting, Railway) and much less on a home connection. The app retries across several internal YouTube player clients automatically, which resolves most cases. If it still happens, add cookies from a logged-in browser session (below).
2. **PO tokens ("The page needs to be reloaded" / "No video formats found")** — YouTube now requires a per-request "proof of origin" token to actually serve video formats, enforced hardest against cloud IPs. The Docker image bundles [Deno](https://deno.com) plus the [bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider) plugin, which generates these tokens locally on every request — no external token server needed, and no action required from you. This also needs yt-dlp itself to stay current, which is why it's intentionally left unpinned in `requirements.txt` (rebuild the image periodically to pick up fixes for YouTube's frequent internal changes).

If downloads still fail with a bot-check error after all that, add cookies:

1. Install a "cookies.txt" export extension in a browser where you're logged into YouTube (e.g. *Get cookies.txt LOCALLY*).
2. Export cookies for youtube.com.
3. In the dashboard, open **⚙ Settings** → expand the bot-check section → paste the cookies file contents → **Save & validate cookies**.

The app immediately makes a real (download-free) request to YouTube with those cookies and tells you right there whether they actually work — no guessing until your next real download. Use **Re-check saved cookies** any time later to confirm they haven't expired.

## How the smart cropping works

- Faces are sampled a few times per second across each clip using [MediaPipe](https://developers.google.com/mediapipe) face detection when available, or OpenCV's built-in Haar cascade detector as an automatic fallback (MediaPipe wheels aren't available for every platform, so the app never hard-fails on this).
- **Auto** mode picks between three layouts based on what it sees: `track` (one dominant face — camera smoothly follows it), `split` (two faces — each speaker gets their own cropped panel stacked vertically), and `general` (no reliable face — centered crop over a blurred full-frame background).
- You can also force a specific mode per batch in the review screen.

## Architecture (for reference)

```
app/
  main.py              FastAPI app + static dashboard serving
  jobs.py              in-memory job state machine (background threads)
  config.py            local settings persistence (/data/config.json)
  models.py            request/response schemas
  pipeline/
    downloader.py      yt-dlp
    transcriber.py     faster-whisper
    scenes.py          PySceneDetect
    llm.py             OpenRouter + Gemini moment-picking
    cropper.py         face detection + smart 9:16 reframing
    captions.py        SRT / VTT / ASS generation
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
