FROM python:3.11-slim

# ffmpeg: encoding/decoding + subtitle burn-in
# libgl1 / libglib2.0-0: required by opencv-python-headless at import time
# curl/git/ca-certificates: needed to install Deno and clone the PO-token
# provider below
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    ca-certificates \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# --- YouTube PO-token support ---
# YouTube now requires a "proof of origin" token to serve video formats,
# enforced most aggressively against datacenter/cloud IPs (exactly where
# this app is usually deployed). Without this, downloads eventually fail
# with confusing errors like "The page needs to be reloaded." even with
# valid cookies. This installs Deno (the JS runtime yt-dlp shells out to)
# and the bgutil-ytdlp-pot-provider script that generates tokens locally -
# no external token server needed. See: https://github.com/Brainicism/bgutil-ytdlp-pot-provider
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/opt/deno sh -s -- -y
ENV PATH="/opt/deno/bin:${PATH}"

RUN git clone --depth 1 https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git /tmp/bgutil \
    && mkdir -p /root/bgutil-ytdlp-pot-provider \
    && cp -r /tmp/bgutil/server /root/bgutil-ytdlp-pot-provider/server \
    && cd /root/bgutil-ytdlp-pot-provider/server && npm ci --omit=dev \
    && rm -rf /tmp/bgutil /root/.npm

WORKDIR /srv

COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && (pip install --no-cache-dir -r requirements-optional.txt || echo "mediapipe unavailable on this platform, falling back to Haar cascade face detection")

COPY app ./app

ENV DATA_DIR=/data
RUN mkdir -p /data

# Note: no VOLUME declaration here — Railway's builder rejects it ("use
# Railway Volumes" instead). Docker Compose still gets persistence via the
# named volume mounted at /data in docker-compose.yml; on Railway, attach a
# Volume mounted at /data from the dashboard for the same effect.

EXPOSE 8000
ENV PORT=8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
