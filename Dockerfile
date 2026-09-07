FROM python:3.11-slim

# ffmpeg: encoding/decoding + subtitle burn-in
# libgl1 / libglib2.0-0: required by opencv-python-headless at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && (pip install --no-cache-dir -r requirements-optional.txt || echo "mediapipe unavailable on this platform, falling back to Haar cascade face detection")

COPY app ./app

ENV DATA_DIR=/data
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000
ENV PORT=8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
