#!/usr/bin/env bash
# One-line launcher for macOS / Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/<you>/freeopus/main/run.sh | bash
#
# or, from a local clone:
#
#   ./run.sh
#
# Requires only Docker Desktop (macOS) or Docker Engine (Linux) to be
# installed and running. Everything else is bundled into the image.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker Desktop from https://www.docker.com/products/docker-desktop/ and re-run this script." >&2
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit it (or use in-app Settings) to add API keys."
fi

echo "Building and starting OpenShorts Lite..."
docker compose up -d --build

echo ""
echo "OpenShorts Lite is running at: http://localhost:8000"
echo "Stop it anytime with: docker compose down"
