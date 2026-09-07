# One-line launcher for Windows (PowerShell).
#
#   .\run.ps1
#
# Requires only Docker Desktop for Windows to be installed and running.
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is required. Install Docker Desktop from https://www.docker.com/products/docker-desktop/ and re-run this script."
    exit 1
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example - edit it (or use in-app Settings) to add API keys."
}

Write-Host "Building and starting OpenShorts Lite..."
docker compose up -d --build

Write-Host ""
Write-Host "OpenShorts Lite is running at: http://localhost:8000"
Write-Host "Stop it anytime with: docker compose down"
