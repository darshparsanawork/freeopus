from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routers import jobs, settings

app = FastAPI(title="OpenShorts Lite")

app.include_router(settings.router)
app.include_router(jobs.router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/{full_path:path}")
def spa(full_path: str):
    return FileResponse(STATIC_DIR / "index.html")
