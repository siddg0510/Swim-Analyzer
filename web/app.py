"""
FastAPI Web Application for Swim Analyzer.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.core.models import AnalysisConfig
from src.analysis.benchmarks import ELITE_PROFILES, BENCHMARKS
from src.ai.gemini_config import resolve_api_key

from .models import JobCreateResponse, JobStatus, JobStatusResponse
from .storage import StorageManager
from .worker import JobWorker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("swim_analyzer.web")

app = FastAPI(
    title="Swim Analyzer API",
    description="Video analytics API for competitive swimming with Gemini-assisted splash tracking.",
    version="1.0.0",
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# Mount frontend static files if directory exists
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

storage = StorageManager()
worker = JobWorker(storage=storage)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the single-page application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Swim Analyzer API</h1><p>Frontend under construction.</p>")


@app.get("/api/health")
async def health_check():
    """Healthcheck endpoint."""
    return {"status": "ok", "app": "swim_analyzer", "version": "1.0.0"}


@app.get("/api/benchmarks")
async def get_benchmarks_meta():
    """Returns available elite athlete profiles and event metadata."""
    profiles = [
        {
            "key": k,
            "name": p.name,
            "country": p.country,
            "flag": p.country_flag,
            "events": p.primary_events,
            "achievement": p.achievement,
        }
        for k, p in ELITE_PROFILES.items()
    ]
    return {
        "profiles": profiles,
        "total_benchmarks": len(BENCHMARKS),
    }


@app.post("/api/jobs", response_model=JobCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis_job(
    video: UploadFile = File(...),
    pool_length_m: float = Form(50.0),
    event_distance_m: Optional[int] = Form(None),
    stroke_override: Optional[str] = Form(None),
    swimmer_sex: str = Form("male"),
    cap_color: Optional[str] = Form(None),
    target_id: Optional[int] = Form(1),
    enable_ai: bool = Form(True),
    enable_multi_swimmer: bool = Form(True),
    user_goal: str = Form("Improve race time and technique efficiency"),
):
    """Upload a race video and start an analysis job."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    valid_extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if not any(video.filename.lower().endswith(ext) for ext in valid_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Allowed: {', '.join(valid_extensions)}",
        )

    # Auto-detect Gemini key
    gemini_key = resolve_api_key()

    # Create job in worker
    temp_id = storage.get_job_dir("temp").name
    job_id = worker.create_job(
        AnalysisConfig(
            video_path="",  # will be updated once saved
            pool_length_m=pool_length_m,
            calibration_keyframes=[],
            cap_color=cap_color,
            wall_position_m=pool_length_m,
            enable_ai=enable_ai,
            gemini_api_key=gemini_key if enable_ai else None,
            stroke_override=stroke_override if stroke_override != "auto" else None,
            event_distance_m=event_distance_m,
            swimmer_sex=swimmer_sex,
            user_goal=user_goal,
            enable_multi_swimmer=enable_multi_swimmer,
            target_id=target_id,
        )
    )

    # Save uploaded file
    video_dest = storage.save_upload(job_id, video.filename, video.file)

    # Update job config with video path
    job_record = worker.jobs[job_id]
    job_record.cfg.video_path = str(video_dest)

    # Start processing in background
    worker.start_job(job_id)

    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=job_record.created_at,
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll for job status, real-time stage progress, and analysis results."""
    res = worker.get_job(job_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return res


@app.get("/api/jobs/{job_id}/csv")
async def download_job_csv(job_id: str):
    """Download CSV export for a completed job."""
    res = worker.get_job(job_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if res.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job is not completed yet.")

    csv_path = storage.get_csv_path(job_id)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV report not generated.")

    return FileResponse(
        path=str(csv_path),
        filename=f"swim_analysis_{job_id[:8]}.csv",
        media_type="text/csv",
    )


@app.delete("/api/jobs/{job_id}")
async def cancel_or_delete_job(job_id: str):
    """Cancel a running job or remove job data."""
    worker.cancel_job(job_id)
    storage.delete_job_files(job_id)
    if job_id in worker.jobs:
        del worker.jobs[job_id]
    return {"status": "deleted", "job_id": job_id}
