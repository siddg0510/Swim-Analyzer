"""
FastAPI Web Application for Swim Analyzer.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
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
from .storage import StorageManager, UploadTooLargeError
from .worker import JobWorker
from .calibration import (
    parse_calibration_keyframes,
    has_usable_manual_calibration,
    CalibrationParseError,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("swim_analyzer.web")

# How often the background retention sweeper runs (minutes). The TTL itself
# lives on the StorageManager (SWIM_ANALYZER_RETENTION_HOURS, default 24h).
SWEEP_INTERVAL_SECONDS = int(os.environ.get("SWIM_ANALYZER_SWEEP_INTERVAL_MIN", 60)) * 60


def _cors_config() -> tuple[list[str], bool]:
    """Resolve CORS origins and whether credentials are allowed.

    The wildcard origin ``*`` and ``allow_credentials=True`` are mutually
    exclusive under the CORS spec — a browser rejects ``Access-Control-Allow-
    Origin: *`` on a credentialed request, so that combination silently breaks
    real requests. This API is token-free (no cookies), so:

      * Default (no ``SWIM_ANALYZER_CORS_ORIGINS``): allow all origins with
        credentials DISABLED — the valid, working wildcard configuration.
      * Explicit comma-separated origins: echo them back with credentials
        ENABLED, which is the only correct way to support credentialed CORS.
    """
    raw = os.environ.get("SWIM_ANALYZER_CORS_ORIGINS", "").strip()
    if not raw:
        return ["*"], False
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins, True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the periodic retention sweeper for the app's lifetime.

    Product decision: auto-sweep expired jobs, 24h default TTL. The sweeper
    deletes on-disk artifacts past their TTL and purges the matching in-memory
    job records, so neither the disk nor the ``jobs`` dict grows unbounded on a
    long-running server. A per-iteration try/except keeps one failed sweep from
    killing the loop.
    """
    async def _sweeper():
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
            try:
                removed = await asyncio.to_thread(storage.cleanup_expired)
                purged = worker.purge_expired(storage.ttl_seconds)
                if removed or purged:
                    logger.info("Retention sweep: %d folder(s), %d record(s) removed", removed, purged)
            except Exception:  # noqa: BLE001 — never let the sweeper die
                logger.exception("Retention sweep iteration failed; continuing")

    task = asyncio.create_task(_sweeper())
    logger.info(
        "Retention sweeper started (interval=%ds, ttl=%ds)",
        SWEEP_INTERVAL_SECONDS, storage.ttl_seconds,
    )
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Swim Analyzer API",
    description="Video analytics API for competitive swimming with Gemini-assisted splash tracking.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS. See _cors_config() for why wildcard + credentials is avoided.
_cors_origins, _cors_credentials = _cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
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
    calibration_json: Optional[str] = Form(None),
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

    # Parse manual calibration (lane corners + distance-labelled reference points)
    # posted by the web picker. Malformed input is a client error, not a crash.
    try:
        calibration_keyframes = parse_calibration_keyframes(calibration_json)
    except CalibrationParseError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid calibration data: {exc}")

    # Auto-detect Gemini key
    gemini_key = resolve_api_key()

    # Product decision #1 ("Manual UI + AI + graceful fail"): a run needs a usable
    # calibration source. Reject early — before uploading a potentially large file
    # — when we can already prove none exists. The core pipeline enforces the same
    # invariant as a backstop (e.g. AI enabled + key present but auto-calibration
    # fails at runtime), so this is a fast-path UX improvement, not the only guard.
    if not has_usable_manual_calibration(calibration_keyframes):
        if not enable_ai:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No calibration provided. Add at least 2 reference points in the "
                    "calibration panel, or enable Gemini AI auto-calibration."
                ),
            )
        if not gemini_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Gemini AI auto-calibration is enabled but no API key is configured "
                    "on the server, so it cannot run. Add manual calibration (at least 2 "
                    "reference points) or configure a Gemini API key."
                ),
            )

    # Create job in worker
    temp_id = storage.get_job_dir("temp").name
    job_id = worker.create_job(
        AnalysisConfig(
            video_path="",  # will be updated once saved
            pool_length_m=pool_length_m,
            calibration_keyframes=calibration_keyframes,
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

    # Save uploaded file (streamed to disk with a hard size cap).
    try:
        video_dest = storage.save_upload(job_id, video.filename, video.file)
    except UploadTooLargeError as exc:
        # Roll back the just-created job so a rejected upload leaves no trace.
        storage.delete_job_files(job_id)
        worker.jobs.pop(job_id, None)
        raise HTTPException(status_code=413, detail=str(exc))

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
