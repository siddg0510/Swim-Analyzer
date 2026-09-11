"""
Background job runner for video analysis.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.core.models import AnalysisConfig, AnalysisResult
from src.core.pipeline import run_analysis
from src.core.metrics import summarize_result
from src.export import export_csv

from .models import (
    JobStatus,
    JobProgress,
    JobStatusResponse,
    RaceAnalysisMetrics,
)
from .storage import StorageManager

logger = logging.getLogger(__name__)


class JobRecord:
    """In-memory state of an analysis job."""

    def __init__(self, job_id: str, cfg: AnalysisConfig, storage: StorageManager):
        self.job_id = job_id
        self.cfg = cfg
        self.storage = storage
        self.status = JobStatus.QUEUED
        self.progress = JobProgress(percent=0, message="Job queued")
        self.created_at = time.time()
        self.completed_at: float | None = None
        self.error: str | None = None
        self.result: RaceAnalysisMetrics | None = None
        self.raw_result: AnalysisResult | None = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        self.status = JobStatus.CANCELLED
        self.progress.message = "Cancelled"

    def is_cancelled(self) -> bool:
        return self._cancelled


class JobWorker:
    """Manages asynchronous job queue and execution."""

    def __init__(self, storage: StorageManager, max_workers: int = 2):
        self.storage = storage
        self.jobs: dict[str, JobRecord] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def create_job(self, cfg: AnalysisConfig) -> str:
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id, cfg, self.storage)
        self.jobs[job_id] = record
        return job_id

    def start_job(self, job_id: str):
        record = self.jobs.get(job_id)
        if not record:
            return
        asyncio.create_task(self._run_job_async(record))

    def get_job(self, job_id: str) -> JobStatusResponse | None:
        record = self.jobs.get(job_id)
        if not record:
            return None
        return JobStatusResponse(
            job_id=record.job_id,
            status=record.status,
            progress=record.progress,
            created_at=record.created_at,
            completed_at=record.completed_at,
            error=record.error,
            result=record.result,
        )

    def cancel_job(self, job_id: str) -> bool:
        record = self.jobs.get(job_id)
        if not record:
            return False
        record.cancel()
        return True

    def purge_expired(self, ttl_seconds: float) -> int:
        """Drop in-memory records for finished jobs older than the TTL.

        The disk sweep (``StorageManager.cleanup_expired``) frees files; this
        frees the matching ``JobRecord`` entries so the ``jobs`` dict does not
        grow without bound over a long-running server. Only terminal jobs
        (completed / failed / cancelled) are purged — active jobs are kept
        regardless of age so a slow analysis is never dropped mid-run.
        """
        now = time.time()
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        stale = [
            jid for jid, rec in self.jobs.items()
            if rec.status in terminal
            and (rec.completed_at or rec.created_at) < now - ttl_seconds
        ]
        for jid in stale:
            del self.jobs[jid]
        if stale:
            logger.info("Purged %d expired in-memory job record(s)", len(stale))
        return len(stale)

    async def _run_job_async(self, record: JobRecord):
        loop = asyncio.get_running_loop()
        record.status = JobStatus.PROCESSING
        record.progress.message = "Starting analysis…"

        def progress_callback(pct: int, msg: str):
            record.progress.percent = pct
            record.progress.message = msg

        try:
            raw_result: AnalysisResult = await loop.run_in_executor(
                self.executor,
                run_analysis,
                record.cfg,
                progress_callback,
                record.is_cancelled,
            )

            if record.is_cancelled():
                record.status = JobStatus.CANCELLED
                return

            record.raw_result = raw_result
            record.result = self._format_metrics(record.cfg, raw_result)
            record.status = JobStatus.COMPLETED
            record.completed_at = time.time()
            record.progress.percent = 100
            record.progress.message = "Analysis complete!"

            # Export CSV to job storage directory
            csv_path = self.storage.get_csv_path(record.job_id)
            export_csv(raw_result, str(csv_path))
            logger.info("Job %s completed successfully", record.job_id)

        except Exception as exc:
            logger.exception("Job %s failed: %s", record.job_id, exc)
            record.status = JobStatus.FAILED
            record.error = str(exc)
            record.completed_at = time.time()
            record.progress.message = f"Failed: {exc}"

    def _format_metrics(self, cfg: AnalysisConfig, res: AnalysisResult) -> RaceAnalysisMetrics:
        """Convert internal AnalysisResult into the API metrics model.

        The numbers themselves come from the shared-core canonical serializer
        (:func:`src.core.metrics.summarize_result`) so the web API and the
        desktop dashboard report identical figures. This method only adapts
        that dict into the Pydantic response model (which coerces the nested
        split / low-confidence dicts into their sub-models automatically).
        """
        return RaceAnalysisMetrics(**summarize_result(cfg, res))
