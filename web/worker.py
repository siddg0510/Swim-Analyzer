"""
Background job runner for video analysis.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.core.models import AnalysisConfig, AnalysisResult
from src.core.pipeline import run_analysis
from src.export import export_csv
from src.analysis.benchmarks import compare_to_benchmark

from .models import (
    JobStatus,
    JobProgress,
    JobStatusResponse,
    RaceAnalysisMetrics,
    SplitItem,
    LowConfidenceSegmentModel,
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
        """Convert internal AnalysisResult into clean API metrics."""
        splits_list: list[SplitItem] = []
        for s in res.split_report.splits:
            splits_list.append(SplitItem(
                marker_m=s.marker_m,
                time_s=round(s.time_s, 3),
                split_time_s=round(s.split_time_s, 3) if s.split_time_s else None,
                interpolated=s.interpolated,
            ))

        low_conf = [
            LowConfidenceSegmentModel(
                start_frame=seg.start_frame,
                end_frame=seg.end_frame,
                start_time_s=round(seg.start_time_s, 2),
                end_time_s=round(seg.end_time_s, 2),
                reason=seg.reason,
                resolved_by=seg.resolved_by,
            )
            for seg in res.low_confidence_segments
        ]

        vp = res.split_report.velocity_profile
        avg_v = sum(v for _, v in vp) / len(vp) if vp else None

        bench_comparison = None
        if avg_v is not None:
            dist = cfg.event_distance_m or int(cfg.pool_length_m)
            bench_comparison = compare_to_benchmark(
                avg_v, res.stroke_classification.stroke,
                dist, sex=cfg.swimmer_sex,
            )

        ai_dict: dict[str, Any] | None = None
        if res.ai_result:
            ai_dict = {}
            if res.ai_result.error:
                ai_dict["error"] = res.ai_result.error
            if res.ai_result.technique_analysis:
                ai_dict["technique_analysis"] = asdict(res.ai_result.technique_analysis)
            if res.ai_result.elite_comparison:
                ai_dict["elite_comparison"] = asdict(res.ai_result.elite_comparison)
            if res.ai_result.improvement_plan:
                ai_dict["improvement_plan"] = asdict(res.ai_result.improvement_plan)
            if res.ai_result.race_strategy:
                ai_dict["race_strategy"] = asdict(res.ai_result.race_strategy)

        return RaceAnalysisMetrics(
            event_distance_m=cfg.event_distance_m or cfg.pool_length_m,
            stroke=res.stroke_classification.stroke,
            stroke_confidence=round(res.stroke_classification.confidence, 2),
            start_method=res.start_detection.method,
            start_reaction_time_s=round(res.start_phase.reaction_time_s, 3) if res.start_phase.reaction_time_s else None,
            start_15m_time_s=round(res.start_phase.time_to_15m_s, 3) if res.start_phase.time_to_15m_s else None,
            turn_time_s=round(res.turn_phase.turn_duration_s, 3) if res.turn_phase and res.turn_phase.turn_duration_s else None,
            total_time_s=round(res.split_report.total_time_s, 3) if res.split_report.total_time_s else None,
            avg_velocity_mps=round(avg_v, 3) if avg_v else None,
            splits=splits_list,
            velocity_profile=[(round(d, 2), round(v, 3)) for d, v in res.split_report.velocity_profile],
            stroke_rates=[(round(t, 2), round(r, 1)) for t, r in res.stroke_rates],
            stroke_lengths=[(round(t, 2), round(l, 2)) for t, l in res.stroke_lengths],
            low_confidence_segments=low_conf,
            gemini_assisted_frames=res.gemini_assisted_frames,
            cv_only_frames=res.cv_only_frames,
            discrepancy_count=res.discrepancy_report.corrected_count + res.discrepancy_report.still_uncertain_count,
            ai_result=ai_dict,
            benchmark_comparison=bench_comparison,
        )
