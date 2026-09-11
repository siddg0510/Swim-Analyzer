"""
Pydantic data models for the Swim Analyzer Web API.
"""
from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float


class JobProgress(BaseModel):
    percent: int = 0
    message: str = "Queued"


class LowConfidenceSegmentModel(BaseModel):
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    reason: str
    resolved_by: str | None = None


class SplitItem(BaseModel):
    marker_m: float
    time_s: float
    split_time_s: float | None = None
    interpolated: bool = False


class RaceAnalysisMetrics(BaseModel):
    event_distance_m: float
    stroke: str
    stroke_confidence: float
    start_method: str
    start_reaction_time_s: float | None = None
    start_15m_time_s: float | None = None
    turn_time_s: float | None = None
    total_time_s: float | None = None
    avg_velocity_mps: float | None = None
    splits: list[SplitItem] = []
    velocity_profile: list[tuple[float, float]] = []  # (distance_m, velocity_mps)
    stroke_rates: list[tuple[float, float]] = []      # (time_s, rate_cpm)
    stroke_lengths: list[tuple[float, float]] = []    # (time_s, length_m)
    low_confidence_segments: list[LowConfidenceSegmentModel] = []
    gemini_assisted_frames: int = 0
    cv_only_frames: int = 0
    discrepancy_count: int = 0
    ai_result: dict[str, Any] | None = None
    benchmark_comparison: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: JobProgress
    created_at: float
    completed_at: float | None = None
    error: str | None = None
    result: RaceAnalysisMetrics | None = None
