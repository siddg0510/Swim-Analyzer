"""
Data models for the analysis pipeline — pure dataclasses with no GUI dependencies.

These are used by both the desktop app and the web backend. Nothing in this
module may import PySide6, Qt, or any other GUI framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..vision.calibration import CalibrationKeyframe
from ..analysis.splits import SplitReport
from ..audio.start_detector import StartDetectionResult
from ..vision.stroke_classifier import StrokeClassification
from ..analysis.biomechanics import StartPhaseMetrics, TurnPhaseMetrics
from ..analysis.error_correction import DiscrepancyReport
from ..analysis.comparator import DTWResult


# ---------------------------------------------------------------------------
# Low-confidence segment tracking (populated by the Kalman/gating layer)
# ---------------------------------------------------------------------------

@dataclass
class LowConfidenceSegment:
    """A contiguous span of frames where tracking confidence was low."""
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    reason: str          # "coast", "splash", "high_innovation"
    resolved_by: str | None = None  # "gemini", "reacquisition", None


# ---------------------------------------------------------------------------
# Analysis configuration — inputs to the pipeline
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    video_path: str
    pool_length_m: float
    calibration_keyframes: list[CalibrationKeyframe]
    cap_color: str | None = None
    wall_position_m: float = None  # set to pool_length_m if turn analysis wanted
    # AI analysis options
    enable_ai: bool = False
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    stroke_override: str | None = None  # user-selected stroke
    event_distance_m: int | None = None  # user-selected event distance
    pool_type: str = "long course"
    swimmer_sex: str = "male"
    elite_compare_key: str | None = None  # key into ELITE_PROFILES
    reference_video_path: str | None = None  # optional reference video
    user_goal: str = "Improve race time and technique efficiency"
    # ── Swimmer tracking mode (Primary Pipeline) ────────────────────────────
    enable_multi_swimmer: bool = True    # If True: use swimmer detection + EKF tracking
    target_id: int | None = None         # Resolved by SwimmerSelectionDialog before analysis


# ---------------------------------------------------------------------------
# AI analysis result — all fields optional
# ---------------------------------------------------------------------------

@dataclass
class AIAnalysisResult:
    """Results from Gemini AI analysis — all optional."""
    technique_analysis: object | None = None   # TechniqueAnalysis
    elite_comparison: object | None = None     # EliteComparison
    improvement_plan: object | None = None     # ImprovementPlan
    race_strategy: object | None = None        # RaceStrategy
    video_comparison: object | None = None     # VideoComparison
    error: str | None = None


# ---------------------------------------------------------------------------
# Full analysis result — output of the pipeline
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    split_report: SplitReport
    start_detection: StartDetectionResult
    stroke_classification: StrokeClassification
    start_phase: StartPhaseMetrics
    turn_phase: TurnPhaseMetrics | None
    stroke_rates: list[tuple[float, float]]
    stroke_lengths: list[tuple[float, float]]
    discrepancy_report: DiscrepancyReport
    pose_available: bool
    csv_rows: list[dict] = field(default_factory=list)
    # AI analysis results (None when AI is disabled or unavailable)
    ai_result: AIAnalysisResult | None = None
    # DTW biomechanical comparison (None when no gold standard is available)
    dtw_result: DTWResult | None = None
    # Low-confidence segment annotations
    low_confidence_segments: list[LowConfidenceSegment] = field(default_factory=list)
    gemini_assisted_frames: int = 0
    cv_only_frames: int = 0


# ---------------------------------------------------------------------------
# Progress callback type (used by both Qt and web backends)
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, str], None]
"""Signature: callback(percent: int, status_message: str) -> None"""
