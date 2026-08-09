"""
Core Gemini-powered swimming video analysis.

This module is the main integration point between the app and Google's
Gemini API. It handles:
  - Uploading video segments to the Gemini File API
  - Sending expert swimming analysis prompts
  - Parsing structured JSON responses
  - Combining AI insights with CV-computed metrics

All Gemini API calls are wrapped with retry logic and graceful fallback.
If the API is unavailable, the app continues with the existing offline
pipeline — AI analysis is strictly additive, never blocking.
"""
from __future__ import annotations

import json
import time
import logging
import tempfile
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for analysis results
# ---------------------------------------------------------------------------

class ModelUnavailableError(Exception):
    pass

@dataclass
class TechniqueElement:
    category: str
    element: str
    rating: str  # excellent | good | needs_improvement | critical_issue
    observation: str
    recommendation: str
    elite_reference: str = ""
    timestamp_hint: str = ""


@dataclass
class PriorityFix:
    priority: int
    element: str
    expected_impact: str
    drill: str


@dataclass
class TechniqueAnalysis:
    stroke_identified: str
    stroke_confidence: float
    overall_rating: str
    elements: list[TechniqueElement]
    top_priorities: list[PriorityFix]
    video_quality_notes: str
    raw_response: str = ""


@dataclass
class EliteStrengthMatch:
    element: str
    similarity: str
    detail: str


@dataclass
class EliteDifference:
    element: str
    user_observation: str
    elite_model: str
    gap_severity: str
    addressable: bool
    how_to_close_gap: str


@dataclass
class RealisticTarget:
    metric: str
    current: str
    target: str
    timeframe: str
    method: str


@dataclass
class EliteComparison:
    comparison_athlete: str
    overall_similarity_pct: float
    strengths: list[EliteStrengthMatch]
    differences: list[EliteDifference]
    metrics_comparison: dict
    realistic_targets: list[RealisticTarget]
    raw_response: str = ""


@dataclass
class Drill:
    name: str
    purpose: str
    description: str
    sets_reps: str
    frequency: str


@dataclass
class ImprovementPlan:
    summary: str
    phase_1: dict
    phase_2: dict
    phase_3: dict
    expected_improvement: dict
    raw_response: str = ""


@dataclass
class RaceStrategy:
    split_pattern: str
    split_analysis: str
    elite_pacing_comparison: dict
    speed_loss_zones: list[dict]
    stroke_rate_analysis: dict
    recommended_race_plan: dict
    raw_response: str = ""


@dataclass
class AISplitPoint:
    distance_m: float
    time_s: float


@dataclass
class AISplitData:
    start_time_s: float | None
    finish_time_s: float | None
    splits: list[AISplitPoint]
    confidence_notes: str


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------
class GeminiSwimAnalyzer:
    """Orchestrates Gemini API calls for swimming video analysis."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from .gemini_config import build_config, get_gemini_client
        self.config = build_config(api_key, model)
        self.client = get_gemini_client(api_key)
        self._uploaded_files: list = []

    def analyze_technique(
        self,
        video_path: str,
        stroke: str = "freestyle",
        distance_m: int = 100,
        pool_type: str = "long course",
    ) -> TechniqueAnalysis | None:
        """Upload video and get comprehensive technique analysis."""
        from .prompts import TECHNIQUE_ANALYSIS_PROMPT, SYSTEM_INSTRUCTION

        try:
            video_file = self._upload_video(video_path)
            if video_file is None:
                return None

            prompt = TECHNIQUE_ANALYSIS_PROMPT.format(
                stroke=stroke, distance=distance_m, pool_type=pool_type,
            )

            response = self._generate(video_file, prompt, SYSTEM_INSTRUCTION)
            if response is None:
                return None

            return self._parse_technique_analysis(response)

        except Exception as e:
            logger.error("Technique analysis failed: %s", e, exc_info=True)
            return None

    def compare_with_elite(
        self,
        video_path: str,
        elite_profile: dict,
        user_metrics: dict,
        stroke: str = "freestyle",
        distance_m: int = 100,
    ) -> EliteComparison | None:
        """Compare user's technique against an elite swimmer profile."""
        from .prompts import ELITE_COMPARISON_PROMPT, SYSTEM_INSTRUCTION

        try:
            video_file = self._upload_video(video_path)
            if video_file is None:
                return None

            prompt = ELITE_COMPARISON_PROMPT.format(
                stroke=stroke,
                distance=distance_m,
                elite_name=elite_profile.get("name", "Unknown"),
                elite_country=elite_profile.get("country", ""),
                elite_achievement=elite_profile.get("achievement", ""),
                elite_technique_description=elite_profile.get("technique", ""),
                elite_performance_data=json.dumps(elite_profile.get("metrics", {}), indent=2),
                user_velocity=user_metrics.get("avg_velocity", "N/A"),
                user_stroke_rate=user_metrics.get("stroke_rate", "N/A"),
                user_stroke_length=user_metrics.get("stroke_length", "N/A"),
            )

            response = self._generate(video_file, prompt, SYSTEM_INSTRUCTION)
            if response is None:
                return None

            return self._parse_elite_comparison(response)

        except Exception as e:
            logger.error("Elite comparison failed: %s", e, exc_info=True)
            return None

    def generate_improvement_plan(
        self,
        video_path: str,
        technique_findings: str,
        user_metrics: dict,
        stroke: str = "freestyle",
        distance_m: int = 100,
        user_goal: str = "Improve race time and technique efficiency",
    ) -> ImprovementPlan | None:
        """Create a personalized multi-phase improvement plan."""
        from .prompts import IMPROVEMENT_PLAN_PROMPT, SYSTEM_INSTRUCTION

        try:
            video_file = self._upload_video(video_path)
            if video_file is None:
                return None

            prompt = IMPROVEMENT_PLAN_PROMPT.format(
                stroke=stroke,
                distance=distance_m,
                technique_findings=technique_findings,
                user_velocity=user_metrics.get("avg_velocity", "N/A"),
                user_stroke_rate=user_metrics.get("stroke_rate", "N/A"),
                user_stroke_length=user_metrics.get("stroke_length", "N/A"),
                start_time=user_metrics.get("start_15m_time", "N/A"),
                turn_time=user_metrics.get("turn_time", "N/A"),
                user_goal=user_goal,
            )

            response = self._generate(video_file, prompt, SYSTEM_INSTRUCTION)
            if response is None:
                return None

            return self._parse_improvement_plan(response)

        except Exception as e:
            logger.error("Improvement plan generation failed: %s", e, exc_info=True)
            return None

    def analyze_race_strategy(
        self,
        video_path: str,
        split_data: str,
        velocity_profile: str,
        stroke_rate_data: str,
        stroke: str = "freestyle",
        distance_m: int = 100,
        pool_type: str = "long course",
    ) -> RaceStrategy | None:
        """Analyze pacing and race strategy."""
        from .prompts import RACE_STRATEGY_PROMPT, SYSTEM_INSTRUCTION

        try:
            video_file = self._upload_video(video_path)
            if video_file is None:
                return None

            prompt = RACE_STRATEGY_PROMPT.format(
                stroke=stroke,
                distance=distance_m,
                pool_type=pool_type,
                split_data=split_data,
                velocity_profile=velocity_profile,
                stroke_rate_data=stroke_rate_data,
            )

            response = self._generate(video_file, prompt, SYSTEM_INSTRUCTION)
            if response is None:
                return None

            return self._parse_race_strategy(response)

        except Exception as e:
            logger.error("Race strategy analysis failed: %s", e, exc_info=True)
            return None

    def identify_stroke(self, video_path: str) -> tuple[str, float]:
        """Use Gemini to identify the swimming stroke (classifier fallback)."""
        from .prompts import STROKE_IDENTIFICATION_PROMPT, SYSTEM_INSTRUCTION

        try:
            video_file = self._upload_video(video_path)
            if video_file is None:
                return "unknown", 0.0

            response = self._generate(
                video_file, STROKE_IDENTIFICATION_PROMPT, SYSTEM_INSTRUCTION
            )
            if response is None:
                return "unknown", 0.0

            data = self._safe_parse_json(response)
            if data:
                return (
                    data.get("stroke", "unknown"),
                    float(data.get("confidence", 0.0)),
                )
            return "unknown", 0.0

        except Exception as e:
            logger.error("Stroke identification failed: %s", e)
            return "unknown", 0.0

    def detect_splits(
        self,
        video_path: str,
        swimmer_identifier: str,
        pool_length: float,
    ) -> AISplitData | None:
        """Use Gemini to identify exactly when the swimmer crosses distance markers."""
        from .prompts import SPLIT_DETECTION_PROMPT, SYSTEM_INSTRUCTION

        try:
            video_file = self._upload_video(video_path)
            if video_file is None:
                return None

            prompt = SPLIT_DETECTION_PROMPT.format(
                swimmer_identifier=swimmer_identifier,
                pool_length=pool_length,
            )

            response = self._generate(video_file, prompt, SYSTEM_INSTRUCTION)
            if response is None:
                return None

            return self._parse_splits(response)

        except ModelUnavailableError:
            raise
        except Exception as e:
            logger.error("Split detection failed: %s", e, exc_info=True)
            return None

    def _parse_splits(self, response: str) -> AISplitData | None:
        data = self._safe_parse_json(response)
        if not data:
            return None

        splits = []
        for s in data.get("splits", []):
            dist = float(s.get("distance_m", 0))
            time_val = float(s.get("time_s", 0))
            splits.append(AISplitPoint(distance_m=dist, time_s=time_val))
            
        # Ensure splits are sorted by time
        splits.sort(key=lambda x: x.time_s)

        return AISplitData(
            start_time_s=data.get("start_time_s"),
            finish_time_s=data.get("finish_time_s"),
            splits=splits,
            confidence_notes=data.get("confidence_notes", ""),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _upload_video(self, video_path: str, max_retries: int = 3):
        """Upload a video to the Gemini File API and wait for it to become ACTIVE."""
        try:
            logger.info("Uploading video to Gemini: %s", video_path)
            video_file = self.client.files.upload(file=video_path)
            self._uploaded_files.append(video_file)

            # Poll until processing is done
            retries = 0
            while video_file.state.name == "PROCESSING":
                if retries > 60:  # ~10 minutes max wait
                    logger.error("Video processing timed out")
                    return None
                time.sleep(10)
                video_file = self.client.files.get(name=video_file.name)
                retries += 1

            if video_file.state.name == "FAILED":
                logger.error("Video processing failed on Gemini's side")
                return None

            logger.info("Video ready: %s", video_file.name)
            return video_file

        except Exception as e:
            logger.error("Video upload failed: %s", e)
            return None

    def _generate(self, video_file, prompt: str, system_instruction: str) -> str | None:
        """Send a generation request with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=[video_file, prompt],
                    config={
                        "system_instruction": system_instruction,
                        "temperature": 0.3,  # Low temp for analytical consistency
                    },
                )
                if response and response.text:
                    return response.text
                logger.warning("Empty response on attempt %d", attempt + 1)

            except Exception as e:
                error_str = str(e)
                logger.warning("API call attempt %d failed: %s", attempt + 1, error_str)
                if "404 NOT_FOUND" in error_str or "is no longer available" in error_str:
                    raise ModelUnavailableError(f"Model {self.config.model} is not available. Please change your AI settings.")
                    
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_s * (attempt + 1))

        return None

    def _safe_parse_json(self, text: str) -> dict | None:
        """Parse JSON from Gemini response, handling common formatting issues."""
        if not text:
            return None

        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (code fences)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            logger.error("Failed to parse JSON from Gemini response")
            logger.debug("Raw response: %s", text[:500])
            return None

    def _parse_technique_analysis(self, response: str) -> TechniqueAnalysis | None:
        data = self._safe_parse_json(response)
        if not data:
            return None

        elements = []
        for elem in data.get("technique_elements", []):
            elements.append(TechniqueElement(
                category=elem.get("category", ""),
                element=elem.get("element", ""),
                rating=elem.get("rating", "needs_improvement"),
                observation=elem.get("observation", ""),
                recommendation=elem.get("recommendation", ""),
                elite_reference=elem.get("elite_reference", ""),
                timestamp_hint=elem.get("timestamp_hint", ""),
            ))

        priorities = []
        for pri in data.get("top_3_priorities", []):
            priorities.append(PriorityFix(
                priority=pri.get("priority", 0),
                element=pri.get("element", ""),
                expected_impact=pri.get("expected_impact", ""),
                drill=pri.get("drill", ""),
            ))

        return TechniqueAnalysis(
            stroke_identified=data.get("stroke_identified", "unknown"),
            stroke_confidence=float(data.get("stroke_confidence", 0.0)),
            overall_rating=data.get("overall_rating", "needs_improvement"),
            elements=elements,
            top_priorities=priorities,
            video_quality_notes=data.get("video_quality_notes", ""),
            raw_response=response,
        )

    def _parse_elite_comparison(self, response: str) -> EliteComparison | None:
        data = self._safe_parse_json(response)
        if not data:
            return None

        strengths = [
            EliteStrengthMatch(
                element=s.get("element", ""),
                similarity=s.get("similarity", ""),
                detail=s.get("detail", ""),
            )
            for s in data.get("strengths_matching_elite", [])
        ]

        differences = [
            EliteDifference(
                element=d.get("element", ""),
                user_observation=d.get("user_observation", ""),
                elite_model=d.get("elite_model", ""),
                gap_severity=d.get("gap_severity", "moderate"),
                addressable=d.get("addressable", True),
                how_to_close_gap=d.get("how_to_close_gap", ""),
            )
            for d in data.get("key_differences", [])
        ]

        targets = [
            RealisticTarget(
                metric=t.get("metric", ""),
                current=t.get("current", ""),
                target=t.get("target", ""),
                timeframe=t.get("timeframe", ""),
                method=t.get("method", ""),
            )
            for t in data.get("realistic_targets", [])
        ]

        return EliteComparison(
            comparison_athlete=data.get("comparison_athlete", ""),
            overall_similarity_pct=float(data.get("overall_similarity_pct", 0)),
            strengths=strengths,
            differences=differences,
            metrics_comparison=data.get("metrics_comparison", {}),
            realistic_targets=targets,
            raw_response=response,
        )

    def _parse_improvement_plan(self, response: str) -> ImprovementPlan | None:
        data = self._safe_parse_json(response)
        if not data:
            return None

        return ImprovementPlan(
            summary=data.get("summary", ""),
            phase_1=data.get("phase_1_immediate", {}),
            phase_2=data.get("phase_2_development", {}),
            phase_3=data.get("phase_3_race_prep", {}),
            expected_improvement=data.get("expected_improvement", {}),
            raw_response=response,
        )

    def _parse_race_strategy(self, response: str) -> RaceStrategy | None:
        data = self._safe_parse_json(response)
        if not data:
            return None

        return RaceStrategy(
            split_pattern=data.get("split_pattern", ""),
            split_analysis=data.get("split_analysis", ""),
            elite_pacing_comparison=data.get("elite_pacing_comparison", {}),
            speed_loss_zones=data.get("speed_loss_zones", []),
            stroke_rate_analysis=data.get("stroke_rate_analysis", {}),
            recommended_race_plan=data.get("recommended_race_plan", {}),
            raw_response=response,
        )

    def cleanup(self) -> None:
        """Delete uploaded files from Gemini to free resources."""
        for f in self._uploaded_files:
            try:
                self.client.files.delete(name=f.name)
                logger.info("Deleted uploaded file: %s", f.name)
            except Exception as e:
                logger.warning("Could not delete file %s: %s", f.name, e)
        self._uploaded_files.clear()
