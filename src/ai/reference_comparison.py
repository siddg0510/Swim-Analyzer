"""
Reference video comparison using Gemini.

Allows the user to upload a second video (e.g. Olympic race footage)
and have Gemini compare technique side-by-side with the user's video.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TechniqueDifference:
    element: str
    user_technique: str
    reference_technique: str
    impact: str
    drill_to_fix: str
    difficulty: str  # easy | moderate | hard


@dataclass
class VideoComparison:
    comparison_summary: str
    technique_differences: list[TechniqueDifference]
    what_user_does_well: list[str]
    overall_technique_gap: str  # small | moderate | large
    top_priority_change: str
    raw_response: str = ""


def compare_videos(
    analyzer,  # GeminiSwimAnalyzer instance
    user_video_path: str,
    reference_video_path: str,
    stroke: str = "freestyle",
) -> VideoComparison | None:
    """Compare user's swimming against a reference video using Gemini.

    Both videos are uploaded to Gemini and analyzed simultaneously. This
    gives the AI direct visual access to both swimmers' techniques for a
    concrete, observation-based comparison.

    Parameters
    ----------
    analyzer : GeminiSwimAnalyzer
        Initialized analyzer with a valid API key.
    user_video_path : str
        Path to the user's swimming video.
    reference_video_path : str
        Path to the reference video (e.g. Olympic footage).
    stroke : str
        The swimming stroke being performed.

    Returns
    -------
    VideoComparison or None
        Structured comparison results, or None if analysis failed.
    """
    from .prompts import REFERENCE_VIDEO_COMPARISON_PROMPT, SYSTEM_INSTRUCTION

    try:
        # Upload both videos
        user_file = analyzer._upload_video(user_video_path)
        ref_file = analyzer._upload_video(reference_video_path)

        if user_file is None or ref_file is None:
            logger.error("Failed to upload one or both videos")
            return None

        prompt = REFERENCE_VIDEO_COMPARISON_PROMPT.format(stroke=stroke)

        # Send both videos with the comparison prompt
        response = None
        for attempt in range(analyzer.config.max_retries):
            try:
                import time
                result = analyzer.client.models.generate_content(
                    model=analyzer.config.model,
                    contents=[
                        user_file,
                        "This is VIDEO 1 (User's swimming).",
                        ref_file,
                        "This is VIDEO 2 (Reference/elite swimmer).",
                        prompt,
                    ],
                    config={
                        "system_instruction": SYSTEM_INSTRUCTION,
                        "temperature": 0.3,
                    },
                )
                if result and result.text:
                    response = result.text
                    break
            except Exception as e:
                logger.warning("Comparison attempt %d failed: %s", attempt + 1, e)
                if attempt < analyzer.config.max_retries - 1:
                    time.sleep(analyzer.config.retry_delay_s * (attempt + 1))

        if response is None:
            return None

        return _parse_comparison(response)

    except Exception as e:
        logger.error("Video comparison failed: %s", e, exc_info=True)
        return None


def _parse_comparison(response: str) -> VideoComparison | None:
    """Parse the structured JSON comparison from Gemini."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                logger.error("Failed to parse comparison JSON")
                return None
        else:
            return None

    differences = [
        TechniqueDifference(
            element=d.get("element", ""),
            user_technique=d.get("user_technique", ""),
            reference_technique=d.get("reference_technique", ""),
            impact=d.get("impact", ""),
            drill_to_fix=d.get("drill_to_fix", ""),
            difficulty=d.get("difficulty", "moderate"),
        )
        for d in data.get("technique_differences", [])
    ]

    return VideoComparison(
        comparison_summary=data.get("comparison_summary", ""),
        technique_differences=differences,
        what_user_does_well=data.get("what_user_does_well", []),
        overall_technique_gap=data.get("overall_technique_gap", "moderate"),
        top_priority_change=data.get("top_priority_change", ""),
        raw_response=response,
    )
