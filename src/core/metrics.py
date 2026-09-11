"""
Canonical metrics serialization — the single source of truth for the
race-analysis numbers shown by *both* the desktop app and the web API.

Why this module exists
----------------------
The web backend (``web/worker.py``) and the desktop dashboard
(``src/gui/dashboard.py``) each used to derive summary figures — average
tracked velocity, the benchmark distance, the elite-benchmark comparison
string, rounded splits — with their own inline copies of the same little
formulae. Two copies of one formula drift: round one place differently, pick
``pool_length_m`` vs ``event_distance_m`` in one surface but not the other,
and the "same race" reports two different numbers.

``summarize_result`` computes those figures once, in the GUI-free core, and
returns a plain JSON-serialisable ``dict`` whose keys match the web API's
``RaceAnalysisMetrics`` fields exactly (so the web layer can build its Pydantic
model straight from it) while remaining equally consumable by the Qt dashboard.
Both surfaces call this; neither re-derives.

Nothing here may import PySide6/Qt or FastAPI — it sits in the shared core.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import AnalysisConfig, AnalysisResult
from ..analysis.benchmarks import compare_to_benchmark


def benchmark_distance_m(cfg: AnalysisConfig) -> int:
    """The event distance used for benchmark lookup.

    Prefer the user-selected event distance; fall back to the pool length
    (a single-length swim). Rounded to an int because benchmark tables are
    keyed by whole-metre events (50/100/200/…).
    """
    if cfg.event_distance_m:
        return int(cfg.event_distance_m)
    return int(round(cfg.pool_length_m))


def average_tracked_velocity(res: AnalysisResult) -> float | None:
    """Mean of the per-marker velocity profile, or None if unavailable.

    This is a tracked-velocity average (mean of the velocity-profile samples),
    not total_distance / total_time — kept identical across surfaces so the
    "Average tracked velocity" line matches everywhere.
    """
    vp = res.split_report.velocity_profile
    if not vp:
        return None
    return sum(v for _, v in vp) / len(vp)


def velocity_benchmark_line(
    res: AnalysisResult, distance_m: float, sex: str
) -> tuple[float | None, str | None]:
    """Shared primitive: average tracked velocity + its benchmark comparison.

    Both the web layer (via :func:`benchmark_comparison`) and the desktop
    dashboard call this so the averaging, the whole-metre distance coercion,
    and the ``compare_to_benchmark`` invocation are defined exactly once.

    Returns ``(avg_velocity_mps, comparison_string)``. The comparison is None
    when there is no velocity to compare, or when no verified figure is
    bundled for this stroke/distance/sex — the caller phrases that case.
    """
    avg_v = average_tracked_velocity(res)
    if avg_v is None:
        return None, None
    comparison = compare_to_benchmark(
        avg_v,
        res.stroke_classification.stroke,
        int(distance_m),
        sex=sex,
    )
    return avg_v, comparison


def benchmark_comparison(cfg: AnalysisConfig, res: AnalysisResult) -> str | None:
    """Elite-benchmark comparison string, or None when no figure applies.

    Returns None both when there is no velocity to compare and when
    ``compare_to_benchmark`` has no verified figure for this
    stroke/distance/sex combination — callers decide how to phrase the
    "no reference bundled" case for their surface.
    """
    _, comparison = velocity_benchmark_line(
        res, benchmark_distance_m(cfg), cfg.swimmer_sex
    )
    return comparison


def _ai_result_dict(res: AnalysisResult) -> dict[str, Any] | None:
    """Flatten the optional Gemini coaching result into a plain dict."""
    ai = res.ai_result
    if not ai:
        return None
    out: dict[str, Any] = {}
    if ai.error:
        out["error"] = ai.error
    if ai.technique_analysis:
        out["technique_analysis"] = asdict(ai.technique_analysis)
    if ai.elite_comparison:
        out["elite_comparison"] = asdict(ai.elite_comparison)
    if ai.improvement_plan:
        out["improvement_plan"] = asdict(ai.improvement_plan)
    if ai.race_strategy:
        out["race_strategy"] = asdict(ai.race_strategy)
    return out or None


def summarize_result(cfg: AnalysisConfig, res: AnalysisResult) -> dict[str, Any]:
    """Canonical, JSON-serialisable summary of an :class:`AnalysisResult`.

    Keys mirror the web API's ``RaceAnalysisMetrics`` model so the web layer
    can construct it directly (``RaceAnalysisMetrics(**summarize_result(...))``)
    and the desktop can read the same figures. This is the one place these
    derived numbers are defined.
    """
    sp = res.split_report
    avg_v = average_tracked_velocity(res)

    # The core Split carries only the cumulative time at each marker. The
    # per-segment "split time" is the delta from the previous marker (the first
    # marker's delta is measured from the race start at t=0). Computed here so
    # both surfaces present the same split deltas.
    splits = []
    prev_time = 0.0
    for s in sp.splits:
        split_time = s.time_s - prev_time
        splits.append({
            "marker_m": s.marker_m,
            "time_s": round(s.time_s, 3),
            "split_time_s": round(split_time, 3),
            "interpolated": s.interpolated,
        })
        prev_time = s.time_s

    low_conf = [
        {
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "start_time_s": round(seg.start_time_s, 2),
            "end_time_s": round(seg.end_time_s, 2),
            "reason": seg.reason,
            "resolved_by": seg.resolved_by,
        }
        for seg in res.low_confidence_segments
    ]

    turn_time = (
        round(res.turn_phase.turn_duration_s, 3)
        if res.turn_phase and res.turn_phase.turn_duration_s
        else None
    )

    return {
        "event_distance_m": float(cfg.event_distance_m or cfg.pool_length_m),
        "stroke": res.stroke_classification.stroke,
        "stroke_confidence": round(res.stroke_classification.confidence, 2),
        "start_method": res.start_detection.method,
        # Block reaction time is not measured: a single side-on camera cannot
        # honestly recover start-signal→first-movement latency (that needs a
        # block sensor or a clear view of the starting strobe). The API field
        # is kept for shape stability but reported as None rather than faked.
        "start_reaction_time_s": None,
        "start_15m_time_s": (
            round(res.start_phase.time_to_15m_s, 3)
            if res.start_phase.time_to_15m_s else None
        ),
        "turn_time_s": turn_time,
        "total_time_s": round(sp.total_time_s, 3) if sp.total_time_s else None,
        "avg_velocity_mps": round(avg_v, 3) if avg_v is not None else None,
        "splits": splits,
        "velocity_profile": [(round(d, 2), round(v, 3)) for d, v in sp.velocity_profile],
        "stroke_rates": [(round(t, 2), round(r, 1)) for t, r in res.stroke_rates],
        "stroke_lengths": [(round(t, 2), round(l, 2)) for t, l in res.stroke_lengths],
        "low_confidence_segments": low_conf,
        "gemini_assisted_frames": res.gemini_assisted_frames,
        "cv_only_frames": res.cv_only_frames,
        "discrepancy_count": (
            res.discrepancy_report.corrected_count
            + res.discrepancy_report.still_uncertain_count
        ),
        "ai_result": _ai_result_dict(res),
        "benchmark_comparison": benchmark_comparison(cfg, res),
    }
