"""
Shared metric labels, units, and terminology.

Both the desktop dashboard and the web frontend import these so the
displayed metric names cannot drift between the two surfaces.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Metric display labels — used by both dashboards
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    # Splits
    "split_time": "Split Time",
    "marker_m": "Distance (m)",
    "time_s": "Time (s)",
    "interpolated": "Interpolated",
    "final_5m": "Final 5 m Split",
    "final_15m": "Final 15 m Split",
    "total_time": "Total Time",

    # Velocity
    "velocity_mps": "Velocity (m/s)",
    "distance_m": "Distance (m)",

    # Stroke metrics
    "stroke_rate_cpm": "Stroke Rate (cycles/min)",
    "stroke_length_m": "Stroke Length (m/cycle)",

    # Start/turn phase
    "time_to_15m": "Time to 15 m",
    "underwater_distance": "Underwater Distance (m)",
    "turn_duration": "Turn Duration (s)",
    "breakout_after_turn": "Breakout After Turn (m)",

    # Tracking quality
    "confidence": "Tracking Confidence",
    "method": "Detection Method",
    "corrected_frames": "Auto-Corrected Frames",
    "uncertain_frames": "Still Uncertain Frames",

    # Confidence segment sources
    "source_detector": "CV Detector",
    "source_ekf_predict": "EKF Coast (Predicted)",
    "source_gemini_recovery": "Gemini-Assisted",
    "source_color": "Color Tracker",
    "source_kalman_predict": "Kalman Coast",
    "source_pose_fallback": "Pose Fallback",
    "source_backward_reconciled": "Backward-Pass Reconciled",
    "source_kalman_interpolated": "Kalman Interpolated",
}


# ---------------------------------------------------------------------------
# Confidence tier labels — for dashboard badges
# ---------------------------------------------------------------------------

CONFIDENCE_TIERS = {
    "high": {"label": "High Confidence", "color": "#27ae60", "icon": "🟢", "threshold": 0.7},
    "medium": {"label": "Medium Confidence", "color": "#f39c12", "icon": "🟡", "threshold": 0.4},
    "low": {"label": "Low Confidence", "color": "#e74c3c", "icon": "🔴", "threshold": 0.0},
}


def confidence_tier(value: float) -> str:
    """Return the tier key ('high', 'medium', 'low') for a confidence value."""
    if value >= CONFIDENCE_TIERS["high"]["threshold"]:
        return "high"
    elif value >= CONFIDENCE_TIERS["medium"]["threshold"]:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Stage labels — progress stages shown during analysis
# ---------------------------------------------------------------------------

ANALYSIS_STAGES = [
    (2, "Opening video…"),
    (5, "Detecting race start…"),
    (10, "Tracking swimmer…"),
    (80, "Tracking complete"),
    (82, "Checking for tracking discrepancies…"),
    (85, "Resolving low-confidence segments (Gemini)…"),
    (88, "Computing splits…"),
    (90, "Classifying stroke…"),
    (93, "Computing stroke rate and length…"),
    (96, "Computing start/turn phase metrics…"),
    (97, "Running AI analysis…"),
    (99, "Running DTW biomechanical comparison…"),
    (100, "Done."),
]
