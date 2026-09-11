"""
Parse manual-calibration input from the web frontend into core
``CalibrationKeyframe`` objects.

The web calibration picker (frontend) lets a user click lane corners and
distance-labelled reference points on a video frame, exactly like the desktop
``CalibrationDialog``. Those clicks are posted to the API as a JSON string in a
multipart form field (``calibration_json``) because multipart forms cannot carry
nested structured values directly. This module validates that JSON and converts
it into the same ``CalibrationKeyframe`` list the desktop produces, so both
surfaces feed the shared pipeline identically.

Accepted JSON shapes (either is fine)::

    {"keyframes": [ <keyframe>, ... ]}      # preferred
    [ <keyframe>, ... ]                     # bare list
    <keyframe>                              # single object

where ``<keyframe>`` is::

    {
      "frame_idx": 0,                                   # optional, default 0
      "reference_points": [ [[x, y], meters], ... ],    # required, >=1 here
      "lane_polygon_px": [ [x, y], ... ],               # optional
      "lane_number": 4                                  # optional
    }

Validation errors raise :class:`CalibrationParseError` with a human-readable
message the API turns into an HTTP 400 — never a 500.
"""
from __future__ import annotations

import json
from typing import Any

from src.vision.calibration import CalibrationKeyframe

# The pipeline needs at least this many reference points to solve any usable
# pixel->metre mapping (2 = linear fallback, 4+ = homography). Kept in sync with
# the graceful-fail guard in src/core/pipeline.py.
MIN_REFERENCE_POINTS = 2


class CalibrationParseError(ValueError):
    """Raised when posted calibration JSON is malformed or incomplete."""


def _as_xy(value: Any, ctx: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CalibrationParseError(f"{ctx} must be a [x, y] pair, got {value!r}")
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        raise CalibrationParseError(f"{ctx} must contain two numbers, got {value!r}")


def _parse_reference_point(value: Any, ctx: str) -> tuple[tuple[float, float], float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CalibrationParseError(
            f"{ctx} must be [[x, y], meters], got {value!r}"
        )
    pixel = _as_xy(value[0], f"{ctx} pixel")
    try:
        meters = float(value[1])
    except (TypeError, ValueError):
        raise CalibrationParseError(f"{ctx} distance must be a number, got {value[1]!r}")
    return pixel, meters


def _parse_keyframe(obj: Any, idx: int) -> CalibrationKeyframe:
    if not isinstance(obj, dict):
        raise CalibrationParseError(f"keyframe #{idx} must be an object, got {type(obj).__name__}")

    raw_refs = obj.get("reference_points", [])
    if not isinstance(raw_refs, list):
        raise CalibrationParseError(f"keyframe #{idx}: reference_points must be a list")
    reference_points = [
        _parse_reference_point(r, f"keyframe #{idx} reference_points[{j}]")
        for j, r in enumerate(raw_refs)
    ]

    raw_lane = obj.get("lane_polygon_px", []) or []
    if not isinstance(raw_lane, list):
        raise CalibrationParseError(f"keyframe #{idx}: lane_polygon_px must be a list")
    lane_polygon_px = [
        _as_xy(p, f"keyframe #{idx} lane_polygon_px[{j}]")
        for j, p in enumerate(raw_lane)
    ]

    frame_idx = obj.get("frame_idx", 0)
    try:
        frame_idx = int(frame_idx)
    except (TypeError, ValueError):
        raise CalibrationParseError(f"keyframe #{idx}: frame_idx must be an integer")
    if frame_idx < 0:
        raise CalibrationParseError(f"keyframe #{idx}: frame_idx must be >= 0")

    lane_number = obj.get("lane_number")
    if lane_number is not None:
        try:
            lane_number = int(lane_number)
        except (TypeError, ValueError):
            raise CalibrationParseError(f"keyframe #{idx}: lane_number must be an integer")

    return CalibrationKeyframe(
        frame_idx=frame_idx,
        reference_points=reference_points,
        lane_polygon_px=lane_polygon_px,
        cap_color=None,
        lane_number=lane_number,
    )


def parse_calibration_keyframes(raw: str | None) -> list[CalibrationKeyframe]:
    """Parse the ``calibration_json`` form field into ``CalibrationKeyframe``s.

    Returns an empty list for empty/blank input (the AI-auto-calibration path).
    Raises :class:`CalibrationParseError` on any malformed structure.
    """
    if raw is None or not raw.strip():
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CalibrationParseError(f"not valid JSON: {exc}")

    if isinstance(data, dict) and "keyframes" in data:
        keyframes_raw = data["keyframes"]
    elif isinstance(data, dict):
        keyframes_raw = [data]           # a single bare keyframe object
    elif isinstance(data, list):
        keyframes_raw = data
    else:
        raise CalibrationParseError("expected an object or a list of keyframes")

    if not isinstance(keyframes_raw, list):
        raise CalibrationParseError("'keyframes' must be a list")

    return [_parse_keyframe(kf, i) for i, kf in enumerate(keyframes_raw)]


def has_usable_manual_calibration(keyframes: list[CalibrationKeyframe]) -> bool:
    """True if any keyframe carries enough reference points to calibrate.

    Mirrors the pipeline's requirement so the API can reject a doomed job early
    (with a clear message) instead of uploading a large file only to fail.
    """
    return any(len(kf.reference_points) >= MIN_REFERENCE_POINTS for kf in keyframes)
