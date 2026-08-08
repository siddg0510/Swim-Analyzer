"""
Split-time computation.

Takes the calibrated distance-vs-time trace (one distance-in-metres value
per frame, from the cap tracker + calibration) and the race-start time
(from audio/visual detection), and computes:
  - split times at each marker in config.SPLIT_MARKERS_M
  - the final-5m and final-15m micro-splits
  - a continuous average-velocity profile

Sub-frame timing: when the swimmer's tracked distance crosses a marker
between frame N and frame N+1, we linearly interpolate the crossing time
rather than snapping to the nearer frame. This is standard practice in
video-based race analysis and is meaningfully more accurate than
frame-snapping, but it is still an estimate bounded by the video's frame
rate — not a hardware timestamp. A 30fps video interpolating within a
33ms window is not the same claim as a 30fps video measuring to the
millisecond; the code and the exported report both say "interpolated,"
not "exact."
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..config import SPLIT_MARKERS_M, FINISH_ZONE_M, CLOSING_ZONE_M


@dataclass
class Split:
    marker_m: float
    time_s: float
    interpolated: bool


@dataclass
class SplitReport:
    splits: list[Split]
    final_5m_s: float | None
    final_15m_s: float | None
    total_time_s: float | None
    velocity_profile: list[tuple[float, float]]  # (distance_m, velocity_mps)


def _interpolate_crossing(t0, d0, t1, d1, target_d) -> tuple[float, bool]:
    if d1 == d0:
        return t0, False
    frac = (target_d - d0) / (d1 - d0)
    frac = float(np.clip(frac, 0.0, 1.0))
    return t0 + frac * (t1 - t0), True


def compute_splits(
    times_s: np.ndarray, distances_m: np.ndarray, pool_length_m: float,
) -> SplitReport:
    """
    times_s, distances_m: equal-length arrays, one entry per analysed
    frame, already zeroed so t=0 is the detected race start and distance
    is cumulative metres travelled along the full race (i.e. already
    unwrapped across turns, not reset to 0 every lap).
    """
    if len(times_s) < 2:
        raise ValueError("Need at least 2 tracked frames to compute splits.")

    order = np.argsort(times_s)
    times_s = times_s[order]
    distances_m = distances_m[order]

    splits: list[Split] = []
    markers = [m for m in SPLIT_MARKERS_M if m <= pool_length_m] or SPLIT_MARKERS_M
    for marker in markers:
        idx = np.searchsorted(distances_m, marker)
        if idx == 0:
            if distances_m[0] >= marker:
                splits.append(Split(marker, float(times_s[0]), interpolated=False))
            continue
        if idx >= len(distances_m):
            continue  # swimmer never confidently reached this marker
        t, interp = _interpolate_crossing(
            times_s[idx - 1], distances_m[idx - 1], times_s[idx], distances_m[idx], marker,
        )
        splits.append(Split(marker, t, interp))

    total_time_s = float(times_s[-1]) if distances_m[-1] >= pool_length_m - 0.25 else None

    final_5m_s = _closing_split_time(times_s, distances_m, pool_length_m, FINISH_ZONE_M)
    final_15m_s = _closing_split_time(times_s, distances_m, pool_length_m, CLOSING_ZONE_M)

    velocity_profile = _velocity_profile(times_s, distances_m)

    return SplitReport(
        splits=splits, final_5m_s=final_5m_s, final_15m_s=final_15m_s,
        total_time_s=total_time_s, velocity_profile=velocity_profile,
    )


def _closing_split_time(times_s, distances_m, pool_length_m, zone_m) -> float | None:
    zone_start = pool_length_m - zone_m
    idx = np.searchsorted(distances_m, zone_start)
    if idx == 0 or idx >= len(distances_m):
        return None
    t_start, _ = _interpolate_crossing(
        times_s[idx - 1], distances_m[idx - 1], times_s[idx], distances_m[idx], zone_start,
    )
    if distances_m[-1] < pool_length_m - 0.25:
        return None  # never confidently finished; don't report a fake split
    return float(times_s[-1] - t_start)


def _velocity_profile(times_s, distances_m, smoothing_window: int = 5) -> list[tuple[float, float]]:
    if len(times_s) < smoothing_window + 1:
        return []
    dt = np.diff(times_s)
    dd = np.diff(distances_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        inst_v = np.where(dt > 1e-6, dd / dt, np.nan)

    kernel = np.ones(smoothing_window) / smoothing_window
    valid = ~np.isnan(inst_v)
    smoothed = np.full_like(inst_v, np.nan)
    if valid.sum() >= smoothing_window:
        smoothed[valid] = np.convolve(inst_v[valid], kernel, mode="same")

    mid_distances = (distances_m[:-1] + distances_m[1:]) / 2.0
    return [
        (float(d), float(v)) for d, v in zip(mid_distances, smoothed) if not np.isnan(v)
    ]
