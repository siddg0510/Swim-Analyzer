"""
Biomechanical metrics: stroke rate, stroke length, start phase, turn phase.

Important, stated plainly rather than buried: breakout distance and
underwater start/turn timing require the swimmer to actually be *visible*
underwater. Typical single side-deck consumer video (phone, GoPro on a
tripod, action-cam) usually loses the swimmer for part or all of the
underwater phase — glare, refraction, depth, and splash all work against
it. The professional systems that measure this reliably (e.g. the AIMsys
rig in Gonjo et al. 2025, Sci Rep) use multiple synchronised underwater
cameras built into the pool itself.

This module does NOT fabricate a breakout distance when the swimmer isn't
visibly tracked underwater. Instead it reports `tracking_confidence` per
phase, and the dashboard/report should show "insufficient underwater
visibility" rather than a number that looks precise but isn't grounded.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..config import MAX_LEGAL_UNDERWATER_M
from ..vision.pose_estimator import PoseFrame, LM_L_WRIST, LM_R_WRIST


@dataclass
class StrokeCycle:
    start_time_s: float
    end_time_s: float
    arm: str  # "left", "right", or "simultaneous"

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s


@dataclass
class StartPhaseMetrics:
    time_to_15m_s: float | None
    underwater_distance_m: float | None
    breakout_confidence: float  # 0..1; 0 means "not measurable from this footage"
    note: str = ""


@dataclass
class TurnPhaseMetrics:
    turn_duration_s: float | None
    wall_contact_time_s: float | None
    breakout_after_turn_m: float | None
    confidence: float
    note: str = ""


def detect_stroke_cycles(
    frames_with_time: list[tuple[float, PoseFrame]], min_cycle_s: float = 0.25,
) -> list[StrokeCycle]:
    """
    Count arm-entry cycles from wrist vertical position peaks. A "cycle"
    is defined as consecutive local minima in a wrist's y-position (i.e.
    the entry point of that arm), which is the standard event race
    analysts count from video.
    """
    times = np.array([t for t, _ in frames_with_time])
    l_wrist = np.array([
        f.landmarks_px.get(LM_L_WRIST, (np.nan, np.nan, 0))[1] for _, f in frames_with_time
    ])
    r_wrist = np.array([
        f.landmarks_px.get(LM_R_WRIST, (np.nan, np.nan, 0))[1] for _, f in frames_with_time
    ])

    cycles: list[StrokeCycle] = []
    for arm_name, series in (("left", l_wrist), ("right", r_wrist)):
        valid = ~np.isnan(series)
        if valid.sum() < 5:
            continue
        idx = np.where(valid)[0]
        vals = series[idx]
        t = times[idx]

        # Local minima = wrist near its highest point in the stroke (image
        # y grows downward, so "minimum y" = highest physical position =
        # hand entry for front-down strokes; for backstroke this instead
        # tends to land on the recovery peak, which is still one
        # consistent, countable event per cycle).
        minima = []
        for i in range(1, len(vals) - 1):
            if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
                minima.append(i)

        last_t = None
        for m in minima:
            if last_t is not None and (t[m] - last_t) < min_cycle_s:
                continue
            if last_t is not None:
                cycles.append(StrokeCycle(last_t, t[m], arm=arm_name))
            last_t = t[m]

    cycles.sort(key=lambda c: c.start_time_s)
    return cycles


def stroke_rate_series(
    cycles: list[StrokeCycle], window_s: float = 6.0,
) -> list[tuple[float, float]]:
    """Rolling stroke rate (strokes/min) sampled every cycle, averaged over
    a trailing window — smoother and more coach-readable than an
    instantaneous 1/duration figure."""
    if not cycles:
        return []
    out = []
    for c in cycles:
        window = [x for x in cycles if c.end_time_s - window_s <= x.end_time_s <= c.end_time_s]
        if len(window) < 2:
            continue
        span_s = window[-1].end_time_s - window[0].start_time_s
        if span_s <= 0:
            continue
        rate_cpm = 60.0 * len(window) / span_s
        out.append((c.end_time_s, rate_cpm))
    return out


def stroke_length_series(
    cycles: list[StrokeCycle], times_s: np.ndarray, distances_m: np.ndarray,
) -> list[tuple[float, float]]:
    """Distance covered per stroke cycle, in metres."""
    out = []
    for c in cycles:
        d0 = float(np.interp(c.start_time_s, times_s, distances_m))
        d1 = float(np.interp(c.end_time_s, times_s, distances_m))
        if c.duration_s > 0:
            out.append((c.end_time_s, d1 - d0))
    return out


def compute_start_phase(
    times_s: np.ndarray, distances_m: np.ndarray,
    pose_frames: list[tuple[float, PoseFrame]] | None = None,
) -> StartPhaseMetrics:
    idx15 = np.searchsorted(distances_m, 15.0)
    if idx15 == 0 or idx15 >= len(distances_m):
        return StartPhaseMetrics(None, None, 0.0, note="Never confidently reached 15 m.")
    time_to_15m = float(np.interp(15.0, distances_m, times_s))

    breakout_m, breakout_conf = _estimate_breakout_distance(
        times_s, distances_m, pose_frames, search_end_m=MAX_LEGAL_UNDERWATER_M,
    )
    note = "" if breakout_conf > 0.4 else (
        "Underwater breakout point not confidently visible in this footage; "
        "distance not reported to avoid showing a fabricated number."
    )
    return StartPhaseMetrics(
        time_to_15m_s=time_to_15m,
        underwater_distance_m=breakout_m if breakout_conf > 0.4 else None,
        breakout_confidence=breakout_conf, note=note,
    )


def _estimate_breakout_distance(
    times_s, distances_m, pose_frames, search_end_m: float,
) -> tuple[float | None, float]:
    """
    Breakout = first frame the pose estimator regains a confident head/
    shoulder landmark after the start/turn, within the legal underwater
    distance. If the pose estimator never regains confidence in that
    window (very common with a single side-deck camera — the swimmer is
    simply not visible), we return confidence 0 rather than guessing.
    """
    if not pose_frames:
        return None, 0.0

    window = [(t, f) for t, f in pose_frames if 0 <= t <= np.interp(search_end_m, distances_m, times_s)]
    for t, f in window:
        if f.present:
            visible_conf = np.mean([lm[2] for lm in f.landmarks_px.values()]) if f.landmarks_px else 0.0
            if visible_conf > 0.6:
                d = float(np.interp(t, times_s, distances_m))
                return d, float(min(visible_conf, 0.95))
    return None, 0.0


def compute_turn_phase(
    times_s: np.ndarray, distances_m: np.ndarray, wall_position_m: float,
    pose_frames: list[tuple[float, PoseFrame]] | None = None,
    pre_wall_m: float = 5.0, post_wall_m: float = 10.0,
) -> TurnPhaseMetrics:
    """
    Turn duration: 5 m before the wall through 10 m after breakout, per
    spec. (Published race-analysis convention typically anchors the
    "turn-in" to 5 m before the wall too — see README Sources — so this
    matches standard usage, not just this project's own definition.)
    """
    start_d = wall_position_m - pre_wall_m
    end_d = wall_position_m + post_wall_m

    if end_d > distances_m[-1] or start_d < distances_m[0]:
        return TurnPhaseMetrics(None, None, None, 0.0,
                                 note="Turn window falls outside tracked range.")

    t_start = float(np.interp(start_d, distances_m, times_s))
    t_end = float(np.interp(end_d, distances_m, times_s))
    t_wall = float(np.interp(wall_position_m, distances_m, times_s))

    breakout_m, breakout_conf = _estimate_breakout_distance(
        times_s - t_wall, distances_m - wall_position_m,
        [(t - t_wall, f) for t, f in (pose_frames or [])],
        search_end_m=MAX_LEGAL_UNDERWATER_M,
    )

    return TurnPhaseMetrics(
        turn_duration_s=t_end - t_start,
        wall_contact_time_s=t_wall,
        breakout_after_turn_m=breakout_m if breakout_conf > 0.4 else None,
        confidence=breakout_conf,
        note="" if breakout_conf > 0.4 else "Post-turn breakout not confidently visible.",
    )
