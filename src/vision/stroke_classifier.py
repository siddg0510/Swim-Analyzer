"""
Auto-event (stroke) recognition.

This is a rule-based heuristic classifier grounded in the actual
biomechanical differences between the four strokes, NOT a trained deep
model — there is no ready-made, verified "stroke classifier" model to
bundle offline, and training one would need a labelled swim-video dataset
this project doesn't have. Treat its output as a strong first guess the
user can override in the UI, not a certified detector; it has not been
validated against real race footage.

Signals used, and why they distinguish the four strokes:
  - Arm phase correlation (left wrist vs right wrist vertical motion):
    near +1 for simultaneous-arm strokes (breaststroke, butterfly),
    near -1 for alternating strokes (freestyle, backstroke).
  - Body orientation (nose vs. shoulder-hip midline depth/position):
    backstroke swims face-up, the other three face-down. We approximate
    this with shoulder-to-hip vertical ordering and nose visibility
    patterns rather than true depth, since we only have a 2D side view.
  - Leg/ankle phase correlation: breaststroke's whip kick is symmetric
    (ankles move together, like the arms); butterfly's dolphin kick is
    also symmetric but at roughly double the arm-cycle frequency; front
    crawl/backstroke flutter kicks are alternating and fast.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .pose_estimator import (
    PoseFrame, LM_L_WRIST, LM_R_WRIST, LM_L_ANKLE, LM_R_ANKLE,
    LM_L_SHOULDER, LM_R_SHOULDER, LM_L_HIP, LM_R_HIP, LM_NOSE,
)


@dataclass
class StrokeClassification:
    stroke: str
    confidence: float
    votes: dict[str, float]


def _series(frames: list[PoseFrame], landmark_id: int, axis: int) -> np.ndarray:
    vals = []
    for f in frames:
        if f.present and landmark_id in f.landmarks_px:
            vals.append(f.landmarks_px[landmark_id][axis])
        else:
            vals.append(np.nan)
    arr = np.array(vals, dtype=float)
    # Forward-fill short gaps so a couple of missed-detection frames don't
    # break the correlation window; leaves long gaps as NaN.
    mask = np.isnan(arr)
    if mask.all():
        return arr
    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 5:
        return 0.0
    a, b = a[valid], b[valid]
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def classify_window(frames: list[PoseFrame]) -> StrokeClassification:
    """
    Classify a ~2-3 second rolling window of pose frames (60-100 frames at
    typical race-video frame rates) into one of the four strokes.
    """
    if len(frames) < 15:
        return StrokeClassification("unknown", 0.0, {})

    l_wrist_y = _series(frames, LM_L_WRIST, 1)
    r_wrist_y = _series(frames, LM_R_WRIST, 1)
    l_ankle_y = _series(frames, LM_L_ANKLE, 1)
    r_ankle_y = _series(frames, LM_R_ANKLE, 1)
    nose_y = _series(frames, LM_NOSE, 1)
    l_sh_y = _series(frames, LM_L_SHOULDER, 1)
    r_sh_y = _series(frames, LM_R_SHOULDER, 1)
    l_hip_y = _series(frames, LM_L_HIP, 1)
    r_hip_y = _series(frames, LM_R_HIP, 1)

    arm_corr = _safe_corr(l_wrist_y, r_wrist_y)
    leg_corr = _safe_corr(l_ankle_y, r_ankle_y)

    # Crude face-up/face-down proxy: in front-down strokes the nose sits
    # close to (often slightly below, mid-breath) the shoulder line; on
    # the back the nose sits well clear of it, and shoulders/hips keep a
    # visibly different relative spacing because the body roll axis
    # flips. This is a coarse proxy, not a solved inverse-kinematics
    # estimate — flagged as such in the module docstring.
    shoulder_mid = (l_sh_y + r_sh_y) / 2.0
    hip_mid = (l_hip_y + r_hip_y) / 2.0
    nose_offset = np.nanmean(nose_y - shoulder_mid)
    face_up_score = float(np.tanh(nose_offset / 30.0))  # >0 suggests supine

    votes = {s: 0.0 for s in ("freestyle", "backstroke", "breaststroke", "butterfly")}

    # Arm phase is the strongest first split.
    if arm_corr > 0.25:
        votes["breaststroke"] += 1.0
        votes["butterfly"] += 1.0
    elif arm_corr < -0.15:
        votes["freestyle"] += 1.0
        votes["backstroke"] += 1.0
    else:
        for s in votes:
            votes[s] += 0.25  # genuinely ambiguous, spread weakly

    # Body orientation splits each pair.
    if face_up_score > 0.15:
        votes["backstroke"] += 1.2
        votes["freestyle"] -= 0.4
    elif face_up_score < -0.15:
        votes["freestyle"] += 0.6
        votes["butterfly"] += 0.3
        votes["breaststroke"] += 0.3
        votes["backstroke"] -= 0.4

    # Leg symmetry splits breaststroke/butterfly.
    if leg_corr > 0.2:
        votes["breaststroke"] += 0.6
        votes["butterfly"] += 0.6
    elif leg_corr < -0.1:
        votes["freestyle"] += 0.4
        votes["backstroke"] += 0.4

    total = sum(max(v, 0.0) for v in votes.values()) or 1.0
    norm_votes = {k: max(v, 0.0) / total for k, v in votes.items()}
    best = max(norm_votes, key=norm_votes.get)
    return StrokeClassification(best, norm_votes[best], norm_votes)


def classify_from_motion(times_s: np.ndarray, dist_m: np.ndarray, y_px: list[float], fps: float = 30.0) -> StrokeClassification:
    """
    Fallback classifier that uses ONLY the 2D trajectory of the cap.
    Used when pose landmarks are unavailable (e.g. model missing or swimmer underwater).
    """
    import scipy.signal
    
    if len(y_px) < int(fps * 2):
        return StrokeClassification("unknown", 0.0, {})
        
    y_arr = np.array(y_px)
    window = max(5, int(fps/5))
    if window % 2 == 0: window += 1
    
    y_smooth = scipy.signal.savgol_filter(y_arr, window_length=window, polyorder=2)
    y_detrend = scipy.signal.detrend(y_smooth)
    
    # Check for strong periodicity in vertical motion
    peaks, _ = scipy.signal.find_peaks(y_detrend, distance=fps/2)
    troughs, _ = scipy.signal.find_peaks(-y_detrend, distance=fps/2)
    
    votes = {s: 0.0 for s in ("freestyle", "backstroke", "breaststroke", "butterfly")}
    
    if len(peaks) > 1 and len(troughs) > 1:
        amp = np.mean(y_detrend[peaks]) - np.mean(y_detrend[troughs])
        if amp > 12.0:  # significant vertical motion (butterfly/breaststroke bob)
            votes["butterfly"] += 1.0
            votes["breaststroke"] += 0.8
        else:
            votes["freestyle"] += 1.0
            votes["backstroke"] += 1.0
            
    # breaststroke has strong surge in forward velocity (distance_m)
    dt = np.diff(times_s)
    dd = np.diff(dist_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dt > 1e-6, dd / dt, 0.0)
        
    if len(v) > int(fps):
        v_window = max(5, int(fps/3))
        if v_window % 2 == 0: v_window += 1
        v_smooth = scipy.signal.savgol_filter(v, window_length=v_window, polyorder=2)
        v_std = np.std(v_smooth)
        
        if v_std > 0.4: # breaststroke velocity surges massively
            votes["breaststroke"] += 1.2
        elif v_std < 0.2: # freestyle/backstroke are much smoother
            votes["freestyle"] += 0.6
            votes["backstroke"] += 0.6
            
    total = sum(max(v, 0.0) for v in votes.values()) or 1.0
    norm_votes = {k: max(v, 0.0) / total for k, v in votes.items()}
    best = max(norm_votes, key=norm_votes.get) if total > 1.0 else "unknown"
    confidence = norm_votes.get(best, 0.0) * 0.6 # lower confidence for fallback
    
    return StrokeClassification(best, confidence, norm_votes)


def classify_with_gemini(
    video_path: str,
    api_key: str | None = None,
) -> StrokeClassification:
    """Use Gemini as a high-confidence stroke classifier fallback.

    This is called when neither the pose-based nor motion-based classifiers
    produce a confident result. Requires an API key and internet access.
    Falls back to 'unknown' if Gemini is unavailable.
    """
    try:
        from ..ai.gemini_analyzer import GeminiSwimAnalyzer

        analyzer = GeminiSwimAnalyzer(api_key=api_key)
        stroke, confidence = analyzer.identify_stroke(video_path)
        analyzer.cleanup()

        if stroke in ("freestyle", "backstroke", "breaststroke", "butterfly"):
            return StrokeClassification(
                stroke=stroke,
                confidence=confidence,
                votes={stroke: confidence},
            )
    except Exception:
        pass

    return StrokeClassification("unknown", 0.0, {})

