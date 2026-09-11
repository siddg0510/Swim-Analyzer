"""
Core analysis pipeline — no GUI dependencies.

This module contains the actual analysis logic extracted from the original
QThread-based AnalysisWorker. It can be called by both the desktop app
(via a QThread wrapper) and the web backend (via an async job runner).

The only interface to the caller is the ``progress_callback`` parameter,
which accepts a simple ``(percent: int, message: str)`` callable. The Qt
version wraps this to emit signals; the web version wraps it to update a
job record.
"""
from __future__ import annotations

import tempfile
import os
import json
import logging
from typing import Callable

import numpy as np
import cv2

from .models import (
    AnalysisConfig, AnalysisResult, AIAnalysisResult,
    LowConfidenceSegment, ProgressCallback,
)
from ..config import DEFAULT_VIDEO_FPS_FALLBACK
from ..vision.calibration import PoolCalibrator, CalibrationResult, CalibrationKeyframe
from ..vision.cap_tracker import CapTracker, FinishLineOpticalFlowTracker
from ..vision.pose_estimator import PoseEstimator, PoseFrame
from ..vision.ego_motion import EgoMotionTracker
from ..vision.stroke_classifier import classify_window, StrokeClassification
from ..audio.start_detector import (
    extract_audio_wav, detect_start_from_audio, detect_start_visual_fallback,
    StartDetectionResult,
)
from ..analysis.splits import compute_splits, SplitReport
from ..analysis.biomechanics import (
    detect_stroke_cycles, stroke_rate_series, stroke_length_series,
    compute_start_phase, compute_turn_phase, StartPhaseMetrics, TurnPhaseMetrics,
)
from ..analysis.error_correction import (
    TrackPoint, flag_outliers, reconcile_with_backward_pass, DiscrepancyReport,
)
from ..analysis.comparator import DTWResult, run_dtw_comparison
from ..config import FINISH_ZONE_M, SPLASH_CONFIDENCE_GATE, SPLASH_RECOVERY_FRAMES, MULTI_SWIMMER_YOLO_MODEL

logger = logging.getLogger(__name__)


def _noop_progress(pct: int, msg: str) -> None:
    """Default no-op progress callback."""
    pass


def run_analysis(
    cfg: AnalysisConfig,
    progress_callback: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> AnalysisResult:
    """
    Run the full swim analysis pipeline.

    Args:
        cfg: Analysis configuration (video path, pool length, etc.)
        progress_callback: Optional callback for progress updates.
            Signature: callback(percent: int, status_message: str)
        cancel_check: Optional callable that returns True if the job should
            be cancelled. Checked periodically during processing.

    Returns:
        AnalysisResult with all computed metrics.

    Raises:
        RuntimeError: If the video cannot be opened or is cancelled.
    """
    progress = progress_callback or _noop_progress
    is_cancelled = cancel_check or (lambda: False)

    progress(2, "Opening video…")
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {cfg.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FPS_FALLBACK
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resolve Gemini API key if not explicitly passed
    if cfg.gemini_api_key is None:
        try:
            from ..ai.gemini_config import get_api_key
            cfg.gemini_api_key = get_api_key()
        except Exception:
            pass
    if cfg.gemini_api_key and not cfg.enable_ai:
        cfg.enable_ai = True

    from ..tracking.tracker import SplashRecoveryAgent
    recovery_agent = SplashRecoveryAgent(
        api_key=cfg.gemini_api_key if cfg.enable_ai else None,
        model=cfg.gemini_model,
    ) if (cfg.enable_ai and cfg.gemini_api_key) else None

    # ---- 1. Start detection (audio, then visual fallback) ---------
    progress(5, "Detecting race start…")
    start_result = _detect_start(cfg.video_path)

    # ---- 2. Frame-by-frame cap tracking + pose --------------------
    progress(10, "Tracking swimmer…")

    keyframes = sorted(cfg.calibration_keyframes, key=lambda k: k.frame_idx)
    ai_splits = None

    if not keyframes:
        if cfg.enable_ai and cfg.gemini_api_key:
            progress(7, "🤖 AI Auto-Calibration: Detecting pool geometry…")
            from ..ai.gemini_analyzer import GeminiSwimAnalyzer
            analyzer = GeminiSwimAnalyzer(api_key=cfg.gemini_api_key, model=cfg.gemini_model)

            ai_calib = analyzer.detect_pool_calibration(
                cfg.video_path, cfg.pool_length_m
            )

            progress(8, "🤖 AI Auto-Calibration: Detecting split timestamps…")
            target_desc = f"swimmer ID {cfg.target_id}" if cfg.target_id is not None else (f"with {cfg.cap_color} cap" if cfg.cap_color else "swimmer")
            ai_splits = analyzer.detect_splits(cfg.video_path, target_desc, cfg.pool_length_m)
            analyzer.cleanup()

            if ai_calib and len(ai_calib.reference_points) >= 2:
                auto_kf = CalibrationKeyframe(
                    frame_idx=0,
                    reference_points=ai_calib.reference_points,
                    lane_polygon_px=ai_calib.lane_polygon_px,
                )
                keyframes = [auto_kf]

        if not keyframes:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            dummy_kf = CalibrationKeyframe(
                frame_idx=0,
                reference_points=[],
                lane_polygon_px=[(0, 0), (w, 0), (w, h), (0, h)]
            )
            keyframes = [dummy_kf]

    current_kf_idx = 0
    active_kf = keyframes[0]

    calibrator = PoolCalibrator()
    if active_kf.reference_points:
        for pixel, dist in active_kf.reference_points:
            calibrator.add_point(pixel, dist)
        calibrator.solve()

    # ── Calibration presence guard (graceful fail) ───────────────────────────
    # A monocular pixel→metre mapping needs either (a) reference points — from
    # the manual picker or AI auto-calibration — or (b) AI split timestamps we
    # can interpolate distance from. With NEITHER, no distance/velocity/split
    # metric can exist. Rather than crash deep in the frame loop when
    # pixel_to_distance() hits an unsolved calibrator, fail up front with an
    # actionable message (product decision: manual UI + AI + graceful fail).
    calibration_solved = calibrator.result is not None
    have_ai_splits = bool(ai_splits and getattr(ai_splits, "splits", None))
    if not calibration_solved and not have_ai_splits:
        cap.release()  # release the capture handle before the early exit
        raise RuntimeError(
            "No usable calibration. Provide at least 2 calibration reference "
            "points (manual lane / reference-point picker) or enable AI "
            "auto-calibration with a valid Gemini API key. Without either, "
            "pixels cannot be mapped to metres and no distance or split "
            "metrics can be produced."
        )

    def _calib_distance(x_shifted: float, y_shifted: float) -> float:
        """Distance-along-lane for a camera-motion-corrected pixel.

        Uses the solved calibrator when one exists; when only AI splits are
        available (no geometry), returns 0.0 here — the authoritative per-frame
        distances for that path are derived post-loop from the AI split
        timeline (see ``get_distance_for_point``). This keeps the in-loop phase
        heuristic from crashing on an unsolved calibrator.
        """
        if calibrator.result is not None:
            return calibrator.pixel_to_distance(x_shifted, y_shifted)
        return 0.0

    initial_color = active_kf.cap_color or cfg.cap_color or "yellow"
    tracker = CapTracker(initial_color, active_kf.lane_polygon_px, recovery_agent=recovery_agent)
    pose_estimator = PoseEstimator()

    # ── Swimmer tracking mode (Primary Pipeline) ──────────────
    anti_splash_tracker = None
    if cfg.enable_multi_swimmer:
        from ..detection.detector import build_detector
        from ..detection.selector import TargetSelector
        from ..tracking.tracker import AntiSplashTracker
        progress(10, "🏊 Initializing swimmer tracking engine…")
        _detector = build_detector(yolo_model=MULTI_SWIMMER_YOLO_MODEL)
        _selector = TargetSelector(target_id=cfg.target_id)
        anti_splash_tracker = AntiSplashTracker(
            detector=_detector,
            selector=_selector,
            recovery_agent=recovery_agent,
            confidence_gate=SPLASH_CONFIDENCE_GATE,
            recovery_frame_threshold=SPLASH_RECOVERY_FRAMES,
        )
        logger.info(
            "Swimmer tracking engine active: target_id=%s, recovery=%s",
            cfg.target_id,
            "Gemini" if (recovery_agent and recovery_agent.available) else "EKF-only",
        )

    frames_cache: dict[int, np.ndarray] = {}
    track_points: list[TrackPoint] = []
    pose_frames: list[tuple[float, PoseFrame]] = []
    finish_tracker = FinishLineOpticalFlowTracker(lane_axis_is_x=True)
    finish_active = False

    # Track low-confidence segments
    low_conf_segments: list[LowConfidenceSegment] = []
    _coast_start_frame: int | None = None
    _coast_start_time: float | None = None
    gemini_assisted_count = 0
    cv_only_count = 0

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(start_result.start_time_s * 1000 - 500, 0))

    ego_tracker = None
    last_distance_m: float = 0.0
    while True:
        if is_cancelled():
            raise RuntimeError("Cancelled by user.")
        ok, frame = cap.read()
        if not ok:
            break

        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - start_result.start_time_s

        # Determine swim phase for adaptive Kalman process noise
        current_phase = "steady"
        if t_s < 3.0 or last_distance_m < 7.5:
            current_phase = "start"
        elif last_distance_m < 15.0:
            current_phase = "breakout"
        elif cfg.wall_position_m is not None and abs(last_distance_m - cfg.wall_position_m) < 5.0:
            current_phase = "turn"

        # Rolling cache of 450 frames (~15s at 30fps) for backward pass
        frames_cache[frame_idx] = frame
        if frame_idx - 450 in frames_cache:
            del frames_cache[frame_idx - 450]

        # Check if we crossed a new keyframe
        if current_kf_idx + 1 < len(keyframes) and frame_idx >= keyframes[current_kf_idx + 1].frame_idx:
            current_kf_idx += 1
            active_kf = keyframes[current_kf_idx]

            # Rebuild calibrator for the new keyframe
            calibrator = PoolCalibrator()
            if active_kf.reference_points:
                for pixel, dist in active_kf.reference_points:
                    calibrator.add_point(pixel, dist)
                calibrator.solve()

            # Update lane constraints and cap color for cap tracker
            if active_kf.cap_color:
                tracker = CapTracker(active_kf.cap_color, active_kf.lane_polygon_px, recovery_agent=recovery_agent)
            else:
                tracker.lane_polygon = np.array(active_kf.lane_polygon_px, dtype=np.int32)

            # Force EgoMotionTracker to re-initialize on this frame, anchoring drift to 0
            ego_tracker = None

        if anti_splash_tracker is not None:
            anti_splash_tracker.set_phase(current_phase)
            # ── Multi-swimmer branch: use EKF + YOLO instead of CapTracker
            ast_result = anti_splash_tracker.process_frame(frame, frame_idx)
            # Map AntiSplashTracker result to the same TrackedPoint API
            from ..vision.cap_tracker import TrackedPoint as _TP
            tp = _TP(
                frame_idx=frame_idx,
                x_px=ast_result.x_px,
                y_px=ast_result.y_px,
                confidence=ast_result.confidence,
                source=ast_result.source,
            )
            # Seed the EKF on the very first detected frame
            if not anti_splash_tracker._ekf._initialized and not (np.isnan(tp.x_px) or np.isnan(tp.y_px)):
                anti_splash_tracker.seed(tp.x_px, tp.y_px)

            # Track confidence segments
            if ast_result.source == "gemini_recovery":
                gemini_assisted_count += 1
            else:
                cv_only_count += 1

            # Coast segment tracking
            if ast_result.confidence < SPLASH_CONFIDENCE_GATE:
                if _coast_start_frame is None:
                    _coast_start_frame = frame_idx
                    _coast_start_time = t_s
            else:
                if _coast_start_frame is not None:
                    resolved_by = "gemini" if ast_result.source == "gemini_recovery" else "reacquisition"
                    low_conf_segments.append(LowConfidenceSegment(
                        start_frame=_coast_start_frame,
                        end_frame=frame_idx - 1,
                        start_time_s=_coast_start_time or 0.0,
                        end_time_s=t_s,
                        reason="coast",
                        resolved_by=resolved_by,
                    ))
                    _coast_start_frame = None
                    _coast_start_time = None
        else:
            tracker.set_phase(current_phase)
            tp = tracker.track_frame(frame, frame_idx)
            if tp.source == "gemini_recovery":
                gemini_assisted_count += 1
            else:
                cv_only_count += 1

            # Coast segment tracking for CapTracker
            if tp.confidence < 0.3:
                if _coast_start_frame is None:
                    _coast_start_frame = frame_idx
                    _coast_start_time = t_s
            else:
                if _coast_start_frame is not None:
                    resolved_by = "gemini" if tp.source == "gemini_recovery" else "reacquisition"
                    low_conf_segments.append(LowConfidenceSegment(
                        start_frame=_coast_start_frame,
                        end_frame=frame_idx - 1,
                        start_time_s=_coast_start_time or 0.0,
                        end_time_s=t_s,
                        reason="splash",
                        resolved_by=resolved_by,
                    ))
                    _coast_start_frame = None
                    _coast_start_time = None

        if ego_tracker is None:
            ego_tracker = EgoMotionTracker(frame, active_kf.lane_polygon_px)

        cam_dx, cam_dy = ego_tracker.update(frame)

        # Skip saving data for pre-start padding frames (tracker state is still updated)
        if t_s < 0:
            frame_idx += 1
            continue

        # Shift the tracked point by the cumulative camera motion to map it
        # back to the coordinate space of the calibration frame
        distance_m = _calib_distance(tp.x_px + cam_dx, tp.y_px + cam_dy)
        last_distance_m = distance_m

        if distance_m >= cfg.pool_length_m - FINISH_ZONE_M and not finish_active:
            finish_tracker.seed(frame, (tp.x_px, tp.y_px))
            finish_active = True
        elif finish_active:
            fv = finish_tracker.step(frame)
            if fv is not None and abs(fv) < 0.4:  # px/frame near-zero
                pass  # caller (analysis.splits) still derives the
                      # actual stop time from the distance/time trace

        pf = pose_estimator.process(frame, frame_idx, int(t_s * 1000))
        pose_frames.append((t_s, pf))

        # Pose-nose fallback — LEGACY (single-swimmer) path ONLY.
        #
        # In multi-swimmer mode the AntiSplashTracker owns the confidence and
        # source, and its coasting EKF must not be second-guessed by a
        # possibly-hallucinated pose landmark — so we leave ast_result intact.
        #
        # Even in the legacy path this used to fire on essentially every frame
        # (`confidence < 1.0`), relabel the point to a fake "0.9" and re-run the
        # tracker's Kalman step a SECOND time (track_frame already advanced it a
        # full predict+correct this frame — a genuine double update). Now it
        # fires only when the colour tracker actually failed (confidence below
        # the splash gate) and MediaPipe reports the nose with decent
        # visibility, assigns an honest visibility-scaled confidence capped well
        # below 1.0, and does NOT touch the filter (no double update).
        if (
            anti_splash_tracker is None
            and tp.confidence < SPLASH_CONFIDENCE_GATE
            and pf.present
            and 0 in pf.landmarks_px
        ):
            nose_x, nose_y, nose_vis = pf.landmarks_px[0]
            if nose_vis >= 0.5:
                tp.x_px = float(nose_x)
                tp.y_px = float(nose_y)
                # Honest: pose during splash/underwater is unreliable, so cap at
                # 0.6 and scale by the landmark's own visibility.
                tp.confidence = float(min(0.6, 0.3 + 0.3 * nose_vis))
                tp.source = "pose_fallback"
                distance_m = _calib_distance(tp.x_px + cam_dx, tp.y_px + cam_dy)

        track_points.append(TrackPoint(
            frame_idx=frame_idx, time_s=t_s, x_px=tp.x_px, y_px=tp.y_px,
            cam_dx=cam_dx, cam_dy=cam_dy,
            distance_m=distance_m, confidence=tp.confidence, method=tp.source,
        ))

        frame_idx += 1
        if total_frames > 0 and frame_idx % 15 == 0:
            pct = 10 + int(70 * frame_idx / total_frames)
            progress(min(pct, 80), f"Tracking… frame {frame_idx}/{total_frames}")

    cap.release()
    pose_estimator.close()

    # Close any open coast segment
    if _coast_start_frame is not None and track_points:
        low_conf_segments.append(LowConfidenceSegment(
            start_frame=_coast_start_frame,
            end_frame=track_points[-1].frame_idx,
            start_time_s=_coast_start_time or 0.0,
            end_time_s=track_points[-1].time_s,
            reason="coast",
            resolved_by=None,
        ))

    # Create a helper to get distance for any frame using the correct keyframe
    if ai_splits and ai_splits.splits:
        ai_times = [0.0] + [s.time_s for s in ai_splits.splits]
        ai_dists = [0.0] + [s.distance_m for s in ai_splits.splits]
        if ai_splits.finish_time_s:
            ai_times.append(ai_splits.finish_time_s)
            ai_dists.append(cfg.pool_length_m)

        def get_distance_for_point(p: TrackPoint, x: float, y: float) -> float:
            return float(np.interp(p.time_s, ai_times, ai_dists))
    else:
        def get_distance_for_point(p: TrackPoint, x: float, y: float) -> float:
            kf = keyframes[0]
            for k in keyframes:
                if k.frame_idx <= p.frame_idx:
                    kf = k
                else:
                    break

            calib = PoolCalibrator()
            if kf.reference_points:
                for pixel, dist in kf.reference_points:
                    calib.add_point(pixel, dist)
                calib.solve()
                return calib.pixel_to_distance(x, y)
            return 0.0

    # 1. Compute offsets per frame to eliminate calibration jumps
    frame_offsets: dict[int, float] = {}
    for i in range(1, len(keyframes)):
        kf_prev = keyframes[i-1]
        kf_curr = keyframes[i]

        idx_curr = -1
        for j, p in enumerate(track_points):
            if p.frame_idx >= kf_curr.frame_idx:
                idx_curr = j
                break

        if idx_curr > 0:
            idx_prev = 0
            for j in range(idx_curr - 1, -1, -1):
                if track_points[j].frame_idx <= kf_prev.frame_idx:
                    idx_prev = j
                    break

            if idx_prev >= 0 and idx_curr > idx_prev:
                v_window = min(5, idx_curr - idx_prev - 1)
                if v_window > 0:
                    dt = track_points[idx_curr-1].time_s - track_points[idx_curr-1-v_window].time_s
                    dd = track_points[idx_curr-1].distance_m - track_points[idx_curr-1-v_window].distance_m
                    v_prev = dd / dt if dt > 1e-6 else 0.0
                else:
                    v_prev = 0.0

                dt_boundary = track_points[idx_curr].time_s - track_points[idx_curr-1].time_s
                true_movement = v_prev * dt_boundary

                jump = track_points[idx_curr].distance_m - track_points[idx_curr-1].distance_m
                error_to_distribute = jump - true_movement

                frame_span = track_points[idx_curr].frame_idx - track_points[idx_prev].frame_idx
                if frame_span > 0:
                    for j in range(idx_prev, idx_curr):
                        f_idx = track_points[j].frame_idx
                        fraction = (f_idx - track_points[idx_prev].frame_idx) / frame_span
                        frame_offsets[f_idx] = frame_offsets.get(f_idx, 0.0) + (error_to_distribute * fraction)

    # Apply the offsets immediately
    for p in track_points:
        p.distance_m += frame_offsets.get(p.frame_idx, 0.0)

    def get_smoothed_distance_for_point(p: TrackPoint, x: float, y: float) -> float:
        base_dist = get_distance_for_point(p, x, y)
        return base_dist + frame_offsets.get(p.frame_idx, 0.0)

    progress(82, "Checking for tracking discrepancies…")
    flagged = flag_outliers(track_points)
    discrepancy_report = reconcile_with_backward_pass(
        track_points, flagged, frames_cache,
        pixel_to_distance_fn=lambda p: get_smoothed_distance_for_point(p, p.x_px + p.cam_dx, p.y_px + p.cam_dy)
    )
    # Recompute distances for any points whose x/y moved during reconciliation
    for p in track_points:
        p.distance_m = get_smoothed_distance_for_point(p, p.x_px + p.cam_dx, p.y_px + p.cam_dy)

    times_arr = np.array([p.time_s for p in track_points])
    dist_arr = np.array([p.distance_m for p in track_points])

    # ---- 4. Splits ---------------------------------------------------
    progress(88, "Computing splits…")
    split_report = compute_splits(times_arr, dist_arr, cfg.pool_length_m)

    # ---- 5. Stroke classification -----------------------------------
    progress(90, "Classifying stroke…")
    window_frames = [f for _, f in pose_frames[: min(150, len(pose_frames))]]
    from ..vision.stroke_classifier import classify_window, classify_from_motion
    stroke_classification = classify_window(window_frames)

    # Fallback to motion-based classifier if pose model failed or no skeleton detected
    if stroke_classification.stroke == "unknown" or stroke_classification.confidence < 0.1:
        y_px = [p.y_px for p in track_points[:min(int(fps * 5), len(track_points))]]
        stroke_classification = classify_from_motion(times_arr, dist_arr, y_px, fps=fps)

    # ---- 6. Stroke cycles / rate / length -----------------------------
    progress(93, "Computing stroke rate and length…")
    cycles = detect_stroke_cycles(pose_frames)
    rates = stroke_rate_series(cycles)
    lengths = stroke_length_series(cycles, times_arr, dist_arr)

    # ---- 7. Start / turn phase -----------------------------------------
    progress(96, "Computing start/turn phase metrics…")
    start_phase = compute_start_phase(times_arr, dist_arr, pose_frames)
    turn_phase = None
    if cfg.wall_position_m and cfg.wall_position_m < cfg.pool_length_m:
        turn_phase = compute_turn_phase(
            times_arr, dist_arr, cfg.wall_position_m, pose_frames,
        )

    pose_available = any(f.present for _, f in pose_frames)
    csv_rows = _build_csv_rows(track_points)

    # ---- 8. AI Analysis (optional, runs after CV pipeline) ----------
    ai_result = None
    if cfg.enable_ai and cfg.gemini_api_key:
        progress(97, "🤖 Running AI analysis with Gemini…")
        ai_result = _run_ai_analysis(
            cfg, stroke_classification, split_report, rates, lengths,
            start_phase, turn_phase, times_arr, dist_arr, progress,
        )

    # ---- 8b. DTW Biomechanical Comparison (optional) -----------------
    dtw_result = None
    stroke_for_dtw = cfg.stroke_override or stroke_classification.stroke
    event_m_for_dtw = cfg.event_distance_m or int(cfg.pool_length_m)
    if pose_available and stroke_for_dtw not in ("unknown", ""):
        progress(99, "🧐 Running DTW biomechanical comparison…")
        try:
            dtw_result = run_dtw_comparison(
                pose_frames, stroke_for_dtw, event_m_for_dtw
            )
        except Exception as _dtw_exc:
            logger.warning("DTW comparison failed: %s", _dtw_exc)

    progress(100, "Done.")

    return AnalysisResult(
        split_report=split_report, start_detection=start_result,
        stroke_classification=stroke_classification, start_phase=start_phase,
        turn_phase=turn_phase, stroke_rates=rates, stroke_lengths=lengths,
        discrepancy_report=discrepancy_report, pose_available=pose_available,
        csv_rows=csv_rows,
        ai_result=ai_result,
        dtw_result=dtw_result,
        low_confidence_segments=low_conf_segments,
        gemini_assisted_frames=gemini_assisted_count,
        cv_only_frames=cv_only_count,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_start(video_path: str) -> StartDetectionResult:
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio.wav")
        has_audio = extract_audio_wav(video_path, wav_path)
        if has_audio:
            result = detect_start_from_audio(wav_path)
            if result is not None:
                return result
    return detect_start_visual_fallback(video_path, blocks_roi=None)


def _build_csv_rows(track_points: list[TrackPoint]) -> list[dict]:
    return [
        {
            "frame": p.frame_idx, "time_s": round(p.time_s, 4),
            "distance_m": round(p.distance_m, 3),
            "confidence": round(p.confidence, 2), "method": p.method,
        }
        for p in track_points
    ]


def _run_ai_analysis(
    cfg: AnalysisConfig,
    stroke_classification: StrokeClassification,
    split_report: SplitReport,
    rates: list[tuple[float, float]],
    lengths: list[tuple[float, float]],
    start_phase: StartPhaseMetrics,
    turn_phase: TurnPhaseMetrics | None,
    times_arr: np.ndarray,
    dist_arr: np.ndarray,
    progress: ProgressCallback,
) -> AIAnalysisResult:
    """Run Gemini AI analysis using the CV results as context."""
    ai_result = AIAnalysisResult()

    try:
        from ..ai.gemini_analyzer import GeminiSwimAnalyzer
        from ..ai.video_segments import extract_full_video_for_upload
        from ..ai.reference_comparison import compare_videos
        from ..analysis.benchmarks import (
            get_elite_profile, build_elite_context_for_prompt,
        )

        analyzer = GeminiSwimAnalyzer(
            api_key=cfg.gemini_api_key,
            model=cfg.gemini_model,
        )

        # Determine stroke and event
        stroke = cfg.stroke_override or stroke_classification.stroke
        distance = cfg.event_distance_m or int(cfg.pool_length_m)
        pool_type = cfg.pool_type

        # Extract a clip for upload
        progress(97, "🤖 Preparing video for AI analysis…")
        clip_path = extract_full_video_for_upload(
            cfg.video_path,
            max_duration_s=60.0,
            start_offset_s=0.0,
        )

        # Compute user metrics for AI context
        vp = split_report.velocity_profile
        avg_v = sum(v for _, v in vp) / len(vp) if vp else None
        avg_sr = sum(r for _, r in rates) / len(rates) if rates else None
        avg_sl = sum(l for _, l in lengths) / len(lengths) if lengths else None

        user_metrics = {
            "avg_velocity": f"{avg_v:.2f}" if avg_v else "N/A",
            "stroke_rate": f"{avg_sr:.1f}" if avg_sr else "N/A",
            "stroke_length": f"{avg_sl:.2f}" if avg_sl else "N/A",
            "start_15m_time": (
                f"{start_phase.time_to_15m_s:.2f}"
                if start_phase.time_to_15m_s else "N/A"
            ),
            "turn_time": (
                f"{turn_phase.turn_duration_s:.2f}"
                if turn_phase and turn_phase.turn_duration_s else "N/A"
            ),
        }

        # 1. Technique Analysis
        progress(97, "🤖 Analyzing technique with Gemini…")
        ai_result.technique_analysis = analyzer.analyze_technique(
            clip_path, stroke=stroke, distance_m=distance, pool_type=pool_type,
        )

        # If Gemini identified the stroke with higher confidence, use it
        if (ai_result.technique_analysis
                and ai_result.technique_analysis.stroke_confidence > stroke_classification.confidence):
            stroke = ai_result.technique_analysis.stroke_identified

        # 2. Elite Comparison (auto-match if no explicit selection)
        progress(98, "🏅 Finding closest elite match…")
        elite_key = cfg.elite_compare_key
        if not elite_key:
            from ..analysis.benchmarks import find_closest_elite_match
            elite_key, auto_profile, match_reason = find_closest_elite_match(
                stroke=stroke,
                distance_m=distance,
                sex=cfg.swimmer_sex,
                user_velocity_mps=avg_v,
                user_stroke_rate=avg_sr,
            )
            logger.info("Auto-match: %s", match_reason)

        if elite_key:
            profile = get_elite_profile(elite_key)
            if profile:
                progress(98, f"🏅 Comparing with {profile.country_flag} {profile.name}…")
                elite_dict = {
                    "name": profile.name,
                    "country": profile.country,
                    "achievement": profile.achievement,
                    "technique": profile.technique_signature,
                    "metrics": {
                        k: {
                            "time_s": v.time_s,
                            "avg_velocity": v.avg_velocity_mps,
                            "stroke_rate": v.stroke_rate_cpm,
                            "stroke_length": v.stroke_length_m,
                        }
                        for k, v in profile.race_data.items()
                    },
                }
                ai_result.elite_comparison = analyzer.compare_with_elite(
                    clip_path, elite_dict, user_metrics,
                    stroke=stroke, distance_m=distance,
                )

        # 3. Improvement Plan
        technique_summary = ""
        if ai_result.technique_analysis:
            technique_summary = "\n".join(
                f"- {e.element}: {e.rating} — {e.observation}"
                for e in ai_result.technique_analysis.elements
            )
        progress(99, "📈 Generating improvement plan…")
        ai_result.improvement_plan = analyzer.generate_improvement_plan(
            clip_path, technique_summary, user_metrics,
            stroke=stroke, distance_m=distance, user_goal=cfg.user_goal,
        )

        # 4. Race Strategy
        split_data_str = json.dumps([
            {"marker_m": s.marker_m, "time_s": s.time_s}
            for s in split_report.splits
        ])
        velocity_str = json.dumps([
            {"distance_m": d, "velocity_mps": v}
            for d, v in (vp[:20] if vp else [])
        ])
        sr_data_str = json.dumps([
            {"time_s": t, "rate_cpm": r}
            for t, r in (rates[:20] if rates else [])
        ])
        ai_result.race_strategy = analyzer.analyze_race_strategy(
            clip_path, split_data_str, velocity_str, sr_data_str,
            stroke=stroke, distance_m=distance, pool_type=pool_type,
        )

        # 5. Reference Video Comparison (optional)
        if cfg.reference_video_path:
            progress(99, "🎥 Comparing with reference video…")
            ai_result.video_comparison = compare_videos(
                analyzer, clip_path, cfg.reference_video_path, stroke=stroke,
            )

        # Cleanup uploaded files
        analyzer.cleanup()

        # Cleanup temp clip
        try:
            os.remove(clip_path)
        except OSError:
            pass

    except ImportError as e:
        ai_result.error = f"AI dependencies not installed: {e}"
        logger.warning(ai_result.error)
    except Exception as e:
        if type(e).__name__ == "ModelUnavailableError":
            raise
        ai_result.error = f"AI analysis error: {e}"
        logger.error(ai_result.error, exc_info=True)

    return ai_result
