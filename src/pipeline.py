"""
Full analysis pipeline, run on a background QThread.

Uses Qt's own signal/slot mechanism for cross-thread communication rather
than raw Python `threading` + manual locking. This is a deliberate
departure from the spec's literal "background worker thread" wording:
Qt's signals are the correct, safe way to push progress/results back to a
GUI thread, and hand-rolled threading + shared mutable state is exactly
the kind of thing that produces intermittent, hard-to-debug UI freezes or
crashes — the opposite of what "keep the GUI smooth" is asking for.
"""
from __future__ import annotations

from .vision.calibration import CalibrationKeyframe
from dataclasses import dataclass, field
import tempfile
import os
import numpy as np
import cv2
from PySide6.QtCore import QThread, Signal

from .config import DEFAULT_VIDEO_FPS_FALLBACK
from .vision.calibration import PoolCalibrator, CalibrationResult
from .vision.cap_tracker import CapTracker, FinishLineOpticalFlowTracker
from .vision.pose_estimator import PoseEstimator, PoseFrame
from .vision.ego_motion import EgoMotionTracker
from .vision.stroke_classifier import classify_window, StrokeClassification
from .audio.start_detector import (
    extract_audio_wav, detect_start_from_audio, detect_start_visual_fallback,
    StartDetectionResult,
)
from .analysis.splits import compute_splits, SplitReport
from .analysis.biomechanics import (
    detect_stroke_cycles, stroke_rate_series, stroke_length_series,
    compute_start_phase, compute_turn_phase, StartPhaseMetrics, TurnPhaseMetrics,
)
from .analysis.error_correction import (
    TrackPoint, flag_outliers, reconcile_with_backward_pass, DiscrepancyReport,
)
from .config import FINISH_ZONE_M


@dataclass
class AnalysisConfig:
    video_path: str
    pool_length_m: float
    calibration_keyframes: list[CalibrationKeyframe]
    cap_color: str
    wall_position_m: float = None  # set to pool_length_m if turn analysis wanted


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


class AnalysisWorker(QThread):
    progress = Signal(int, str)          # percent, status message
    finished_ok = Signal(object)          # AnalysisResult
    finished_error = Signal(str)

    def __init__(self, cfg: AnalysisConfig):
        super().__init__()
        self.cfg = cfg
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            result = self._run_pipeline()
            if not self._cancelled:
                self.finished_ok.emit(result)
        except Exception as exc:  # surfaced to the user, not swallowed
            self.finished_error.emit(f"{type(exc).__name__}: {exc}")

    # -- internal ------------------------------------------------------
    def _run_pipeline(self) -> AnalysisResult:
        cfg = self.cfg
        self.progress.emit(2, "Opening video…")
        cap = cv2.VideoCapture(cfg.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {cfg.video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FPS_FALLBACK
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ---- 1. Start detection (audio, then visual fallback) ---------
        self.progress.emit(5, "Detecting race start…")
        start_result = self._detect_start(cfg.video_path)

        # ---- 2. Frame-by-frame cap tracking + pose --------------------
        self.progress.emit(10, "Tracking swimmer…")
        
        # Sort keyframes by frame_idx just to be safe
        keyframes = sorted(cfg.calibration_keyframes, key=lambda k: k.frame_idx)
        current_kf_idx = 0
        active_kf = keyframes[0]
        
        calibrator = PoolCalibrator()
        for pixel, dist in active_kf.reference_points:
            calibrator.add_point(pixel, dist)
        calibrator.solve()

        tracker = CapTracker(cfg.cap_color, active_kf.lane_polygon_px)
        pose_estimator = PoseEstimator()

        frames_cache: dict[int, np.ndarray] = {}
        track_points: list[TrackPoint] = []
        pose_frames: list[tuple[float, PoseFrame]] = []
        finish_tracker = FinishLineOpticalFlowTracker(lane_axis_is_x=True)
        finish_active = False

        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_MSEC, max(start_result.start_time_s * 1000 - 500, 0))
        
        ego_tracker = None
        while True:
            if self._cancelled:
                raise RuntimeError("Cancelled by user.")
            ok, frame = cap.read()
            if not ok:
                break

            t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - start_result.start_time_s
            
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
                for pixel, dist in active_kf.reference_points:
                    calibrator.add_point(pixel, dist)
                calibrator.solve()
                
                # Update lane constraints for cap tracker
                tracker.lane_polygon = np.array(active_kf.lane_polygon_px, dtype=np.int32)
                
                # Force EgoMotionTracker to re-initialize on this frame, anchoring drift to 0
                ego_tracker = None

            tp = tracker.track_frame(frame, frame_idx)
            
            if ego_tracker is None:
                ego_tracker = EgoMotionTracker(frame, active_kf.lane_polygon_px)
                
            cam_dx, cam_dy = ego_tracker.update(frame)
            
            # Skip saving data for pre-start padding frames (tracker state is still updated)
            if t_s < 0:
                frame_idx += 1
                continue
                
            # Shift the tracked point by the cumulative camera motion to map it 
            # back to the coordinate space of the calibration frame
            distance_m = calibrator.pixel_to_distance(tp.x_px + cam_dx, tp.y_px + cam_dy)

            if distance_m >= cfg.pool_length_m - FINISH_ZONE_M and not finish_active:
                finish_tracker.seed(frame, (tp.x_px, tp.y_px))
                finish_active = True
            elif finish_active:
                fv = finish_tracker.step(frame)
                if fv is not None and abs(fv) < 0.4:  # px/frame near-zero
                    pass  # caller (analysis.splits) still derives the
                          # actual stop time from the distance/time trace;
                          # this loop just keeps the flow tracker fed.

            pf = pose_estimator.process(frame, frame_idx, int(t_s * 1000))
            pose_frames.append((t_s, pf))

            if tp.confidence < 1.0 and pf.present and 0 in pf.landmarks_px:
                # Fallback to pose estimator's nose if color tracker fails
                tp.x_px = float(pf.landmarks_px[0][0])
                tp.y_px = float(pf.landmarks_px[0][1])
                tp.confidence = 0.9
                tp.source = "pose_fallback"
                tracker.kalman.correct(tp.x_px, tp.y_px)
                # Recalculate distance_m
                distance_m = calibrator.pixel_to_distance(tp.x_px + cam_dx, tp.y_px + cam_dy)

            track_points.append(TrackPoint(
                frame_idx=frame_idx, time_s=t_s, x_px=tp.x_px, y_px=tp.y_px,
                cam_dx=cam_dx, cam_dy=cam_dy,
                distance_m=distance_m, confidence=tp.confidence, method=tp.source,
            ))

            frame_idx += 1
            if total_frames > 0 and frame_idx % 15 == 0:
                pct = 10 + int(70 * frame_idx / total_frames)
                self.progress.emit(min(pct, 80), f"Tracking… frame {frame_idx}/{total_frames}")

        cap.release()
        pose_estimator.close()

        # Create a helper to get distance for any frame using the correct keyframe
        def get_distance_for_frame(f_idx: int, x: float, y: float) -> float:
            kf = keyframes[0]
            for k in keyframes:
                if k.frame_idx <= f_idx:
                    kf = k
                else:
                    break
            
            calib = PoolCalibrator()
            for pixel, dist in kf.reference_points:
                calib.add_point(pixel, dist)
            calib.solve()
            return calib.pixel_to_distance(x, y)

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

        def get_smoothed_distance_for_frame(f_idx: int, x: float, y: float) -> float:
            base_dist = get_distance_for_frame(f_idx, x, y)
            return base_dist + frame_offsets.get(f_idx, 0.0)

        self.progress.emit(82, "Checking for tracking discrepancies…")
        flagged = flag_outliers(track_points)
        discrepancy_report = reconcile_with_backward_pass(
            track_points, flagged, frames_cache,
            pixel_to_distance_fn=lambda p: get_smoothed_distance_for_frame(p.frame_idx, p.x_px + p.cam_dx, p.y_px + p.cam_dy)
        )
        # Recompute distances for any points whose x/y moved during
        # reconciliation.
        for p in track_points:
            p.distance_m = get_smoothed_distance_for_frame(p.frame_idx, p.x_px + p.cam_dx, p.y_px + p.cam_dy)

        times_arr = np.array([p.time_s for p in track_points])
        dist_arr = np.array([p.distance_m for p in track_points])

        # ---- 4. Splits ---------------------------------------------------
        self.progress.emit(88, "Computing splits…")
        split_report = compute_splits(times_arr, dist_arr, cfg.pool_length_m)

        # ---- 5. Stroke classification -----------------------------------
        self.progress.emit(90, "Classifying stroke…")
        window_frames = [f for _, f in pose_frames[: min(150, len(pose_frames))]]
        from .vision.stroke_classifier import classify_window, classify_from_motion
        stroke_classification = classify_window(window_frames)
        
        # Fallback to motion-based classifier if pose model failed or no skeleton detected
        if stroke_classification.stroke == "unknown" or stroke_classification.confidence < 0.1:
            y_px = [p.y_px for p in track_points[:min(int(fps * 5), len(track_points))]]
            stroke_classification = classify_from_motion(times_arr, dist_arr, y_px, fps=fps)

        # ---- 6. Stroke cycles / rate / length -----------------------------
        self.progress.emit(93, "Computing stroke rate and length…")
        cycles = detect_stroke_cycles(pose_frames)
        rates = stroke_rate_series(cycles)
        lengths = stroke_length_series(cycles, times_arr, dist_arr)

        # ---- 7. Start / turn phase -----------------------------------------
        self.progress.emit(96, "Computing start/turn phase metrics…")
        start_phase = compute_start_phase(times_arr, dist_arr, pose_frames)
        turn_phase = None
        if cfg.wall_position_m and cfg.wall_position_m < cfg.pool_length_m:
            turn_phase = compute_turn_phase(
                times_arr, dist_arr, cfg.wall_position_m, pose_frames,
            )

        self.progress.emit(100, "Done.")
        pose_available = any(f.present for _, f in pose_frames)

        csv_rows = self._build_csv_rows(track_points)

        return AnalysisResult(
            split_report=split_report, start_detection=start_result,
            stroke_classification=stroke_classification, start_phase=start_phase,
            turn_phase=turn_phase, stroke_rates=rates, stroke_lengths=lengths,
            discrepancy_report=discrepancy_report, pose_available=pose_available,
            csv_rows=csv_rows,
        )

    def _detect_start(self, video_path: str) -> StartDetectionResult:
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "audio.wav")
            has_audio = extract_audio_wav(video_path, wav_path)
            if has_audio:
                result = detect_start_from_audio(wav_path)
                if result is not None:
                    return result
        return detect_start_visual_fallback(video_path, blocks_roi=None)

    def _build_csv_rows(self, track_points: list[TrackPoint]) -> list[dict]:
        return [
            {
                "frame": p.frame_idx, "time_s": round(p.time_s, 4),
                "distance_m": round(p.distance_m, 3),
                "confidence": round(p.confidence, 2), "method": p.method,
            }
            for p in track_points
        ]
