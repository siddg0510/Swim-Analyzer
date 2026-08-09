"""
Full analysis pipeline, run on a background QThread.

Uses Qt's own signal/slot mechanism for cross-thread communication rather
than raw Python `threading` + manual locking. This is a deliberate
departure from the spec's literal "background worker thread" wording:
Qt's signals are the correct, safe way to push progress/results back to a
GUI thread, and hand-rolled threading + shared mutable state is exactly
the kind of thing that produces intermittent, hard-to-debug UI freezes or
crashes — the opposite of what "keep the GUI smooth" is asking for.

ENHANCEMENT (AI Integration):
The pipeline now optionally runs Gemini AI analysis after the CV pipeline
completes. AI analysis is strictly additive — if the API key is missing
or the network is unavailable, the app falls back gracefully to offline-
only results. The AI layer adds: technique analysis, elite comparison,
improvement plan, and race strategy analysis.
"""
from __future__ import annotations

from .vision.calibration import CalibrationKeyframe
from dataclasses import dataclass, field
import tempfile
import os
import json
import logging
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

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    video_path: str
    pool_length_m: float
    calibration_keyframes: list[CalibrationKeyframe]
    cap_color: str
    wall_position_m: float = None  # set to pool_length_m if turn analysis wanted
    # AI analysis options
    enable_ai: bool = False
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    stroke_override: str | None = None  # user-selected stroke
    event_distance_m: int | None = None  # user-selected event distance
    pool_type: str = "long course"
    swimmer_sex: str = "male"
    elite_compare_key: str | None = None  # key into ELITE_PROFILES
    reference_video_path: str | None = None  # optional reference video
    user_goal: str = "Improve race time and technique efficiency"


@dataclass
class AIAnalysisResult:
    """Results from Gemini AI analysis — all optional."""
    technique_analysis: object | None = None   # TechniqueAnalysis
    elite_comparison: object | None = None     # EliteComparison
    improvement_plan: object | None = None     # ImprovementPlan
    race_strategy: object | None = None        # RaceStrategy
    video_comparison: object | None = None     # VideoComparison
    error: str | None = None


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
    # AI analysis results (None when AI is disabled or unavailable)
    ai_result: AIAnalysisResult | None = None


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
        
        keyframes = sorted(cfg.calibration_keyframes, key=lambda k: k.frame_idx)
        ai_splits = None

        if not keyframes:
            if cfg.enable_ai and cfg.gemini_api_key:
                self.progress.emit(7, "🤖 AI Auto-Calibration: Detecting pool markers…")
                from .ai.gemini_analyzer import GeminiSwimAnalyzer
                analyzer = GeminiSwimAnalyzer(api_key=cfg.gemini_api_key, model=cfg.gemini_model)
                ai_splits = analyzer.detect_splits(cfg.video_path, f"with {cfg.cap_color} cap", cfg.pool_length_m)
                analyzer.cleanup()

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

        self.progress.emit(82, "Checking for tracking discrepancies…")
        flagged = flag_outliers(track_points)
        discrepancy_report = reconcile_with_backward_pass(
            track_points, flagged, frames_cache,
            pixel_to_distance_fn=lambda p: get_smoothed_distance_for_point(p, p.x_px + p.cam_dx, p.y_px + p.cam_dy)
        )
        # Recompute distances for any points whose x/y moved during
        # reconciliation.
        for p in track_points:
            p.distance_m = get_smoothed_distance_for_point(p, p.x_px + p.cam_dx, p.y_px + p.cam_dy)

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

        pose_available = any(f.present for _, f in pose_frames)
        csv_rows = self._build_csv_rows(track_points)

        # ---- 8. AI Analysis (optional, runs after CV pipeline) ----------
        ai_result = None
        if cfg.enable_ai and cfg.gemini_api_key:
            self.progress.emit(97, "🤖 Running AI analysis with Gemini…")
            ai_result = self._run_ai_analysis(
                cfg, stroke_classification, split_report, rates, lengths,
                start_phase, turn_phase, times_arr, dist_arr,
            )

        self.progress.emit(100, "Done.")

        return AnalysisResult(
            split_report=split_report, start_detection=start_result,
            stroke_classification=stroke_classification, start_phase=start_phase,
            turn_phase=turn_phase, stroke_rates=rates, stroke_lengths=lengths,
            discrepancy_report=discrepancy_report, pose_available=pose_available,
            csv_rows=csv_rows,
            ai_result=ai_result,
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

    def _run_ai_analysis(
        self,
        cfg: AnalysisConfig,
        stroke_classification: StrokeClassification,
        split_report: SplitReport,
        rates: list[tuple[float, float]],
        lengths: list[tuple[float, float]],
        start_phase: StartPhaseMetrics,
        turn_phase: TurnPhaseMetrics | None,
        times_arr: np.ndarray,
        dist_arr: np.ndarray,
    ) -> AIAnalysisResult:
        """Run Gemini AI analysis using the CV results as context."""
        ai_result = AIAnalysisResult()

        try:
            from .ai.gemini_analyzer import GeminiSwimAnalyzer
            from .ai.video_segments import extract_full_video_for_upload
            from .ai.reference_comparison import compare_videos
            from .analysis.benchmarks import (
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
            self.progress.emit(97, "🤖 Preparing video for AI analysis…")
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
            self.progress.emit(97, "🤖 Analyzing technique with Gemini…")
            ai_result.technique_analysis = analyzer.analyze_technique(
                clip_path, stroke=stroke, distance_m=distance, pool_type=pool_type,
            )

            # If Gemini identified the stroke with higher confidence, use it
            if (ai_result.technique_analysis
                    and ai_result.technique_analysis.stroke_confidence > stroke_classification.confidence):
                stroke = ai_result.technique_analysis.stroke_identified

            # 2. Elite Comparison (auto-match if no explicit selection)
            self.progress.emit(98, "🏅 Finding closest elite match…")
            elite_key = cfg.elite_compare_key
            if not elite_key:
                # Auto-select based on stroke, event, gender, and performance
                from .analysis.benchmarks import find_closest_elite_match
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
                    self.progress.emit(98, f"🏅 Comparing with {profile.country_flag} {profile.name}…")
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
            self.progress.emit(99, "📈 Generating improvement plan…")
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
                self.progress.emit(99, "🎥 Comparing with reference video…")
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
