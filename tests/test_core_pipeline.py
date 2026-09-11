"""
Test headless core pipeline execution on synthetic video.
"""
import os
import tempfile
import cv2
import numpy as np
import pytest

from src.core.models import AnalysisConfig, AnalysisResult
from src.core.pipeline import run_analysis
from src.core.metrics import summarize_result
from src.vision.calibration import CalibrationKeyframe
from web.models import RaceAnalysisMetrics


def _create_synthetic_swim_video(num_frames=30, width=640, height=480, fps=30.0) -> str:
    """Create a short synthetic video with a moving cap."""
    tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tf.close()
    path = tf.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), 220, dtype=np.uint8)  # Pool water background
        # Lane lines
        cv2.line(frame, (50, 100), (590, 100), (0, 0, 200), 2)
        cv2.line(frame, (50, 380), (590, 380), (0, 0, 200), 2)

        # Swimmer moving across lane with yellow cap
        x = int(100 + (i / num_frames) * 400)
        y = 240
        cv2.circle(frame, (x, y), 14, (0, 255, 255), -1)  # BGR Yellow cap
        writer.write(frame)

    writer.release()
    return path


def test_core_pipeline_headless_run():
    """Verify that run_analysis runs purely headlessly and produces AnalysisResult."""
    video_path = _create_synthetic_swim_video(num_frames=20)
    try:
        keyframes = [
            CalibrationKeyframe(
                frame_idx=0,
                reference_points=[((100, 240), 0.0), ((500, 240), 50.0)],
                lane_polygon_px=[(50, 100), (590, 100), (590, 380), (50, 380)],
                cap_color="yellow",
            )
        ]

        cfg = AnalysisConfig(
            video_path=video_path,
            pool_length_m=50.0,
            calibration_keyframes=keyframes,
            cap_color="yellow",
            wall_position_m=50.0,
            enable_ai=False,
            enable_multi_swimmer=False,
        )

        progress_log = []

        def on_progress(pct, msg):
            progress_log.append((pct, msg))

        result = run_analysis(cfg, progress_callback=on_progress)

        assert isinstance(result, AnalysisResult)
        assert len(progress_log) > 0
        assert result.stroke_classification is not None
        assert len(result.csv_rows) > 0
        assert result.split_report is not None

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


def test_pipeline_fails_gracefully_without_calibration_or_ai():
    """With no manual calibration AND AI disabled, the pipeline must fail with a
    clear, actionable message BEFORE the frame loop — not crash on an unsolved
    calibrator deep inside tracking (product decision: graceful fail)."""
    video_path = _create_synthetic_swim_video(num_frames=20)
    try:
        cfg = AnalysisConfig(
            video_path=video_path,
            pool_length_m=50.0,
            calibration_keyframes=[],   # no manual calibration
            enable_ai=False,            # and no AI auto-calibration
            enable_multi_swimmer=False,
        )
        with pytest.raises(RuntimeError) as excinfo:
            run_analysis(cfg)
        msg = str(excinfo.value).lower()
        assert "calibration" in msg
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


def test_canonical_summary_matches_api_model():
    """The shared-core serializer must produce exactly the structure the web
    API model expects — RaceAnalysisMetrics(**summarize_result(...)) validates
    with no missing/extra fields — so desktop and web report identical numbers."""
    video_path = _create_synthetic_swim_video(num_frames=20)
    try:
        keyframes = [
            CalibrationKeyframe(
                frame_idx=0,
                reference_points=[((100, 240), 0.0), ((500, 240), 50.0)],
                lane_polygon_px=[(50, 100), (590, 100), (590, 380), (50, 380)],
                cap_color="yellow",
            )
        ]
        cfg = AnalysisConfig(
            video_path=video_path,
            pool_length_m=50.0,
            calibration_keyframes=keyframes,
            cap_color="yellow",
            event_distance_m=50,
            enable_ai=False,
            enable_multi_swimmer=False,
        )
        result = run_analysis(cfg)

        summary = summarize_result(cfg, result)

        # Structural parity: keys equal the API model's declared fields.
        assert set(summary.keys()) == set(RaceAnalysisMetrics.model_fields.keys())

        # And it actually validates into the model.
        model = RaceAnalysisMetrics(**summary)
        assert model.stroke == result.stroke_classification.stroke
        assert model.event_distance_m == 50.0
        assert model.cv_only_frames == result.cv_only_frames
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
