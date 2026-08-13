"""
Smoke tests using synthetic data — no real swim footage available (or
appropriate) to test against, so these validate that the math and the
OpenCV/NumPy/SciPy plumbing behave correctly on controlled inputs,
catching real bugs before shipping. They do NOT validate real-world
tracking accuracy on an actual race video — that can only be checked by
running the app on real footage, which the README flags clearly as an
expected first step.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2


def test_calibration_linear():
    from src.vision.calibration import PoolCalibrator
    calib = PoolCalibrator()
    calib.add_point((100, 300), 0.0)
    calib.add_point((900, 300), 50.0)
    result = calib.solve()
    assert result.mode == "linear"
    d = calib.pixel_to_distance(500, 300)
    assert abs(d - 25.0) < 0.5, f"expected ~25m, got {d}"
    print("test_calibration_linear: OK  (25m point ->", round(d, 2), "m)")


def test_calibration_homography():
    from src.vision.calibration import PoolCalibrator
    calib = PoolCalibrator()
    # 4 points along a lane with mild perspective (near end wider than far end)
    calib.add_point((80, 400), 0.0)
    calib.add_point((80, 200), 0.0)
    calib.add_point((920, 380), 50.0)
    calib.add_point((920, 220), 50.0)
    result = calib.solve()
    assert result.mode == "homography"
    d = calib.pixel_to_distance(500, 300)
    assert 15 < d < 35, f"expected roughly mid-pool, got {d}"
    print("test_calibration_homography: OK  (mid-lane ->", round(d, 2), "m)")


def test_hsv_color_range_and_detection():
    from src.vision.cap_tracker import CapTracker, hex_or_keyword_to_hsv_ranges

    ranges = hex_or_keyword_to_hsv_ranges("#FF0000")
    for lower, upper in ranges:
        assert lower[0] <= upper[0]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (255, 255, 255)  # white-ish background (BGR)
    cv2.circle(frame, (320, 240), 15, (0, 0, 255), -1)  # red cap, BGR

    tracker = CapTracker(cap_color="red", lane_polygon=None)
    det = tracker.detect(frame)
    assert det is not None, "expected to detect the synthetic red blob"
    x, y, conf = det
    assert abs(x - 320) < 10 and abs(y - 240) < 10
    print("test_hsv_color_range_and_detection: OK  (detected at", round(x), round(y), ")")


def test_kalman_bridges_occlusion():
    from src.vision.cap_tracker import CapTracker
    tracker = CapTracker(cap_color="red", lane_polygon=None)

    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    for i, x in enumerate([100, 120, 140]):
        f = frame.copy()
        cv2.circle(f, (x, 240), 15, (0, 0, 255), -1)
        tracker.track_frame(f, i)

    # occluded frame: no red blob at all (simulates splash covering the cap)
    occluded = frame.copy()
    tp = tracker.track_frame(occluded, 3)
    assert tp.source == "kalman_predict"
    assert 140 < tp.x_px < 200, f"Kalman prediction should extrapolate forward, got {tp.x_px}"
    print("test_kalman_bridges_occlusion: OK  (predicted x =", round(tp.x_px, 1), ")")


def test_audio_onset_detection_synthetic():
    import wave
    import tempfile
    from src.audio.start_detector import detect_start_from_audio

    sr = 44100
    duration_s = 5.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    signal = 0.01 * np.random.randn(len(t))  # quiet ambient noise
    beep_start_s = 2.0
    beep_mask = (t >= beep_start_s) & (t < beep_start_s + 0.2)
    signal[beep_mask] += 0.8 * np.sin(2 * np.pi * 2000 * t[beep_mask])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((signal * 32767).astype(np.int16).tobytes())

    result = detect_start_from_audio(path, baseline_s=1.0)
    os.unlink(path)
    assert result is not None, "expected a confident onset detection"
    assert abs(result.start_time_s - beep_start_s) < 0.05, \
        f"expected ~{beep_start_s}s, got {result.start_time_s}s"
    print("test_audio_onset_detection_synthetic: OK  (detected at",
          round(result.start_time_s, 3), "s, true onset", beep_start_s, "s)")


def test_splits_computation():
    from src.analysis.splits import compute_splits

    # Synthetic swimmer: constant 1.5 m/s for 50m, sampled at 30fps
    fps = 30.0
    total_time = 50.0 / 1.5
    times = np.arange(0, total_time, 1 / fps)
    distances = 1.5 * times

    report = compute_splits(times, distances, pool_length_m=50.0)
    marker_times = {s.marker_m: s.time_s for s in report.splits}
    assert abs(marker_times[25] - 25 / 1.5) < 0.05
    assert abs(marker_times[50] - 50 / 1.5) < 0.05
    assert report.total_time_s is not None
    assert abs(report.total_time_s - total_time) < 0.05
    assert report.final_5m_s is not None
    assert abs(report.final_5m_s - 5 / 1.5) < 0.05
    print("test_splits_computation: OK  (50m split =", round(marker_times[50], 2),
          "s, expected", round(total_time, 2), "s)")


def test_outlier_flagging_catches_impossible_jump():
    from src.analysis.error_correction import flag_outliers, TrackPoint

    points = []
    for i in range(30):
        points.append(TrackPoint(i, i * 0.1, x_px=i * 5, y_px=200, cam_dx=0.0, cam_dy=0.0,
                                 distance_m=i * 0.15, confidence=0.9, method="tracked"))
    # inject one impossible jump (teleport) at frame 15
    points[15].distance_m = points[14].distance_m + 8.0  # ~8m in 0.1s = 80 m/s, impossible

    flagged = flag_outliers(points)
    assert 15 in flagged, f"expected frame 15 to be flagged, got {flagged}"
    print("test_outlier_flagging_catches_impossible_jump: OK  (flagged:", flagged, ")")


def test_stroke_classifier_runs_on_synthetic_landmarks():
    from src.vision.stroke_classifier import classify_window
    from src.vision.pose_estimator import (
        PoseFrame, LM_L_WRIST, LM_R_WRIST, LM_L_ANKLE, LM_R_ANKLE,
        LM_L_SHOULDER, LM_R_SHOULDER, LM_L_HIP, LM_R_HIP, LM_NOSE,
    )

    frames = []
    for i in range(60):
        t = i / 30.0
        # alternating arms (freestyle/backstroke-like): out of phase sine waves
        l_wrist_y = 200 + 40 * np.sin(2 * np.pi * 1.0 * t)
        r_wrist_y = 200 + 40 * np.sin(2 * np.pi * 1.0 * t + np.pi)
        l_ankle_y = 300 + 10 * np.sin(2 * np.pi * 2.0 * t)
        r_ankle_y = 300 + 10 * np.sin(2 * np.pi * 2.0 * t + np.pi)
        landmarks = {
            LM_L_WRIST: (100, l_wrist_y, 0.9), LM_R_WRIST: (120, r_wrist_y, 0.9),
            LM_L_ANKLE: (100, l_ankle_y, 0.9), LM_R_ANKLE: (120, r_ankle_y, 0.9),
            LM_L_SHOULDER: (100, 150, 0.9), LM_R_SHOULDER: (120, 150, 0.9),
            LM_L_HIP: (100, 250, 0.9), LM_R_HIP: (120, 250, 0.9),
            LM_NOSE: (110, 140, 0.9),  # nose above shoulder line -> face-down proxy
        }
        frames.append(PoseFrame(i, int(t * 1000), present=True, landmarks_px=landmarks))

    result = classify_window(frames)
    assert result.stroke in ("freestyle", "backstroke"), \
        f"alternating-arm synthetic data should lean freestyle/backstroke, got {result.stroke}"
    print("test_stroke_classifier_runs_on_synthetic_landmarks: OK  (classified:",
          result.stroke, f", confidence {result.confidence:.2f})")


def test_full_synthetic_video_pipeline_pieces():
    """
    Not a full QThread pipeline run (that needs a running Qt event loop),
    but exercises the same sequence of calls the pipeline makes: read a
    synthetic video frame by frame, track a moving colored blob, calibrate,
    and compute splits — the exact chain that matters for correctness.
    """
    import tempfile
    from src.vision.cap_tracker import CapTracker
    from src.vision.calibration import PoolCalibrator
    from src.analysis.splits import compute_splits

    fps = 30
    width, height = 960, 240
    pool_length_m = 25.0
    speed_mps = 1.6
    duration_s = pool_length_m / speed_mps
    n_frames = int(duration_s * fps)

    path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    px_per_m = (width - 80) / pool_length_m
    for i in range(n_frames):
        frame = np.full((height, width, 3), 200, dtype=np.uint8)  # light "water"
        x = int(40 + speed_mps * (i / fps) * px_per_m)
        cv2.circle(frame, (x, height // 2), 10, (0, 255, 255), -1)  # yellow cap
        writer.write(frame)
    writer.release()

    calib = PoolCalibrator()
    calib.add_point((40, height // 2), 0.0)
    calib.add_point((40 + pool_length_m * px_per_m, height // 2), pool_length_m)
    calib.solve()

    tracker = CapTracker(cap_color="yellow", lane_polygon=None)
    cap = cv2.VideoCapture(path)
    times, dists = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tp = tracker.track_frame(frame, idx)
        d = calib.pixel_to_distance(tp.x_px, tp.y_px)
        times.append(idx / fps)
        dists.append(d)
        idx += 1
    cap.release()
    os.unlink(path)

    report = compute_splits(np.array(times), np.array(dists), pool_length_m=pool_length_m)
    assert report.total_time_s is not None
    assert abs(report.total_time_s - duration_s) < 0.2, \
        f"expected ~{duration_s:.2f}s, got {report.total_time_s:.2f}s"
    print("test_full_synthetic_video_pipeline_pieces: OK  (measured",
          round(report.total_time_s, 2), "s vs true", round(duration_s, 2), "s)")


def test_calibration_keyframe_serialization():
    from src.vision.calibration import CalibrationKeyframe
    import dataclasses
    
    # Test old style (no cap_color or lane_number)
    kf_old = CalibrationKeyframe(
        frame_idx=10,
        reference_points=[((1.0, 2.0), 5.0)],
        lane_polygon_px=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    )
    assert kf_old.cap_color is None
    assert kf_old.lane_number is None
    
    # Test new style
    kf_new = CalibrationKeyframe(
        frame_idx=15,
        reference_points=[((10.0, 20.0), 25.0)],
        lane_polygon_px=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        cap_color="#FF0000",
        lane_number=5
    )
    assert kf_new.cap_color == "#FF0000"
    assert kf_new.lane_number == 5
    
    # Test to dict and from dict
    d = dataclasses.asdict(kf_new)
    kf_restored = CalibrationKeyframe(**d)
    assert kf_restored.cap_color == "#FF0000"
    assert kf_restored.lane_number == 5
    
    # Test backward compatibility deserialization
    d_old = dataclasses.asdict(kf_old)
    # Remove the None defaults as if loaded from old JSON
    del d_old["cap_color"]
    del d_old["lane_number"]
    
    kf_old_restored = CalibrationKeyframe(**d_old)
    assert kf_old_restored.cap_color is None
    assert kf_old_restored.lane_number is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"{t.__name__}: FAILED — {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
