"""
Unit tests for ExtendedKalmanFilter6D (src/tracking/tracker.py).

Tests verify:
  1. Initialisation from the first measurement
  2. Prediction advances state smoothly (no NaNs)
  3. Update corrects toward measurement
  4. After N prediction-only steps the position drifts predictably
"""
import math
import sys
from pathlib import Path

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.tracking.tracker import ExtendedKalmanFilter6D


class TestEKFInitialisation:
    def test_predict_before_init_returns_nan(self):
        ekf = ExtendedKalmanFilter6D()
        x, y = ekf.predict()
        assert math.isnan(x) and math.isnan(y)

    def test_update_initialises_on_first_call(self):
        ekf = ExtendedKalmanFilter6D()
        rx, ry = ekf.update(100.0, 200.0)
        assert ekf._initialized
        # After the first update the corrected position should be near (100, 200)
        assert abs(rx - 100.0) < 10.0
        assert abs(ry - 200.0) < 10.0

    def test_manual_initialize(self):
        ekf = ExtendedKalmanFilter6D()
        ekf.initialize(50.0, 75.0)
        assert ekf._initialized
        x, y, *_ = ekf.state
        assert abs(x - 50.0) < 1e-3
        assert abs(y - 75.0) < 1e-3


class TestEKFPrediction:
    def _make_moving_ekf(self, vx=5.0, vy=0.0):
        """Set up an EKF at origin with an initial velocity."""
        ekf = ExtendedKalmanFilter6D()
        # Seed with two measurements to let the filter learn velocity
        ekf.update(0.0, 0.0)
        ekf.update(vx, vy)
        return ekf

    def test_predict_produces_no_nan(self):
        ekf = self._make_moving_ekf()
        for _ in range(30):
            px, py = ekf.predict()
            assert not math.isnan(px), "predict() returned NaN for x"
            assert not math.isnan(py), "predict() returned NaN for y"

    def test_predict_extrapolates_forward(self):
        """After learning a rightward velocity, predictions should keep moving right."""
        ekf = self._make_moving_ekf(vx=10.0)
        prev_x = ekf.state[0]
        for _ in range(10):
            px, py = ekf.predict()
            assert px >= prev_x - 0.5, f"EKF x moved backwards: {px:.2f} < {prev_x:.2f}"
            prev_x = px

    def test_predict_state_has_six_components(self):
        ekf = ExtendedKalmanFilter6D()
        ekf.initialize(0.0, 0.0)
        ekf.predict()
        state = ekf.state
        assert len(state) == 6


class TestEKFUpdate:
    def test_update_pulls_toward_measurement(self):
        """After many prediction-only steps the filter drifts; an update should
        pull the estimate back toward the true measurement."""
        ekf = ExtendedKalmanFilter6D()
        ekf.initialize(0.0, 0.0)
        # Drift the filter far from origin with predictions only
        for _ in range(50):
            ekf.predict()

        # True position is at origin — update should snap back
        rx, ry = ekf.update(0.0, 0.0)
        assert abs(rx) < 50.0, "update() did not pull estimate toward measurement"
        assert abs(ry) < 50.0

    def test_repeated_updates_converge(self):
        """Repeated updates at a fixed point should converge to that point."""
        ekf = ExtendedKalmanFilter6D()
        target_x, target_y = 300.0, 150.0
        for _ in range(30):
            rx, ry = ekf.update(target_x, target_y)
        assert abs(rx - target_x) < 5.0
        assert abs(ry - target_y) < 5.0


class TestMahalanobisGate:
    def test_gate_accepts_plausible_measurement(self):
        ekf = ExtendedKalmanFilter6D()
        ekf.initialize(100.0, 100.0)
        # Small step consistent with tracking
        rx, ry, accepted = ekf.gated_update(105.0, 100.0)
        assert accepted
        assert abs(rx - 105.0) < 5.0

    def test_gate_rejects_wild_outlier(self):
        ekf = ExtendedKalmanFilter6D()
        ekf.initialize(100.0, 100.0)
        # Settle filter with a few frames
        ekf.gated_update(102.0, 100.0)
        ekf.gated_update(104.0, 100.0)
        # Huge sudden jump (simulating splash artifact on another lane)
        rx, ry, accepted = ekf.gated_update(800.0, 500.0)
        assert not accepted
        # Should retain predicted position, not jump to (800, 500)
        assert rx < 150.0


class TestAdaptiveQ:
    def test_phase_changes_process_noise(self):
        from src.tracking.tracker import SwimmerKalmanFilter, PHASE_Q
        kf = SwimmerKalmanFilter()
        steady_q = kf.kf.processNoiseCov.copy()

        kf.set_phase("start")
        start_q = kf.kf.processNoiseCov.copy()
        assert np.allclose(start_q, PHASE_Q["start"])
        assert not np.allclose(start_q, steady_q)

        kf.set_phase("turn")
        assert np.allclose(kf.kf.processNoiseCov, PHASE_Q["turn"])

        kf.set_phase("steady")
        assert np.allclose(kf.kf.processNoiseCov, PHASE_Q["steady"])


class TestCoastMode:
    def test_covariance_grows_during_coast(self):
        from src.tracking.tracker import SwimmerKalmanFilter
        kf = SwimmerKalmanFilter()
        kf.initialize(50.0, 50.0)

        # Settle filter
        for i in range(5):
            kf.gated_update(50.0 + i * 2, 50.0)

        initial_trace = kf.covariance_trace
        for _ in range(10):
            kf.predict()

        assert kf.coast_frames == 10
        assert kf.covariance_trace > initial_trace

        # Update resets coast frames
        kf.gated_update(70.0, 50.0)
        assert kf.coast_frames == 0

    def test_prolonged_coast_stays_bounded(self):
        """Regression: a constant-acceleration filter must NOT extrapolate the
        position to infinity during a long predict-only coast.

        Before the coast-bounding fix, a poisoned velocity/acceleration made
        the position diverge quadratically — on the splash fixture the tracked
        point ran ~1,000,000 px off a 1280px frame. Here we hand the filter a
        strong rightward run, then coast 200 frames and assert the estimate
        stays physically plausible (bounded) and finite, and that the per-frame
        step shrinks toward a position-hold rather than growing.
        """
        from src.tracking.tracker import SwimmerKalmanFilter, MAX_PLAUSIBLE_SPEED_PX
        kf = SwimmerKalmanFilter()
        kf.initialize(200.0, 100.0)
        # Establish a fast rightward velocity (near the plausible ceiling).
        for x in range(210, 460, 50):
            kf.gated_update(float(x), 100.0)

        xs = []
        for _ in range(200):
            px, py = kf.predict()
            assert math.isfinite(px) and math.isfinite(py)
            xs.append(px)

        # Bounded: nowhere near the old ~1e6 runaway. Total coast travel is
        # capped by the velocity clamp + decay, so a few thousand px at most.
        assert abs(xs[-1]) < 5000.0, f"coast ran away to x={xs[-1]:.1f}"
        # Per-frame step must never exceed the physical velocity clamp...
        steps = [abs(xs[i] - xs[i - 1]) for i in range(1, len(xs))]
        assert max(steps) <= MAX_PLAUSIBLE_SPEED_PX + 1e-3
        # ...and should decay toward a hold (late steps far smaller than early).
        assert steps[-1] < steps[0]
        assert steps[-1] < 1.0


class TestKalmanPointTrackerGating:
    def test_cap_tracker_gated_step_and_phase(self):
        from src.vision.cap_tracker import KalmanPointTracker
        kp = KalmanPointTracker()
        kp.init(100.0, 200.0)

        # Plausible updates accepted
        x, y, acc = kp.gated_step((105.0, 200.0))
        assert acc
        x, y, acc = kp.gated_step((110.0, 200.0))
        assert acc

        # Wild jump rejected
        x, y, acc = kp.gated_step((900.0, 900.0))
        assert not acc
        assert x < 150.0

        # Phase change works
        kp.set_phase("start")
        assert kp._current_phase == "start"

