"""
Module C: Anti-Splash Tracking Engine

Implements the three-layer tracking strategy from the blueprint:

  Layer 1 — Detector observation (YOLOv8 or HOG).
             When confidence ≥ SPLASH_CONFIDENCE_GATE, we trust the detector
             and update the EKF with the real measurement.

  Layer 2 — EKF prediction (constant-acceleration motion model).
             When the detector drops below the gate, we keep the bounding
             box moving smoothly using the filter's own predicted trajectory.
             This prevents the tracker from "freezing" during a splash burst.

  Layer 3 — Gemini Vision Recovery.
             If the detector has been below the confidence gate for more than
             SPLASH_RECOVERY_FRAMES consecutive frames, we crop the frame and
             call the Gemini Vision API to ask it to locate the swimmer's
             centre-of-mass in the turbulent water.  The returned coordinate
             re-seeds the EKF so it doesn't drift indefinitely.

The EKF uses a 6-state constant-acceleration model:
    state  = [x, y, vx, vy, ax, ay]
    motion = x_{t+1} = x_t + vx*dt + 0.5*ax*dt²  (and analogously for y)
    observation = [x, y]  (centre of bounding box)

Design decisions (documented per implementation plan):
  - Named SwimmerKalmanFilter (not "EKF") because the motion model is linear.
    A UKF would add sigma-point overhead with zero accuracy gain for linear
    kinematics. If a non-linear observation model is added later (e.g.,
    fisheye lens distortion), the UKF switch becomes justified.
  - Mahalanobis distance gating: reject measurements that are statistically
    implausible given the filter's predicted state and uncertainty.
  - Adaptive process noise (Q): different tuning for start/turn/steady phases.
  - Coast mode: covariance grows during predict-only frames; consecutive coast
    frames are counted and surfaced for low-confidence segment annotation.
"""
from __future__ import annotations

import logging
import tempfile
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..detection.detector import Track, DetectorProtocol
from ..detection.selector import TargetSelector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptive process noise presets by swim phase
# ---------------------------------------------------------------------------

PHASE_Q = {
    "steady":   np.diag([1.0, 1.0, 5.0, 5.0, 5.0, 5.0]).astype(np.float32),
    "start":    np.diag([4.0, 4.0, 40.0, 40.0, 80.0, 80.0]).astype(np.float32),
    "turn":     np.diag([4.0, 4.0, 40.0, 40.0, 80.0, 80.0]).astype(np.float32),
    "breakout": np.diag([2.0, 2.0, 20.0, 20.0, 40.0, 40.0]).astype(np.float32),
}

# Maximum plausible swimmer speed in px/frame for re-ID validation
MAX_PLAUSIBLE_SPEED_PX = 60.0  # ~2m/s at typical resolution


# ---------------------------------------------------------------------------
# SwimmerKalmanFilter — 6-state constant-acceleration model
# ---------------------------------------------------------------------------

class SwimmerKalmanFilter:
    """
    Kalman filter with state vector [x, y, vx, vy, ax, ay].

    Named SwimmerKalmanFilter (not "EKF") because the motion model is linear
    (constant acceleration). The transition Jacobian ≡ F, making this
    mathematically equivalent to a standard Kalman filter. A UKF would add
    sigma-point computation overhead with zero accuracy gain for a linear
    system. If a non-linear observation model is added later (e.g., fisheye
    lens distortion correction), the UKF switch becomes justified.

    Includes:
      - Mahalanobis distance gating for measurement validation
      - Adaptive process noise (Q) for different swim phases
      - Coast mode with covariance growth tracking
    """

    DT: float = 1.0  # time step in frames; can be overridden per-call

    # Default Mahalanobis gate threshold: chi-squared(2 DOF, p=0.99) = 9.21
    DEFAULT_GATE_THRESHOLD: float = 9.21

    # Covariance growth factor per coast frame (P *= this each predict-only step)
    COAST_COV_GROWTH: float = 1.08

    # Coast-prediction bounding. A constant-acceleration model in pure
    # predict-only mode diverges *quadratically* (position ∝ ½·a·t²), so a
    # single poisoned state extrapolates the swimmer off to infinity. Real
    # unobserved motion is not unbounded acceleration, so during coast we:
    #   • decay the (unobserved) acceleration each frame so it cannot compound,
    #   • hard-clamp velocity to a plausible swim speed, and
    #   • once the coast is prolonged (target effectively lost) decay velocity
    #     toward zero so the estimate settles into a position-hold with growing
    #     covariance rather than sliding off-screen.
    COAST_ACCEL_DECAY: float = 0.5     # per coast frame
    COAST_VEL_DECAY: float = 0.85      # per coast frame, once prolonged
    COAST_VEL_DECAY_AFTER: int = 8     # coast frames before velocity decay kicks in

    def __init__(self) -> None:
        n, m = 6, 2
        self.kf = cv2.KalmanFilter(n, m)

        dt = self.DT
        # Transition matrix: constant-acceleration kinematics
        F = np.eye(n, dtype=np.float32)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * dt * dt
        F[1, 5] = 0.5 * dt * dt
        F[2, 4] = dt
        F[3, 5] = dt
        self.kf.transitionMatrix = F

        # Measurement matrix: we observe [x, y]
        H = np.zeros((m, n), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        self.kf.measurementMatrix = H

        # Process noise — start with "steady" preset
        self.kf.processNoiseCov = PHASE_Q["steady"].copy()
        self._current_phase = "steady"

        # Measurement noise — ~5px bounding-box jitter
        self.kf.measurementNoiseCov = np.eye(m, dtype=np.float32) * 25.0

        # Initial state uncertainty — large, so the first measurement dominates
        self.kf.errorCovPost = np.eye(n, dtype=np.float32) * 500.0

        self._initialized = False
        self._coast_frames = 0
        self._last_bbox_area: float | None = None  # for re-ID size check

    def set_phase(self, phase: str) -> None:
        """Set the swim phase to adjust process noise accordingly.

        Args:
            phase: One of "steady", "start", "turn", "breakout".
        """
        if phase in PHASE_Q and phase != self._current_phase:
            self.kf.processNoiseCov = PHASE_Q[phase].copy()
            self._current_phase = phase
            logger.debug("Kalman phase → %s", phase)

    def initialize(self, x: float, y: float) -> None:
        state = np.array([[x], [y], [0], [0], [0], [0]], dtype=np.float32)
        self.kf.statePost = state
        self.kf.statePre = state.copy()
        self._initialized = True
        self._coast_frames = 0
        logger.debug("KF initialised at (%.1f, %.1f)", x, y)

    def predict(self) -> tuple[float, float]:
        """Advance the filter one step (no measurement). Returns predicted (x, y).

        During coast, covariance grows by COAST_COV_GROWTH per frame, and the
        state is bounded (see the COAST_* constants) so a poisoned velocity /
        acceleration cannot extrapolate the position off-screen.
        """
        if not self._initialized:
            return float("nan"), float("nan")
        self.kf.predict()
        self._coast_frames += 1
        # Grow covariance during coast to reflect increasing uncertainty
        if self._coast_frames > 0:
            self.kf.errorCovPost *= self.COAST_COV_GROWTH

        # Bound the coast so it stays physically plausible. cv2's predict()
        # advances statePost (this is what makes successive predicts chain and,
        # unbounded, diverge). We damp the higher-order terms in-place so the
        # NEXT predict starts from a sane state.
        s = self.kf.statePost
        s[4, 0] *= self.COAST_ACCEL_DECAY   # ax — unobserved, must not compound
        s[5, 0] *= self.COAST_ACCEL_DECAY   # ay
        s[2, 0] = float(np.clip(s[2, 0], -MAX_PLAUSIBLE_SPEED_PX, MAX_PLAUSIBLE_SPEED_PX))
        s[3, 0] = float(np.clip(s[3, 0], -MAX_PLAUSIBLE_SPEED_PX, MAX_PLAUSIBLE_SPEED_PX))
        if self._coast_frames > self.COAST_VEL_DECAY_AFTER:
            # Prolonged coast → target effectively lost: relax velocity toward a
            # position-hold so the estimate stops sliding away while covariance
            # keeps growing (honest "I no longer know where it is").
            s[2, 0] *= self.COAST_VEL_DECAY
            s[3, 0] *= self.COAST_VEL_DECAY
        self.kf.statePost = s
        return float(s[0, 0]), float(s[1, 0])

    def gated_update(
        self, x: float, y: float,
        gate_threshold: float | None = None,
    ) -> tuple[float, float, bool]:
        """Correct the filter with a measurement, applying Mahalanobis gating.

        Args:
            x, y: Measurement coordinates.
            gate_threshold: Chi-squared threshold for the 2-DOF gate.
                Defaults to DEFAULT_GATE_THRESHOLD (9.21, p=0.99).

        Returns:
            (corrected_x, corrected_y, accepted) where accepted is True if
            the measurement passed the gate and was used for correction.
        """
        if not self._initialized:
            self.initialize(x, y)
            return x, y, True

        threshold = gate_threshold or self.DEFAULT_GATE_THRESHOLD

        # Predict first (required before correct)
        self.kf.predict()

        # Compute innovation (measurement residual)
        H = self.kf.measurementMatrix
        z = np.array([[x], [y]], dtype=np.float32)
        z_pred = H @ self.kf.statePre
        innovation = z - z_pred

        # Innovation covariance: S = H @ P_pred @ H^T + R
        P_pred = self.kf.errorCovPre
        R = self.kf.measurementNoiseCov
        S = H @ P_pred @ H.T + R

        # Mahalanobis distance: d² = innovation^T @ S^{-1} @ innovation
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular S — accept measurement unconditionally
            corrected = self.kf.correct(z)
            self._coast_frames = 0
            return float(corrected[0, 0]), float(corrected[1, 0]), True

        mahal_dist_sq = float((innovation.T @ S_inv @ innovation)[0, 0])

        if mahal_dist_sq > threshold:
            # Measurement rejected — revert to prediction
            logger.debug(
                "Mahalanobis gate REJECT: d²=%.1f > threshold=%.1f "
                "(measurement=(%.1f, %.1f), predicted=(%.1f, %.1f))",
                mahal_dist_sq, threshold, x, y,
                float(z_pred[0, 0]), float(z_pred[1, 0]),
            )
            # Don't reset coast counter — this was a rejected noisy measurement
            return float(self.kf.statePre[0, 0]), float(self.kf.statePre[1, 0]), False

        # Measurement accepted — correct the filter
        corrected = self.kf.correct(z)
        self._coast_frames = 0
        return float(corrected[0, 0]), float(corrected[1, 0]), True

    def update(self, x: float, y: float) -> tuple[float, float]:
        """Correct the filter with a measurement (no gating, backward compat).

        Returns corrected (x, y).
        """
        if not self._initialized:
            self.initialize(x, y)
            return x, y
        self.kf.predict()
        meas = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kf.correct(meas)
        self._coast_frames = 0
        return float(corrected[0, 0]), float(corrected[1, 0])

    @property
    def state(self) -> tuple[float, float, float, float, float, float]:
        """Return current state estimate (x, y, vx, vy, ax, ay)."""
        s = self.kf.statePost
        return tuple(float(s[i, 0]) for i in range(6))  # type: ignore[return-value]

    @property
    def coast_frames(self) -> int:
        """Number of consecutive frames without a valid measurement."""
        return self._coast_frames

    @property
    def covariance_trace(self) -> float:
        """Trace of the position sub-block of the error covariance.
        Higher = more uncertain. Useful for confidence estimation."""
        P = self.kf.errorCovPost
        return float(P[0, 0] + P[1, 1])


# Backward-compatible alias
ExtendedKalmanFilter6D = SwimmerKalmanFilter


# ---------------------------------------------------------------------------
# Gemini Vision Recovery Agent
# ---------------------------------------------------------------------------

@dataclass
class SplashRecoveryResult:
    x: float
    y: float
    confidence: float
    source: str = "gemini_recovery"


class SplashRecoveryAgent:
    """
    Calls the Gemini Vision API to locate a swimmer whose bounding-box
    detector has lost confidence due to splash/bubble occlusion.

    Now sends short video clips (the coast segment) instead of single frames,
    giving Gemini temporal context for better swimmer localization.

    This is an additive layer — if the API is unavailable, the EKF prediction
    continues as Layer 2, and we simply accept more drift until the detector
    recovers.
    """

    SINGLE_FRAME_PROMPT = (
        "This is a frame from a competitive swimming race. "
        "The swimmer is partially obscured by splash and turbulent water. "
        "Locate the swimmer's centre of mass (the approximate geometric centre "
        "of their body). "
        "Return ONLY a JSON object with keys 'x' and 'y' as pixel coordinates "
        "(integers, 0-indexed from the top-left corner). "
        "Do not include any explanation, markdown, or extra text."
    )

    VIDEO_CLIP_PROMPT = (
        "This is a short video clip from a competitive swimming race showing "
        "a segment where the swimmer is partially obscured by splash and "
        "turbulent water. "
        "For the MIDDLE frame of this clip, locate the swimmer's centre of mass "
        "(the approximate geometric centre of their body). "
        "Return ONLY a JSON object with keys 'x' and 'y' as pixel coordinates "
        "(integers, 0-indexed from the top-left corner), and 'confidence' "
        "as a float 0-1 indicating how certain you are. "
        "Do not include any explanation, markdown, or extra text."
    )

    def __init__(self, api_key: str | None, model: str | None = None) -> None:
        self._api_key = api_key
        self._model = model or "gemini-2.0-flash"
        self._client = None
        self._available = False
        if api_key:
            try:
                from google import genai  # noqa: PLC0415
                self._client = genai.Client(api_key=api_key)
                self._available = True
                logger.info("SplashRecoveryAgent: Gemini client ready (model=%s)", self._model)
            except ImportError:
                logger.warning("google-genai not installed; Gemini splash recovery disabled.")
            except Exception as exc:
                logger.warning("Gemini client init failed: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def locate_swimmer(self, frame_bgr: np.ndarray) -> SplashRecoveryResult | None:
        """
        Encode the frame as JPEG, send to Gemini, parse the returned JSON.
        Returns None on any error (API unavailable, parse failure, etc.).
        """
        if not self._available or self._client is None:
            return None

        try:
            import json  # noqa: PLC0415
            from google.genai import types  # noqa: PLC0415

            ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None
            image_bytes = buf.tobytes()

            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    self.SINGLE_FRAME_PROMPT,
                ],
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()

            data = json.loads(raw)
            x = float(data["x"])
            y = float(data["y"])
            h, w = frame_bgr.shape[:2]
            x = max(0.0, min(x, float(w)))
            y = max(0.0, min(y, float(h)))
            logger.debug("Gemini recovery: swimmer at (%.1f, %.1f)", x, y)
            return SplashRecoveryResult(x=x, y=y, confidence=0.7)

        except Exception as exc:  # noqa: BLE001
            logger.warning("SplashRecoveryAgent.locate_swimmer failed: %s", exc)
            return None

    def locate_swimmer_in_segment(
        self,
        frames: list[np.ndarray],
        fps: float = 30.0,
    ) -> SplashRecoveryResult | None:
        """
        Send a short video clip (the coast segment) to Gemini for swimmer
        localization. This gives Gemini temporal context (motion direction,
        stroke phase) instead of a single ambiguous frame.

        Falls back to single-frame mode if video encoding fails.

        Args:
            frames: List of BGR frames comprising the coast segment.
            fps: Frame rate for encoding the temporary video clip.

        Returns:
            SplashRecoveryResult for the middle frame, or None on failure.
        """
        if not self._available or self._client is None:
            return None

        if len(frames) < 3:
            # Too short for video context — use single frame
            mid = len(frames) // 2
            return self.locate_swimmer(frames[mid])

        try:
            import json  # noqa: PLC0415
            import time  # noqa: PLC0415

            # Encode frames as a temporary video clip
            h, w = frames[0].shape[:2]
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                temp_path = tf.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()

            # Upload to Gemini
            logger.info(
                "Uploading %d-frame coast segment to Gemini for recovery",
                len(frames),
            )
            video_file = self._client.files.upload(file=temp_path)

            # Poll until processing is done
            retries = 0
            while video_file.state.name == "PROCESSING":
                if retries > 30:
                    logger.warning("Coast segment processing timed out")
                    os.remove(temp_path)
                    return None
                time.sleep(2)
                video_file = self._client.files.get(name=video_file.name)
                retries += 1

            if video_file.state.name == "FAILED":
                logger.warning("Coast segment processing failed on Gemini's side")
                os.remove(temp_path)
                return None

            # Send query
            response = self._client.models.generate_content(
                model=self._model,
                contents=[video_file, self.VIDEO_CLIP_PROMPT],
                config={"temperature": 0.2},
            )

            # Cleanup
            try:
                self._client.files.delete(name=video_file.name)
            except Exception:
                pass
            try:
                os.remove(temp_path)
            except OSError:
                pass

            if not response or not response.text:
                return None

            raw = response.text.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines)

            # Try JSON parse
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
            else:
                data = json.loads(raw)

            x = float(data["x"])
            y = float(data["y"])
            conf = float(data.get("confidence", 0.7))
            x = max(0.0, min(x, float(w)))
            y = max(0.0, min(y, float(h)))

            logger.info(
                "Gemini video recovery: swimmer at (%.1f, %.1f) conf=%.2f",
                x, y, conf,
            )
            return SplashRecoveryResult(x=x, y=y, confidence=conf)

        except Exception as exc:  # noqa: BLE001
            logger.warning("SplashRecoveryAgent.locate_swimmer_in_segment failed: %s", exc)
            # Fall back to single-frame
            mid = len(frames) // 2
            return self.locate_swimmer(frames[mid])


# ---------------------------------------------------------------------------
# Main tracker: ties Detector + KF + Gemini together
# ---------------------------------------------------------------------------

@dataclass
class TrackResult:
    """Output of AntiSplashTracker for a single frame."""
    frame_idx: int
    x_px: float
    y_px: float
    confidence: float
    source: str          # "detector", "ekf_predict", "gemini_recovery"
    target_id: int | None = None
    gate_rejected: bool = False  # True if measurement was rejected by Mahalanobis gate


class AntiSplashTracker:
    """
    Per-frame tracking loop that implements the blueprint's Decision Gate:

        IF detector_confidence >= gate  → update KF with gating (Layer 1 + Layer 2 fusion)
        ELSE (splash detected)          → KF predict only (Layer 2)
        IF splash_frames > threshold    → call Gemini, re-seed KF (Layer 3)

    Includes:
      - Mahalanobis gating: reject statistically implausible measurements
      - Adaptive Q: different process noise for start/turn/steady phases
      - Coast mode: growing covariance during predict-only
      - Multi-signal re-ID: validate recovered detection against multiple criteria

    Call process_frame() for each frame in the video loop.
    """

    def __init__(
        self,
        detector: DetectorProtocol,
        selector: TargetSelector,
        recovery_agent: SplashRecoveryAgent | None = None,
        confidence_gate: float = 0.4,
        recovery_frame_threshold: int = 10,
    ) -> None:
        self._detector = detector
        self._selector = selector
        self._recovery = recovery_agent
        self._gate = confidence_gate
        self._recovery_threshold = recovery_frame_threshold

        self._ekf = SwimmerKalmanFilter()
        self._splash_frames: int = 0    # consecutive low-confidence frames
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._last_bbox_area: float | None = None
        self._coast_buffer: list[np.ndarray] = []  # frames accumulated during coast

    def _bbox_center(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, h = bbox
        return x + w / 2, y + h / 2

    def _bbox_area(self, bbox: tuple[float, float, float, float]) -> float:
        _, _, w, h = bbox
        return w * h

    def _validate_reacquisition(
        self,
        candidate_bbox: tuple[float, float, float, float],
        candidate_x: float,
        candidate_y: float,
    ) -> bool:
        """
        Multi-signal validation for detection recovery after coast.
        Reject if ≥2 of 3 checks fail (even if YOLO confidence is high).

        Checks:
          1. Position: within covariance-scaled search window
          2. Size: bounding box area within 2× of last known
          3. Velocity: implied speed from last-known to candidate is plausible
        """
        if not self._ekf._initialized:
            return True  # No prior state to validate against

        fails = 0
        state = self._ekf.state
        pred_x, pred_y = state[0], state[1]
        vx, vy = state[2], state[3]

        # 1. Position check: is the candidate within a reasonable distance?
        # Use covariance trace as a scale factor for the search radius
        cov_scale = max(1.0, self._ekf.covariance_trace ** 0.5)
        max_dist = min(cov_scale * 2.0, 200.0)  # cap at 200px
        dist = ((candidate_x - pred_x) ** 2 + (candidate_y - pred_y) ** 2) ** 0.5
        if dist > max_dist:
            fails += 1

        # 2. Size check: bounding box area within 2× of last known
        if self._last_bbox_area is not None:
            candidate_area = self._bbox_area(candidate_bbox)
            ratio = candidate_area / self._last_bbox_area if self._last_bbox_area > 0 else 1.0
            if ratio < 0.3 or ratio > 3.0:
                fails += 1

        # 3. Velocity check: implied speed must be physically plausible
        # (coast frames × speed should roughly match the position jump)
        if self._splash_frames > 0:
            implied_speed = dist / max(self._splash_frames, 1)
            if implied_speed > MAX_PLAUSIBLE_SPEED_PX:
                fails += 1

        # 4. Direction check: if the swimmer had real momentum when the coast
        # began, a re-detection *behind* the direction of travel is almost
        # certainly a different object (residual splash, a deck person, an
        # adjacent-lane swimmer) rather than our target reversing. Only fires
        # when |v| is meaningful, so a swimmer who submerged near-stationary
        # (e.g. plateauing before a breakout) is not penalised — that ambiguous
        # case is left for the Gemini resolver / appearance re-ID.
        speed = (vx * vx + vy * vy) ** 0.5
        if speed > 5.0:
            dx, dy = candidate_x - pred_x, candidate_y - pred_y
            along = (dx * vx + dy * vy) / speed  # signed projection onto heading
            if along < -30.0:  # clearly behind, beyond position noise
                fails += 1

        if fails >= 2:
            logger.debug(
                "Re-ID validation REJECT: %d/4 checks failed "
                "(dist=%.1f, last_area=%s, coast=%d frames)",
                fails, dist, self._last_bbox_area, self._splash_frames,
            )
            return False
        return True

    def set_phase(self, phase: str) -> None:
        """Set the current swim phase for adaptive Q tuning."""
        self._ekf.set_phase(phase)

    def process_frame(
        self, frame_bgr: np.ndarray, frame_idx: int
    ) -> TrackResult:
        """
        Run one iteration of the three-layer tracking loop.

        Returns a TrackResult with the best available (x, y) estimate and
        a source string indicating which layer provided it.
        """
        # 1. Run detector on the full frame
        all_tracks = self._detector.detect(frame_bgr)

        # 2. Find the target track (if selector has a lock)
        target_track = self._selector.get_track(all_tracks)

        # Auto-lock onto most confident track if no target was pre-selected
        if target_track is None and self._selector.target_id is None and all_tracks:
            best_track = max(all_tracks, key=lambda t: t.conf)
            target_track = best_track
            self._selector.target_id = best_track.id

        # Re-associate to closest plausible track if current ID is lost
        if target_track is None and self._ekf._initialized and all_tracks:
            state = self._ekf.state
            px, py = state[0], state[1]
            candidates = []
            for trk in all_tracks:
                if trk.conf >= self._gate:
                    cx, cy = self._bbox_center(trk.bbox)
                    d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if d < 180.0 and self._validate_reacquisition(trk.bbox, cx, cy):
                        candidates.append((d, trk))
            if candidates:
                candidates.sort(key=lambda c: c[0])
                target_track = candidates[0][1]
                self._selector.target_id = target_track.id

        # 3. Decision gate
        if target_track is not None and target_track.conf >= self._gate:
            cx, cy = self._bbox_center(target_track.bbox)

            # If we were coasting, validate the re-acquisition
            if self._splash_frames > 3:
                if not self._validate_reacquisition(target_track.bbox, cx, cy):
                    # Reject — keep coasting
                    self._splash_frames += 1
                    px, py = self._ekf.predict()
                    self._coast_buffer.append(frame_bgr.copy())
                    return TrackResult(
                        frame_idx=frame_idx,
                        x_px=px if not np.isnan(px) else 0.0,
                        y_px=py if not np.isnan(py) else 0.0,
                        confidence=0.1,
                        source="ekf_predict",
                        target_id=self._selector.target_id,
                        gate_rejected=True,
                    )

            # ── Layer 1+2: real measurement → update KF with gating ──────
            rx, ry, accepted = self._ekf.gated_update(cx, cy)
            if accepted:
                self._splash_frames = 0
                self._last_bbox = target_track.bbox
                self._last_bbox_area = self._bbox_area(target_track.bbox)
                self._coast_buffer.clear()
                return TrackResult(
                    frame_idx=frame_idx, x_px=rx, y_px=ry,
                    confidence=target_track.conf, source="detector",
                    target_id=target_track.id,
                )
            else:
                # Measurement rejected by Mahalanobis gate — treat as coast
                self._splash_frames += 1
                self._coast_buffer.append(frame_bgr.copy())
                return TrackResult(
                    frame_idx=frame_idx, x_px=rx, y_px=ry,
                    confidence=0.15, source="ekf_predict",
                    target_id=target_track.id,
                    gate_rejected=True,
                )

        else:
            # ── Layer 2: splash — KF predict ─────────────────────────────
            self._splash_frames += 1
            px, py = self._ekf.predict()
            source = "ekf_predict"

            # Buffer frames for potential video-based Gemini recovery
            self._coast_buffer.append(frame_bgr.copy())
            # Cap coast buffer to prevent unbounded memory growth
            if len(self._coast_buffer) > 90:  # ~3s at 30fps
                self._coast_buffer = self._coast_buffer[-60:]

            # ── Layer 3: prolonged splash — ask Gemini ─────────────────────
            if (
                self._splash_frames > self._recovery_threshold
                and self._recovery is not None
                and self._recovery.available
            ):
                # Try video clip first (gives Gemini temporal context)
                recovery = None
                if len(self._coast_buffer) >= 3:
                    recovery = self._recovery.locate_swimmer_in_segment(
                        self._coast_buffer,
                        fps=30.0,
                    )
                if recovery is None:
                    # Fall back to single frame
                    recovery = self._recovery.locate_swimmer(frame_bgr)

                if recovery is not None:
                    # Re-seed the KF with Gemini's coordinate
                    rx, ry = self._ekf.update(recovery.x, recovery.y)
                    self._splash_frames = 0
                    self._coast_buffer.clear()
                    logger.info(
                        "Frame %d: Gemini re-seeded KF at (%.1f, %.1f)",
                        frame_idx, rx, ry,
                    )
                    return TrackResult(
                        frame_idx=frame_idx, x_px=rx, y_px=ry,
                        confidence=recovery.confidence,
                        source="gemini_recovery",
                        target_id=self._selector.target_id,
                    )

            # Return KF prediction (may be nan if never initialized)
            return TrackResult(
                frame_idx=frame_idx,
                x_px=px if not np.isnan(px) else 0.0,
                y_px=py if not np.isnan(py) else 0.0,
                confidence=0.1,
                source=source,
                target_id=self._selector.target_id,
            )

    def seed(self, x: float, y: float) -> None:
        """Manually seed the KF (e.g. from the first detection after target selection)."""
        self._ekf.initialize(x, y)

    @property
    def consecutive_splash_frames(self) -> int:
        return self._splash_frames
