"""
Swim-cap tracking.

Design choice, stated plainly: this uses HSV colour-threshold detection +
a constant-velocity Kalman filter, not a YOLOv8 "swim-specific" detector.
Why:
  - There is no publicly available pretrained YOLO model trained
    specifically for swim caps / swimmers-in-lanes. What exists is generic
    person/pose detectors trained on COCO (see README "Sources"). Bundling
    one wouldn't add lane-of-interest discrimination a colour filter
    already gives you for free, and Ultralytics' YOLOv8 weights are
    AGPL-3.0 (or require a paid Enterprise licence for closed-source
    redistribution) — a real licensing cost for a "downloadable .exe"
    product that a permissive colour-tracking approach avoids entirely.
  - Real published race-analysis systems (e.g. the AIMsys system described
    in Gonjo et al., Sci Rep 2025) track swimmers by head/cap colour, not
    generic person detection. This isn't a corner cut; it's the standard
    approach.

A pluggable `Detector` protocol is included at the bottom so a
custom-trained YOLO/ONNX model can be swapped in later without touching
the tracking or Kalman logic — see plug_in_custom_detector().
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class TrackedPoint:
    frame_idx: int
    x_px: float
    y_px: float
    confidence: float          # 0..1, detector confidence for this frame
    source: str                 # "color", "kalman_predict", or "optical_flow"


def hex_or_keyword_to_hsv_ranges(color: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Convert a hex code ('#FF6600') or a common colour keyword ('red',
    'yellow', 'neon green', ...) into a list of HSV lower/upper bound pairs for
    cv2.inRange, with a generous tolerance band (splash + lighting shifts
    the apparent colour of a wet cap a lot more than a dry one).
    Returns a list because red wraps around the 180 hue boundary.
    """
    keywords = {
        "red": (0, 100, 80), "orange": (15, 100, 80), "yellow": (28, 100, 80),
        "green": (55, 100, 60), "neon green": (45, 150, 150),
        "olive": (45, 80, 60), "dark green": (50, 80, 60), 
        "cyan": (90, 100, 80), "blue": (110, 100, 60), "navy": (115, 80, 40),
        "purple": (135, 80, 60), "pink": (155, 80, 100), "white": (0, 0, 200),
        "black": (0, 0, 0), "silver": (0, 0, 140), "gray": (0, 0, 120),
    }

    if color.lower() in keywords:
        h, s, v = keywords[color.lower()]
        bgr = None
    else:
        hexstr = color.lstrip("#")
        if len(hexstr) != 6:
            raise ValueError(f"Unrecognised cap colour: {color!r}")
        r, g, b = (int(hexstr[i:i + 2], 16) for i in (0, 2, 4))
        bgr = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    ranges = []
    # White/black/gray are hue-independent (low saturation); everything
    # else gets a wide +/-25 deg hue window with a low saturation/value floor
    # to tolerate wet-cap specular highlights and pool-water colour cast.
    if color.lower() in ("white", "black", "silver", "gray"):
        lower = np.array([0, 0, max(v - 80, 0)], dtype=np.uint8)
        upper = np.array([180, 80, min(v + 80, 255)], dtype=np.uint8)
        ranges.append((lower, upper))
    else:
        # Much wider tolerance: ±25 hue, saturation can drop very low (wet caps)
        h_margin = 25
        min_s = 30
        min_v = 30
        
        lower_h = h - h_margin
        upper_h = h + h_margin
        
        if lower_h < 0:
            ranges.append((np.array([0, min_s, min_v], dtype=np.uint8), 
                           np.array([upper_h, 255, 255], dtype=np.uint8)))
            ranges.append((np.array([180 + lower_h, min_s, min_v], dtype=np.uint8), 
                           np.array([179, 255, 255], dtype=np.uint8)))
        elif upper_h > 179:
            ranges.append((np.array([lower_h, min_s, min_v], dtype=np.uint8), 
                           np.array([179, 255, 255], dtype=np.uint8)))
            ranges.append((np.array([0, min_s, min_v], dtype=np.uint8), 
                           np.array([upper_h - 180, 255, 255], dtype=np.uint8)))
        else:
            ranges.append((np.array([lower_h, min_s, min_v], dtype=np.uint8), 
                           np.array([upper_h, 255, 255], dtype=np.uint8)))
            
    return ranges


CAP_PHASE_Q = {
    "steady":   np.diag([1.0, 1.0, 5.0, 5.0]).astype(np.float32),
    "start":    np.diag([4.0, 4.0, 25.0, 25.0]).astype(np.float32),
    "turn":     np.diag([4.0, 4.0, 25.0, 25.0]).astype(np.float32),
    "breakout": np.diag([2.0, 2.0, 15.0, 15.0]).astype(np.float32),
}


class KalmanPointTracker:
    """Constant-velocity 2D Kalman filter with Mahalanobis gating.

    The gating rejects implausible measurements (e.g., splash artifacts
    matching the cap color at wrong locations) instead of accepting
    whatever the detector returns unconditionally.
    """

    # Chi-squared threshold for 2 DOF at p=0.99
    DEFAULT_GATE_THRESHOLD: float = 9.21

    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        # Process noise deliberately larger on the velocity terms than
        # position: real cap motion isn't constant-velocity (stroke
        # cycles, starts, wall pushes all accelerate/decelerate), so the
        # filter needs freedom to revise its velocity estimate quickly
        # rather than clinging to an outdated one. The first tuning here
        # (1e-2 process vs 4.0 measurement noise) made the filter trust
        # its own constant-velocity assumption ~400x more than new
        # detections and effectively never learn velocity — caught by
        # test_kalman_bridges_occlusion, fixed by rebalancing these two.
        self.kf.processNoiseCov = CAP_PHASE_Q["steady"].copy()
        self._current_phase = "steady"
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        self.kf.errorCovPost = np.diag([10.0, 10.0, 400.0, 400.0]).astype(np.float32)
        self._initialized = False
        self._coast_frames = 0
        self._frames_tracked = 0

    def set_phase(self, phase: str) -> None:
        """Set the swim phase to adjust process noise accordingly."""
        if phase in CAP_PHASE_Q and phase != self._current_phase:
            self.kf.processNoiseCov = CAP_PHASE_Q[phase].copy()
            self._current_phase = phase

    def init(self, x: float, y: float) -> None:
        self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.errorCovPost = np.diag([10.0, 10.0, 400.0, 400.0]).astype(np.float32)
        self._initialized = True
        self._coast_frames = 0
        self._frames_tracked = 1

    def step(self, measurement: tuple[float, float] | None) -> tuple[float, float]:
        """
        Advance exactly one frame. Pass (x, y) when a detection was found
        this frame, or None to predict-only (bridging an occluded frame).

        predict() must run every single frame — including frames that
        also have a measurement — for the velocity component of the
        state to actually be learned from frame-to-frame motion. Calling
        correct() without a preceding predict() on the same frame leaves
        the filter's internal statePre stale, which silently breaks the
        forward-extrapolation this exists for.
        """
        if not self._initialized:
            if measurement is None:
                return float("nan"), float("nan")
            self.init(*measurement)
            return measurement

        pred = self.kf.predict()
        if measurement is not None:
            c = self.kf.correct(np.array([[measurement[0]], [measurement[1]]], np.float32))
            self._coast_frames = 0
            self._frames_tracked += 1
            return float(c[0, 0]), float(c[1, 0])
        self._coast_frames += 1
        self.kf.errorCovPost *= 1.05
        return float(pred[0, 0]), float(pred[1, 0])

    def gated_step(
        self,
        measurement: tuple[float, float] | None,
        gate_threshold: float | None = None,
    ) -> tuple[float, float, bool]:
        """
        Like step(), but applies Mahalanobis distance gating to the
        measurement. Returns (x, y, accepted) where accepted is False
        if the measurement was rejected as statistically implausible.

        When rejected, the filter uses prediction only (as if no
        measurement was available).
        """
        if not self._initialized:
            if measurement is None:
                return float("nan"), float("nan"), False
            self.init(*measurement)
            return measurement[0], measurement[1], True

        threshold = gate_threshold or self.DEFAULT_GATE_THRESHOLD

        pred = self.kf.predict()

        if measurement is None:
            self._coast_frames += 1
            self.kf.errorCovPost *= 1.05
            return float(pred[0, 0]), float(pred[1, 0]), False

        # Compute Mahalanobis distance
        H = self.kf.measurementMatrix
        z = np.array([[measurement[0]], [measurement[1]]], np.float32)
        z_pred = H @ self.kf.statePre
        innovation = z - z_pred

        P_pred = self.kf.errorCovPre
        R = self.kf.measurementNoiseCov
        S = H @ P_pred @ H.T + R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            c = self.kf.correct(z)
            self._coast_frames = 0
            self._frames_tracked += 1
            return float(c[0, 0]), float(c[1, 0]), True

        mahal_dist_sq = float((innovation.T @ S_inv @ innovation)[0, 0])

        if self._frames_tracked >= 3 and mahal_dist_sq > threshold:
            # Reject — use prediction
            self._coast_frames += 1
            self.kf.errorCovPost *= 1.05
            return float(pred[0, 0]), float(pred[1, 0]), False

        # Accept
        c = self.kf.correct(z)
        self._coast_frames = 0
        self._frames_tracked += 1
        return float(c[0, 0]), float(c[1, 0]), True

    # Back-compat aliases used by a couple of call sites that only ever
    # need one half of step().
    def predict(self) -> tuple[float, float]:
        return self.step(None)

    def correct(self, x: float, y: float) -> tuple[float, float]:
        return self.step((x, y))

    @property
    def coast_frames(self) -> int:
        """Number of consecutive frames without a valid measurement."""
        return self._coast_frames


class CapTracker:
    """
    Per-frame swim-cap centroid tracker within a user-defined lane ROI.

    lane_polygon: list of (x, y) pixel points defining the lane boundary
    (drawn during calibration). Detections outside it are discarded, which
    is what lets colour tracking stay locked onto the right swimmer even
    when an adjacent-lane swimmer wears a similar cap colour.
    """

    def __init__(
        self,
        cap_color: str,
        lane_polygon: list[tuple[float, float]] | None,
        recovery_agent: object | None = None,
    ):
        self.hsv_ranges = hex_or_keyword_to_hsv_ranges(cap_color)
        self.lane_polygon = (
            np.array(lane_polygon, dtype=np.int32) if lane_polygon else None
        )
        self.kalman = KalmanPointTracker()
        self.last_good: tuple[float, float] | None = None
        self.min_contour_area = 100  # px^2; increased from 25 to reject tiny noise
        self.max_contour_area = 5000 # reject huge blobs (e.g. half the pool)
        self._recovery = recovery_agent
        self._coast_buffer: list[np.ndarray] = []

    def set_phase(self, phase: str) -> None:
        """Set the swim phase for adaptive Kalman process noise."""
        self.kalman.set_phase(phase)

    def _in_lane(self, x: float, y: float) -> bool:
        if self.lane_polygon is None:
            return True
        return cv2.pointPolygonTest(self.lane_polygon, (x, y), False) >= 0

    def detect(self, frame_bgr: np.ndarray) -> tuple[float, float, float] | None:
        """Return (x, y, confidence) of the best colour-match blob, or None."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.hsv_ranges:
            mask |= cv2.inRange(hsv, lower, upper)
            
        # Stronger morphological ops to reject noise and close splash holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
                
            # Aspect ratio check (caps are roughly elliptical, not super long/skinny strings of noise)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
                
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            if not self._in_lane(cx, cy):
                continue
            candidates.append((cx, cy, area))

        if not candidates:
            return None

        # Prefer the candidate closest to the last known good position (if
        # any) over the simply-largest blob — this stops the tracker
        # jumping to a stray patch of similarly-coloured splash/reflection.
        if self.last_good is not None:
            lx, ly = self.last_good
            candidates.sort(key=lambda c: (c[0] - lx) ** 2 + (c[1] - ly) ** 2)
        else:
            candidates.sort(key=lambda c: -c[2])

        cx, cy, area = candidates[0]
        max_area = max(c[2] for c in candidates)
        confidence = min(1.0, 0.4 + 0.6 * min(area / 400.0, 1.0))
        return cx, cy, confidence

    def track_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> TrackedPoint:
        det = self.detect(frame_bgr)
        if det is not None:
            x, y, conf = det
            fx, fy, accepted = self.kalman.gated_step((x, y))
            if accepted:
                self.last_good = (x, y)
                self._coast_buffer.clear()
                return TrackedPoint(frame_idx, fx, fy, conf, source="color")
            else:
                # Measurement rejected by Mahalanobis gate — treat as splash
                self._coast_buffer.append(frame_bgr.copy())
                if len(self._coast_buffer) > 90:
                    self._coast_buffer = self._coast_buffer[-60:]
                return TrackedPoint(frame_idx, fx, fy, confidence=0.15, source="kalman_predict")
        else:
            px, py, _ = self.kalman.gated_step(None)
            self._coast_buffer.append(frame_bgr.copy())
            if len(self._coast_buffer) > 90:
                self._coast_buffer = self._coast_buffer[-60:]

            # Prolonged splash occlusion: ask Gemini for recovery if available
            if (
                self.kalman.coast_frames > 10
                and self._recovery is not None
                and getattr(self._recovery, "available", False)
            ):
                recovery = None
                if len(self._coast_buffer) >= 3:
                    recovery = self._recovery.locate_swimmer_in_segment(
                        self._coast_buffer, fps=30.0
                    )
                if recovery is None:
                    recovery = self._recovery.locate_swimmer(frame_bgr)

                if recovery is not None:
                    rx, ry = self.kalman.correct(recovery.x, recovery.y)
                    self.last_good = (rx, ry)
                    self._coast_buffer.clear()
                    return TrackedPoint(
                        frame_idx, rx, ry,
                        confidence=recovery.confidence,
                        source="gemini_recovery",
                    )

            return TrackedPoint(frame_idx, px, py, confidence=0.15, source="kalman_predict")


class FinishLineOpticalFlowTracker:
    """
    High-precision finish sub-routine (spec section 3, "pinpoint finish").

    Standard blob tracking degrades right at the wall because splashing
    fragments/hides the cap. Once the swimmer is inside the finish zone,
    we switch to Lucas-Kanade sparse optical flow on a small cluster of
    feature points seeded around the last confident detection, and track
    their *forward* (along-lane) velocity frame to frame. "Touch" is
    declared where that forward velocity crosses to ~0 and stays there,
    which for a fixed camera is the frame the swimmer's motion toward the
    wall physically stops.

    Precision honesty: this gives sub-frame *interpolated* timing, not
    literal hardware-clock milliseconds. The video's own frame rate is the
    real resolution floor (a 30fps clip samples the world every ~33ms; a
    60fps clip every ~17ms). Linear interpolation between the last-moving
    and first-stopped frame estimates a time inside that window assuming
    locally constant deceleration — a standard, defensible technique, not
    a hardware timestamp.
    """

    def __init__(self, lane_axis_is_x: bool = True):
        self.lane_axis_is_x = lane_axis_is_x
        self.lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self._prev_gray: np.ndarray | None = None
        self._points: np.ndarray | None = None

    def seed(self, frame_bgr: np.ndarray, center: tuple[float, float]) -> None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cx, cy = int(center[0]), int(center[1])
        h, w = gray.shape
        x0, x1 = max(cx - 30, 0), min(cx + 30, w)
        y0, y1 = max(cy - 30, 0), min(cy + 30, h)
        roi = gray[y0:y1, x0:x1]
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=15, qualityLevel=0.15, minDistance=4)
        if pts is None:
            self._points = np.array([[[cx, cy]]], dtype=np.float32)
        else:
            pts[:, 0, 0] += x0
            pts[:, 0, 1] += y0
            self._points = pts.astype(np.float32)
        self._prev_gray = gray

    def step(self, frame_bgr: np.ndarray) -> float | None:
        """Feed the next frame; returns mean forward velocity in px/frame,
        or None if tracking was lost (caller should re-seed)."""
        if self._prev_gray is None or self._points is None:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._points, None, **self.lk_params
        )
        good_new = new_pts[status.flatten() == 1]
        good_old = self._points[status.flatten() == 1]
        if len(good_new) < 3:
            self._prev_gray = gray
            self._points = new_pts
            return None

        delta = (good_new - good_old).reshape(-1, 2)
        axis = 0 if self.lane_axis_is_x else 1
        forward_velocity = float(np.median(delta[:, axis]))

        self._prev_gray = gray
        self._points = good_new.reshape(-1, 1, 2)
        return forward_velocity


# ---------------------------------------------------------------------------
# Pluggable detector interface for anyone who wants to swap in their own
# trained model instead of colour tracking. Not implemented by default —
# see the licensing note at the top of this file for why.
# ---------------------------------------------------------------------------
class Detector:
    def detect(self, frame_bgr: np.ndarray) -> tuple[float, float, float] | None:
        raise NotImplementedError


def plug_in_custom_detector(tracker: CapTracker, detector: Detector) -> None:
    """Monkey-patch a CapTracker to use a custom Detector.detect() instead
    of HSV thresholding, while keeping the Kalman/lane-polygon logic."""
    tracker.detect = detector.detect  # type: ignore[method-assign]
