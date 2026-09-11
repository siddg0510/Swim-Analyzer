"""
Module A: Detection & Identification Engine

Detects ALL swimmers in a frame and assigns persistent integer track IDs using
YOLOv8 + ByteTrack.  The design deliberately provides a clean fallback chain:

  Tier 1 — YOLOv8 (ultralytics) + supervision ByteTrack
            → Best accuracy; requires `ultralytics` and `supervision` pip packages.
  Tier 2 — OpenCV HOG + simple IoU tracker
            → No extra packages; lower accuracy but always available.

The `Detector` protocol at the bottom mirrors the one already defined in
cap_tracker.py, so the existing CapTracker / KalmanPointTracker path continues
to work unchanged when multi-swimmer mode is off.

License note (mirrors cap_tracker.py comment): YOLOv8 is AGPL-3.0. This module
is only imported when the user explicitly enables "Multi-Swimmer Mode" in the GUI;
the default pipeline still uses HSV colour tracking without touching YOLO at all.
This makes the licensing posture identical to what it was before: AGPL code is
only used (and only needs to be distributed) when the operator/user opts in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Contour fallback is a last-resort hint with no appearance model, so its
# confidence is capped *below* the splash confidence gate (config default 0.4).
# This means an adaptive-threshold blob can never, on its own, satisfy the
# tracker's decision gate — during splash the tracker coasts on the EKF instead
# of snapping to a foam/shadow artifact. Kept as a local constant (rather than
# importing the gate) to avoid a lower→upper layer import; see the comment at
# the call site if the gate value in src/config.py changes.
_CONTOUR_FALLBACK_MAX_CONF = 0.35
_CONTOUR_FALLBACK_MIN_CONF = 0.12


def _sigmoid(x: float) -> float:
    """Squash an unbounded SVM decision margin into a (0, 1) pseudo-confidence.

    OpenCV's HOG ``detectMultiScale`` returns raw SVM margins (roughly 0..3+),
    not probabilities. A logistic map is monotonic and bounded, giving an
    honest 0-1 value for the ``Track.conf`` contract. This is a heuristic
    squashing, NOT a calibrated probability.
    """
    return 1.0 / (1.0 + float(np.exp(-x)))


# ---------------------------------------------------------------------------
# Shared data structure
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """A single swimmer track returned by the detector each frame.

    ``conf`` is always in [0.0, 1.0]. For YOLO it is the model's class
    probability; for HOG it is a logistic-squashed SVM margin (see
    :func:`_sigmoid`); for the adaptive-threshold contour fallback it is a
    shape-plausibility score capped below the splash gate.
    """
    id: int
    bbox: tuple[float, float, float, float]  # (x, y, w, h) in pixels
    conf: float                               # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Tier 1: YOLOv8 + ByteTrack (optional, fails gracefully)
# ---------------------------------------------------------------------------

class YOLOByteTrackDetector:
    """
    Uses `ultralytics` YOLOv8 for person detection and `supervision` ByteTracker
    for persistent swimmer ID assignment across frames.

    Requires:  pip install ultralytics supervision
    Falls back transparently to HOGDetector if imports fail.
    """

    def __init__(self, model_size: str = "yolov8n.pt", conf_threshold: float = 0.25) -> None:
        from ultralytics import YOLO                         # noqa: PLC0415
        import supervision as sv                             # noqa: PLC0415

        self._model = YOLO(model_size)
        self._tracker = sv.ByteTracker()
        self._conf = conf_threshold
        logger.info("YOLOByteTrackDetector initialised with model=%s", model_size)

    def detect(self, frame_bgr: np.ndarray) -> list[Track]:
        import supervision as sv  # noqa: PLC0415

        results = self._model(frame_bgr, verbose=False, conf=self._conf, classes=[0])[0]
        # classes=[0] → "person" in COCO
        detections = sv.Detections.from_ultralytics(results)
        tracked = self._tracker.update_with_detections(detections)

        out: list[Track] = []
        for i, track_id in enumerate(tracked.tracker_id or []):
            if track_id is None:
                continue
            x1, y1, x2, y2 = tracked.xyxy[i]
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
            out.append(Track(
                id=int(track_id),
                bbox=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                conf=conf,
            ))
        return out


# ---------------------------------------------------------------------------
# Tier 2: OpenCV HOG + IoU tracker (always available)
# ---------------------------------------------------------------------------

@dataclass
class _SimpleTrack:
    id: int
    bbox: tuple[float, float, float, float]
    age: int = 0            # frames since last matched detection
    conf: float = 0.5


def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class HOGIoUDetector:
    """
    Fallback detector using OpenCV's HOG person detector (no external packages).
    Tracks IDs through simple greedy IoU matching between consecutive frames.
    Accuracy is lower than YOLO in cluttered aquatic scenes, but it always works.
    """

    def __init__(self, conf_threshold: float = 0.3, max_age: int = 5) -> None:
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._conf = conf_threshold
        self._max_age = max_age
        self._tracks: list[_SimpleTrack] = []
        self._next_id = 1
        logger.info("HOGIoUDetector initialised (YOLO/supervision not available)")

    def detect(self, frame_bgr: np.ndarray) -> list[Track]:
        rects, weights = self._hog.detectMultiScale(
            frame_bgr, winStride=(8, 8), padding=(4, 4), scale=1.05
        )

        detections: list[tuple[float, float, float, float, float]] = []
        for i, (x, y, w, h) in enumerate(rects):
            raw = float(weights[i]) if len(weights) > i else 0.0
            # Gate on the raw SVM margin (unchanged recall), but report a
            # bounded 0-1 confidence so downstream consumers and the splash
            # gate see a value that actually honours the Track.conf contract.
            if raw >= self._conf:
                detections.append((float(x), float(y), float(w), float(h), _sigmoid(raw)))

        # Fallback for swimmers in water (prone posture where upright HOG detector yields 0 detections)
        if not detections and frame_bgr is not None and frame_bgr.size > 0:
            h_img, w_img = frame_bgr.shape[:2]
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
            )
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates: list[tuple[float, float, float, float, float]] = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 120 < area < (h_img * w_img * 0.35):
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = w / max(h, 1)
                    if 0.25 < aspect < 4.0:
                        # Shape-plausibility confidence, deliberately capped
                        # below the splash gate. A prone swimmer viewed from
                        # poolside is a solid, elongated dark blob; foam/shadow
                        # fragments are blobby or spidery. Grade on solidity
                        # (area / bbox area) and aspect closeness to a swimmer-
                        # like ~2.2, then squash into
                        # [_CONTOUR_FALLBACK_MIN_CONF, _CONTOUR_FALLBACK_MAX_CONF].
                        solidity = area / float(w * h) if w * h > 0 else 0.0
                        aspect_score = max(0.0, 1.0 - abs(aspect - 2.2) / 2.2)
                        shape = 0.5 * min(solidity, 1.0) + 0.5 * aspect_score
                        conf = (
                            _CONTOUR_FALLBACK_MIN_CONF
                            + (_CONTOUR_FALLBACK_MAX_CONF - _CONTOUR_FALLBACK_MIN_CONF) * shape
                        )
                        candidates.append((float(x), float(y), float(w), float(h), conf))
            candidates.sort(key=lambda c: c[2] * c[3], reverse=True)
            detections = candidates[:8]

        # Greedy IoU matching
        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()
        updated_tracks: list[_SimpleTrack] = []

        for trk in self._tracks:
            best_iou = 0.4  # IoU gate
            best_di = -1
            for di, det in enumerate(detections):
                if di in matched_det_indices:
                    continue
                iou = _iou(trk.bbox, det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_di = di
            if best_di >= 0:
                det = detections[best_di]
                trk.bbox = det[:4]
                trk.conf = det[4]
                trk.age = 0
                matched_track_ids.add(trk.id)
                matched_det_indices.add(best_di)
                updated_tracks.append(trk)
            else:
                trk.age += 1
                if trk.age <= self._max_age:
                    updated_tracks.append(trk)

        # New unmatched detections → new tracks
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                updated_tracks.append(_SimpleTrack(
                    id=self._next_id,
                    bbox=det[:4],
                    conf=det[4],
                ))
                self._next_id += 1

        self._tracks = updated_tracks
        return [Track(id=t.id, bbox=t.bbox, conf=t.conf) for t in self._tracks]


# ---------------------------------------------------------------------------
# Public factory — returns whichever tier is available
# ---------------------------------------------------------------------------

def build_detector(
    yolo_model: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
    prefer_hog: bool = False,
) -> "DetectorProtocol":
    """
    Return the best available swimmer detector.

    Args:
        yolo_model:      YOLO weights file name or path.  The ultralytics
                         package auto-downloads it on first use.
        conf_threshold:  Detection confidence gate (applied by both tiers).
        prefer_hog:      Force the HOG fallback (useful for testing or when
                         AGPL licensing is a concern for the calling context).
    """
    if not prefer_hog:
        try:
            return YOLOByteTrackDetector(yolo_model, conf_threshold)
        except ImportError:
            logger.warning(
                "ultralytics / supervision not installed; "
                "falling back to OpenCV HOG detector. "
                "Install with: pip install ultralytics supervision"
            )
        except Exception as exc:
            logger.warning("YOLOByteTrackDetector init failed (%s); using HOG.", exc)
    return HOGIoUDetector(conf_threshold)


# ---------------------------------------------------------------------------
# Protocol (for type-checking; mirrors cap_tracker.Detector)
# ---------------------------------------------------------------------------

class DetectorProtocol(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Track]: ...
