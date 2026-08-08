"""
Pose landmark extraction via MediaPipe's Tasks API.

Verified while building this project: the older `mp.solutions.pose` API
(the one most tutorials/StackOverflow answers still show) no longer exists
in current MediaPipe — `import mediapipe as mp; mp.solutions` raises
AttributeError on the version this project was built and tested against
(mediapipe 0.10.33). Pose estimation now goes exclusively through
`mediapipe.tasks.python.vision.PoseLandmarker`, which requires a separate
.task model file (not bundled in the pip package). See
models/DOWNLOAD_MODEL.md.

This wrapper exposes just the landmarks the rest of the app needs (wrists,
shoulders, hips, ankles, nose) as plain floats, so nothing else in the
codebase has to know MediaPipe's result-object structure.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)

from ..config import POSE_MODEL_PATH

# Indices into MediaPipe's 33-point BlazePose topology
# (see: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models)
LM_NOSE = 0
LM_L_SHOULDER, LM_R_SHOULDER = 11, 12
LM_L_ELBOW, LM_R_ELBOW = 13, 14
LM_L_WRIST, LM_R_WRIST = 15, 16
LM_L_HIP, LM_R_HIP = 23, 24
LM_L_ANKLE, LM_R_ANKLE = 27, 28


@dataclass
class PoseFrame:
    frame_idx: int
    timestamp_ms: int
    present: bool
    landmarks_px: dict[int, tuple[float, float, float]]  # id -> (x, y, visibility)


class PoseEstimator:
    """Thin, offline wrapper around PoseLandmarker in VIDEO mode."""

    def __init__(self, model_path=None, min_detection_confidence: float = 0.5):
        path = str(model_path or POSE_MODEL_PATH)
        try:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=path),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
            self.available = True
        except Exception as exc:  # model file missing/corrupt, etc.
            self._landmarker = None
            self.available = False
            self._init_error = str(exc)

    def process(self, frame_bgr: np.ndarray, frame_idx: int, timestamp_ms: int) -> PoseFrame:
        if not self.available:
            return PoseFrame(frame_idx, timestamp_ms, present=False, landmarks_px={})

        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return PoseFrame(frame_idx, timestamp_ms, present=False, landmarks_px={})

        h, w = frame_bgr.shape[:2]
        lm_list = result.pose_landmarks[0]
        pts = {
            i: (lm.x * w, lm.y * h, getattr(lm, "visibility", 1.0))
            for i, lm in enumerate(lm_list)
        }
        return PoseFrame(frame_idx, timestamp_ms, present=True, landmarks_px=pts)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
