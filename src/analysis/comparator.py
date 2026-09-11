"""
Module D: Biomechanical Comparison via Dynamic Time Warping

Implements the three-step comparison pipeline from the blueprint:

  Step 1 — Feature Vector extraction:
            Convert the 33 MediaPipe keypoints for each frame into a compact
            8-element angular vector (joint angles at both elbows, shoulders,
            hips, and knees).  Angles are expressed in degrees and normalized
            to [-1, 1] so all joints have equal weight in the DTW distance.

  Step 2 — Gold-Standard loading:
            Load a pre-computed .npy array from data/olympic_gold/ that
            represents the angular sequence of an elite swimmer.  The array
            shape is (T_elite, 8).

  Step 3 — DTW Comparison:
            Use `fastdtw` to align the user's sequence (T_user, 8) to the
            template (T_elite, 8).  The returned "distance" is the cumulative
            alignment cost — lower = more similar technique.
            A normalized "flaw_score" (0–100) is derived so coaches can read
            it as a percentage deviation from the gold standard.

Graceful degradation:
  - If `fastdtw` is not installed → falls back to simple Euclidean distance
    on mean-pooled frames (rough but always available).
  - If no gold-standard .npy exists for a given stroke/event → returns None
    so the dashboard simply hides the DTW tab.

.npy file format:
    shape : (T, 8) float32
    axes  : T = number of frames, 8 joint angles (see JOINT_NAMES below)
    units : normalised to [-1, 1]  (angle_deg / 180.0)
    To build templates run:  python scripts/generate_templates.py <video> <stroke> <event_m>
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..vision.pose_estimator import (
    PoseFrame,
    LM_L_SHOULDER, LM_R_SHOULDER,
    LM_L_ELBOW,    LM_R_ELBOW,
    LM_L_WRIST,    LM_R_WRIST,
    LM_L_HIP,      LM_R_HIP,
    LM_L_ANKLE,    LM_R_ANKLE,
)

logger = logging.getLogger(__name__)

# ── Joint angle indices in the feature vector ──────────────────────────────
JOINT_NAMES = [
    "left_elbow",       # 0
    "right_elbow",      # 1
    "left_shoulder",    # 2
    "right_shoulder",   # 3
    "left_hip",         # 4
    "right_hip",        # 5
    "left_knee",        # 6
    "right_knee",       # 7
]

# MediaPipe landmark indices for the three-point angle calculations
_ANGLE_TRIPLES = [
    # (A, vertex, B)
    (LM_L_SHOULDER,  LM_L_ELBOW,    LM_L_WRIST),    # left elbow
    (LM_R_SHOULDER,  LM_R_ELBOW,    LM_R_WRIST),    # right elbow
    (LM_L_HIP,       LM_L_SHOULDER, LM_L_ELBOW),    # left shoulder
    (LM_R_HIP,       LM_R_SHOULDER, LM_R_ELBOW),    # right shoulder
    # Hips and knees: use a spine reference for hip, and lower-leg for knee
    # MediaPipe indices: left hip=23, left knee=25, left ankle=27
    (LM_L_SHOULDER,  LM_L_HIP,      25),             # left hip extension
    (LM_R_SHOULDER,  LM_R_HIP,      26),             # right hip extension
    (LM_L_HIP,       25,             LM_L_ANKLE),    # left knee
    (LM_R_HIP,       26,             LM_R_ANKLE),    # right knee
]


def _angle_deg(
    a: tuple[float, float],
    vertex: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """
    Return the interior angle at `vertex` formed by the two rays to `a` and `b`,
    in degrees [0, 180].  Returns NaN if any point is invalid.
    """
    va = np.array([a[0] - vertex[0], a[1] - vertex[1]], dtype=np.float64)
    vb = np.array([b[0] - vertex[0], b[1] - vertex[1]], dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-6 or nb < 1e-6:
        return float("nan")
    cos_theta = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def extract_feature_vector(pose: PoseFrame) -> np.ndarray | None:
    """
    Convert a PoseFrame into a normalised 8-element float32 feature vector.
    Returns None if too many keypoints are missing (< 5 valid angles).
    """
    if not pose.present or not pose.landmarks_px:
        return None

    angles: list[float] = []
    for a_idx, v_idx, b_idx in _ANGLE_TRIPLES:
        pts = pose.landmarks_px
        if a_idx not in pts or v_idx not in pts or b_idx not in pts:
            angles.append(float("nan"))
            continue
        a_xy = pts[a_idx][:2]
        v_xy = pts[v_idx][:2]
        b_xy = pts[b_idx][:2]
        angles.append(_angle_deg(a_xy, v_xy, b_xy))

    arr = np.array(angles, dtype=np.float32)
    if np.sum(np.isfinite(arr)) < 5:
        return None
    # Normalise to [-1, 1] (angle range 0–180 → 0–1 → centred at 0 = 90°)
    arr = arr / 180.0
    # Fill NaN with 0.5 (neutral, mid-range angle) so DTW doesn't choke
    arr = np.where(np.isfinite(arr), arr, 0.5)
    return arr


def build_temporal_sequence(
    pose_frames: list[tuple[float, PoseFrame]],
) -> np.ndarray:
    """
    Convert a list of (time_s, PoseFrame) pairs into a (T, 8) float32 array.
    Frames with no valid pose are filled with the previous frame's values
    (forward-fill), or with 0.5 (neutral) if the sequence starts with missing data.
    """
    rows: list[np.ndarray] = []
    last_valid = np.full(8, 0.5, dtype=np.float32)

    for _t, pf in pose_frames:
        vec = extract_feature_vector(pf)
        if vec is not None:
            last_valid = vec
        rows.append(last_valid.copy())

    if not rows:
        return np.zeros((0, 8), dtype=np.float32)
    return np.stack(rows, axis=0)  # (T, 8)


# ---------------------------------------------------------------------------
# Gold Standard loading
# ---------------------------------------------------------------------------

def _gold_standard_path(stroke: str, event_m: int) -> Path:
    from ..config import resource_path  # noqa: PLC0415
    return resource_path("data", "olympic_gold", f"{stroke}_{event_m}m_gold_standard.npy")


def load_gold_standard(stroke: str, event_m: int) -> np.ndarray | None:
    """
    Load the (T_elite, 8) gold-standard array for a given stroke + event.
    Returns None if the file doesn't exist (DTW tab will be hidden in the GUI).
    """
    path = _gold_standard_path(stroke, event_m)
    if not path.exists():
        logger.info("No gold standard found at %s — DTW comparison skipped.", path)
        return None
    try:
        arr = np.load(str(path)).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 8:
            logger.warning("Gold standard %s has wrong shape %s — skipping.", path, arr.shape)
            return None
        logger.info("Loaded gold standard %s (shape %s)", path, arr.shape)
        return arr
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load gold standard %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# DTW result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DTWResult:
    """Result of a DTW comparison between user sequence and gold standard."""
    flaw_score: float                           # 0 (perfect) – 100 (very different)
    raw_distance: float                         # unnormalized DTW distance
    per_joint_errors: list[float]               # length-8 mean alignment error per joint
    joint_names: list[str] = field(default_factory=lambda: list(JOINT_NAMES))
    gold_label: str = ""                        # e.g. "freestyle 100m"
    method: str = "fastdtw"                     # "fastdtw" or "euclidean_fallback"
    aligned_user_seq: np.ndarray | None = None  # (T_user, 8) reindexed by DTW path
    aligned_gold_seq: np.ndarray | None = None  # (T_gold, 8) reindexed by DTW path


# ---------------------------------------------------------------------------
# DTW Comparator
# ---------------------------------------------------------------------------

class DTWComparator:
    """
    Compare a user's temporal angular sequence against an Olympic gold standard
    using Dynamic Time Warping, which accounts for speed differences between
    the user and the elite athlete (a faster swimmer's stroke cycle will have
    fewer frames).

    The DTW distance is normalized by dividing by the path length × 8 joints,
    giving a per-joint-per-frame mean error in the [0, 1] normalised angle
    space.  This value × 100 is the flaw_score.
    """

    def compare(
        self,
        user_seq: np.ndarray,
        gold_seq: np.ndarray,
        gold_label: str = "",
    ) -> DTWResult:
        """
        Args:
            user_seq:   (T_user, 8) float32 — user's normalised joint angles
            gold_seq:   (T_elite, 8) float32 — gold standard angles
            gold_label: human-readable label for the template

        Returns a DTWResult.  Falls back to Euclidean similarity if fastdtw
        is not installed.
        """
        if user_seq.shape[0] == 0 or gold_seq.shape[0] == 0:
            return DTWResult(
                flaw_score=100.0, raw_distance=float("inf"),
                per_joint_errors=[1.0] * 8, gold_label=gold_label,
                method="empty_sequence",
            )

        try:
            return self._dtw_compare(user_seq, gold_seq, gold_label)
        except ImportError:
            logger.warning(
                "fastdtw not installed; using Euclidean fallback. "
                "Install with: pip install fastdtw"
            )
            return self._euclidean_fallback(user_seq, gold_seq, gold_label)

    def _dtw_compare(
        self, user_seq: np.ndarray, gold_seq: np.ndarray, gold_label: str
    ) -> DTWResult:
        from fastdtw import fastdtw  # noqa: PLC0415
        from scipy.spatial.distance import euclidean  # noqa: PLC0415

        distance, path = fastdtw(user_seq, gold_seq, dist=euclidean)
        path = np.array(path)               # (P, 2)  — (user_idx, gold_idx)
        path_len = len(path)

        # Reconstruct aligned sequences so the GUI can overlay them
        aligned_user = user_seq[path[:, 0]]   # (P, 8)
        aligned_gold = gold_seq[path[:, 1]]   # (P, 8)

        per_joint = np.mean(np.abs(aligned_user - aligned_gold), axis=0).tolist()
        mean_error = np.mean(per_joint)       # in normalised space [0, 1]
        flaw_score = float(np.clip(mean_error * 100.0, 0.0, 100.0))

        return DTWResult(
            flaw_score=flaw_score,
            raw_distance=float(distance),
            per_joint_errors=per_joint,
            gold_label=gold_label,
            method="fastdtw",
            aligned_user_seq=aligned_user,
            aligned_gold_seq=aligned_gold,
        )

    def _euclidean_fallback(
        self, user_seq: np.ndarray, gold_seq: np.ndarray, gold_label: str
    ) -> DTWResult:
        """
        Fallback: mean-pool both sequences to the same length, compute MAE.
        Much less accurate than DTW (ignores temporal warping) but usable
        when fastdtw is unavailable.
        """
        T_target = min(user_seq.shape[0], gold_seq.shape[0])
        user_ds = _downsample(user_seq, T_target)
        gold_ds = _downsample(gold_seq, T_target)

        per_joint = np.mean(np.abs(user_ds - gold_ds), axis=0).tolist()
        mean_error = float(np.mean(per_joint))
        flaw_score = float(np.clip(mean_error * 100.0, 0.0, 100.0))

        return DTWResult(
            flaw_score=flaw_score,
            raw_distance=mean_error * T_target * 8,
            per_joint_errors=per_joint,
            gold_label=gold_label,
            method="euclidean_fallback",
        )


def _downsample(arr: np.ndarray, T_target: int) -> np.ndarray:
    """Linearly interpolate arr (T, C) to T_target frames."""
    T, C = arr.shape
    if T == T_target:
        return arr
    idx = np.linspace(0, T - 1, T_target)
    out = np.zeros((T_target, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(idx, np.arange(T), arr[:, c])
    return out


# ---------------------------------------------------------------------------
# Convenience function used by the pipeline
# ---------------------------------------------------------------------------

def run_dtw_comparison(
    pose_frames: list[tuple[float, PoseFrame]],
    stroke: str,
    event_m: int,
) -> DTWResult | None:
    """
    Build the user's temporal sequence from pose frames, load the gold
    standard, and run DTW comparison.  Returns None if no template exists
    or the pose sequence is empty.
    """
    gold = load_gold_standard(stroke, event_m)
    if gold is None:
        return None

    user_seq = build_temporal_sequence(pose_frames)
    if user_seq.shape[0] < 10:
        logger.info("User pose sequence too short (%d frames) for DTW.", user_seq.shape[0])
        return None

    comparator = DTWComparator()
    label = f"{stroke} {event_m}m"
    result = comparator.compare(user_seq, gold, gold_label=label)
    logger.info(
        "DTW comparison (%s): flaw_score=%.1f, method=%s",
        label, result.flaw_score, result.method,
    )
    return result
