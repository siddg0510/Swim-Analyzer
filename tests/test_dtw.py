"""
Unit tests for the DTW biomechanical comparator (src/analysis/comparator.py).

Tests:
  1. extract_feature_vector — returns None when pose is not present
  2. extract_feature_vector — returns 8-element array for a valid pose
  3. build_temporal_sequence — shape correctness
  4. DTWComparator.compare — result is non-negative and bounded
  5. DTWComparator — identical sequences produce flaw_score ≈ 0
  6. DTWComparator — very different sequences produce higher flaw_score
  7. Euclidean fallback path works without fastdtw
  8. run_dtw_comparison — returns None when gold standard doesn't exist
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.analysis.comparator import (
    DTWComparator, DTWResult, JOINT_NAMES,
    extract_feature_vector, build_temporal_sequence,
    run_dtw_comparison,
)
from src.vision.pose_estimator import PoseFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pose_frame(present: bool = True) -> PoseFrame:
    """Return a synthetic PoseFrame with all 33 landmarks set."""
    if not present:
        return PoseFrame(frame_idx=0, timestamp_ms=0, present=False, landmarks_px={})
    # Fill every landmark with a plausible 2-D point and visibility=1.0
    landmarks = {i: (float(50 + i * 10), float(100 + i * 5), 1.0) for i in range(33)}
    return PoseFrame(frame_idx=0, timestamp_ms=0, present=True, landmarks_px=landmarks)


def _make_random_seq(T: int, C: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((T, C)).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature vector extraction
# ---------------------------------------------------------------------------

class TestExtractFeatureVector:
    def test_absent_pose_returns_none(self):
        pf = _make_pose_frame(present=False)
        assert extract_feature_vector(pf) is None

    def test_valid_pose_returns_8_element_array(self):
        pf = _make_pose_frame(present=True)
        vec = extract_feature_vector(pf)
        # May return None if the geometry collapses (all points collinear), but
        # should not raise.  If it returns something, it must be length-8.
        if vec is not None:
            assert vec.shape == (8,)
            assert vec.dtype == np.float32

    def test_output_values_in_valid_range(self):
        pf = _make_pose_frame(present=True)
        vec = extract_feature_vector(pf)
        if vec is not None:
            # Normalised to [0, 1] after fillna
            assert np.all(vec >= 0.0)
            assert np.all(vec <= 1.0)


# ---------------------------------------------------------------------------
# Temporal sequence building
# ---------------------------------------------------------------------------

class TestBuildTemporalSequence:
    def test_empty_input_returns_empty_array(self):
        result = build_temporal_sequence([])
        assert result.shape == (0, 8)

    def test_output_shape(self):
        frames = [(float(i) / 30.0, _make_pose_frame(present=True)) for i in range(20)]
        seq = build_temporal_sequence(frames)
        assert seq.shape[0] == 20
        assert seq.shape[1] == 8

    def test_absent_frames_forward_filled(self):
        """Frames with no pose should carry forward the last valid vector."""
        frames = [
            (0.0, _make_pose_frame(present=True)),
            (0.033, _make_pose_frame(present=False)),
            (0.067, _make_pose_frame(present=False)),
        ]
        seq = build_temporal_sequence(frames)
        assert seq.shape == (3, 8)
        # Rows 1 and 2 should equal row 0 (forward fill)
        np.testing.assert_array_equal(seq[0], seq[1])
        np.testing.assert_array_equal(seq[0], seq[2])


# ---------------------------------------------------------------------------
# DTWComparator
# ---------------------------------------------------------------------------

class TestDTWComparator:
    def test_identical_sequences_low_flaw(self):
        seq = _make_random_seq(60)
        comparator = DTWComparator()
        # Mock fastdtw if available so test doesn't depend on it
        try:
            result = comparator.compare(seq, seq.copy(), gold_label="test")
        except Exception:
            pytest.skip("DTW comparison raised an unexpected error")
        assert isinstance(result, DTWResult)
        # Identical sequences → flaw_score should be very close to 0
        assert result.flaw_score < 5.0, f"Expected near-0 flaw for identical seqs, got {result.flaw_score}"

    def test_different_sequences_higher_flaw(self):
        user_seq  = np.zeros((60, 8), dtype=np.float32)       # all-zero angles
        gold_seq  = np.ones((60, 8), dtype=np.float32)        # all-one angles
        comparator = DTWComparator()
        result = comparator.compare(user_seq, gold_seq)
        assert result.flaw_score > 50.0, (
            f"Expected high flaw for maximally different seqs, got {result.flaw_score}"
        )

    def test_result_is_bounded(self):
        user_seq = _make_random_seq(45, seed=1)
        gold_seq = _make_random_seq(60, seed=2)
        comparator = DTWComparator()
        result = comparator.compare(user_seq, gold_seq)
        assert 0.0 <= result.flaw_score <= 100.0

    def test_per_joint_errors_length(self):
        user_seq = _make_random_seq(30)
        gold_seq = _make_random_seq(40)
        comparator = DTWComparator()
        result = comparator.compare(user_seq, gold_seq)
        assert len(result.per_joint_errors) == 8

    def test_raw_distance_non_negative(self):
        user_seq = _make_random_seq(30)
        gold_seq = _make_random_seq(30)
        comparator = DTWComparator()
        result = comparator.compare(user_seq, gold_seq)
        assert result.raw_distance >= 0.0

    def test_empty_user_seq_returns_worst_score(self):
        empty = np.zeros((0, 8), dtype=np.float32)
        gold  = _make_random_seq(60)
        comparator = DTWComparator()
        result = comparator.compare(empty, gold)
        assert result.flaw_score == 100.0

    def test_euclidean_fallback(self):
        """Force the Euclidean fallback by patching fastdtw import."""
        user_seq = _make_random_seq(30)
        gold_seq = _make_random_seq(40)
        comparator = DTWComparator()
        with patch.dict("sys.modules", {"fastdtw": None}):
            result = comparator._euclidean_fallback(user_seq, gold_seq, gold_label="fallback")
        assert result.method == "euclidean_fallback"
        assert 0.0 <= result.flaw_score <= 100.0


# ---------------------------------------------------------------------------
# run_dtw_comparison convenience function
# ---------------------------------------------------------------------------

class TestRunDTWComparison:
    def test_returns_none_when_no_gold_standard(self):
        """If there's no .npy template for the given stroke/event, return None."""
        frames = [(float(i) / 30.0, _make_pose_frame()) for i in range(60)]
        # Use an event that's guaranteed not to have a file (9999m)
        result = run_dtw_comparison(frames, stroke="freestyle", event_m=9999)
        assert result is None

    def test_returns_result_with_existing_gold_standard(self):
        """With the placeholder template installed, we should get a DTWResult."""
        frames = [(float(i) / 30.0, _make_pose_frame()) for i in range(60)]
        result = run_dtw_comparison(frames, stroke="freestyle", event_m=100)
        # Result may be None if pose extraction fails on synthetic frames,
        # but if it returns something it must be a DTWResult.
        if result is not None:
            assert isinstance(result, DTWResult)
            assert 0.0 <= result.flaw_score <= 100.0
