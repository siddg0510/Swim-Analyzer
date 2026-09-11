"""
Unit tests for TargetSelector (src/detection/selector.py).

Tests:
  1. select_by_click — exact containment
  2. select_by_click — nearest centroid fallback
  3. select_by_click — empty track list returns None
  4. select_by_lane — correct x-band matching
  5. select_by_lane — empty tracks returns None
  6. get_track — retrieves by ID
  7. reset — clears target_id
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.detection.detector import Track
from src.detection.selector import TargetSelector


def _make_tracks(*bboxes_with_ids):
    """Helper: list of Track objects from (id, x, y, w, h) tuples."""
    return [Track(id=t[0], bbox=(t[1], t[2], t[3], t[4]), conf=0.8) for t in bboxes_with_ids]


class TestSelectByClick:
    def test_exact_containment(self):
        tracks = _make_tracks(
            (1, 100, 100, 50, 80),   # swimmer 1: x 100-150, y 100-180
            (2, 300, 100, 50, 80),   # swimmer 2
        )
        sel = TargetSelector()
        result = sel.select_by_click(125, 140, tracks)
        assert result == 1
        assert sel.target_id == 1

    def test_second_swimmer_selected(self):
        tracks = _make_tracks(
            (1, 100, 100, 50, 80),
            (2, 300, 100, 50, 80),
        )
        sel = TargetSelector()
        result = sel.select_by_click(325, 140, tracks)
        assert result == 2

    def test_nearest_centroid_fallback(self):
        """Click between boxes → nearest centroid wins."""
        tracks = _make_tracks(
            (1, 0,   100, 50, 80),   # centroid at (25, 140)
            (2, 500, 100, 50, 80),   # centroid at (525, 140)
        )
        sel = TargetSelector()
        # Click at x=100 — closer to track 1 centroid (25) than track 2 (525)
        result = sel.select_by_click(100, 140, tracks)
        assert result == 1

    def test_empty_tracks_returns_none(self):
        sel = TargetSelector()
        result = sel.select_by_click(100, 100, [])
        assert result is None
        assert sel.target_id is None


class TestSelectByLane:
    def test_selects_correct_lane(self):
        # 8-lane pool, frame width 1600 → lane width 200px
        # Lane 1 centre ≈ 100, Lane 4 centre ≈ 700
        tracks = _make_tracks(
            (1, 50,  200, 100, 150),   # centre x ≈ 100 → Lane 1
            (4, 650, 200, 100, 150),   # centre x ≈ 700 → Lane 4
            (7, 1250,200, 100, 150),   # centre x ≈ 1300 → Lane 7
        )
        sel = TargetSelector()
        result = sel.select_by_lane(4, tracks, frame_width=1600, num_lanes=8)
        assert result == 4

    def test_empty_tracks_returns_none(self):
        sel = TargetSelector()
        result = sel.select_by_lane(3, [], frame_width=1920)
        assert result is None


class TestGetTrack:
    def test_returns_correct_track(self):
        tracks = _make_tracks(
            (1, 0, 0, 50, 80),
            (2, 100, 0, 50, 80),
        )
        sel = TargetSelector(target_id=2)
        t = sel.get_track(tracks)
        assert t is not None
        assert t.id == 2

    def test_returns_none_when_no_target(self):
        tracks = _make_tracks((1, 0, 0, 50, 80))
        sel = TargetSelector()
        assert sel.get_track(tracks) is None

    def test_returns_none_when_id_not_in_tracks(self):
        tracks = _make_tracks((1, 0, 0, 50, 80))
        sel = TargetSelector(target_id=99)
        assert sel.get_track(tracks) is None


class TestReset:
    def test_reset_clears_target(self):
        sel = TargetSelector(target_id=5)
        sel.reset()
        assert sel.target_id is None
        assert not sel.is_locked()
