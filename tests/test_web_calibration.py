"""
Unit tests for web/calibration.py — parsing manual-calibration JSON posted by
the web picker into core CalibrationKeyframe objects.
"""
import json
import pytest

from web.calibration import (
    parse_calibration_keyframes,
    has_usable_manual_calibration,
    CalibrationParseError,
    MIN_REFERENCE_POINTS,
)
from src.vision.calibration import CalibrationKeyframe


def test_empty_input_returns_empty_list():
    assert parse_calibration_keyframes(None) == []
    assert parse_calibration_keyframes("") == []
    assert parse_calibration_keyframes("   ") == []


def test_parses_wrapped_keyframes():
    raw = json.dumps({
        "keyframes": [{
            "frame_idx": 3,
            "reference_points": [[[100, 240], 0.0], [[500, 240], 50.0]],
            "lane_polygon_px": [[50, 100], [590, 100], [590, 380], [50, 380]],
            "lane_number": 4,
        }]
    })
    kfs = parse_calibration_keyframes(raw)
    assert len(kfs) == 1
    kf = kfs[0]
    assert isinstance(kf, CalibrationKeyframe)
    assert kf.frame_idx == 3
    assert kf.lane_number == 4
    assert kf.reference_points == [((100.0, 240.0), 0.0), ((500.0, 240.0), 50.0)]
    assert kf.lane_polygon_px == [(50.0, 100.0), (590.0, 100.0), (590.0, 380.0), (50.0, 380.0)]


def test_parses_bare_list_and_single_object():
    single = json.dumps({"reference_points": [[[1, 2], 0.0], [[3, 4], 25.0]]})
    assert len(parse_calibration_keyframes(single)) == 1
    bare = json.dumps([{"reference_points": [[[1, 2], 0.0], [[3, 4], 25.0]]}])
    assert len(parse_calibration_keyframes(bare)) == 1


def test_frame_idx_defaults_to_zero():
    raw = json.dumps({"reference_points": [[[1, 2], 0.0], [[3, 4], 25.0]]})
    assert parse_calibration_keyframes(raw)[0].frame_idx == 0


def test_lane_polygon_optional():
    raw = json.dumps({"reference_points": [[[1, 2], 0.0], [[3, 4], 25.0]]})
    assert parse_calibration_keyframes(raw)[0].lane_polygon_px == []


@pytest.mark.parametrize("bad", [
    "{not json",
    json.dumps(42),
    json.dumps({"reference_points": "nope"}),
    json.dumps({"reference_points": [[[1, 2]]]}),          # missing meters
    json.dumps({"reference_points": [[["x", 2], 0.0]]}),   # non-numeric pixel
    json.dumps({"reference_points": [[[1, 2], "far"]]}),   # non-numeric meters
    json.dumps({"reference_points": [], "frame_idx": -1}), # negative frame
    json.dumps({"reference_points": [], "lane_polygon_px": [[1, 2, 3]]}),  # bad xy
])
def test_malformed_input_raises(bad):
    with pytest.raises(CalibrationParseError):
        parse_calibration_keyframes(bad)


def test_has_usable_manual_calibration():
    enough = parse_calibration_keyframes(
        json.dumps({"reference_points": [[[1, 2], 0.0], [[3, 4], 25.0]]})
    )
    assert has_usable_manual_calibration(enough) is True

    too_few = parse_calibration_keyframes(
        json.dumps({"reference_points": [[[1, 2], 0.0]]})
    )
    assert len(too_few[0].reference_points) < MIN_REFERENCE_POINTS
    assert has_usable_manual_calibration(too_few) is False
    assert has_usable_manual_calibration([]) is False
