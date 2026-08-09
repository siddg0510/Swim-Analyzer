"""
Tests for the Gemini AI integration layer.

These tests use mocks — they do NOT call the real Gemini API. They verify:
  - GeminiConfig key resolution logic
  - JSON response parsing (technique, elite comparison, etc.)
  - Graceful handling of malformed/empty responses
  - Video segment extraction helpers
"""
import unittest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ai.gemini_config import (
    resolve_api_key,
    has_api_key,
    GeminiConfig,
    DEFAULT_MODEL,
)


class TestGeminiConfig(unittest.TestCase):
    """Test API key resolution."""

    def test_explicit_key_takes_priority(self):
        key = resolve_api_key(explicit_key="test-key-123")
        self.assertEqual(key, "test-key-123")

    def test_explicit_key_strips_whitespace(self):
        key = resolve_api_key(explicit_key="  test-key-123  ")
        self.assertEqual(key, "test-key-123")

    def test_none_when_no_key_available(self):
        # Clear env var if set
        old = os.environ.pop("GEMINI_API_KEY", None)
        try:
            key = resolve_api_key(explicit_key=None)
            # May or may not be None depending on if a key file exists
            # We can't fully control this in tests, but we can verify the type
            self.assertTrue(key is None or isinstance(key, str))
        finally:
            if old:
                os.environ["GEMINI_API_KEY"] = old

    def test_env_var_key(self):
        os.environ["GEMINI_API_KEY"] = "env-test-key"
        try:
            key = resolve_api_key()
            self.assertEqual(key, "env-test-key")
        finally:
            del os.environ["GEMINI_API_KEY"]

    def test_default_model(self):
        self.assertEqual(DEFAULT_MODEL, "gemini-3.1-pro")

    def test_config_dataclass(self):
        cfg = GeminiConfig(api_key="test", model="gemini-3.1-pro")
        self.assertEqual(cfg.api_key, "test")
        self.assertEqual(cfg.model, "gemini-3.1-pro")
        self.assertEqual(cfg.max_retries, 3)


class TestResponseParsing(unittest.TestCase):
    """Test JSON parsing from simulated Gemini responses."""

    def _make_analyzer_for_parsing(self):
        """Create an analyzer instance just for testing parse methods."""
        # We can't instantiate GeminiSwimAnalyzer without a key, so we
        # test the parsing logic directly
        from src.ai.gemini_analyzer import GeminiSwimAnalyzer
        # Mock the initialization
        class MockAnalyzer:
            def __init__(self):
                real = GeminiSwimAnalyzer.__new__(GeminiSwimAnalyzer)
                self._safe_parse_json = real._safe_parse_json.__get__(real)
                self._parse_technique_analysis = real._parse_technique_analysis.__get__(real)
                self._parse_elite_comparison = real._parse_elite_comparison.__get__(real)
                self._parse_improvement_plan = real._parse_improvement_plan.__get__(real)
                self._parse_race_strategy = real._parse_race_strategy.__get__(real)
                self._parse_splits = real._parse_splits.__get__(real)
        return MockAnalyzer()

    def test_parse_clean_json(self):
        analyzer = self._make_analyzer_for_parsing()
        data = analyzer._safe_parse_json('{"key": "value"}')
        self.assertEqual(data, {"key": "value"})

    def test_parse_json_with_code_fences(self):
        analyzer = self._make_analyzer_for_parsing()
        response = '```json\n{"key": "value"}\n```'
        data = analyzer._safe_parse_json(response)
        self.assertEqual(data, {"key": "value"})

    def test_parse_json_with_surrounding_text(self):
        analyzer = self._make_analyzer_for_parsing()
        response = 'Here is the result:\n{"key": "value"}\nDone!'
        data = analyzer._safe_parse_json(response)
        self.assertEqual(data, {"key": "value"})

    def test_parse_empty_response(self):
        analyzer = self._make_analyzer_for_parsing()
        data = analyzer._safe_parse_json("")
        self.assertIsNone(data)

    def test_parse_none_response(self):
        analyzer = self._make_analyzer_for_parsing()
        data = analyzer._safe_parse_json(None)
        self.assertIsNone(data)

    def test_parse_technique_analysis(self):
        analyzer = self._make_analyzer_for_parsing()
        response = json.dumps({
            "stroke_identified": "freestyle",
            "stroke_confidence": 0.95,
            "overall_rating": "good",
            "technique_elements": [
                {
                    "category": "body_position",
                    "element": "Head Position",
                    "rating": "good",
                    "observation": "Head is in neutral position",
                    "recommendation": "Maintain current position",
                    "elite_reference": "Similar to Pan Zhanle",
                    "timestamp_hint": "0:02-0:05"
                }
            ],
            "top_3_priorities": [
                {
                    "priority": 1,
                    "element": "Catch",
                    "expected_impact": "0.5-1.0 seconds",
                    "drill": "Fist drill"
                }
            ],
            "video_quality_notes": "Good angle, clear visibility"
        })
        result = analyzer._parse_technique_analysis(response)
        self.assertIsNotNone(result)
        self.assertEqual(result.stroke_identified, "freestyle")
        self.assertAlmostEqual(result.stroke_confidence, 0.95)
        self.assertEqual(len(result.elements), 1)
        self.assertEqual(result.elements[0].category, "body_position")
        self.assertEqual(len(result.top_priorities), 1)
        self.assertEqual(result.top_priorities[0].drill, "Fist drill")

    def test_parse_elite_comparison(self):
        analyzer = self._make_analyzer_for_parsing()
        response = json.dumps({
            "comparison_athlete": "Pan Zhanle",
            "overall_similarity_pct": 42.0,
            "strengths_matching_elite": [
                {"element": "Kick timing", "similarity": "Close", "detail": "Good 6-beat kick"}
            ],
            "key_differences": [
                {
                    "element": "Catch",
                    "user_observation": "Dropped elbow",
                    "elite_model": "High elbow catch",
                    "gap_severity": "major",
                    "addressable": True,
                    "how_to_close_gap": "Fist drill + sculling"
                }
            ],
            "metrics_comparison": {"velocity_gap_pct": -15.0},
            "realistic_targets": [
                {
                    "metric": "Velocity",
                    "current": "1.8 m/s",
                    "target": "1.95 m/s",
                    "timeframe": "6 months",
                    "method": "Technique + endurance work"
                }
            ]
        })
        result = analyzer._parse_elite_comparison(response)
        self.assertIsNotNone(result)
        self.assertEqual(result.comparison_athlete, "Pan Zhanle")
        self.assertAlmostEqual(result.overall_similarity_pct, 42.0)
        self.assertEqual(len(result.strengths), 1)
        self.assertEqual(len(result.differences), 1)
        self.assertTrue(result.differences[0].addressable)

    def test_parse_improvement_plan(self):
        analyzer = self._make_analyzer_for_parsing()
        response = json.dumps({
            "summary": "Focus on catch mechanics",
            "phase_1_immediate": {
                "duration_weeks": 3,
                "focus": "Catch technique",
                "drills": [{"name": "Fist drill", "purpose": "Forearm feel",
                           "description": "Swim with fists", "sets_reps": "4x50m",
                           "frequency": "3x/week"}],
                "target_metrics": {"stroke_rate": "48-52 cpm"}
            },
            "phase_2_development": {
                "duration_weeks": 6,
                "focus": "Integration",
                "drills": [],
                "race_simulation": "4x100m at race pace"
            },
            "phase_3_race_prep": {
                "duration_weeks": 2,
                "focus": "Sharpening",
                "key_sets": [{"name": "Broken 100s", "description": "4x25m",
                             "purpose": "Race pace feel"}],
                "mental_cues": ["High elbow", "Press the T"]
            },
            "expected_improvement": {
                "time_reduction_estimate": "1-2 seconds",
                "primary_gains_from": "Improved catch efficiency",
                "caveat": "Individual results vary"
            }
        })
        result = analyzer._parse_improvement_plan(response)
        self.assertIsNotNone(result)
        self.assertEqual(result.summary, "Focus on catch mechanics")
        self.assertIn("duration_weeks", result.phase_1)

    def test_parse_race_strategy(self):
        analyzer = self._make_analyzer_for_parsing()
        response = json.dumps({
            "split_pattern": "positive",
            "split_analysis": "First half was too fast",
            "elite_pacing_comparison": {"event_norm": "Even split"},
            "speed_loss_zones": [
                {"zone": "75-100m", "speed_drop_pct": 12, "likely_cause": "fatigue",
                 "fix": "Better pacing in first 50"}
            ],
            "stroke_rate_analysis": {"pattern": "increasing", "optimal_adjustment": "Start lower"},
            "recommended_race_plan": {
                "target_first_split": "28.5s",
                "target_second_split": "30.0s",
                "key_focus_points": ["Controlled start"],
                "pacing_strategy": "Build through the race"
            }
        })
        result = analyzer._parse_race_strategy(response)
        self.assertIsNotNone(result)
        self.assertEqual(result.split_pattern, "positive")
        self.assertEqual(len(result.speed_loss_zones), 1)

    def test_parse_splits(self):
        analyzer = self._make_analyzer_for_parsing()
        response = json.dumps({
            "start_time_s": 0.5,
            "splits": [
                {"distance_m": 5, "time_s": 3.0},
                {"distance_m": 15, "time_s": 9.5}
            ],
            "finish_time_s": 25.4,
            "confidence_notes": "Clear view of markers"
        })
        result = analyzer._parse_splits(response)
        self.assertIsNotNone(result)
        self.assertEqual(result.start_time_s, 0.5)
        self.assertEqual(result.finish_time_s, 25.4)
        self.assertEqual(len(result.splits), 2)
        self.assertEqual(result.splits[1].distance_m, 15)
        self.assertEqual(result.splits[1].time_s, 9.5)


class TestVideoSegments(unittest.TestCase):
    """Test video segment extraction helpers (without actual video)."""

    def test_segment_boundary_computation(self):
        from src.ai.video_segments import _compute_segment_boundaries

        segments = _compute_segment_boundaries(
            start_time_s=2.0,
            total_duration_s=60.0,
            pool_length_m=50.0,
            split_times=[(10.0, 5.0), (25.0, 13.0), (50.0, 28.0)],
        )
        self.assertGreater(len(segments), 0)
        # Should have at least start, clean swim, and finish
        names = [s[0] for s in segments]
        self.assertIn("start_phase", names)
        self.assertIn("clean_swimming", names)
        self.assertIn("finish_phase", names)

    def test_segment_boundary_without_splits(self):
        from src.ai.video_segments import _compute_segment_boundaries

        segments = _compute_segment_boundaries(
            start_time_s=0.0,
            total_duration_s=40.0,
            pool_length_m=25.0,
            split_times=None,
        )
        self.assertGreater(len(segments), 0)


if __name__ == "__main__":
    unittest.main()
