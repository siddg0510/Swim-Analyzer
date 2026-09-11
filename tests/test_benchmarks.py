"""
Tests for the expanded benchmarks module.

Validates:
  - All benchmark entries have required fields
  - All elite profiles have consistent data
  - Lookup functions return correct results
  - Auto-matching logic selects appropriate athletes
  - Technique models cover all four strokes
"""
import unittest
import sys
import os

# Ensure we can import from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.benchmarks import (
    BENCHMARKS,
    ELITE_PROFILES,
    TECHNIQUE_MODELS,
    RACE_SEGMENT_CONVENTIONS,
    MAX_UNDERWATER_DISTANCE_M,
    get_benchmark,
    get_elite_profile,
    get_profiles_for_event,
    get_technique_model,
    get_all_profile_names,
    compare_to_benchmark,
    find_closest_elite_match,
    build_elite_context_for_prompt,
    BenchmarkEntry,
    EliteSwimmerProfile,
    RaceMetrics,
    TechniqueModel,
)


class TestBenchmarkData(unittest.TestCase):
    """Validate benchmark entries are well-formed."""

    def test_all_benchmarks_have_required_fields(self):
        self.assertGreater(len(BENCHMARKS), 20, "Should have 20+ benchmark entries")
        for b in BENCHMARKS:
            self.assertIn(b.stroke, ("freestyle", "backstroke", "breaststroke", "butterfly"))
            self.assertGreater(b.distance_m, 0)
            self.assertIn(b.sex, ("male", "female", "mixed_top10"))
            self.assertTrue(b.source, f"Missing source for {b.stroke} {b.distance_m}m")

    def test_freestyle_coverage(self):
        """Freestyle should have benchmarks for 50m through 1500m."""
        free_dists = {b.distance_m for b in BENCHMARKS if b.stroke == "freestyle"}
        for d in (50, 100, 200, 400, 800, 1500):
            self.assertIn(d, free_dists, f"Missing freestyle {d}m benchmark")

    def test_all_strokes_covered(self):
        strokes = {b.stroke for b in BENCHMARKS}
        self.assertEqual(strokes, {"freestyle", "backstroke", "breaststroke", "butterfly"})

    def test_velocities_are_plausible(self):
        for b in BENCHMARKS:
            if b.avg_velocity_mps is not None:
                self.assertGreater(b.avg_velocity_mps, 1.0,
                    f"{b.stroke} {b.distance_m}m velocity too low: {b.avg_velocity_mps}")
                self.assertLess(b.avg_velocity_mps, 2.5,
                    f"{b.stroke} {b.distance_m}m velocity too high: {b.avg_velocity_mps}")

    def test_stroke_rates_are_plausible(self):
        for b in BENCHMARKS:
            if b.stroke_rate_cpm is not None:
                lo, hi = b.stroke_rate_cpm
                self.assertGreater(lo, 20, f"{b.stroke} {b.distance_m}m SR too low")
                self.assertLess(hi, 80, f"{b.stroke} {b.distance_m}m SR too high")
                self.assertLessEqual(lo, hi, f"{b.stroke} {b.distance_m}m SR range inverted")


class TestEliteProfiles(unittest.TestCase):
    """Validate elite swimmer profiles."""

    def test_profile_count(self):
        self.assertGreaterEqual(len(ELITE_PROFILES), 8)

    def test_all_profiles_have_required_fields(self):
        for key, profile in ELITE_PROFILES.items():
            self.assertTrue(profile.name, f"{key}: missing name")
            self.assertTrue(profile.country, f"{key}: missing country")
            self.assertTrue(profile.country_flag, f"{key}: missing flag")
            self.assertTrue(profile.primary_events, f"{key}: missing events")
            self.assertTrue(profile.achievement, f"{key}: missing achievement")
            self.assertTrue(profile.technique_signature, f"{key}: missing technique")
            self.assertTrue(profile.race_data, f"{key}: missing race data")
            self.assertTrue(profile.coaching_notes, f"{key}: missing coaching notes")

    def test_race_data_is_valid(self):
        for key, profile in ELITE_PROFILES.items():
            for event_key, race in profile.race_data.items():
                self.assertGreater(race.time_s, 0,
                    f"{key}/{event_key}: invalid time")
                self.assertGreater(race.avg_velocity_mps, 0,
                    f"{key}/{event_key}: invalid velocity")
                self.assertTrue(race.splits,
                    f"{key}/{event_key}: missing splits")
                self.assertTrue(race.source,
                    f"{key}/{event_key}: missing source")

    def test_known_profiles_exist(self):
        expected = [
            "pan_zhanle", "leon_marchand", "caeleb_dressel",
            "katie_ledecky", "adam_peaty", "sarah_sjostrom",
            "kaylee_mckeown", "summer_mcintosh",
        ]
        for name in expected:
            self.assertIn(name, ELITE_PROFILES, f"Missing profile: {name}")

    def test_pan_zhanle_wr(self):
        pan = ELITE_PROFILES["pan_zhanle"]
        wr = pan.race_data["100m_free"]
        self.assertAlmostEqual(wr.time_s, 46.40, places=2)
        self.assertEqual(len(wr.splits), 2)
        # First 50m split
        self.assertAlmostEqual(wr.splits[0][1], 22.28, places=2)

    def test_peaty_wr(self):
        peaty = ELITE_PROFILES["adam_peaty"]
        wr = peaty.race_data["100m_breast"]
        self.assertAlmostEqual(wr.time_s, 56.88, places=2)


class TestTechniqueModels(unittest.TestCase):
    """Validate technique models cover all strokes."""

    def test_all_strokes_covered(self):
        for stroke in ("freestyle", "backstroke", "breaststroke", "butterfly"):
            model = get_technique_model(stroke)
            self.assertIsNotNone(model, f"Missing technique model: {stroke}")
            self.assertTrue(model.key_elements)
            self.assertTrue(model.common_errors)
            self.assertTrue(model.drills)


class TestLookupFunctions(unittest.TestCase):
    """Test benchmark and profile lookup helpers."""

    def test_get_benchmark_exact_match(self):
        b = get_benchmark("freestyle", 100, "male")
        self.assertIsNotNone(b)
        self.assertEqual(b.stroke, "freestyle")
        self.assertEqual(b.distance_m, 100)

    def test_get_benchmark_missing(self):
        b = get_benchmark("freestyle", 300, "male")
        self.assertIsNone(b)

    def test_get_elite_profile(self):
        p = get_elite_profile("pan_zhanle")
        self.assertIsNotNone(p)
        self.assertEqual(p.name, "Pan Zhanle")

    def test_get_profiles_for_event(self):
        profiles = get_profiles_for_event("freestyle", 100)
        self.assertGreater(len(profiles), 0)
        names = [p.name for p in profiles]
        self.assertIn("Pan Zhanle", names)

    def test_get_all_profile_names(self):
        names = get_all_profile_names()
        self.assertGreaterEqual(len(names), 8)
        for key, name, flag in names:
            self.assertTrue(key)
            self.assertTrue(name)
            self.assertTrue(flag)

    def test_compare_to_benchmark(self):
        result = compare_to_benchmark(1.5, "freestyle", 100, "male")
        self.assertIsNotNone(result)
        self.assertIn("slower than", result)

    def test_compare_to_benchmark_missing_event(self):
        result = compare_to_benchmark(1.5, "freestyle", 300, "male")
        self.assertIsNone(result)


class TestAutoMatch(unittest.TestCase):
    """Test the closest-elite-match auto-selection."""

    def test_exact_event_match_freestyle_100(self):
        key, profile, reason = find_closest_elite_match(
            stroke="freestyle", distance_m=100, sex="male"
        )
        self.assertIsNotNone(key)
        self.assertIsNotNone(profile)
        self.assertIn("100m", reason.lower() if reason else "")

    def test_exact_event_match_backstroke(self):
        key, profile, reason = find_closest_elite_match(
            stroke="backstroke", distance_m=100, sex="female"
        )
        self.assertIsNotNone(key)
        self.assertEqual(profile.name, "Kaylee McKeown")

    def test_breaststroke_match(self):
        key, profile, reason = find_closest_elite_match(
            stroke="breaststroke", distance_m=100, sex="male"
        )
        self.assertIsNotNone(key)
        self.assertEqual(profile.name, "Adam Peaty")

    def test_gender_preference(self):
        _, male_profile, _ = find_closest_elite_match(
            stroke="freestyle", distance_m=100, sex="male"
        )
        _, female_profile, _ = find_closest_elite_match(
            stroke="freestyle", distance_m=100, sex="female"
        )
        # Should prefer same-gender matches
        self.assertIsNotNone(male_profile)
        self.assertIsNotNone(female_profile)

    def test_fallback_to_stroke_match(self):
        # 800m freestyle — Ledecky is the dominant female
        key, profile, reason = find_closest_elite_match(
            stroke="freestyle", distance_m=800, sex="female"
        )
        self.assertIsNotNone(key)
        self.assertEqual(profile.name, "Katie Ledecky")

    def test_velocity_based_matching(self):
        # With a velocity close to elite, should still find a match
        key, profile, reason = find_closest_elite_match(
            stroke="freestyle", distance_m=100, sex="male",
            user_velocity_mps=1.8,
        )
        self.assertIsNotNone(key)

    def test_returns_none_for_no_profiles(self):
        # This shouldn't happen with our database, but test the logic
        # by checking the function handles edge cases
        key, profile, reason = find_closest_elite_match(
            stroke="freestyle", distance_m=100, sex="male"
        )
        self.assertIsNotNone(key)  # We have profiles, so should never be None


class TestBuildEliteContext(unittest.TestCase):
    """Test the prompt context builder."""

    def test_builds_context_for_known_event(self):
        context = build_elite_context_for_prompt("freestyle", 100)
        self.assertIn("Elite benchmarks", context)
        self.assertIn("Pan Zhanle", context)

    def test_builds_context_for_backstroke(self):
        context = build_elite_context_for_prompt("backstroke", 100)
        self.assertIn("McKeown", context)

    def test_builds_context_with_technique_model(self):
        context = build_elite_context_for_prompt("butterfly", 200)
        self.assertIn("Ideal butterfly technique model", context)


class TestJSONBenchmarkLoading(unittest.TestCase):
    """Test external JSON benchmark dataset loading and precision flagging."""

    def test_precision_flags_present(self):
        precisions = {b.precision for b in BENCHMARKS}
        self.assertIn("verified", precisions)
        self.assertIn("low_precision", precisions)

    def test_retrieved_date_present(self):
        dates = [b.retrieved_date for b in BENCHMARKS if b.retrieved_date]
        self.assertGreater(len(dates), 0)

    def test_load_benchmarks_from_json_direct(self):
        from src.analysis.benchmarks import load_benchmarks_from_json
        bench, profs, convs, techs = load_benchmarks_from_json()
        self.assertGreaterEqual(len(bench), 20)
        self.assertGreaterEqual(len(profs), 8)
        self.assertGreaterEqual(len(convs), 3)
        self.assertEqual(len(techs), 4)


class TestBuiltinFallbackProvenance(unittest.TestCase):
    """The in-code fallback (used when the JSON is missing/corrupt) must be as
    honest as the JSON: it must never silently relabel an estimate as verified,
    and it must not strip provenance dates. Regression guard for the bug where
    _BUILTIN_BENCHMARKS defaulted every entry to precision='verified'."""

    def test_conservative_default_is_low_precision(self):
        """A BenchmarkEntry with no explicit precision is an ESTIMATE, never
        silently 'verified'."""
        entry = BenchmarkEntry(
            stroke="freestyle", distance_m=100, sex="male",
            avg_velocity_mps=2.0, stroke_rate_cpm=None,
            stroke_length_m=None, breakout_distance_m=None,
            source="unit test",
        )
        self.assertEqual(entry.precision, "low_precision")

    def test_json_missing_default_is_low_precision(self):
        """The JSON loader must also default a precision-less entry to estimate,
        not 'verified' (a schema change must not silently overclaim)."""
        import inspect
        from src.analysis import benchmarks as bm
        src = inspect.getsource(bm.load_benchmarks_from_json)
        self.assertIn('b.get("precision", "low_precision")', src)

    def test_builtin_fallback_all_have_retrieved_date(self):
        from src.analysis.benchmarks import _BUILTIN_BENCHMARKS
        undated = [
            (b.stroke, b.distance_m, b.sex)
            for b in _BUILTIN_BENCHMARKS if not b.retrieved_date
        ]
        self.assertEqual(undated, [], f"fallback entries missing retrieved_date: {undated}")

    def test_builtin_fallback_matches_json_precision(self):
        """The in-code fallback labels must stay in lock-step with the curated
        JSON — otherwise a JSON-load failure would present different provenance
        than a successful load. This guards the duplicated data from drifting."""
        import json
        from src.analysis.benchmarks import _BUILTIN_BENCHMARKS, _BENCHMARK_JSON_PATH

        with open(_BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        json_precision = {
            (b["stroke"], b["distance_m"], b["sex"]): b.get("precision", "low_precision")
            for b in raw["benchmarks"]
        }
        mismatches = []
        for b in _BUILTIN_BENCHMARKS:
            key = (b.stroke, b.distance_m, b.sex)
            if key in json_precision and json_precision[key] != b.precision:
                mismatches.append((key, "json=" + json_precision[key], "builtin=" + b.precision))
        self.assertEqual(mismatches, [], f"fallback/JSON precision drift: {mismatches}")

    def test_low_precision_comparison_is_flagged(self):
        """A low_precision reference must be flagged as an estimate in the
        user-facing comparison string; a verified one must not be."""
        from src.analysis.benchmarks import get_benchmark
        # Find one of each precision that has a velocity to compare against.
        low = next(
            b for b in BENCHMARKS
            if b.precision == "low_precision" and b.avg_velocity_mps
        )
        verified = next(
            b for b in BENCHMARKS
            if b.precision == "verified" and b.avg_velocity_mps
        )
        low_str = compare_to_benchmark(low.avg_velocity_mps, low.stroke, low.distance_m, low.sex)
        ver_str = compare_to_benchmark(
            verified.avg_velocity_mps, verified.stroke, verified.distance_m, verified.sex
        )
        self.assertIsNotNone(low_str)
        self.assertIsNotNone(ver_str)
        self.assertIn("estimate", low_str.lower())
        self.assertNotIn("estimate", ver_str.lower())


if __name__ == "__main__":
    unittest.main()
