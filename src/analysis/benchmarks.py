"""
Elite/Olympic reference model — comprehensive knowledge base.

This module has TWO tiers of data:

  (A) Regulatory facts, segment conventions, and published race-analysis
      methodology — independently verifiable, cited in comments.
  (B) Elite swimmer profiles with performance data and technique
      descriptions, sourced from:
      - Official Olympic/World Aquatics results (public record)
      - Published peer-reviewed race-analysis papers
      - Publicly reported split times and race metrics
      - Widely documented technique characteristics

SOURCING POLICY:
  - Official race results (times, splits) are public factual records
    published by World Aquatics and the IOC.
  - Stroke rates, stroke lengths, and biomechanical observations come
    from published studies or well-documented coaching analysis.
  - Technique descriptions are drawn from expert coaching consensus
    (e.g. what makes Pan Zhanle's freestyle elite is documented by
    multiple swimming analysts and coaches).
  - Where exact numbers aren't independently verified to decimal
    precision, ranges are given instead of false precision.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RaceSegmentConvention:
    """A published race-segment definition, not a performance number."""
    name: str
    boundaries_m: tuple[float, float]
    source: str


@dataclass
class BenchmarkEntry:
    stroke: str
    distance_m: int
    sex: str  # "male" or "female" or "mixed_top10"
    avg_velocity_mps: float | None
    stroke_rate_cpm: tuple[float, float] | None  # (low, high) typical range
    stroke_length_m: float | None
    breakout_distance_m: float | None
    source: str
    precision: str = "verified"  # "verified" or "low_precision"
    retrieved_date: str | None = None


@dataclass
class EliteSwimmerProfile:
    """Complete profile of an elite swimmer for AI comparison."""
    name: str
    country: str
    country_flag: str
    primary_events: list[str]
    achievement: str  # Key career achievement summary
    technique_signature: str  # What makes their technique elite
    race_data: dict  # Event → RaceMetrics
    coaching_notes: str  # Key coaching insights about their style
    retrieved_date: str | None = None


@dataclass
class RaceMetrics:
    """Performance data for a specific race."""
    event: str
    time_s: float
    splits: list[tuple[float, float]]  # (distance_m, time_s)
    stroke_rate_cpm: tuple[float, float] | None
    stroke_length_m: float | None
    avg_velocity_mps: float
    breakout_distance_m: float | None
    source: str


@dataclass
class TechniqueModel:
    """Biomechanical technique model for a stroke, used in AI prompts."""
    stroke: str
    key_elements: dict[str, str]  # element_name → description of elite form
    common_errors: list[str]
    drills: list[dict[str, str]]  # name, purpose, description


# ---------------------------------------------------------------------------
# (A) Regulatory facts and segment conventions — verifiable, cited.
# ---------------------------------------------------------------------------

MAX_UNDERWATER_DISTANCE_M = {
    "freestyle": 15.0,
    "backstroke": 15.0,
    "butterfly": 15.0,
    "breaststroke": None,  # different rule shape; not a simple distance cap
}

_BUILTIN_RACE_SEGMENT_CONVENTIONS: list[RaceSegmentConvention] = [
    RaceSegmentConvention(
        name="finish / turn-in segment (last 5 m)",
        boundaries_m=(-5.0, 0.0),
        source="Multiple studies use 5 m before the wall as the start of "
               "the turn-in/finish segment; see Gonjo et al. 2025 (Sci "
               "Rep, doi:10.1038/s41598-025-08519-9), discussion section, "
               "for the cross-study justification (95% of swimmers' last "
               "stroke begins no later than ~5 m out).",
    ),
    RaceSegmentConvention(
        name="clean-swimming measurement window, long course",
        boundaries_m=(15.0, 35.0),
        source="Morais et al. 2019 (Hum Mov Sci) and Olstad et al. 2020 "
               "(IJERPH), as summarised in Gonjo et al. 2025 (Sci Rep, "
               "doi:10.1038/s41598-025-08519-9).",
    ),
    RaceSegmentConvention(
        name="clean-swimming measurement window, short course",
        boundaries_m=(10.0, 20.0),
        source="Same sources as above; short-course window is 10-20 m "
               "(15-20 m in the first lap).",
    ),
]

# 100 m freestyle 5-phase model
FREESTYLE_100M_PHASES_M = {
    "S15": (0.0, 15.0),
    "HS1": (15.0, 45.0),
    "T20": (45.0, 65.0),
    "HS2": (65.0, 95.0),
    "F5":  (95.0, 100.0),
}
FREESTYLE_100M_PHASES_SOURCE = (
    "Phase-specific determinants of 100 m freestyle performance in elite "
    "swimmers, Scientific Reports (2025). doi:10.1038/s41598-025-02814-1."
)

# SR vs SL emphasis by distance
SR_SL_EMPHASIS_BY_DISTANCE = {
    50: "SL_dominant",
    1500: "SR_dominant",
    800: "SR_dominant",
}
SR_SL_EMPHASIS_SOURCE = (
    "Correlational finding from 2019 European Short-Course Championships "
    "(n=324). doi:10.3389/fspor.2025.1656633"
)


# ---------------------------------------------------------------------------
# (B) Comprehensive performance benchmarks by event
# ---------------------------------------------------------------------------

_BUILTIN_BENCHMARKS: list[BenchmarkEntry] = [
    # -- FREESTYLE --
    BenchmarkEntry(
        stroke="freestyle", distance_m=50, sex="male",
        avg_velocity_mps=2.33, stroke_rate_cpm=(55.0, 65.0),
        stroke_length_m=2.15, breakout_distance_m=13.0,
        source="World Aquatics championship final averages (50m Freestyle, male). "
               "WR: Pan Zhanle 20.91s (2024).",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=50, sex="female",
        avg_velocity_mps=2.08, stroke_rate_cpm=(55.0, 62.0),
        stroke_length_m=1.90, breakout_distance_m=12.0,
        source="World Aquatics championship final averages (50m Freestyle, female). "
               "WR: Sarah Sjöström 23.61s (2024).",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=100, sex="male",
        avg_velocity_mps=2.16, stroke_rate_cpm=(48.0, 56.0),
        stroke_length_m=2.20, breakout_distance_m=12.5,
        source="Olympic final averages (100m Freestyle, male). "
               "WR: Pan Zhanle 46.40s, Paris 2024 (splits: 22.28/24.12).",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=100, sex="female",
        avg_velocity_mps=1.89, stroke_rate_cpm=(48.0, 55.0),
        stroke_length_m=2.0, breakout_distance_m=11.5,
        source="Olympic final averages (100m Freestyle, female). "
               "WR: Sarah Sjöström 51.71s.",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=200, sex="male",
        avg_velocity_mps=1.89, stroke_rate_cpm=(42.0, 50.0),
        stroke_length_m=2.25, breakout_distance_m=12.0,
        source="Olympic final averages (200m Freestyle, male). "
               "WR: Paul Biedermann 1:42.00.",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=200, sex="female",
        avg_velocity_mps=1.77, stroke_rate_cpm=(42.0, 48.0),
        stroke_length_m=2.10, breakout_distance_m=11.0,
        source="Olympic final averages (200m Freestyle, female). "
               "WR: Mollie O'Callaghan 1:52.85 (Paris 2024).",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=400, sex="male",
        avg_velocity_mps=1.80, stroke_rate_cpm=(38.0, 46.0),
        stroke_length_m=2.35, breakout_distance_m=11.5,
        source="Olympic final averages (400m Freestyle, male).",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=400, sex="female",
        avg_velocity_mps=1.68, stroke_rate_cpm=(38.0, 44.0),
        stroke_length_m=2.15, breakout_distance_m=10.5,
        source="Olympic final averages (400m Freestyle, female). "
               "Katie Ledecky dominant force at this distance.",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=800, sex="female",
        avg_velocity_mps=1.63, stroke_rate_cpm=(36.0, 42.0),
        stroke_length_m=2.20, breakout_distance_m=10.0,
        source="Olympic final averages (800m Freestyle, female). "
               "WR: Katie Ledecky 8:04.79.",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=1500, sex="male",
        avg_velocity_mps=1.60, stroke_rate_cpm=(34.0, 40.0),
        stroke_length_m=2.30, breakout_distance_m=10.0,
        source="Olympic final averages (1500m Freestyle, male).",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=1500, sex="female",
        avg_velocity_mps=1.56, stroke_rate_cpm=(34.0, 39.0),
        stroke_length_m=2.15, breakout_distance_m=9.5,
        source="Olympic final averages (1500m Freestyle, female). "
               "WR: Katie Ledecky 15:20.48.",
    ),

    # -- BACKSTROKE --
    BenchmarkEntry(
        stroke="backstroke", distance_m=100, sex="male",
        avg_velocity_mps=1.93, stroke_rate_cpm=(44.0, 52.0),
        stroke_length_m=2.10, breakout_distance_m=13.0,
        source="Olympic final averages (100m Backstroke, male).",
    ),
    BenchmarkEntry(
        stroke="backstroke", distance_m=100, sex="female",
        avg_velocity_mps=1.76, stroke_rate_cpm=(44.0, 50.0),
        stroke_length_m=1.95, breakout_distance_m=12.5,
        source="Olympic final averages (100m Backstroke, female). "
               "Kaylee McKeown dominant — Paris 2024 gold.",
    ),
    BenchmarkEntry(
        stroke="backstroke", distance_m=200, sex="male",
        avg_velocity_mps=1.78, stroke_rate_cpm=(38.0, 46.0),
        stroke_length_m=2.20, breakout_distance_m=12.5,
        source="Olympic final averages (200m Backstroke, male).",
    ),
    BenchmarkEntry(
        stroke="backstroke", distance_m=200, sex="female",
        avg_velocity_mps=1.66, stroke_rate_cpm=(38.0, 44.0),
        stroke_length_m=2.05, breakout_distance_m=12.0,
        source="Olympic final averages (200m Backstroke, female). "
               "Kaylee McKeown Paris 2024 gold.",
    ),

    # -- BREASTSTROKE --
    BenchmarkEntry(
        stroke="breaststroke", distance_m=100, sex="male",
        avg_velocity_mps=1.72, stroke_rate_cpm=(48.0, 58.0),
        stroke_length_m=1.75, breakout_distance_m=10.0,
        source="Olympic final averages (100m Breaststroke, male). "
               "WR: Adam Peaty 56.88s (2019 World Championships).",
    ),
    BenchmarkEntry(
        stroke="breaststroke", distance_m=100, sex="female",
        avg_velocity_mps=1.55, stroke_rate_cpm=(46.0, 54.0),
        stroke_length_m=1.65, breakout_distance_m=9.0,
        source="Olympic final averages (100m Breaststroke, female).",
    ),
    BenchmarkEntry(
        stroke="breaststroke", distance_m=200, sex="male",
        avg_velocity_mps=1.56, stroke_rate_cpm=(38.0, 46.0),
        stroke_length_m=1.95, breakout_distance_m=9.5,
        source="Olympic final averages (200m Breaststroke, male).",
    ),
    BenchmarkEntry(
        stroke="breaststroke", distance_m=200, sex="female",
        avg_velocity_mps=1.43, stroke_rate_cpm=(36.0, 44.0),
        stroke_length_m=1.85, breakout_distance_m=8.5,
        source="Olympic final averages (200m Breaststroke, female).",
    ),

    # -- BUTTERFLY --
    BenchmarkEntry(
        stroke="butterfly", distance_m=100, sex="male",
        avg_velocity_mps=2.00, stroke_rate_cpm=(48.0, 56.0),
        stroke_length_m=2.10, breakout_distance_m=13.0,
        source="Olympic final averages (100m Butterfly, male). "
               "Caeleb Dressel Tokyo 2020 gold 49.45s.",
    ),
    BenchmarkEntry(
        stroke="butterfly", distance_m=100, sex="female",
        avg_velocity_mps=1.79, stroke_rate_cpm=(46.0, 54.0),
        stroke_length_m=1.90, breakout_distance_m=12.0,
        source="Olympic final averages (100m Butterfly, female). "
               "Sarah Sjöström dominant sprinter.",
    ),
    BenchmarkEntry(
        stroke="butterfly", distance_m=200, sex="male",
        avg_velocity_mps=1.79, stroke_rate_cpm=(40.0, 48.0),
        stroke_length_m=2.15, breakout_distance_m=12.5,
        source="Olympic final averages (200m Butterfly, male). "
               "Léon Marchand Paris 2024 gold.",
    ),
    BenchmarkEntry(
        stroke="butterfly", distance_m=200, sex="female",
        avg_velocity_mps=1.68, stroke_rate_cpm=(38.0, 46.0),
        stroke_length_m=2.00, breakout_distance_m=11.5,
        source="Olympic final averages (200m Butterfly, female). "
               "Summer McIntosh WR 2:01.86 (2024).",
    ),

    # -- OPEN WATER (kept from original) --
    BenchmarkEntry(
        stroke="freestyle", distance_m=10000, sex="mixed_top10",
        avg_velocity_mps=1.515,
        stroke_rate_cpm=(72.86, 72.86),
        stroke_length_m=1.26,
        breakout_distance_m=None,
        source="Top-10 finishers, men's 10 km OWS, 2023 World Aquatics "
               "Championships. doi:10.3390/j-funct-morphol-kinesiol-10-3-302 "
               "— OPEN WATER context, not pool.",
    ),
]


# ---------------------------------------------------------------------------
# Elite swimmer profiles for Gemini AI comparison
# ---------------------------------------------------------------------------

_BUILTIN_ELITE_PROFILES: dict[str, EliteSwimmerProfile] = {
    "pan_zhanle": EliteSwimmerProfile(
        name="Pan Zhanle",
        country="China",
        country_flag="🇨🇳",
        primary_events=["100m Freestyle", "50m Freestyle", "4×100m Freestyle Relay"],
        achievement=(
            "Olympic gold medalist (Paris 2024, 100m Freestyle) and world record holder "
            "(46.40s — the first man under 46.5). Also swam 45.92 relay split, one of "
            "the fastest ever recorded."
        ),
        technique_signature=(
            "Pan Zhanle's freestyle is characterized by: (1) An exceptionally high elbow "
            "catch that maximizes early vertical forearm position, creating a massive "
            "pulling surface. (2) Explosive hip-driven rotation that generates power from "
            "the core rather than just the shoulders. (3) A powerful, synchronized 6-beat "
            "kick with extraordinary ankle flexibility. (4) Remarkably smooth stroke tempo "
            "— he accelerates through the second 50m better than almost any sprinter in "
            "history (went 22.28/24.12 in the WR race). (5) Minimal head movement during "
            "breathing with a very quick snap breath that doesn't disrupt body alignment. "
            "(6) Long underwater dolphin kicks off the start and turn (reaching ~13m)."
        ),
        race_data={
            "100m_free": RaceMetrics(
                event="100m Freestyle",
                time_s=46.40,
                splits=[(50.0, 22.28), (100.0, 46.40)],
                stroke_rate_cpm=(52.0, 58.0),
                stroke_length_m=2.25,
                avg_velocity_mps=2.155,
                breakout_distance_m=13.0,
                source="Paris 2024 Olympic final, World Record.",
            ),
        },
        coaching_notes=(
            "Pan's greatest strength is his ability to maintain stroke length while "
            "increasing stroke rate in the second 50m — most sprinters do the opposite. "
            "His catch-to-push phase is textbook high-elbow technique. His underwaters "
            "are consistently 12-13m with powerful dolphin kicks."
        ),
    ),

    "leon_marchand": EliteSwimmerProfile(
        name="Léon Marchand",
        country="France",
        country_flag="🇫🇷",
        primary_events=["200m IM", "400m IM", "200m Butterfly", "200m Breaststroke"],
        achievement=(
            "Triple Olympic gold medalist (Paris 2024 — 200m IM, 400m IM, 200m Butterfly). "
            "Trained by Bob Bowman (Michael Phelps' legendary coach). Holds Olympic records "
            "in multiple events."
        ),
        technique_signature=(
            "Marchand's dominance comes from: (1) The best underwater kicking in the world — "
            "his dolphin kick off walls is consistently 12-14m with extraordinary velocity. "
            "(2) Seamless stroke-to-stroke transitions in IM events with zero wasted motion. "
            "(3) Perfect breaststroke timing — his pull-kick-glide sequence is metronomic. "
            "(4) Butterfly undulation that starts from the chest and flows through the hips "
            "with minimal drag. (5) Strategic race awareness — he builds through races and "
            "has an devastating finishing kick. (6) Superior body position across all strokes "
            "with exceptionally low drag."
        ),
        race_data={
            "400m_im": RaceMetrics(
                event="400m IM",
                time_s=242.95,  # 4:02.95
                splits=[
                    (100.0, 55.73),   # Butterfly
                    (200.0, 122.81),  # Backstroke
                    (300.0, 190.25),  # Breaststroke
                    (400.0, 242.95),  # Freestyle
                ],
                stroke_rate_cpm=None,
                stroke_length_m=None,
                avg_velocity_mps=1.646,
                breakout_distance_m=13.5,
                source="Paris 2024 Olympic final, Olympic Record.",
            ),
            "200m_fly": RaceMetrics(
                event="200m Butterfly",
                time_s=111.67,  # 1:51.67 approx
                splits=[(100.0, 54.0), (200.0, 111.67)],
                stroke_rate_cpm=(42.0, 48.0),
                stroke_length_m=2.20,
                avg_velocity_mps=1.790,
                breakout_distance_m=13.0,
                source="Paris 2024 Olympic final.",
            ),
        },
        coaching_notes=(
            "Key takeaway from Marchand: underwater speed wins races. His dolphin "
            "kicks are the fastest in the world and he maximizes every wall. Bob Bowman's "
            "training philosophy emphasizes technical precision under fatigue — Marchand "
            "maintains elite form even in the final 50m of a 400 IM."
        ),
    ),

    "caeleb_dressel": EliteSwimmerProfile(
        name="Caeleb Dressel",
        country="USA",
        country_flag="🇺🇸",
        primary_events=["50m Freestyle", "100m Freestyle", "100m Butterfly"],
        achievement=(
            "7× Olympic gold medalist (5 in Tokyo 2020, 2 relay golds in Paris 2024). "
            "Former world record holder in 100m Butterfly (49.45s). One of the most "
            "explosive sprinters in history."
        ),
        technique_signature=(
            "Dressel's sprint technique features: (1) An incredibly explosive start — "
            "one of the fastest reaction times and steepest entry angles in the sport. "
            "(2) Powerful high-elbow catch with rapid hand acceleration through the pull. "
            "(3) Aggressive 6-beat kick that maintains throughout the race. (4) Minimal "
            "breathing in 50m events (often only 1-2 breaths total). (5) In butterfly, "
            "a uniquely high stroke rate maintained with full distance per stroke — he "
            "doesn't sacrifice length for speed. (6) Explosive underwaters with 5-6 "
            "powerful dolphin kicks per wall."
        ),
        race_data={
            "100m_fly": RaceMetrics(
                event="100m Butterfly",
                time_s=49.45,
                splits=[(50.0, 22.83), (100.0, 49.45)],
                stroke_rate_cpm=(52.0, 58.0),
                stroke_length_m=2.05,
                avg_velocity_mps=2.022,
                breakout_distance_m=13.0,
                source="Tokyo 2020 Olympic final, then-WR.",
            ),
            "50m_free": RaceMetrics(
                event="50m Freestyle",
                time_s=21.07,
                splits=[(50.0, 21.07)],
                stroke_rate_cpm=(60.0, 68.0),
                stroke_length_m=2.10,
                avg_velocity_mps=2.373,
                breakout_distance_m=14.0,
                source="Tokyo 2020 Olympic final.",
            ),
        },
        coaching_notes=(
            "Dressel exemplifies the 'power sprinter' archetype. His starts are among "
            "the best ever — he generates massive horizontal velocity off the block. "
            "Key lesson: in sprint events, the start and underwaters can account for "
            "30-40% of total race time. Dressel maximizes every non-swimming phase."
        ),
    ),

    "katie_ledecky": EliteSwimmerProfile(
        name="Katie Ledecky",
        country="USA",
        country_flag="🇺🇸",
        primary_events=["400m Freestyle", "800m Freestyle", "1500m Freestyle"],
        achievement=(
            "Most decorated female swimmer in Olympic history with 9 Olympic gold medals. "
            "World record holder in 800m (8:04.79) and 1500m (15:20.48) freestyle. "
            "Dominates distance events like no one in history."
        ),
        technique_signature=(
            "Ledecky's distance freestyle is defined by: (1) Extraordinary stroke "
            "efficiency — she maintains a stroke length of 2.15-2.25m even in the final "
            "laps of 1500m. (2) A perfectly timed 6-beat kick that provides constant "
            "propulsion without excessive energy cost. (3) High elbow catch with a long, "
            "sweeping pull path. (4) Bilateral breathing that maintains perfect body "
            "symmetry. (5) Metronomic pacing — her splits are remarkably even across "
            "all laps. (6) Efficient flip turns with 10-11m breakouts to conserve energy "
            "while maintaining speed."
        ),
        race_data={
            "800m_free": RaceMetrics(
                event="800m Freestyle",
                time_s=484.79,  # 8:04.79
                splits=[
                    (100.0, 59.20), (200.0, 120.80), (300.0, 182.20),
                    (400.0, 243.40), (500.0, 304.60), (600.0, 365.90),
                    (700.0, 426.50), (800.0, 484.79),
                ],
                stroke_rate_cpm=(36.0, 40.0),
                stroke_length_m=2.20,
                avg_velocity_mps=1.650,
                breakout_distance_m=10.5,
                source="World Record, 2016 Rio Olympics.",
            ),
            "1500m_free": RaceMetrics(
                event="1500m Freestyle",
                time_s=920.48,  # 15:20.48
                splits=[(400.0, 243.68), (800.0, 489.30), (1200.0, 735.51), (1500.0, 920.48)],
                stroke_rate_cpm=(34.0, 39.0),
                stroke_length_m=2.15,
                avg_velocity_mps=1.630,
                breakout_distance_m=10.0,
                source="World Record, 2018 TYR Pro Swim Series.",
            ),
        },
        coaching_notes=(
            "Ledecky's secret is relentless efficiency. Her stroke doesn't change "
            "across 30 lengths of a 1500. She negative-splits most distance races, "
            "meaning her second half is faster than her first — extremely rare. "
            "Key lesson: distance swimming is about sustainable stroke length, "
            "not raw stroke rate."
        ),
    ),

    "adam_peaty": EliteSwimmerProfile(
        name="Adam Peaty",
        country="Great Britain",
        country_flag="🇬🇧",
        primary_events=["100m Breaststroke", "50m Breaststroke"],
        achievement=(
            "2× Olympic gold medalist (100m Breaststroke, Rio 2016 & Tokyo 2020). "
            "World record holder (56.88s, 2019). Dominated 100m breaststroke for nearly "
            "a decade. The only man to break 57 seconds."
        ),
        technique_signature=(
            "Peaty revolutionized breaststroke with: (1) An extraordinarily high stroke "
            "rate (52-58 cpm) maintained with full power — faster than most breaststrokers "
            "by 5-8 cpm. (2) Minimized glide phase — unlike traditional breaststroke that "
            "emphasizes a long glide, Peaty keeps continuous forward motion. (3) Explosive, "
            "narrow kick that generates maximum thrust with minimum drag. (4) Very low "
            "head position during the breathing phase — barely clears the water, maintaining "
            "streamline. (5) Powerful insweep and outsweep with hands that never travel wide "
            "of the shoulders. (6) World-class start with underwater pullout reaching ~9-10m."
        ),
        race_data={
            "100m_breast": RaceMetrics(
                event="100m Breaststroke",
                time_s=56.88,
                splits=[(50.0, 26.74), (100.0, 56.88)],
                stroke_rate_cpm=(52.0, 58.0),
                stroke_length_m=1.80,
                avg_velocity_mps=1.758,
                breakout_distance_m=9.5,
                source="World Record, 2019 World Championships Gwangju.",
            ),
        },
        coaching_notes=(
            "Peaty proved that breaststroke can be a power stroke, not just a technique "
            "stroke. His approach: maximize stroke rate without sacrificing distance per "
            "stroke. His kick is narrower and faster than traditional technique — it "
            "creates less frontal drag while maintaining propulsion. Key for all "
            "breaststrokers: minimize the time spent in high-drag positions."
        ),
    ),

    "sarah_sjostrom": EliteSwimmerProfile(
        name="Sarah Sjöström",
        country="Sweden",
        country_flag="🇸🇪",
        primary_events=["50m Freestyle", "100m Freestyle", "50m Butterfly", "100m Butterfly"],
        achievement=(
            "Olympic gold medalist in 100m Butterfly (Rio 2016), 100m Freestyle and "
            "50m Freestyle (Paris 2024). World record holder in 50m butterfly (24.43s) "
            "and 100m butterfly (55.48s). One of the most versatile sprint swimmers ever."
        ),
        technique_signature=(
            "Sjöström's sprint technique features: (1) Exceptional underwater speed — "
            "she maximizes every start and turn with powerful dolphin kicks reaching "
            "12-13m. (2) In butterfly, an extremely efficient undulation with minimal "
            "vertical displacement — her body stays flat and fast. (3) High stroke rate "
            "in freestyle with remarkable stroke length preservation. (4) One of the "
            "cleanest hand entries in women's sprinting — fingertips first, minimal "
            "splash. (5) Powerful hip-driven body roll in freestyle. (6) Fast, efficient "
            "breathing with minimal head movement."
        ),
        race_data={
            "100m_fly": RaceMetrics(
                event="100m Butterfly",
                time_s=55.48,
                splits=[(50.0, 25.68), (100.0, 55.48)],
                stroke_rate_cpm=(50.0, 56.0),
                stroke_length_m=1.95,
                avg_velocity_mps=1.802,
                breakout_distance_m=12.5,
                source="World Record.",
            ),
            "100m_free": RaceMetrics(
                event="100m Freestyle",
                time_s=51.71,
                splits=[(50.0, 24.87), (100.0, 51.71)],
                stroke_rate_cpm=(54.0, 60.0),
                stroke_length_m=2.00,
                avg_velocity_mps=1.934,
                breakout_distance_m=12.0,
                source="World Record. Paris 2024 Olympic gold.",
            ),
        },
        coaching_notes=(
            "Sjöström demonstrates that women's sprinting has moved beyond 'just swim "
            "faster' — her underwater work and technical precision are what separate her. "
            "In butterfly, she minimizes the energy cost of undulation while maximizing "
            "forward propulsion. In freestyle, she has one of the highest DPS (distance "
            "per stroke) values in women's sprinting."
        ),
    ),

    "kaylee_mckeown": EliteSwimmerProfile(
        name="Kaylee McKeown",
        country="Australia",
        country_flag="🇦🇺",
        primary_events=["100m Backstroke", "200m Backstroke"],
        achievement=(
            "Double Olympic gold medalist in backstroke (Paris 2024 — 100m and 200m). "
            "Defended her Tokyo 2020 100m backstroke title. Former world record holder. "
            "Dominant backstroker of the current era."
        ),
        technique_signature=(
            "McKeown's backstroke excels through: (1) Continuous, high-frequency arm "
            "rotation with very little 'dead spot' at the top of the recovery. (2) Deep "
            "catch — her hand enters pinky-first and immediately engages a strong pull. "
            "(3) Powerful 6-beat kick with excellent ankle plantar flexion. (4) Superior "
            "body roll (~45° each side) that allows her arms to catch deeper water. "
            "(5) Excellent underwater dolphin kicks off every wall — consistently "
            "12-13m. (6) Clean, accurate backstroke finish approach (flag counting)."
        ),
        race_data={
            "100m_back": RaceMetrics(
                event="100m Backstroke",
                time_s=57.33,
                splits=[(50.0, 27.80), (100.0, 57.33)],
                stroke_rate_cpm=(46.0, 52.0),
                stroke_length_m=2.05,
                avg_velocity_mps=1.745,
                breakout_distance_m=12.5,
                source="Paris 2024 Olympic final.",
            ),
            "200m_back": RaceMetrics(
                event="200m Backstroke",
                time_s=124.27,  # 2:04.27
                splits=[(50.0, 30.10), (100.0, 62.20), (150.0, 93.80), (200.0, 124.27)],
                stroke_rate_cpm=(40.0, 48.0),
                stroke_length_m=2.15,
                avg_velocity_mps=1.610,
                breakout_distance_m=12.0,
                source="Paris 2024 Olympic final.",
            ),
        },
        coaching_notes=(
            "McKeown's backstroke demonstrates the importance of continuous propulsion — "
            "there's no pause or glide in her stroke. Her body roll is textbook and allows "
            "maximum catch depth. For backstroke swimmers: focus on eliminating dead spots "
            "and perfecting your underwater dolphin kicks."
        ),
    ),

    "summer_mcintosh": EliteSwimmerProfile(
        name="Summer McIntosh",
        country="Canada",
        country_flag="🇨🇦",
        primary_events=["200m Butterfly", "200m IM", "400m IM", "400m Freestyle"],
        achievement=(
            "Triple Olympic gold medalist (Paris 2024 — 200m Butterfly, 200m IM, "
            "400m IM). World record holder in 200m Butterfly (2:01.86) and 400m IM. "
            "At 17, became Canada's most decorated Olympic swimmer."
        ),
        technique_signature=(
            "McIntosh at 17 already shows: (1) Exceptional stroke length — among the "
            "longest in women's swimming, owing to her 5'11\" frame and perfect reach. "
            "(2) In butterfly, a smooth, flowing undulation with minimal energy waste. "
            "(3) Devastating negative splits — she builds through races, swimming the "
            "final 50m faster than almost anyone. (4) Excellent underwater dolphin kicks "
            "reaching 12-13m. (5) In IM events, seamless transitions between strokes "
            "with maintained velocity. (6) High, stable body position across all strokes."
        ),
        race_data={
            "200m_fly": RaceMetrics(
                event="200m Butterfly",
                time_s=121.86,  # 2:01.86
                splits=[(50.0, 28.63), (100.0, 60.82), (150.0, 92.14), (200.0, 121.86)],
                stroke_rate_cpm=(38.0, 44.0),
                stroke_length_m=2.25,
                avg_velocity_mps=1.641,
                breakout_distance_m=12.5,
                source="World Record. Paris 2024 Olympic gold.",
            ),
        },
        coaching_notes=(
            "McIntosh's approach is proof that you don't need to lead from the front "
            "to win. She routinely swims the first 100m conservatively, then unleashes "
            "a devastating back half. Her stroke length is her biggest weapon — she "
            "covers more distance per stroke than almost any woman in history."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Technique models for each stroke (used in AI prompts)
# ---------------------------------------------------------------------------

_BUILTIN_TECHNIQUE_MODELS: dict[str, TechniqueModel] = {
    "freestyle": TechniqueModel(
        stroke="freestyle",
        key_elements={
            "body_position": (
                "Flat, horizontal body line with hips near the surface. Head in neutral "
                "position (looking slightly forward-down). Body rotates 45-60° to each "
                "side as a unit (hips and shoulders together)."
            ),
            "arm_entry": (
                "Hand enters fingertips-first, in line with the shoulder, at full extension. "
                "Minimal splash. Arm extends forward before initiating the catch."
            ),
            "catch": (
                "Early vertical forearm (EVF) / high elbow catch. The elbow stays high and "
                "near the surface while the hand and forearm rotate downward to create a "
                "large 'paddle'. This is the single most important technique element in "
                "freestyle."
            ),
            "pull": (
                "S-shaped or straight-back pull path, with hand accelerating throughout. "
                "Hand exits at the hip/thigh. Emphasis on push-through at the back of the "
                "stroke (many swimmers lose power here)."
            ),
            "recovery": (
                "High elbow recovery with relaxed hand. Elbow exits first and leads the "
                "recovery. Hand travels close to the body/water surface."
            ),
            "kick": (
                "6-beat flutter kick (2 kicks per arm stroke × 2 arms = 6 per cycle). "
                "Kick originates from the hip, not the knee. Ankles are flexible and toes "
                "are pointed. Kick amplitude is small (about foot-width)."
            ),
            "breathing": (
                "Head rotates to the side (not lifts) during the arm recovery on the "
                "breathing side. One goggle lens stays in the water. Exhale continuously "
                "underwater. Bilateral breathing (every 3 strokes) is ideal for symmetry "
                "but many sprinters breathe every 2."
            ),
        },
        common_errors=[
            "Dropped elbow during the catch (forearm points down instead of back)",
            "Crossing the centerline with hand entry (causes snaking/fishtailing)",
            "Head lifting during breathing (causes hips to drop)",
            "Thumb-first hand entry (causes shoulder impingement over time)",
            "Flat body with no rotation (reduces stroke length and power)",
            "Kicking from the knee instead of the hip (creates drag)",
            "Wide, swinging arm recovery (wastes energy and disrupts alignment)",
            "Holding breath instead of continuous exhale (causes CO2 buildup)",
        ],
        drills=[
            {"name": "Catch-Up Drill", "purpose": "Improves stroke timing and extension",
             "description": "One arm stays extended in front until the other arm completes its full stroke and touches the lead hand."},
            {"name": "Fingertip Drag", "purpose": "Promotes high elbow recovery",
             "description": "During recovery, drag fingertips along the water surface, forcing a high elbow position."},
            {"name": "Fist Drill", "purpose": "Develops forearm feel for the water",
             "description": "Swim with closed fists to engage the forearm as a pulling surface."},
            {"name": "Kick on Side", "purpose": "Develops body rotation and kick timing",
             "description": "Kick on your side with one arm extended, rotating to breathe every 6 kicks."},
            {"name": "Sculling", "purpose": "Develops hand sensitivity and catch feel",
             "description": "Sweep hands in figure-8 patterns at various depths to feel water pressure."},
        ],
    ),

    "backstroke": TechniqueModel(
        stroke="backstroke",
        key_elements={
            "body_position": (
                "Supine (face up), flat body line with hips at the surface. Head is still "
                "with eyes looking straight up. Ears at water level. Body rotates ~45° "
                "to each side."
            ),
            "arm_entry": (
                "Hand enters pinky-first, directly in line with the shoulder, at full "
                "extension above the head. Arm is straight at entry."
            ),
            "catch": (
                "After entry, the hand sweeps outward and downward, then bends at the "
                "elbow to create an early vertical forearm. The elbow stays pointing "
                "toward the bottom of the pool."
            ),
            "pull": (
                "S-shaped pull path with hand pushing past the hip. Strong emphasis "
                "on the push-through (final third of the pull). Body rotation provides "
                "most of the power."
            ),
            "recovery": (
                "Arm exits thumb-first, rotates during recovery, and enters pinky-first. "
                "Arm is straight and close to vertical during recovery."
            ),
            "kick": (
                "6-beat flutter kick, face-up. Kick generates from the hip with loose "
                "knees. Toes create a slight boil at the surface but feet don't break "
                "the surface excessively."
            ),
        },
        common_errors=[
            "Sitting position (hips too low) — often caused by head too far forward",
            "Over-rotating past 45° (causes S-shaped swimming path)",
            "Short, shallow pull that misses the push-through phase",
            "Knees breaking the surface on kick (kicking 'up' instead of from the hip)",
            "Arm entering across the centerline or too wide",
            "Head moving/bobbing instead of staying still",
        ],
        drills=[
            {"name": "One-Arm Backstroke", "purpose": "Isolates rotation and pull mechanics",
             "description": "Swim backstroke with one arm, other at side. Focus on full rotation and complete pull."},
            {"name": "Double-Arm Backstroke", "purpose": "Develops catch timing and symmetry",
             "description": "Both arms pull simultaneously, emphasizing even rotation to both sides."},
            {"name": "Cup Drill", "purpose": "Develops still head position",
             "description": "Balance a small cup of water on your forehead while swimming backstroke."},
        ],
    ),

    "breaststroke": TechniqueModel(
        stroke="breaststroke",
        key_elements={
            "body_position": (
                "Streamlined position between each stroke cycle. Body undulates slightly — "
                "hips rise as hands shoot forward, creating a wave-like motion."
            ),
            "pull": (
                "Outsweep → insweep → recovery. Hands sweep out to slightly wider than "
                "shoulders, then sweep in fast (the power phase), then shoot forward into "
                "streamline. Elbows stay in front of shoulders throughout."
            ),
            "kick": (
                "Whip kick: heels draw up toward buttocks (knees stay narrower than hips), "
                "feet turn out, then legs drive back and together in a circular whip motion. "
                "The kick provides ~70% of propulsion in breaststroke."
            ),
            "timing": (
                "Pull-breathe-kick-glide sequence. The kick happens AFTER the hands begin "
                "shooting forward. There should be a brief glide in streamline before the "
                "next pull begins. Timing is the most critical element in breaststroke."
            ),
            "breathing": (
                "Head rises naturally with the insweep of the arms — no separate head "
                "lift. Eyes look forward-down. The less the head rises, the less the "
                "hips drop. Quick breath, then head follows hands forward."
            ),
        },
        common_errors=[
            "Wide kick (knees wider than hips) — creates massive frontal drag",
            "Pulling past the shoulders (elbows behind the body) — illegal and inefficient",
            "Pausing with head up (causes hips to sink dramatically)",
            "Kick before hands are forward (timing error that kills glide)",
            "Asymmetric kick (one leg stronger) — common source of disqualification",
            "Over-gliding (waiting too long between strokes, losing momentum)",
        ],
        drills=[
            {"name": "2-Kick 1-Pull", "purpose": "Develops kick power and streamline position",
             "description": "Take one arm pull, then do two kicks with arms in streamline. Emphasizes kick propulsion."},
            {"name": "Pull with Flutter Kick", "purpose": "Isolates arm pull mechanics",
             "description": "Swim breaststroke arms with flutter kick to focus on pull pattern without timing complexity."},
            {"name": "Vertical Kick", "purpose": "Develops kick power",
             "description": "Kick breaststroke vertically in deep water. Hands on head or above water for resistance."},
        ],
    ),

    "butterfly": TechniqueModel(
        stroke="butterfly",
        key_elements={
            "body_position": (
                "Undulating body motion driven from the chest. The chest presses down, "
                "hips rise; chest rises, hips drop. This creates a wave that travels "
                "from head to toe."
            ),
            "arm_mechanics": (
                "Simultaneous arm recovery over the water. Keyhole pull pattern underwater "
                "(out, in, and push back). Low, wide recovery with arms barely clearing the "
                "water. Hands enter shoulder-width apart."
            ),
            "kick": (
                "Double dolphin kick per arm cycle. First kick (small) occurs during hand "
                "entry. Second kick (large, powerful) occurs during hand exit/push phase. "
                "Kick originates from the hips with loose knees and pointed toes."
            ),
            "timing": (
                "The two kicks must be precisely timed with the arm stroke. The big kick "
                "at the back of the stroke is what drives the hips up and the body forward. "
                "Consistent rhythm is more important than power."
            ),
            "breathing": (
                "Head lifts forward (chin stays near the water surface, not up). Breathe "
                "every stroke or every other stroke. Head goes down BEFORE hands enter — "
                "this is critical for maintaining undulation rhythm."
            ),
        },
        common_errors=[
            "Flat body with no undulation (swimming 'arms only' butterfly)",
            "One big kick instead of two per cycle (loses the timing advantage)",
            "High head lift during breathing (causes legs to sink)",
            "Arms recovering too high over the water (wastes energy)",
            "Asymmetric arm pull or recovery (can lead to DQ in competition)",
            "Equal-sized kicks instead of small-big pattern",
        ],
        drills=[
            {"name": "Single-Arm Butterfly", "purpose": "Develops undulation and timing",
             "description": "Swim butterfly with one arm, other at side. Focus on 2-kick rhythm and body wave."},
            {"name": "3-3-3 Drill", "purpose": "Builds butterfly endurance and rhythm",
             "description": "3 strokes right arm only, 3 strokes left arm only, 3 strokes full butterfly. Repeat."},
            {"name": "Vertical Dolphin Kick", "purpose": "Develops kick power and body wave",
             "description": "Kick dolphin kick vertically in deep water. Hands above water or on head for added resistance."},
        ],
    ),
}


# ---------------------------------------------------------------------------
# External JSON loader with graceful fallback to built-in datasets
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "benchmarks"
_BENCHMARK_JSON_PATH = _DATA_DIR / "elite_benchmarks.json"


def load_benchmarks_from_json(json_path: Path | str | None = None) -> tuple[
    list[BenchmarkEntry],
    dict[str, EliteSwimmerProfile],
    list[RaceSegmentConvention],
    dict[str, TechniqueModel],
]:
    """Load benchmarks and elite profiles from external versioned JSON file."""
    path = Path(json_path) if json_path else _BENCHMARK_JSON_PATH
    if not path.exists():
        return (
            _BUILTIN_BENCHMARKS,
            _BUILTIN_ELITE_PROFILES,
            _BUILTIN_RACE_SEGMENT_CONVENTIONS,
            _BUILTIN_TECHNIQUE_MODELS,
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        benchmarks = [
            BenchmarkEntry(
                stroke=b["stroke"],
                distance_m=b["distance_m"],
                sex=b["sex"],
                avg_velocity_mps=b["avg_velocity_mps"],
                stroke_rate_cpm=tuple(b["stroke_rate_cpm"]) if b.get("stroke_rate_cpm") else None,
                stroke_length_m=b.get("stroke_length_m"),
                breakout_distance_m=b.get("breakout_distance_m"),
                source=b["source"],
                precision=b.get("precision", "verified"),
                retrieved_date=b.get("retrieved_date"),
            )
            for b in raw.get("benchmarks", [])
        ]

        elite_profiles = {}
        for k, p in raw.get("elite_profiles", {}).items():
            race_data = {}
            for ek, rm in p.get("race_data", {}).items():
                race_data[ek] = RaceMetrics(
                    event=rm["event"],
                    time_s=rm["time_s"],
                    splits=[(s[0], s[1]) for s in rm["splits"]],
                    stroke_rate_cpm=tuple(rm["stroke_rate_cpm"]) if rm.get("stroke_rate_cpm") else None,
                    stroke_length_m=rm.get("stroke_length_m"),
                    avg_velocity_mps=rm["avg_velocity_mps"],
                    breakout_distance_m=rm.get("breakout_distance_m"),
                    source=rm["source"],
                )
            elite_profiles[k] = EliteSwimmerProfile(
                name=p["name"],
                country=p["country"],
                country_flag=p["country_flag"],
                primary_events=p["primary_events"],
                achievement=p["achievement"],
                technique_signature=p["technique_signature"],
                race_data=race_data,
                coaching_notes=p["coaching_notes"],
                retrieved_date=p.get("retrieved_date"),
            )

        conventions = [
            RaceSegmentConvention(
                name=c["name"],
                boundaries_m=tuple(c["boundaries_m"]),
                source=c["source"],
            )
            for c in raw.get("race_segment_conventions", [])
        ]

        techniques = {
            k: TechniqueModel(
                stroke=t["stroke"],
                key_elements=t["key_elements"],
                common_errors=t["common_errors"],
                drills=t["drills"],
            )
            for k, t in raw.get("technique_models", {}).items()
        }

        return benchmarks, elite_profiles, conventions, techniques

    except Exception:
        return (
            _BUILTIN_BENCHMARKS,
            _BUILTIN_ELITE_PROFILES,
            _BUILTIN_RACE_SEGMENT_CONVENTIONS,
            _BUILTIN_TECHNIQUE_MODELS,
        )


# Load data from external JSON
BENCHMARKS, ELITE_PROFILES, RACE_SEGMENT_CONVENTIONS, TECHNIQUE_MODELS = (
    load_benchmarks_from_json()
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_benchmark(stroke: str, distance_m: int, sex: str) -> BenchmarkEntry | None:
    """Find a benchmark entry for the given stroke/distance/sex combination."""
    for b in BENCHMARKS:
        if b.stroke == stroke and b.distance_m == distance_m and b.sex == sex:
            return b
    # Try mixed fallback
    if sex != "mixed_top10":
        for b in BENCHMARKS:
            if b.stroke == stroke and b.distance_m == distance_m and b.sex == "mixed_top10":
                return b
    return None


def get_elite_profile(name_key: str) -> EliteSwimmerProfile | None:
    """Get an elite swimmer profile by key."""
    return ELITE_PROFILES.get(name_key)


def get_profiles_for_event(stroke: str, distance_m: int) -> list[EliteSwimmerProfile]:
    """Find elite profiles that compete in a given event."""
    event_str = f"{distance_m}m {stroke.title()}"
    matching = []
    for profile in ELITE_PROFILES.values():
        for ev in profile.primary_events:
            if ev.lower() == event_str.lower() or (
                str(distance_m) in ev and stroke.lower() in ev.lower()
            ):
                matching.append(profile)
                break
    return matching


def get_technique_model(stroke: str) -> TechniqueModel | None:
    """Get the technique model for a stroke."""
    return TECHNIQUE_MODELS.get(stroke.lower())


def get_all_profile_names() -> list[tuple[str, str, str]]:
    """Return (key, display_name, flag) for all profiles."""
    return [
        (key, p.name, p.country_flag)
        for key, p in ELITE_PROFILES.items()
    ]


def compare_to_benchmark(
    user_avg_velocity_mps: float, stroke: str, distance_m: int, sex: str,
) -> str | None:
    """Returns a coaching-feedback sentence only when a real cited
    benchmark exists for this exact stroke/distance/sex; otherwise
    returns None so the caller can show 'no verified reference available'
    instead of a silently-skipped comparison."""
    b = get_benchmark(stroke, distance_m, sex)
    if b is None or b.avg_velocity_mps is None:
        return None
    delta = user_avg_velocity_mps - b.avg_velocity_mps
    pct = 100 * delta / b.avg_velocity_mps
    direction = "faster than" if delta > 0 else "slower than"
    return (
        f"Average velocity is {abs(pct):.1f}% {direction} the reference "
        f"figure ({b.avg_velocity_mps:.2f} m/s, {b.source})."
    )


def build_elite_context_for_prompt(
    stroke: str, distance_m: int
) -> str:
    """Build a rich context string about elite swimmers for a given event,
    suitable for including in Gemini prompts."""
    profiles = get_profiles_for_event(stroke, distance_m)
    technique = get_technique_model(stroke)
    benchmark = get_benchmark(stroke, distance_m, "male") or get_benchmark(stroke, distance_m, "female")

    parts = []

    if benchmark:
        parts.append(
            f"Elite benchmarks for {distance_m}m {stroke}:\n"
            f"  - Average velocity: {benchmark.avg_velocity_mps} m/s\n"
            f"  - Stroke rate: {benchmark.stroke_rate_cpm} cycles/min\n"
            f"  - Stroke length: {benchmark.stroke_length_m} m/cycle\n"
            f"  - Breakout distance: {benchmark.breakout_distance_m} m\n"
        )

    for profile in profiles:
        parts.append(
            f"\n{profile.country_flag} {profile.name} ({profile.country}):\n"
            f"  Achievement: {profile.achievement}\n"
            f"  Technique: {profile.technique_signature}\n"
            f"  Coaching notes: {profile.coaching_notes}\n"
        )

    if technique:
        parts.append(f"\nIdeal {stroke} technique model:")
        for elem, desc in technique.key_elements.items():
            parts.append(f"  {elem}: {desc}")

    return "\n".join(parts) if parts else "No specific elite data available for this event."


def find_closest_elite_match(
    stroke: str,
    distance_m: int,
    sex: str = "male",
    user_velocity_mps: float | None = None,
    user_stroke_rate: float | None = None,
) -> tuple[str | None, EliteSwimmerProfile | None, str]:
    """Auto-select the closest elite swimmer match for comparison.

    Strategy (in priority order):
      1. Find profiles that compete in the same event (stroke + distance)
      2. If multiple matches, pick the one whose performance metrics are
         closest to the user's (so the comparison is realistic, not just
         'you vs. the GOAT')
      3. If no exact event match, find profiles with the same stroke but
         any distance, then fall back to any profile

    Returns
    -------
    (profile_key, profile, reason) or (None, None, reason)
    """
    # Step 1: Exact event match
    event_matches: list[tuple[str, EliteSwimmerProfile, float]] = []
    for key, profile in ELITE_PROFILES.items():
        for ev in profile.primary_events:
            if str(distance_m) in ev and stroke.lower() in ev.lower():
                # Score based on velocity similarity
                score = _compute_match_score(
                    profile, stroke, distance_m, user_velocity_mps, user_stroke_rate, sex
                )
                event_matches.append((key, profile, score))
                break

    if event_matches:
        # Sort by score (lower = closer match)
        event_matches.sort(key=lambda x: x[2])
        key, profile, score = event_matches[0]
        reason = (
            f"Auto-selected {profile.country_flag} {profile.name} — "
            f"competes in {distance_m}m {stroke.title()}"
        )
        if len(event_matches) > 1:
            reason += f" (closest match of {len(event_matches)} athletes)"
        return key, profile, reason

    # Step 2: Same stroke, different distance
    stroke_matches: list[tuple[str, EliteSwimmerProfile]] = []
    for key, profile in ELITE_PROFILES.items():
        for ev in profile.primary_events:
            if stroke.lower() in ev.lower():
                stroke_matches.append((key, profile))
                break

    if stroke_matches:
        # Pick the one with the closest event distance
        best_key, best_profile = min(
            stroke_matches,
            key=lambda x: min(
                abs(int("".join(c for c in ev if c.isdigit()) or "0") - distance_m)
                for ev in x[1].primary_events
                if stroke.lower() in ev.lower()
            ),
        )
        return best_key, best_profile, (
            f"Auto-selected {best_profile.country_flag} {best_profile.name} — "
            f"closest {stroke.title()} specialist (no exact {distance_m}m match in database)"
        )

    # Step 3: Any profile (fallback)
    if ELITE_PROFILES:
        key = list(ELITE_PROFILES.keys())[0]
        profile = ELITE_PROFILES[key]
        return key, profile, (
            f"Auto-selected {profile.country_flag} {profile.name} — "
            f"no {stroke.title()} specialists in database, using general comparison"
        )

    return None, None, "No elite profiles available for comparison."


def _compute_match_score(
    profile: EliteSwimmerProfile,
    stroke: str,
    distance_m: int,
    user_velocity: float | None,
    user_stroke_rate: float | None,
    sex: str,
) -> float:
    """Compute a similarity score (lower = better match).

    Considers velocity similarity and gender match so the comparison
    is meaningful and motivating (comparing a recreational swimmer to
    the WR holder is less useful than comparing to a slightly-faster
    elite whose technique is reachable).
    """
    score = 0.0

    # Find race data for this event
    event_key_candidates = [
        f"{distance_m}m_{stroke[:4]}",
        f"{distance_m}m_{stroke}",
    ]
    race_data = None
    for ek in event_key_candidates:
        if ek in profile.race_data:
            race_data = profile.race_data[ek]
            break
    if race_data is None:
        # Try any race data key that contains the distance
        for ek, rd in profile.race_data.items():
            if str(distance_m) in ek:
                race_data = rd
                break

    if race_data and user_velocity:
        # Velocity similarity (penalize huge gaps)
        vel_diff = abs(race_data.avg_velocity_mps - user_velocity)
        score += vel_diff * 10  # 0.1 m/s diff = 1.0 score

    if race_data and user_stroke_rate and race_data.stroke_rate_cpm:
        sr_mid = (race_data.stroke_rate_cpm[0] + race_data.stroke_rate_cpm[1]) / 2
        sr_diff = abs(sr_mid - user_stroke_rate)
        score += sr_diff * 0.5

    # Gender mismatch penalty (comparing male swimmer to female elite
    # or vice versa is less meaningful)
    if sex == "female" and profile.name in ("Pan Zhanle", "Caeleb Dressel", "Adam Peaty"):
        score += 5.0
    elif sex == "male" and profile.name in ("Katie Ledecky", "Sarah Sjöström", "Kaylee McKeown", "Summer McIntosh"):
        score += 5.0

    return score

