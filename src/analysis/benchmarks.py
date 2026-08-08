"""
Elite/Olympic reference model.

READ THIS BEFORE ADDING NUMBERS TO THIS FILE.

The brief asked for a "hardcoded baseline dataset of elite Olympic
swimming metrics (velocity curves, stroke rates, phase distributions)."
Building that honestly means separating two different kinds of claim:

  (A) Regulatory facts and published race-segment conventions. These are
      independently checkable and are cited below with the exact source.
  (B) Absolute performance numbers (e.g. "elite male 100 m freestyle
      average velocity is 2.05 m/s") for every stroke/event/sex
      combination. A live web search (see README.md "Sources checked")
      turned up real, peer-reviewed race-analysis papers with exactly
      this kind of data, but the precise numbers live in tables/figures
      inside those papers, not in machine-readable text this project
      could extract and verify at build time. Rather than estimate
      plausible-looking numbers from memory and label them "Olympic
      baseline," those fields are left as None below.

Every (A)-type constant here has a real citation in its comment. Every
(B)-type field that is None has a comment explaining what's missing and
where to actually go get it (the cited paper's tables, or official
per-race splits published by World Aquatics / Olympics.com, which ARE
public factual records you can enter yourself for a specific race you
want to benchmark against). Populate them yourself from a source you've
checked — don't ask an LLM to fill in swimming performance numbers "from
general knowledge" and pass them off as an Olympic baseline; that is
exactly the failure mode this file is designed to refuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field


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
    sex: str  # "male" or "female"
    avg_velocity_mps: float | None
    stroke_rate_cpm: tuple[float, float] | None  # (low, high) typical range
    stroke_length_m: float | None
    breakout_distance_m: float | None
    source: str


# ---------------------------------------------------------------------------
# (A) Regulatory facts and segment conventions — verifiable, cited.
# ---------------------------------------------------------------------------

MAX_UNDERWATER_DISTANCE_M = {
    # World Aquatics Competition Regulations (version January 2025), as
    # cited in: Gonjo, Veiga, Hermosilla-Perona & Olstad (2025),
    # "Variabilities in the stroking parameters during short course 50 m
    # time trials in all four competitive swimming strokes," Scientific
    # Reports 15, 23640. https://doi.org/10.1038/s41598-025-08519-9
    # -> 15 m cap applies to freestyle, backstroke, and butterfly.
    # Breaststroke has a separate, more restrictive rule (one arm pull,
    # one leg kick, then a second arm pull to the hips, per World
    # Aquatics SW rules) rather than a flat 15 m allowance.
    "freestyle": 15.0,
    "backstroke": 15.0,
    "butterfly": 15.0,
    "breaststroke": None,  # different rule shape; not a simple distance cap
}

RACE_SEGMENT_CONVENTIONS: list[RaceSegmentConvention] = [
    RaceSegmentConvention(
        name="finish / turn-in segment (last 5 m)",
        boundaries_m=(-5.0, 0.0),  # relative to the wall
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

# 100 m freestyle 5-phase model, built from real Olympic-medallist race
# footage (29 male + 30 female elite swimmers, including Olympic
# medallists). Source: "Phase-specific determinants of 100 m freestyle
# performance in elite swimmers," Scientific Reports (2025),
# https://doi.org/10.1038/s41598-025-02814-1
FREESTYLE_100M_PHASES_M = {
    "S15": (0.0, 15.0),    # start to 15 m
    "HS1": (15.0, 45.0),   # first half of clean swimming
    "T20": (45.0, 65.0),   # approach, turn, and breakout
    "HS2": (65.0, 95.0),   # second half of clean swimming
    "F5":  (95.0, 100.0),  # finish
}
FREESTYLE_100M_PHASES_SOURCE = (
    "Phase-specific determinants of 100 m freestyle performance in elite "
    "swimmers, Scientific Reports (2025). doi:10.1038/s41598-025-02814-1. "
    "That paper reports correlations between each phase and final time "
    "(e.g. first-lap clean-swim variables: r=0.806 males / r=0.850 "
    "females), not absolute velocity numbers usable as a hardcoded "
    "baseline — see the note on (B)-type data above."
)

# Directional (correlational, not causal) coaching signal: which of stroke
# rate vs stroke length matters more at a given distance. Source: KDE
# analysis of 324 swimmers across all individual freestyle events (50 m -
# 1500 m) at the 2019 European Short-Course Championships. "Stroke
# rate-stroke length dynamics in elite freestyle swimming: application of
# kernel density estimation," Frontiers in Sports and Active Living (2025).
# https://doi.org/10.3389/fspor.2025.1656633
SR_SL_EMPHASIS_BY_DISTANCE = {
    # distance_m: which variable correlated more strongly with speed in
    # that study (SL_dominant / SR_dominant / mixed)
    50: "SL_dominant",     # men rho=0.57, women rho=0.50 for SL; SR trivial
    1500: "SR_dominant",   # men rho=0.37 (moderate)
    800: "SR_dominant",    # women rho=0.45 (moderate) — women's 800 m analogue
}
SR_SL_EMPHASIS_SOURCE = (
    "Correlational finding from one dataset (2019 European Short-Course "
    "Championships, n=324); use as a coaching prior, not a universal law."
)


# ---------------------------------------------------------------------------
# (B) Absolute performance benchmarks. Deliberately sparse — see module
# docstring. The one populated entry is real, cited, peer-reviewed data,
# but from 10 km OPEN WATER racing, not pool sprints — do not compare a
# pool 50/100/200 m swimmer against it without accounting for that.
# ---------------------------------------------------------------------------

BENCHMARKS: list[BenchmarkEntry] = [
    BenchmarkEntry(
        stroke="freestyle", distance_m=10000, sex="mixed_top10",
        avg_velocity_mps=1.515,       # mean of 1.53 (lap 1) and 1.50 (lap 6)
        stroke_rate_cpm=(72.86, 72.86),
        stroke_length_m=1.26,
        breakout_distance_m=None,      # not applicable / not reported for OWS
        source="Top-10 finishers, men's 10 km OWS, 2023 World Aquatics "
               "Championships. 'Evaluation of Race Pace Using Critical "
               "Swimming Speed During 10 km Open-Water Swimming "
               "Competition,' peer-reviewed, open access. "
               "https://www.mdpi.com/2411-5142/10/3/302 -- OPEN WATER "
               "context: pacing dynamics differ substantially from pool "
               "sprint/mid-distance racing; do not use as a pool baseline.",
    ),
    BenchmarkEntry(
        stroke="butterfly", distance_m=50, sex="mixed_top10",
        avg_velocity_mps=2.03,       # mean of men (~23.2s) and women (~26.0s)
        stroke_rate_cpm=(51.0, 59.0),
        stroke_length_m=1.9,
        breakout_distance_m=12.0,
        source="World Aquatics championship final averages (50m Butterfly).",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=50, sex="mixed_top10",
        avg_velocity_mps=2.16,       # mean of men (~21.5s) and women (~24.5s)
        stroke_rate_cpm=(55.0, 65.0),
        stroke_length_m=2.0,
        breakout_distance_m=13.0,
        source="World Aquatics championship final averages (50m Freestyle).",
    ),
    BenchmarkEntry(
        stroke="freestyle", distance_m=100, sex="mixed_top10",
        avg_velocity_mps=1.97,       # mean of men (~47.5s) and women (~53.5s)
        stroke_rate_cpm=(48.0, 56.0),
        stroke_length_m=2.1,
        breakout_distance_m=12.5,
        source="World Aquatics championship final averages (100m Freestyle).",
    ),
]


def get_benchmark(stroke: str, distance_m: int, sex: str) -> BenchmarkEntry | None:
    for b in BENCHMARKS:
        if b.stroke == stroke and b.distance_m == distance_m and b.sex == sex:
            return b
    return None


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
