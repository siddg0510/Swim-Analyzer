"""
Central configuration and path resolution.

Every path that needs to work both when run as `python main.py` and when
frozen into a PyInstaller onefile/onedir build goes through resource_path()
and user_data_dir(). Nothing else in the codebase should build a path to a
bundled asset by hand.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path


def resource_path(*parts: str) -> Path:
    """
    Resolve a path to a read-only bundled asset (models, icons, reference
    data). Works in three situations:
      - normal `python main.py` execution
      - PyInstaller --onefile  (assets unpacked to sys._MEIPASS at runtime)
      - PyInstaller --onedir   (assets sit next to the executable)
    """
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).resolve().parent.parent
    return base.joinpath(*parts)


def user_data_dir() -> Path:
    """
    Writable directory for logs, temp audio extraction, and CSV exports.
    Never write into resource_path() — it may be inside a read-only,
    signed app bundle on macOS or an unpacked temp dir on Windows.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    d = base / "SwimRaceAnalyzer"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# MediaPipe Pose Landmarker model.
#
# Current MediaPipe (0.10.x, verified against the installed package while
# building this project) removed the old bundled `mp.solutions.pose` API.
# Pose estimation now goes through the Tasks API, which requires an explicit
# .task model file — MediaPipe does not ship it inside the pip package.
#
# This file is NOT included in this project's download. You must fetch it
# once (requires internet, one time only, at setup — not at field/runtime
# use) from Google's official model bundle and place it at the path below.
# See models/DOWNLOAD_MODEL.md for the exact command and a fallback plan
# if that URL ever moves.
# ---------------------------------------------------------------------------
POSE_MODEL_PATH = resource_path("models", "pose_landmarker.task")

# ---------------------------------------------------------------------------
# Pool / race segment scheme.
#
# Marker distances match the ones specified for this project. The 15 m and
# 35 m marks, and the "5 m before the wall" finish/turn-in convention, are
# not arbitrary — they match the segment boundaries used in published race
# -analysis methodology (World Aquatics caps underwater swimming at 15 m
# after the start/turn for every stroke except breaststroke; 15-35 m is the
# standard long-course "clean swimming" measurement window). See README.md
# "Sources" section for the papers this was checked against.
# ---------------------------------------------------------------------------
SPLIT_MARKERS_M = [5, 10, 15, 25, 35, 50]
FINISH_ZONE_M = 5      # final-5m micro-split / optical-flow finish hand-off
CLOSING_ZONE_M = 15    # final-15m micro-split (project-specific, not a
                       # standard literature segment — implemented as asked,
                       # not claimed as a citable convention)
MAX_LEGAL_UNDERWATER_M = 15  # World Aquatics rule, all strokes but breaststroke

STROKES = ["freestyle", "backstroke", "breaststroke", "butterfly"]

# Physical/video sanity bounds used by the self-correction pass. These are
# generous outlier gates, not physiological constants — deliberately wide
# so a real fast swimmer is never "corrected" into a slower, wrong number.
MAX_PLAUSIBLE_SPEED_MPS = 2.6      # fastest human swim speeds sit under this
MAX_PLAUSIBLE_ACCEL_MPS2 = 6.0     # generous bound for a start/wall push-off

DEFAULT_VIDEO_FPS_FALLBACK = 30.0

# ---------------------------------------------------------------------------
# AI analysis (Gemini) — optional, the offline pipeline still works without it
# ---------------------------------------------------------------------------
EVENTS = [
    "50m", "100m", "200m", "400m", "800m", "1500m",
]
POOL_TYPES = ["long course (50m)", "short course (25m)", "short course yards (25y)"]
GENDERS = ["male", "female"]

# Maximum video duration to upload for AI analysis (seconds).
# Short clips are both cheaper and produce better results than full-length
# race video (Gemini focuses on visible technique rather than processing
# minutes of footage).
AI_MAX_CLIP_DURATION_S = 60

CAP_COLOR_PRESETS = [
    "red", "orange", "yellow", "green", "neon green", "olive", "dark green", "cyan", "blue",
    "navy", "purple", "pink", "white", "black", "silver", "gray", "Custom hex…",
]
