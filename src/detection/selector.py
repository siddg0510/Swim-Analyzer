"""
Module B: Target Acquisition

Manages a single `TARGET_ID` that tells the rest of the pipeline which track to
follow.  Two selection methods are supported:

  1. Manual / click-based: the user clicks a bounding box in the GUI's
     SwimmerSelectionDialog and we resolve that click to the closest track ID.
  2. Lane-based: the user specifies "Lane 4" and we pick the track whose
     horizontal centre falls in that lane's x-band.

The `TargetSelector` object is a plain dataclass — not a global singleton —
so the pipeline can instantiate one per analysis run without shared-state bugs.
The GUI hands the resolved `target_id` into `AnalysisConfig`; the pipeline
then passes it to `AntiSplashTracker` for the frame loop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .detector import Track

logger = logging.getLogger(__name__)


@dataclass
class TargetSelector:
    """
    Maintains the selected track ID and provides helper methods to resolve it
    from a user interaction (click or lane number).
    """
    target_id: int | None = None

    # ------------------------------------------------------------------
    # Selection methods
    # ------------------------------------------------------------------

    def select_by_click(
        self,
        click_x: float,
        click_y: float,
        tracks: list[Track],
    ) -> int | None:
        """
        Find the track whose bounding box contains (click_x, click_y), or the
        nearest one if the click missed all boxes.

        Returns the resolved ID and also stores it in self.target_id.
        """
        # First pass: exact containment
        for t in tracks:
            x, y, w, h = t.bbox
            if x <= click_x <= x + w and y <= click_y <= y + h:
                self.target_id = t.id
                logger.info("Target selected by click: id=%d", t.id)
                return t.id

        # Second pass: nearest centroid (fallback for imprecise clicks)
        if tracks:
            best = min(
                tracks,
                key=lambda t: (
                    (t.bbox[0] + t.bbox[2] / 2 - click_x) ** 2
                    + (t.bbox[1] + t.bbox[3] / 2 - click_y) ** 2
                ),
            )
            self.target_id = best.id
            logger.info(
                "Target selected by nearest centroid: id=%d (no box contained click)", best.id
            )
            return best.id

        logger.warning("select_by_click called with no tracks available.")
        return None

    def select_by_lane(
        self,
        lane_number: int,
        tracks: list[Track],
        frame_width: int,
        num_lanes: int = 8,
    ) -> int | None:
        """
        Select the track whose horizontal centre falls closest to the expected
        x-position of `lane_number` (1-indexed, left to right).

        The lane width is assumed to be uniform across the frame.  For cameras
        with a perspective effect this is an approximation, but it is good
        enough to seed the EKF which then tracks the actual swimmer.
        """
        if not tracks:
            return None

        lane_width_px = frame_width / num_lanes
        # Centre of requested lane in pixel x
        target_x = (lane_number - 0.5) * lane_width_px

        best = min(
            tracks,
            key=lambda t: abs(t.bbox[0] + t.bbox[2] / 2 - target_x),
        )
        self.target_id = best.id
        logger.info(
            "Target selected by lane %d: id=%d (track cx=%.1f, expected x=%.1f)",
            lane_number, best.id, best.bbox[0] + best.bbox[2] / 2, target_x,
        )
        return best.id

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def get_track(self, tracks: list[Track]) -> Track | None:
        """Return the Track object for the current target_id, or None."""
        if self.target_id is None:
            return None
        for t in tracks:
            if t.id == self.target_id:
                return t
        return None

    def is_locked(self) -> bool:
        return self.target_id is not None

    def reset(self) -> None:
        self.target_id = None
        logger.info("TargetSelector reset.")
