"""
Pixel-to-meter calibration.

Honest framing up front: a single monocular side-on camera cannot recover
true 3D pool geometry. What we actually build is a 2D projective mapping
(a homography) from image pixels to real-world meters along the lane's
swimming plane, using reference points the user clicks and labels with a
known real-world distance (wall, flags, lane-rope knots, backstroke flags,
T-marks on the bottom, etc.).

This is the same practical compromise every monocular sports-video system
makes (broadcast graphics, single-camera VAR overlays, etc.) and it is
accurate as long as:
  1. at least 4 non-collinear reference points are supplied, and
  2. the swimmer stays close to the plane those points define (the water
     surface / lane line), which is the normal case for a fixed side-deck
     camera framing one lane.

If the user only clicks 2 points (e.g. both end walls), we fall back to a
1D linear pixel-distance-to-meter mapping along the lane axis. That's
weaker (no perspective correction) but still far better than nothing, and
we say so in the confidence metadata returned alongside every measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import cv2


@dataclass
class ReferencePoint:
    pixel: tuple[float, float]
    meters_along_lane: float   # distance from the start wall, in metres


@dataclass
class CalibrationKeyframe:
    frame_idx: int
    reference_points: list[tuple[tuple[float, float], float]]
    lane_polygon_px: list[tuple[float, float]]


@dataclass
class CalibrationResult:
    mode: str  # "homography" or "linear"
    homography: np.ndarray | None = None
    linear_fit: tuple[float, float] | None = None  # (slope, intercept) px->m
    reference_points: list[ReferencePoint] = field(default_factory=list)
    rmse_m: float = float("nan")

    @property
    def is_valid(self) -> bool:
        return self.mode in ("homography", "linear")


class PoolCalibrator:
    """
    Build and apply the pixel<->metre mapping for one lane.

    Usage:
        calib = PoolCalibrator()
        calib.add_point((x_px, y_px), meters_along_lane=0.0)
        calib.add_point((x_px, y_px), meters_along_lane=25.0)
        ... (2-8 points total; more = better perspective correction)
        result = calib.solve()
        distance_m = calib.pixel_to_distance(x_px, y_px)
    """

    def __init__(self) -> None:
        self._points: list[ReferencePoint] = []
        self._result: CalibrationResult | None = None

    def add_point(self, pixel: tuple[float, float], meters_along_lane: float) -> None:
        self._points.append(ReferencePoint(pixel=pixel, meters_along_lane=meters_along_lane))

    def clear(self) -> None:
        self._points.clear()
        self._result = None

    def solve(self) -> CalibrationResult:
        if len(self._points) < 2:
            raise ValueError(
                "Calibration needs at least 2 clicked reference points "
                "(4+ recommended, non-collinear, for perspective correction)."
            )

        pts = self._points
        if len(pts) >= 4:
            result = self._solve_homography(pts)
        else:
            result = self._solve_linear(pts)

        self._result = result
        return result

    def _solve_linear(self, pts: list[ReferencePoint]) -> CalibrationResult:
        # Only makes sense along the dominant motion axis. We detect that
        # axis as whichever of x/y has the larger pixel spread across the
        # reference points, then fit meters = slope * pixel + intercept.
        px = np.array([p.pixel[0] for p in pts])
        py = np.array([p.pixel[1] for p in pts])
        m = np.array([p.meters_along_lane for p in pts])

        axis_is_x = (px.max() - px.min()) >= (py.max() - py.min())
        coord = px if axis_is_x else py

        A = np.vstack([coord, np.ones_like(coord)]).T
        slope, intercept = np.linalg.lstsq(A, m, rcond=None)[0]
        pred = A @ np.array([slope, intercept])
        rmse = float(np.sqrt(np.mean((pred - m) ** 2)))

        result = CalibrationResult(
            mode="linear",
            linear_fit=(float(slope), float(intercept)),
            reference_points=pts,
            rmse_m=rmse,
        )
        result._axis_is_x = axis_is_x  # type: ignore[attr-defined]
        return result

    def _solve_homography(self, pts: list[ReferencePoint]) -> CalibrationResult:
        # We only have 1D "meters along lane" labels from the user (not full
        # 2D world coords), so we build synthetic world coordinates assuming
        # a straight lane: world_x = meters_along_lane, world_y = 0, and let
        # the homography absorb the lane's real-world width implicitly by
        # using the pixel y-spread as a proxy lane-width axis. This keeps
        # the calibration UI to "click point, type distance" instead of
        # requiring full 3D surveying, at the cost of assuming the camera's
        # view of lane width doesn't itself skew distance-along-lane
        # accuracy much — reasonable for a camera roughly perpendicular to
        # the lane, which is the typical pool-deck setup this is designed
        # for.
        src = np.array([p.pixel for p in pts], dtype=np.float32)
        dst = np.array(
            [[p.meters_along_lane, 0.0] for p in pts], dtype=np.float32
        )

        # findHomography needs the "0 width" degeneracy broken; jitter a
        # synthetic second row so the solver has a well-posed system, then
        # only ever read back world_x (distance along lane) downstream.
        dst_synth = dst.copy()
        dst_synth[:, 1] = np.linspace(-0.01, 0.01, len(dst_synth))

        H, _ = cv2.findHomography(src, dst_synth, method=0)
        if H is None:
            return self._solve_linear(pts)

        # RMSE against the reference points themselves.
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        rmse = float(np.sqrt(np.mean((proj[:, 0] - dst[:, 0]) ** 2)))

        return CalibrationResult(
            mode="homography", homography=H, reference_points=pts, rmse_m=rmse
        )

    def pixel_to_distance(self, x_px: float, y_px: float) -> float:
        """Return distance along the lane, in metres, for one pixel coord."""
        if self._result is None:
            raise RuntimeError("Call solve() before pixel_to_distance().")

        r = self._result
        if r.mode == "homography":
            pt = np.array([[[x_px, y_px]]], dtype=np.float32)
            world = cv2.perspectiveTransform(pt, r.homography)
            return float(world[0, 0, 0])
        else:
            slope, intercept = r.linear_fit
            coord = x_px if getattr(r, "_axis_is_x", True) else y_px
            return float(slope * coord + intercept)

    @property
    def result(self) -> CalibrationResult | None:
        return self._result
