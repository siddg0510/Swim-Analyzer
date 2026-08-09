"""
Video segment extraction for efficient Gemini analysis.

Rather than uploading a full race video (which can be many minutes and
would waste tokens), this module extracts the specific segments that
matter for technique analysis:
  - Start phase (dive → breakout → first few strokes)
  - Clean swimming (mid-pool, technique-representative section)
  - Turn phase (approach → flip → push-off → breakout)
  - Finish phase (last ~10 m approach and touch)

Uses OpenCV for frame-level extraction and ffmpeg (via imageio-ffmpeg)
for efficient re-encoding of the clips.
"""
from __future__ import annotations

import os
import tempfile
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoSegment:
    """A named segment of the race video."""
    name: str
    path: str
    start_time_s: float
    end_time_s: float
    description: str


def extract_race_segments(
    video_path: str,
    start_time_s: float,
    pool_length_m: float,
    fps: float = 30.0,
    split_times: list[tuple[float, float]] | None = None,
    output_dir: str | None = None,
) -> list[VideoSegment]:
    """Extract key segments from the race video.

    Parameters
    ----------
    video_path : str
        Path to the full race video.
    start_time_s : float
        Detected race start time (seconds from video start).
    pool_length_m : float
        Pool length in metres.
    fps : float
        Video frame rate.
    split_times : list of (distance_m, time_s) pairs, optional
        If available, used to compute accurate segment boundaries.
    output_dir : str, optional
        Where to write clips. If None, uses a temp directory.

    Returns
    -------
    list of VideoSegment
        Extracted clips with metadata.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="swim_segments_")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_duration_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (
        cap.get(cv2.CAP_PROP_FPS) or fps
    )
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute segment boundaries
    segments_spec = _compute_segment_boundaries(
        start_time_s, total_duration_s, pool_length_m, split_times
    )

    segments: list[VideoSegment] = []
    for name, t_start, t_end, description in segments_spec:
        # Clamp to valid range
        t_start = max(0.0, t_start)
        t_end = min(total_duration_s, t_end)
        if t_end <= t_start + 0.5:
            logger.warning("Segment '%s' too short (%.1fs), skipping", name, t_end - t_start)
            continue

        clip_path = os.path.join(output_dir, f"{name}.mp4")
        _extract_clip(cap, clip_path, t_start, t_end, actual_fps, frame_w, frame_h)
        segments.append(VideoSegment(
            name=name, path=clip_path,
            start_time_s=t_start, end_time_s=t_end,
            description=description,
        ))

    cap.release()
    return segments


def extract_full_video_for_upload(
    video_path: str,
    max_duration_s: float = 60.0,
    start_offset_s: float = 0.0,
    output_dir: str | None = None,
) -> str:
    """Extract a clip of at most ``max_duration_s`` from the video.

    For short race videos (under 60s), this just copies the relevant
    portion. For longer videos, it takes the first ``max_duration_s``
    starting from ``start_offset_s``.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="swim_clip_")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / actual_fps
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t_start = max(0.0, start_offset_s - 1.0)  # 1s before start for context
    t_end = min(total_duration, t_start + max_duration_s)

    clip_path = os.path.join(output_dir, "race_clip.mp4")
    _extract_clip(cap, clip_path, t_start, t_end, actual_fps, frame_w, frame_h)
    cap.release()
    return clip_path


def _compute_segment_boundaries(
    start_time_s: float,
    total_duration_s: float,
    pool_length_m: float,
    split_times: list[tuple[float, float]] | None,
) -> list[tuple[str, float, float, str]]:
    """Return (name, start_s, end_s, description) for each segment."""
    segments = []

    # Estimate race duration from splits or default
    if split_times and len(split_times) >= 2:
        race_end = start_time_s + max(t for _, t in split_times)
    else:
        # Rough estimate: ~1.5 m/s average → pool_length_m / 1.5
        race_end = start_time_s + pool_length_m / 1.5

    # 1. Start segment: 1s before start to ~5s after
    seg_start = max(0, start_time_s - 1.0)
    seg_end = min(total_duration_s, start_time_s + 8.0)
    segments.append((
        "start_phase", seg_start, seg_end,
        "Start phase: reaction, dive, underwater, breakout, first strokes"
    ))

    # 2. Clean swimming: mid-pool section (roughly 15m-35m timing)
    if split_times:
        # Find times at ~15m and ~35m
        t15 = _interp_split_time(split_times, 15.0) or (start_time_s + 7.0)
        t35 = _interp_split_time(split_times, 35.0) or (start_time_s + 18.0)
    else:
        # Estimate
        avg_speed = 1.8  # m/s rough average
        t15 = start_time_s + 15.0 / avg_speed
        t35 = start_time_s + 35.0 / avg_speed

    seg_start = max(0, t15 - 1.0)
    seg_end = min(total_duration_s, t35 + 1.0)
    # Cap at 20 seconds
    if seg_end - seg_start > 20.0:
        seg_end = seg_start + 20.0
    segments.append((
        "clean_swimming", seg_start, seg_end,
        "Clean swimming phase: mid-pool technique at race pace"
    ))

    # 3. Turn segment (only for 50m+ pools or multi-lap races)
    if pool_length_m >= 50:
        # Turn at the far wall
        if split_times:
            t_wall = _interp_split_time(split_times, pool_length_m) or race_end
        else:
            t_wall = start_time_s + pool_length_m / 1.8
        seg_start = max(0, t_wall - 4.0)
        seg_end = min(total_duration_s, t_wall + 8.0)
        segments.append((
            "turn_phase", seg_start, seg_end,
            "Turn phase: approach, flip/open turn, push-off, streamline, breakout"
        ))

    # 4. Finish segment: last ~8 seconds of the race
    seg_end = min(total_duration_s, race_end + 2.0)
    seg_start = max(0, seg_end - 10.0)
    segments.append((
        "finish_phase", seg_start, seg_end,
        "Finish phase: final approach and wall touch"
    ))

    return segments


def _interp_split_time(
    split_times: list[tuple[float, float]], target_m: float
) -> float | None:
    """Linearly interpolate time for a given distance from split data."""
    if not split_times:
        return None
    distances = [d for d, _ in split_times]
    times = [t for _, t in split_times]
    if target_m < distances[0] or target_m > distances[-1]:
        return None
    return float(np.interp(target_m, distances, times))


def _extract_clip(
    cap: cv2.VideoCapture,
    output_path: str,
    t_start: float,
    t_end: float,
    fps: float,
    width: int,
    height: int,
) -> None:
    """Write frames from ``t_start`` to ``t_end`` into a new MP4 file."""
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_MSEC, t_start * 1000.0)
    while True:
        pos_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pos_s > t_end:
            break
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)

    writer.release()
    logger.info("Extracted clip %s (%.1fs → %.1fs)", output_path, t_start, t_end)


def cleanup_segments(segments: list[VideoSegment]) -> None:
    """Delete temporary segment files."""
    for seg in segments:
        try:
            os.remove(seg.path)
        except OSError:
            pass
