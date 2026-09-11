#!/usr/bin/env python3
"""
scripts/generate_templates.py
─────────────────────────────
Offline tool to build Olympic Gold Standard templates from real reference video.

Usage:
    python scripts/generate_templates.py \\
        --video path/to/elite_swimmer.mp4 \\
        --stroke freestyle \\
        --event 100 \\
        [--output data/olympic_gold/freestyle_100m_gold_standard.npy] \\
        [--fps 30] \\
        [--start-sec 0] \\
        [--end-sec 60]

The script:
  1. Opens the video with OpenCV
  2. Runs MediaPipe PoseLandmarker on every frame (or the specified window)
  3. Extracts the 8-joint angular feature vector from each frame
  4. Saves the (T, 8) float32 array to the output .npy file

Run once per reference video; the resulting file is checked in and ships with
the app so users don't need to reprocess the footage.
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Make project root importable when run as a script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DTW gold-standard .npy templates.")
    p.add_argument("--video",      required=True,  help="Path to reference video file.")
    p.add_argument("--stroke",     required=True,
                   choices=["freestyle", "backstroke", "breaststroke", "butterfly"],
                   help="Swimming stroke.")
    p.add_argument("--event",      required=True,  type=int,
                   help="Event distance in metres (e.g. 100).")
    p.add_argument("--output",     default=None,
                   help="Output .npy path. Defaults to data/olympic_gold/<stroke>_<event>m_gold_standard.npy")
    p.add_argument("--fps",        type=float, default=None,
                   help="Override video FPS (default: read from file).")
    p.add_argument("--start-sec",  type=float, default=0.0,
                   help="Start time in seconds (default: 0).")
    p.add_argument("--end-sec",    type=float, default=None,
                   help="End time in seconds (default: end of file).")
    p.add_argument("--model",      default=None,
                   help="Path to pose_landmarker.task model file.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --- resolve output path ---
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = _ROOT / "data" / "olympic_gold" / f"{args.stroke}_{args.event}m_gold_standard.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}", file=sys.stderr)
        return 1

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps or video_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(args.start_sec * fps)
    end_frame   = int(args.end_sec   * fps) if args.end_sec else total_frames

    print(f"Video: {args.video}  ({fps:.1f} fps, {total_frames} frames)")
    print(f"Processing frames {start_frame}–{end_frame} → {out_path}")

    # --- pose estimator ---
    from src.vision.pose_estimator import PoseEstimator  # noqa: PLC0415
    pe = PoseEstimator(model_path=args.model)
    if not pe.available:
        print("ERROR: PoseEstimator not available. Ensure pose_landmarker.task is downloaded.", file=sys.stderr)
        return 1

    # --- feature extractor ---
    from src.analysis.comparator import extract_feature_vector  # noqa: PLC0415

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows: list[np.ndarray] = []
    last_valid = np.full(8, 0.5, dtype=np.float32)
    fi = start_frame

    while fi < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_ms = int((fi / fps) * 1000)
        pf = pe.process(frame, fi, timestamp_ms)
        vec = extract_feature_vector(pf)
        if vec is not None:
            last_valid = vec
        rows.append(last_valid.copy())

        fi += 1
        if fi % 100 == 0:
            print(f"  frame {fi}/{end_frame} …", end="\r")

    cap.release()
    pe.close()

    if not rows:
        print("ERROR: No pose data extracted.", file=sys.stderr)
        return 1

    arr = np.stack(rows, axis=0).astype(np.float32)  # (T, 8)
    np.save(str(out_path), arr)
    print(f"\nSaved {out_path}  shape={arr.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
