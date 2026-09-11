"""
Tracking-confidence regression harness.

Runs the primary (multi-swimmer) tracking path on a clip and reports how the
tracker's *confidence* and *stability* behave, with attention to the coast
(predict-only) frames where the swimmer is occluded by splash.

Why these metrics: the Step 3 requirement is "splash frames must produce
no/low-confidence detection" and "coast/predict-only mode with growing
covariance marked low-confidence." Crucially we do NOT have hand-labelled
ground truth for where the swimmer actually is, and we will not fabricate it,
so we measure things that need no ground truth:

  • Stability  — frame-to-frame positional jitter (px). A tracker that
                 teleports between the swimmer and splash artifacts shows huge
                 jitter; a tracker that coasts smoothly shows small jitter.
  • Bounded    — the max excursion of the tracked point *outside* the frame
                 rectangle. A constant-acceleration filter that runs away
                 during coast extrapolates hundreds/thousands of px off-screen;
                 a bounded coast stays near the frame.
  • Coast honesty — of the frames the tracker coasted on (no accepted
                 detection), the fraction correctly flagged below the
                 confidence gate. Should be ~100%: a coasting tracker must not
                 claim confidence it does not have.

Interpret accordingly: this measures confidence *calibration* and *stability*,
NOT absolute positional accuracy (which would need the ground truth we lack).

Usage:
    python scripts/measure_tracking_confidence.py [clip.mp4] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

# Make `src` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.detector import build_detector
from src.detection.selector import TargetSelector
from src.tracking.tracker import AntiSplashTracker
from src.config import SPLASH_CONFIDENCE_GATE, SPLASH_RECOVERY_FRAMES


def measure(clip_path: str) -> dict:
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open clip: {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    # CV-only baseline: no recovery agent, so this measures the pure
    # detector+EKF confidence without Gemini masking the result.
    detector = build_detector(prefer_hog=True)  # deterministic; YOLO weights absent here anyway
    selector = TargetSelector(target_id=None)   # auto-lock onto most confident track
    tracker = AntiSplashTracker(
        detector=detector,
        selector=selector,
        recovery_agent=None,
        confidence_gate=SPLASH_CONFIDENCE_GATE,
        recovery_frame_threshold=SPLASH_RECOVERY_FRAMES,
    )

    confidences: list[float] = []
    sources: dict[str, int] = {}
    gate_rejects = 0
    positions: list[tuple[float, float]] = []
    coast_total = 0        # frames the tracker coasted (source == ekf_predict)
    coast_low_conf = 0     # ...of those, correctly flagged below the gate
    max_offframe = 0.0     # max px excursion outside the frame rectangle

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = tracker.process_frame(frame, idx)
        confidences.append(r.confidence)
        sources[r.source] = sources.get(r.source, 0) + 1
        if r.gate_rejected:
            gate_rejects += 1
        positions.append((r.x_px, r.y_px))

        if r.source == "ekf_predict":
            coast_total += 1
            if r.confidence < SPLASH_CONFIDENCE_GATE:
                coast_low_conf += 1

        # How far outside the frame did the estimate stray? (0 if inside.)
        if not (np.isnan(r.x_px) or np.isnan(r.y_px)):
            ox = max(0.0, -r.x_px, r.x_px - frame_w)
            oy = max(0.0, -r.y_px, r.y_px - frame_h)
            max_offframe = max(max_offframe, (ox * ox + oy * oy) ** 0.5)
        idx += 1

    cap.release()

    # Frame-to-frame positional jitter (stability proxy): median jump in px.
    jumps = [
        ((positions[i][0] - positions[i - 1][0]) ** 2 + (positions[i][1] - positions[i - 1][1]) ** 2) ** 0.5
        for i in range(1, len(positions))
        if not (np.isnan(positions[i][0]) or np.isnan(positions[i - 1][0]))
    ]

    return {
        "clip": clip_path,
        "frames": idx,
        "fps": round(fps, 2),
        "frame_size": [frame_w, frame_h],
        "gate": SPLASH_CONFIDENCE_GATE,
        "mean_confidence": round(statistics.mean(confidences), 4) if confidences else None,
        "median_confidence": round(statistics.median(confidences), 4) if confidences else None,
        "pct_frames_below_gate": round(100 * sum(c < SPLASH_CONFIDENCE_GATE for c in confidences) / len(confidences), 1) if confidences else None,
        "sources": sources,
        "gate_rejects": gate_rejects,
        "coast_frames": coast_total,
        "coast_low_conf_pct": round(100 * coast_low_conf / coast_total, 1) if coast_total else None,
        "max_offframe_px": round(max_offframe, 1),
        "median_jitter_px": round(statistics.median(jumps), 2) if jumps else None,
        "max_jitter_px": round(max(jumps), 2) if jumps else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip", nargs="?", default="data/testclips/splash_heavy_start.mp4")
    ap.add_argument("--json", default=None, help="Optional path to write the raw JSON report.")
    args = ap.parse_args()

    report = measure(args.clip)

    print("=" * 62)
    print("TRACKING CONFIDENCE REPORT (primary / multi-swimmer path, CV-only)")
    print("=" * 62)
    print(f"  clip                : {report['clip']}")
    print(f"  frames              : {report['frames']}  @ {report['fps']} fps  {report['frame_size']}")
    print(f"  splash gate         : {report['gate']}")
    print(f"  mean confidence     : {report['mean_confidence']}")
    print(f"  median confidence   : {report['median_confidence']}")
    print(f"  % frames below gate : {report['pct_frames_below_gate']}%")
    print(f"  sources             : {report['sources']}")
    print(f"  Mahalanobis rejects : {report['gate_rejects']}")
    print("  --- coast honesty (predict-only frames should be low-confidence) ---")
    print(f"    coast frames         : {report['coast_frames']}")
    print(f"    flagged low-conf %   : {report['coast_low_conf_pct']}%   <-- want ~100%")
    print("  --- boundedness (runaway detector; no ground truth needed) ---")
    print(f"    max off-frame excursion (px) : {report['max_offframe_px']}   <-- want small")
    print("  --- stability (frame-to-frame jitter; no ground truth needed) ---")
    print(f"    median jitter (px)   : {report['median_jitter_px']}   <-- want small")
    print(f"    max jitter (px)      : {report['max_jitter_px']}")
    print("=" * 62)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
