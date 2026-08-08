# Swim Race Analyzer

Offline desktop app for analyzing swim race video: splits, stroke rate/
length, start and turn phase metrics, and a reference-comparison
dashboard. Built with OpenCV, MediaPipe, PySide6, and matplotlib. No
network calls at runtime — everything happens on the machine it runs on.

## Before you do anything else: read this

A few parts of the original spec described things that either don't
exist as off-the-shelf assets or can't be measured the way they were
phrased. Rather than fake them, this build implements the closest honest
version of each and says so in the code:

| Spec asked for | What's actually here | Why |
|---|---|---|
| "YOLOv8-swim ONNX models" | HSV colour tracking + Kalman filter (default); pluggable `Detector` interface if you want to bring your own model | No pretrained swim-specific YOLO model exists publicly. Generic YOLOv8 (Ultralytics) is AGPL-3.0 / needs a paid licence to ship closed-source — colour tracking avoids that and is what real race-analysis systems (e.g. AIMsys) actually use for cap tracking. |
| Millisecond-precision finish | Sub-frame **interpolated** timing | A camera's frame rate is the real resolution floor (30fps ≈ 33ms between samples). Interpolation estimates within that window; it isn't a hardware timestamp. |
| "Map the pool's 3D geometry" | 2D homography from your clicked reference points | A single side-on camera can't recover true 3D geometry. The homography is the same practical compromise every monocular sports-video system makes. |
| Hardcoded Olympic velocity-curve/stroke-rate baseline dataset | A framework populated with real, cited, regulatory facts and one real (open-water) data point — see `src/analysis/benchmarks.py` | Search turned up real peer-reviewed race-analysis papers, but the exact numbers live in tables/figures those searches couldn't extract as verifiable text. Rather than fill the gaps with plausible-sounding numbers, the fields are `None` with a comment on exactly where to get the real ones. |
| Auto stroke/event recognition | A rule-based heuristic (arm-phase correlation, body orientation, leg-phase correlation) | No pretrained stroke classifier exists to bundle offline. The heuristic is grounded in real biomechanical differences between strokes but hasn't been validated against real race footage — treat its output as a first guess, not a certified answer. |
| Breakout distance / underwater metrics | Reported only when the pose estimator is actually confident the swimmer is visible; otherwise `None` with a note | Most single-side-deck consumer video loses the swimmer underwater. Showing a number there would look precise while being invented. |

Everything else in the spec — GUI with the 3 required inputs, audio+visual
start detection, calibration overlay, cap tracking, split/micro-split
timing, stroke rate & length, self-correction pass, multithreaded
pipeline, CSV export, PyInstaller packaging — is implemented as
described, and the algorithmic core (calibration math, colour tracking,
Kalman occlusion bridging, audio onset detection, split interpolation,
outlier detection) is covered by synthetic tests in `tests/test_core.py`
that actually run and pass — see "Testing," below. There is no real swim
video to test against, so none of this has been validated on an actual
race yet; that's the honest state of things, not a hedge.

## Setup

```bash
pip install -r requirements.txt
```

Then, once, download the pose model (see `models/DOWNLOAD_MODEL.md` for
why this can't be bundled automatically):

```bash
curl -L -o models/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

Run it:

```bash
python main.py
```

## Using it

1. **Browse** to a race video.
2. Enter **pool length**, **lane number**, **swim cap colour** (pick a
   preset or type a hex code).
3. **Open calibration overlay** — click 2+ points on the first frame
   (4+, not in a straight line, for full perspective correction) and
   type the real-world distance each one is from the start wall. Switch
   to "Lane boundary" mode and click the 4 corners of the target lane so
   tracking ignores other lanes.
4. **Run Analysis**. Progress runs on a background thread — the window
   stays responsive.
5. Read the dashboard: velocity/stroke-rate/stroke-length charts, a
   splits table, a coaching-feedback panel, and a tracking-quality tab
   that tells you plainly when something wasn't measurable. Export CSV
   from the button at the bottom.

## Building the .exe / .app

PyInstaller does not cross-compile — build on the OS you're targeting.

```bash
# after downloading the model and installing requirements on that machine
pyinstaller build.spec
```

Output: `dist/SwimRaceAnalyzer/` (onedir build — chosen over onefile
because mediapipe + OpenCV + PySide6 unpack a lot of native libraries,
and onefile's unpack-to-temp-on-every-launch cost isn't worth it here).
On macOS this also produces `SwimRaceAnalyzer.app`.

## Testing

```bash
python tests/test_core.py
```

These are synthetic-data tests (generated video frames, generated audio
tones, generated pose landmark sequences) — they check the math and the
OpenCV/NumPy/SciPy plumbing are correct, not real-world tracking accuracy
on an actual swimmer. All 9 currently pass. Two real bugs were caught and
fixed while building this (a numpy 2.x scalar-conversion break in the
Kalman filter, and a mistuned process/measurement noise ratio that made
the filter almost ignore new detections) — exactly the kind of thing
that "looks done" until you actually run it.

## Project layout

```
main.py                    entry point
src/config.py               paths, PyInstaller resource resolution, constants
src/vision/                 calibration, cap tracking, pose, stroke classification
src/audio/start_detector.py audio onset + visual fallback start detection
src/analysis/                splits, biomechanics, self-correction, benchmarks
src/gui/                     PySide6 windows, calibration overlay, dashboard
src/pipeline.py             QThread worker tying it all together
src/export.py               CSV report
tests/test_core.py          synthetic smoke tests
build.spec                  PyInstaller config
models/DOWNLOAD_MODEL.md    pose model download instructions
```

## Sources checked while building this

Real citations, not general recollection — used for the regulatory facts
and segment conventions in `src/analysis/benchmarks.py`, and to sanity-
check the marker/segment choices in `src/config.py` against how race
analysts actually define them:

- Gonjo, Veiga, Hermosilla-Perona & Olstad (2025). "Variabilities in the
  stroking parameters during short course 50 m time trials in all four
  competitive swimming strokes." *Scientific Reports* 15, 23640.
  doi:10.1038/s41598-025-08519-9 — World Aquatics' 15 m underwater rule,
  the 5-m-before-the-wall turn-in/finish convention, and the fact that a
  real professional race-analysis system (AIMsys) tracks swimmers by cap
  colour, the same approach this project uses.
- "Phase-specific determinants of 100 m freestyle performance in elite
  swimmers." *Scientific Reports* (2025). doi:10.1038/s41598-025-02814-1
  — the 5-phase (S15/HS1/T20/HS2/F5) segmentation model, from real
  Olympic-medallist footage.
- "Stroke rate–stroke length dynamics in elite freestyle swimming:
  application of kernel density estimation." *Frontiers in Sports and
  Active Living* (2025). doi:10.3389/fspor.2025.1656633 — correlational
  data (not a hardcoded baseline) on when stroke length vs. stroke rate
  matters more, by race distance.
- "Evaluation of Race Pace Using Critical Swimming Speed During 10 km
  Open-Water Swimming Competition." Open access, peer-reviewed.
  https://www.mdpi.com/2411-5142/10/3/302 — the one populated numeric
  entry in `benchmarks.py` (10 km open-water top-10, 2023 World Aquatics
  Championships), clearly scoped to that context, not pool sprints.
- Google AI Edge / MediaPipe official docs
  (ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — the
  model download URL and confirmation that the Tasks API (not the older
  `mp.solutions.pose`, which no longer exists in current MediaPipe) is
  the current interface.

Sources checked and deliberately **not** used: a "swimming stroke rate
calculator" site that cited World Aquatics without a traceable source —
skipped per the instruction to stick to sites with actual credibility,
even though its numbers were roughly plausible.

## Licensing note if you plan to ship this

Everything used by default (OpenCV, MediaPipe, NumPy/SciPy, PySide6,
matplotlib, imageio-ffmpeg) is permissively licensed (BSD/Apache/LGPL) —
fine for a closed-source distributed executable. If you later plug in
Ultralytics YOLOv8 via the `Detector` interface, note that it's
AGPL-3.0 (or requires a paid Enterprise licence for closed-source
redistribution) — that's a real legal consideration for a "downloadable
.exe," not this project's default path.
