# Downloading the pose model (one-time, before your first build)

This app uses MediaPipe's Pose Landmarker for stroke classification,
stroke-rate/length, and breakout-distance metrics. Current MediaPipe
(0.10.x, verified while building this project) no longer bundles a pose
model inside the pip package — it moved to a separate `.task` model file
you download once. This is a real, official Google asset, not something
this project can legally re-host or embed for you sight-unseen, and the
download host is outside the sandboxed environment this project was
built in, so it could not be fetched automatically.

**This is the only step in the whole build/run process that needs
internet, and only once, before your first run.** After the file is in
place, the compiled app runs completely offline.

## Get it

```bash
# From this project's root folder:
curl -L -o models/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

(or `pose_landmarker_full` / `pose_landmarker_lite` instead of `_heavy`
for a smaller, faster, slightly-less-accurate model — trade-off table is
on Google's page linked below. `_heavy` is the recommended default for
this app since accuracy matters more than speed for a one-off race
analysis.)

If that URL ever changes, the canonical, current link is always
published on Google's own docs page:
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
(verified against that page while building this project, July 2026.)

## Verify it landed correctly

```bash
ls -la models/pose_landmarker.task
# should be several MB, not a few hundred bytes (a few hundred bytes
# usually means the URL 404'd and curl saved an HTML error page instead)
```

## What happens if you skip this

The app still runs. `PoseEstimator` catches the missing-model error and
sets `available = False`; the Tracking Quality tab will tell you no pose
landmarks were found. You'll still get audio/visual start detection,
cap-based tracking, splits, and velocity — everything that doesn't need
a skeleton. You'll lose stroke classification, stroke rate/length, and
breakout-distance estimates.
