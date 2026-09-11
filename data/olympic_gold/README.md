# Olympic Gold Standard Templates

This directory contains pre-processed `.npy` keypoint angle sequences extracted
from elite swimmer reference footage using MediaPipe Pose.

## File Format

| Property  | Value                                           |
|-----------|-------------------------------------------------|
| Shape     | `(T, 8)` — T frames × 8 joint angles           |
| dtype     | `float32`                                       |
| Units     | Normalised to `[-1, 1]` (angle_deg / 180.0)     |

## Joint Index Mapping

| Index | Joint            | MediaPipe landmarks used          |
|-------|------------------|-----------------------------------|
| 0     | Left Elbow       | L_Shoulder → L_Elbow → L_Wrist   |
| 1     | Right Elbow      | R_Shoulder → R_Elbow → R_Wrist   |
| 2     | Left Shoulder    | L_Hip → L_Shoulder → L_Elbow     |
| 3     | Right Shoulder   | R_Hip → R_Shoulder → R_Elbow     |
| 4     | Left Hip Ext.    | L_Shoulder → L_Hip → L_Knee      |
| 5     | Right Hip Ext.   | R_Shoulder → R_Hip → R_Knee      |
| 6     | Left Knee        | L_Hip → L_Knee → L_Ankle         |
| 7     | Right Knee       | R_Hip → R_Knee → R_Ankle         |

## Naming Convention

```
{stroke}_{event_m}m_gold_standard.npy
```

Examples:
- `freestyle_100m_gold_standard.npy`
- `butterfly_100m_gold_standard.npy`
- `backstroke_200m_gold_standard.npy`
- `breaststroke_100m_gold_standard.npy`

## Generating Templates

Run the offline template generator on any reference video:

```bash
python scripts/generate_templates.py \
    --video path/to/elite_swimmer.mp4 \
    --stroke freestyle \
    --event 100 \
    --output data/olympic_gold/freestyle_100m_gold_standard.npy
```

The script runs MediaPipe on the video, extracts the 8 joint angles for each
frame, normalises them, and saves the resulting array.

## Placeholder Files

The included `*_gold_standard.npy` files are **synthetic placeholders** built
from biomechanically plausible sinusoidal patterns for each stroke. They allow
the DTW comparison tab to render and the full test suite to pass without
requiring proprietary reference footage.

Replace them with templates extracted from actual elite race footage for
production use.
