"""
Generate synthetic placeholder gold-standard .npy templates.

Run once after installation to create demo templates that allow the DTW
Comparison tab to render even before real elite footage is available.

Replace outputs with real templates using generate_templates.py.
"""
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data" / "olympic_gold"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Template parameters per stroke — chosen to produce biomechanically
# plausible sinusoidal patterns at each joint.
# (T_frames, phase_offsets[8], amplitudes[8])
STROKE_PARAMS = {
    "freestyle": {
        "event": [50, 100, 200],
        "T": 90,           # ~3 seconds of a stroke cycle at 30fps
        "phases": [0.0, np.pi, 0.3, np.pi+0.3, 0.1, np.pi+0.1, 0.5, np.pi+0.5],
        "amps":   [0.25, 0.25, 0.15, 0.15, 0.10, 0.10, 0.08, 0.08],
        "base":   [0.50, 0.50, 0.60, 0.60, 0.55, 0.55, 0.75, 0.75],
    },
    "backstroke": {
        "event": [100, 200],
        "T": 90,
        "phases": [0.0, np.pi, 0.4, np.pi+0.4, 0.2, np.pi+0.2, 0.6, np.pi+0.6],
        "amps":   [0.22, 0.22, 0.18, 0.18, 0.12, 0.12, 0.09, 0.09],
        "base":   [0.48, 0.48, 0.58, 0.58, 0.52, 0.52, 0.72, 0.72],
    },
    "breaststroke": {
        "event": [100, 200],
        "T": 100,          # breaststroke is slower
        "phases": [0.0, 0.05, 0.5, 0.55, 0.3, 0.35, 0.7, 0.75],
        "amps":   [0.30, 0.30, 0.20, 0.20, 0.25, 0.25, 0.30, 0.30],
        "base":   [0.50, 0.50, 0.55, 0.55, 0.60, 0.60, 0.65, 0.65],
    },
    "butterfly": {
        "event": [100, 200],
        "T": 80,
        "phases": [0.0, 0.1, 0.0, 0.1, 0.5, 0.6, 0.5, 0.6],
        "amps":   [0.28, 0.28, 0.22, 0.22, 0.20, 0.20, 0.25, 0.25],
        "base":   [0.50, 0.50, 0.60, 0.60, 0.55, 0.55, 0.70, 0.70],
    },
}

rng = np.random.default_rng(42)  # reproducible

for stroke, params in STROKE_PARAMS.items():
    T = params["T"]
    t = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
    seq = np.zeros((T, 8), dtype=np.float32)
    for j in range(8):
        freq = 1.0 if stroke != "breaststroke" else 0.8
        seq[:, j] = (
            params["base"][j]
            + params["amps"][j] * np.sin(freq * t + params["phases"][j])
            + rng.normal(0, 0.02, T)   # small realistic jitter
        ).astype(np.float32)
    # Clip to valid normalised range [0, 1] (0° → 0.0, 180° → 1.0)
    seq = np.clip(seq, 0.0, 1.0)

    for event_m in params["event"]:
        out_path = OUT_DIR / f"{stroke}_{event_m}m_gold_standard.npy"
        np.save(str(out_path), seq)
        print(f"Saved {out_path}  shape={seq.shape}")

print("Done — placeholder gold standards generated.")
