# Swim Race Analyzer (AI-Enhanced)

By: Siddharth Gulati  
Reach out to me at: https://www.linkedin.com/in/siddharth-gulati-4742143a1/

An advanced offline and AI-powered desktop app for analyzing swim race video. It tracks swimmer splits, stroke rate/length, start and turn phase metrics using computer vision (OpenCV, MediaPipe). With the new **Google Gemini AI integration**, it also provides expert technique analysis, compares your technique against Olympic gold medalists (like Pan Zhanle, Katie Ledecky, and Léon Marchand), and delivers personalized, multi-phase improvement plans.

## Features

- **Computer Vision Pipeline (Offline)**: Cap tracking, pose estimation, split/micro-split timing, and stroke metrics computed entirely on your machine.
- **Gemini AI Coaching (Optional)**: Automatically extracts key race segments (start, clean swim, finish) and sends them to Gemini for:
  - **Technique Analysis**: Detailed breakdown by body part with rating badges (🟢🟡🟠🔴).
  - **Elite Comparison**: Side-by-side comparison with an auto-matched Olympian based on your stroke, event, gender, and performance.
  - **Improvement Plan**: 3-phase training program with specific drills and expected time improvement.
  - **Race Strategy**: Pacing analysis, speed loss zones, and tactical feedback.
  - **Reference Video Compare**: Upload an Olympic footage video for a direct side-by-side AI analysis.
- **Elite Benchmark Database**: Built-in 25+ benchmark entries across all 4 strokes and Olympic distances, plus 8 detailed elite swimmer profiles with real race data, splits, and technique signatures.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This includes `google-genai` for the new AI features.*

2. **Download the Pose Model** (one-time setup):
   See `models/DOWNLOAD_MODEL.md` for why this isn't bundled automatically.
   ```bash
   curl -L -o models/pose_landmarker.task \
     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
   ```

3. **Get a Gemini API Key** (optional, but required for AI features):
   Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey). You can enter this directly in the app's UI.

4. **Run it**:
   ```bash
   python main.py
   ```

## Using it

1. **Browse** to a race video.
2. Enter **pool length**, **lane number**, **swim cap colour** (pick a preset or type a hex code).
3. **Configure AI** (optional): Check "Enable AI-powered technique analysis", add your API key, and set your event details (Stroke, Gender, Goal). You can pick a specific elite swimmer to compare against, or leave it on "Auto-select best match".
4. **Open calibration overlay** — click 2+ points on the first frame (4+, not in a straight line, for full perspective correction) and type the real-world distance each one is from the start wall. Switch to "Lane boundary" mode and click the 4 corners of the target lane so tracking ignores other lanes.
5. **Run Analysis**. The CV pipeline runs first (progress on a background thread), followed by the AI analysis (video upload and inference).
6. **Read the dashboard**:
   - **Original Tabs**: Velocity/stroke-rate/stroke-length charts, a splits table, tracking-quality, and a coaching-feedback panel.
   - **AI Tabs**: 🤖 Technique Analysis, 🏅 Elite Comparison, 📈 Improvement Plan, 🏊 Race Strategy, 🎥 Video Comparison.
   - Export your results to a CSV report from the bottom bar.

## Building the .exe / .app

PyInstaller does not cross-compile — build on the OS you're targeting.

```bash
pyinstaller build.spec
```

Output: `dist/SwimRaceAnalyzer/` (onedir build). On macOS this also produces `SwimRaceAnalyzer.app`.

## Testing

The project includes an extensive test suite covering the algorithmic CV core, the new benchmark databases, and the Gemini integration.

```bash
# Run core CV synthetic tests (9 tests)
python tests/test_core.py

# Run benchmark database validation and auto-match logic (29 tests)
python tests/test_benchmarks.py

# Run Gemini parsing and API integration tests (mocked, 17 tests)
python tests/test_gemini.py
```

All 55 tests should pass. The core tests check the math and the OpenCV/NumPy plumbing using synthetic data, while the benchmark and Gemini tests ensure data integrity and robust JSON parsing of AI outputs.

## Project layout

```
main.py                       entry point
src/config.py                 paths, PyInstaller resource resolution, AI constants
src/vision/                   calibration, cap tracking, pose, stroke classification (w/ Gemini fallback)
src/audio/start_detector.py   audio onset + visual fallback start detection
src/analysis/                 splits, biomechanics, self-correction, expanded benchmarks
src/ai/                       Gemini config, prompts, analyzer, reference comparison, video segmentation
src/gui/                      PySide6 windows, calibration overlay, enhanced AI dashboard
src/pipeline.py               QThread worker tying CV and AI together
src/export.py                 CSV report
tests/                        test suites for core, benchmarks, and gemini
build.spec                    PyInstaller config
models/DOWNLOAD_MODEL.md      pose model download instructions
```

## Sources checked while building this

Real citations, not general recollection — used for the regulatory facts and segment conventions in `src/analysis/benchmarks.py`:

- Gonjo, Veiga, Hermosilla-Perona & Olstad (2025). "Variabilities in the stroking parameters during short course 50 m time trials in all four competitive swimming strokes." *Scientific Reports* 15, 23640. doi:10.1038/s41598-025-08519-9
- "Phase-specific determinants of 100 m freestyle performance in elite swimmers." *Scientific Reports* (2025). doi:10.1038/s41598-025-02814-1
- "Stroke rate–stroke length dynamics in elite freestyle swimming: application of kernel density estimation." *Frontiers in Sports and Active Living* (2025). doi:10.3389/fspor.2025.1656633

## Licensing note if you plan to ship this

Everything used by default (OpenCV, MediaPipe, NumPy/SciPy, PySide6, matplotlib, imageio-ffmpeg, google-genai) is permissively licensed (BSD/Apache/LGPL/MIT) — fine for a closed-source distributed executable. If you later plug in Ultralytics YOLOv8 via the `Detector` interface, note that it's AGPL-3.0 (or requires a paid Enterprise licence for closed-source redistribution).
