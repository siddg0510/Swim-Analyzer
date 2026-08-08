"""
Race-start detection: audio primary, visual fallback.

Audio path: extract the video's audio track with ffmpeg (via
imageio-ffmpeg, which bundles a real static binary per-platform — no
system ffmpeg install or internet needed at runtime), then find the first
onset where short-time energy jumps well above the ambient noise floor.
Starter horns/beeps are short (<300ms), loud relative to crowd noise, and
usually sit in a mid-high frequency band, so a bandpass filter before
onset detection meaningfully cuts false triggers from crowd noise/splash.

Visual fallback (spec section 2): if there's no audio track, or the
person disables audio detection, track frame-to-frame motion energy in a
"blocks" ROI (near the calibrated 0 m mark) and flag the first frame
where motion jumps sharply — the explosive push off the blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
import subprocess
import wave
import numpy as np
from scipy.signal import butter, filtfilt
import imageio_ffmpeg
import cv2


@dataclass
class StartDetectionResult:
    method: str                 # "audio" or "visual_fallback"
    start_time_s: float
    confidence: float
    notes: str = ""


def extract_audio_wav(video_path: str, out_wav_path: str, sample_rate: int = 44100) -> bool:
    """Extract a mono WAV track from the video using the bundled ffmpeg
    binary. Returns False (no exception) if the video has no audio
    stream — that's the normal trigger for the visual fallback, not an
    error."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1",
        out_wav_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False
    # ffmpeg can "succeed" and still write a near-empty file if there was
    # no audio stream at all; catch that case too.
    try:
        with wave.open(out_wav_path, "rb") as wf:
            return wf.getnframes() > sample_rate * 0.5  # at least 0.5s of audio
    except (wave.Error, EOFError, FileNotFoundError):
        return False


def _bandpass_energy(signal: np.ndarray, sr: int, low_hz=800, high_hz=6000,
                      window_ms=10) -> np.ndarray:
    nyq = sr / 2.0
    low, high = max(low_hz / nyq, 1e-4), min(high_hz / nyq, 0.999)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)

    win = max(int(sr * window_ms / 1000), 1)
    energy = np.convolve(filtered ** 2, np.ones(win) / win, mode="same")
    return energy


def detect_start_from_audio(wav_path: str, baseline_s: float = 1.0) -> StartDetectionResult | None:
    """
    Return the estimated start time (seconds from video start) of the
    starter horn/beep, or None if no confident onset was found (caller
    should fall back to visual detection).
    """
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        signal = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        signal /= 32768.0

    if len(signal) < sr * (baseline_s + 0.5):
        return None

    energy = _bandpass_energy(signal, sr)

    baseline_n = int(sr * baseline_s)
    noise_floor = float(np.mean(energy[:baseline_n]))
    noise_std = float(np.std(energy[:baseline_n])) + 1e-9
    threshold = noise_floor + 8 * noise_std  # conservative: real onset, not
                                              # a crowd murmur bump

    search = energy[baseline_n:]
    above = np.where(search > threshold)[0]
    if len(above) == 0:
        return None

    onset_sample = baseline_n + int(above[0])
    onset_time_s = onset_sample / sr

    # Confidence scales with how far above the noise floor the trigger was
    # and how sharply it rose (a real horn is a near-instant step, not a
    # slow swell — crowd cheering ramps up over ~1s and this rejects that
    # shape somewhat via the same threshold, but confidence still reflects
    # the margin).
    margin = (float(search[above[0]]) - noise_floor) / (noise_std + 1e-9)
    confidence = float(np.clip(margin / 20.0, 0.3, 0.98))

    return StartDetectionResult(
        method="audio", start_time_s=onset_time_s, confidence=confidence,
        notes=f"Onset {margin:.1f} std-devs above ambient noise floor.",
    )


def detect_start_visual_fallback(
    video_path: str, blocks_roi: tuple[int, int, int, int] | None,
    max_search_s: float = 15.0,
) -> StartDetectionResult:
    """
    No-audio fallback: frame-differencing motion spike near the starting
    blocks. blocks_roi is (x, y, w, h) in pixels; if None, uses the full
    frame (noisier, but still workable for a locked-off tripod shot).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_search_s * fps)

    prev_gray = None
    motion_series = []
    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if blocks_roi:
            x, y, w, h = blocks_roi
            frame = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_series.append(float(np.mean(diff)))
        else:
            motion_series.append(0.0)
        prev_gray = gray
        frame_idx += 1
    cap.release()

    motion = np.array(motion_series)
    if len(motion) < 5:
        return StartDetectionResult("visual_fallback", 0.0, 0.1,
                                     notes="Too few frames to detect motion reliably.")

    baseline_n = max(int(fps * 0.5), 3)
    noise_floor = float(np.mean(motion[:baseline_n]))
    noise_std = float(np.std(motion[:baseline_n])) + 1e-9
    threshold = noise_floor + 6 * noise_std

    above = np.where(motion[baseline_n:] > threshold)[0]
    if len(above) == 0:
        return StartDetectionResult(
            "visual_fallback", 0.0, 0.15,
            notes="No clear motion spike found in the search window; "
                  "defaulting start to t=0 — verify manually.",
        )

    onset_frame = baseline_n + int(above[0])
    onset_time_s = onset_frame / fps
    margin = (motion[baseline_n + above[0]] - noise_floor) / noise_std
    confidence = float(np.clip(margin / 15.0, 0.25, 0.85))  # visual fallback
                                                             # is inherently
                                                             # less certain
                                                             # than audio
    return StartDetectionResult(
        "visual_fallback", onset_time_s, confidence,
        notes=f"Motion spike {margin:.1f} std-devs above baseline at frame {onset_frame}.",
    )
