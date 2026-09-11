"""CSV export for a completed AnalysisResult."""
from __future__ import annotations

import csv
from .core.models import AnalysisResult


def export_csv(result: AnalysisResult, out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["Swim Race Analysis Report"])
        writer.writerow([])
        writer.writerow(["Start detection method", result.start_detection.method])
        writer.writerow(["Start detection confidence", f"{result.start_detection.confidence:.2f}"])
        writer.writerow(["Start detection notes", result.start_detection.notes])
        writer.writerow([])

        writer.writerow(["Classified stroke", result.stroke_classification.stroke])
        writer.writerow(["Classification confidence", f"{result.stroke_classification.confidence:.2f}"])
        writer.writerow(["(heuristic classifier — not validated on real race footage; verify manually)"])
        writer.writerow([])

        writer.writerow(["-- Split Times --"])
        writer.writerow(["Marker (m)", "Time (s)", "Interpolated"])
        for s in result.split_report.splits:
            writer.writerow([s.marker_m, f"{s.time_s:.3f}", s.interpolated])
        writer.writerow(["Final 5 m split (s)", result.split_report.final_5m_s or "n/a — swimmer not confidently tracked to finish"])
        writer.writerow(["Final 15 m split (s)", result.split_report.final_15m_s or "n/a"])
        writer.writerow(["Total time (s)", result.split_report.total_time_s or "n/a — see notes"])
        writer.writerow([])

        writer.writerow(["-- Start Phase --"])
        writer.writerow(["Time to 15 m (s)", result.start_phase.time_to_15m_s])
        writer.writerow(["Underwater distance (m)", result.start_phase.underwater_distance_m or "not measurable from this footage"])
        writer.writerow(["Note", result.start_phase.note])
        writer.writerow([])

        if result.turn_phase is not None:
            writer.writerow(["-- Turn Phase --"])
            writer.writerow(["Turn duration (s)", result.turn_phase.turn_duration_s])
            writer.writerow(["Breakout after turn (m)", result.turn_phase.breakout_after_turn_m or "not measurable from this footage"])
            writer.writerow(["Note", result.turn_phase.note])
            writer.writerow([])

        writer.writerow(["-- Stroke Rate (rolling, cycles/min) --"])
        writer.writerow(["Time (s)", "Stroke rate (cpm)"])
        for t, r in result.stroke_rates:
            writer.writerow([f"{t:.2f}", f"{r:.1f}"])
        writer.writerow([])

        writer.writerow(["-- Stroke Length (m per cycle) --"])
        writer.writerow(["Time (s)", "Stroke length (m)"])
        for t, l in result.stroke_lengths:
            writer.writerow([f"{t:.2f}", f"{l:.2f}"])
        writer.writerow([])

        writer.writerow(["-- Velocity Profile --"])
        writer.writerow(["Distance (m)", "Velocity (m/s)"])
        for d, v in result.split_report.velocity_profile:
            writer.writerow([f"{d:.2f}", f"{v:.2f}"])
        writer.writerow([])

        writer.writerow(["-- Tracking Quality --"])
        writer.writerow(["Frames auto-corrected", result.discrepancy_report.corrected_count])
        writer.writerow(["Frames still uncertain after correction", result.discrepancy_report.still_uncertain_count])
        writer.writerow([])

        writer.writerow(["-- Raw Per-Frame Track --"])
        writer.writerow(["frame", "time_s", "distance_m", "confidence", "method"])
        for row in result.csv_rows:
            writer.writerow([row["frame"], row["time_s"], row["distance_m"], row["confidence"], row["method"]])
