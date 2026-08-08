"""Results dashboard shown after a completed analysis run."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QTextEdit, QFileDialog, QTabWidget, QHeaderView,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..pipeline import AnalysisResult
from ..export import export_csv
from ..analysis.benchmarks import compare_to_benchmark


class DashboardWidget(QWidget):
    def __init__(self, result: AnalysisResult, pool_length_m: float, parent=None):
        super().__init__(parent)
        self.result = result
        self.pool_length_m = pool_length_m

        layout = QVBoxLayout(self)

        total_frames = len(result.csv_rows)
        dr = result.discrepancy_report
        uncertain_pct = dr.still_uncertain_count / total_frames if total_frames > 0 else 0
        warning = ""
        if uncertain_pct > 0.15:
            warning = f'<br><b style="color:#e84118;">⚠️ LOW TRACKING CONFIDENCE: {uncertain_pct:.0%} of frames lost. Velocity/split data may be unreliable.</b>'
            
        header = QLabel(
            f"<h2>Race Analysis</h2>"
            f"<b>Start detection:</b> {result.start_detection.method} "
            f"(confidence {result.start_detection.confidence:.0%}) &nbsp;|&nbsp; "
            f"<b>Stroke:</b> {result.stroke_classification.stroke} "
            f"(confidence {result.stroke_classification.confidence:.0%})"
            f"{warning}"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(self._build_charts_tab(), "Velocity && Stroke Charts")
        tabs.addTab(self._build_splits_tab(), "Splits")
        tabs.addTab(self._build_feedback_tab(), "Coaching Feedback")
        tabs.addTab(self._build_quality_tab(), "Tracking Quality")
        layout.addWidget(tabs)

        export_row = QHBoxLayout()
        export_btn = QPushButton("Export CSV report")
        export_btn.clicked.connect(self._export_csv)
        export_row.addStretch(1)
        export_row.addWidget(export_btn)
        layout.addLayout(export_row)

    # ------------------------------------------------------------------
    def _build_charts_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        fig = Figure(figsize=(7, 7), tight_layout=True)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        vp = self.result.split_report.velocity_profile
        if vp:
            xs, ys = zip(*vp)
            ax1.plot(xs, ys, color="#1f77b4", label="Tracked velocity")
        ax1.set_xlabel("Distance (m)")
        ax1.set_ylabel("Velocity (m/s)")
        ax1.set_title("Velocity profile")
        ax1.grid(alpha=0.3)

        if self.result.stroke_rates:
            xs, ys = zip(*self.result.stroke_rates)
            ax2.plot(xs, ys, color="#d62728")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Stroke rate (cycles/min)")
        ax2.set_title("Stroke rate over the race")
        ax2.grid(alpha=0.3)

        if self.result.stroke_lengths:
            xs, ys = zip(*self.result.stroke_lengths)
            ax3.plot(xs, ys, color="#2ca02c")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Stroke length (m/cycle)")
        ax3.set_title("Stroke length over the race")
        ax3.grid(alpha=0.3)

        canvas = FigureCanvasQTAgg(fig)
        v.addWidget(canvas)
        return w

    def _build_splits_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        table = QTableWidget()
        splits = self.result.split_report.splits
        table.setRowCount(len(splits) + 2)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Marker (m)", "Time (s)", "Interpolated"])
        for i, s in enumerate(splits):
            table.setItem(i, 0, QTableWidgetItem(str(s.marker_m)))
            table.setItem(i, 1, QTableWidgetItem(f"{s.time_s:.3f}"))
            table.setItem(i, 2, QTableWidgetItem("yes" if s.interpolated else "no"))
        r = len(splits)
        table.setItem(r, 0, QTableWidgetItem("Final 5 m"))
        table.setItem(r, 1, QTableWidgetItem(
            f"{self.result.split_report.final_5m_s:.3f}"
            if self.result.split_report.final_5m_s is not None else "n/a"
        ))
        table.setItem(r + 1, 0, QTableWidgetItem("Final 15 m"))
        table.setItem(r + 1, 1, QTableWidgetItem(
            f"{self.result.split_report.final_15m_s:.3f}"
            if self.result.split_report.final_15m_s is not None else "n/a"
        ))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v.addWidget(table)

        total = self.result.split_report.total_time_s
        v.addWidget(QLabel(
            f"<b>Total time:</b> {total:.3f} s" if total is not None else
            "<b>Total time:</b> not available — swimmer wasn't confidently "
            "tracked all the way to the wall (check cap colour/lane "
            "settings and re-run, or verify manually)."
        ))
        return w

    def _build_feedback_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        text = QTextEdit()
        text.setReadOnly(True)

        lines = []
        sp = self.result.start_phase
        lines.append(f"Start: reached 15 m at {sp.time_to_15m_s:.2f} s"
                      if sp.time_to_15m_s is not None else "Start: 15 m mark not reached in tracked footage.")
        if sp.underwater_distance_m is not None:
            lines.append(f"Estimated underwater breakout distance: {sp.underwater_distance_m:.2f} m")
        else:
            lines.append("Underwater breakout distance: not reported — " + sp.note)

        if self.result.turn_phase is not None:
            tp = self.result.turn_phase
            if tp.turn_duration_s is not None:
                lines.append(f"Turn duration (5 m in to 10 m out): {tp.turn_duration_s:.2f} s")
            if tp.breakout_after_turn_m is not None:
                lines.append(f"Post-turn breakout distance: {tp.breakout_after_turn_m:.2f} m")
            elif tp.note:
                lines.append("Post-turn breakout: " + tp.note)

        vp = self.result.split_report.velocity_profile
        avg_v = sum(v for _, v in vp) / len(vp) if vp else None
        if avg_v is not None:
            lines.append(f"\nAverage tracked velocity: {avg_v:.2f} m/s")
            comparison = compare_to_benchmark(
                avg_v, self.result.stroke_classification.stroke,
                int(round(self.pool_length_m)), sex="mixed_top10",
            )
            if comparison:
                lines.append(comparison)
            else:
                lines.append(
                    "No verified reference figure is bundled for this exact "
                    "stroke/distance/sex combination — see "
                    "src/analysis/benchmarks.py for how to add one from a "
                    "source you've checked yourself (this app will not "
                    "invent a plausible-looking number to compare against)."
                )

        text.setPlainText("\n".join(lines))
        v.addWidget(text)
        return w

    def _build_quality_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        dr = self.result.discrepancy_report
        v.addWidget(QLabel(
            f"Frames auto-corrected via backward re-tracking / interpolation: {dr.corrected_count}\n"
            f"Frames still uncertain after correction: {dr.still_uncertain_count}"
        ))
        if not self.result.pose_available:
            v.addWidget(QLabel(
                "<b>Note:</b> no pose landmarks were detected in this "
                "video. Stroke classification, stroke rate/length, and "
                "breakout-distance metrics will be empty or low-confidence. "
                "Check that models/pose_landmarker.task was downloaded "
                "(see README) and that the swimmer is reasonably visible "
                "and unobstructed in frame."
            ))
        return w

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV report", "race_report.csv", "CSV files (*.csv)")
        if path:
            export_csv(self.result, path)
