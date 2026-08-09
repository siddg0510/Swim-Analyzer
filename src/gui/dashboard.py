"""Results dashboard shown after a completed analysis run.

Enhanced with AI-powered tabs for technique analysis, elite comparison,
improvement plans, and race strategy when Gemini analysis is available.
Falls back to the original CV-only dashboard gracefully.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QTextEdit, QFileDialog, QTabWidget, QHeaderView, QScrollArea,
    QFrame, QGridLayout, QGroupBox,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from ..pipeline import AnalysisResult, AIAnalysisResult
from ..export import export_csv
from ..analysis.benchmarks import compare_to_benchmark


# ---------------------------------------------------------------------------
# Colour helpers for rating badges
# ---------------------------------------------------------------------------
RATING_COLORS = {
    "excellent": "#27ae60",
    "good": "#2ecc71",
    "needs_improvement": "#f39c12",
    "critical_issue": "#e74c3c",
}

RATING_ICONS = {
    "excellent": "🟢",
    "good": "🟡",
    "needs_improvement": "🟠",
    "critical_issue": "🔴",
}

GAP_COLORS = {
    "minor": "#27ae60",
    "moderate": "#f39c12",
    "major": "#e74c3c",
}


def _badge(rating: str) -> str:
    color = RATING_COLORS.get(rating, "#7f8fa6")
    icon = RATING_ICONS.get(rating, "⚪")
    label = rating.replace("_", " ").title()
    return f'{icon} <span style="color:{color}; font-weight:bold;">{label}</span>'


class DashboardWidget(QWidget):
    def __init__(
        self,
        result: AnalysisResult,
        pool_length_m: float,
        event_distance: int = 100,
        stroke: str | None = None,
        swimmer_sex: str = "male",
        parent=None,
    ):
        super().__init__(parent)
        self.result = result
        self.pool_length_m = pool_length_m
        self.event_distance = event_distance
        self.swimmer_sex = swimmer_sex
        self.stroke = stroke or result.stroke_classification.stroke

        layout = QVBoxLayout(self)

        # -- Header --
        total_frames = len(result.csv_rows)
        dr = result.discrepancy_report
        uncertain_pct = dr.still_uncertain_count / total_frames if total_frames > 0 else 0
        warning = ""
        if uncertain_pct > 0.15:
            warning = f'<br><b style="color:#e84118;">⚠️ LOW TRACKING CONFIDENCE: {uncertain_pct:.0%} of frames lost. Velocity/split data may be unreliable.</b>'

        ai_status = ""
        if result.ai_result and not result.ai_result.error:
            ai_status = ' &nbsp;|&nbsp; <b style="color:#27ae60;">🤖 AI Analysis Available</b>'
        elif result.ai_result and result.ai_result.error:
            ai_status = f' &nbsp;|&nbsp; <b style="color:#f39c12;">🤖 AI: {result.ai_result.error[:60]}</b>'

        header = QLabel(
            f"<h2>Race Analysis — {self.event_distance}m {self.stroke.title()}</h2>"
            f"<b>Start detection:</b> {result.start_detection.method} "
            f"(confidence {result.start_detection.confidence:.0%}) &nbsp;|&nbsp; "
            f"<b>Stroke:</b> {result.stroke_classification.stroke} "
            f"(confidence {result.stroke_classification.confidence:.0%})"
            f"{ai_status}{warning}"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # -- Tabs --
        tabs = QTabWidget()

        # Original tabs
        tabs.addTab(self._build_charts_tab(), "📊 Velocity && Stroke Charts")
        tabs.addTab(self._build_splits_tab(), "⏱ Splits")
        tabs.addTab(self._build_feedback_tab(), "💬 Coaching Feedback")
        tabs.addTab(self._build_quality_tab(), "🔍 Tracking Quality")

        # AI-powered tabs (only shown when AI results are available)
        if result.ai_result:
            ai = result.ai_result
            if ai.technique_analysis:
                tabs.addTab(self._build_technique_tab(ai), "🤖 AI Technique Analysis")
            if ai.elite_comparison:
                tabs.addTab(self._build_elite_tab(ai), "🏅 Elite Comparison")
            if ai.improvement_plan:
                tabs.addTab(self._build_improvement_tab(ai), "📈 Improvement Plan")
            if ai.race_strategy:
                tabs.addTab(self._build_strategy_tab(ai), "🏊 Race Strategy")
            if ai.video_comparison:
                tabs.addTab(self._build_video_comparison_tab(ai), "🎥 Video Comparison")

        layout.addWidget(tabs)

        # -- Bottom bar --
        bottom_row = QHBoxLayout()

        back_btn = QPushButton("← New Analysis")
        back_btn.clicked.connect(self._go_back)
        bottom_row.addWidget(back_btn)

        bottom_row.addStretch(1)

        export_btn = QPushButton("Export CSV report")
        export_btn.clicked.connect(self._export_csv)
        bottom_row.addWidget(export_btn)

        layout.addLayout(bottom_row)

    # ------------------------------------------------------------------
    # Original tabs (preserved from existing code)
    # ------------------------------------------------------------------
    def _build_charts_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        fig = Figure(figsize=(7, 7), tight_layout=True)
        fig.patch.set_facecolor("#2f3640")
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        for ax in (ax1, ax2, ax3):
            ax.set_facecolor("#353b48")
            ax.tick_params(colors="#f5f6fa")
            ax.xaxis.label.set_color("#f5f6fa")
            ax.yaxis.label.set_color("#f5f6fa")
            ax.title.set_color("#f5f6fa")
            for spine in ax.spines.values():
                spine.set_color("#4a5568")

        vp = self.result.split_report.velocity_profile
        if vp:
            xs, ys = zip(*vp)
            ax1.plot(xs, ys, color="#00a8ff", linewidth=1.5, label="Tracked velocity")
            ax1.fill_between(xs, ys, alpha=0.15, color="#00a8ff")
        ax1.set_xlabel("Distance (m)")
        ax1.set_ylabel("Velocity (m/s)")
        ax1.set_title("Velocity Profile")
        ax1.grid(alpha=0.2, color="#7f8fa6")

        if self.result.stroke_rates:
            xs, ys = zip(*self.result.stroke_rates)
            ax2.plot(xs, ys, color="#e84118", linewidth=1.5)
            ax2.fill_between(xs, ys, alpha=0.15, color="#e84118")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Stroke rate (cycles/min)")
        ax2.set_title("Stroke Rate Over the Race")
        ax2.grid(alpha=0.2, color="#7f8fa6")

        if self.result.stroke_lengths:
            xs, ys = zip(*self.result.stroke_lengths)
            ax3.plot(xs, ys, color="#44bd32", linewidth=1.5)
            ax3.fill_between(xs, ys, alpha=0.15, color="#44bd32")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Stroke length (m/cycle)")
        ax3.set_title("Stroke Length Over the Race")
        ax3.grid(alpha=0.2, color="#7f8fa6")

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
                self.event_distance, sex=self.swimmer_sex,
            )
            if comparison:
                lines.append(comparison)
            else:
                lines.append(
                    "No verified reference figure is bundled for this exact "
                    "stroke/distance/sex combination — see "
                    "src/analysis/benchmarks.py for how to add one."
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

    # ------------------------------------------------------------------
    # AI-powered tabs
    # ------------------------------------------------------------------
    def _build_technique_tab(self, ai: AIAnalysisResult) -> QWidget:
        """AI Technique Analysis — detailed breakdown by body part/phase."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        v = QVBoxLayout(content)

        tech = ai.technique_analysis
        if not tech:
            v.addWidget(QLabel("No technique analysis available."))
            scroll.setWidget(content)
            return scroll

        # Overall rating
        v.addWidget(QLabel(
            f"<h3>Overall Technique Rating: {_badge(tech.overall_rating)}</h3>"
            f"<p>Stroke identified: <b>{tech.stroke_identified}</b> "
            f"(confidence: {tech.stroke_confidence:.0%})</p>"
        ))

        # Top 3 priorities
        if tech.top_priorities:
            prio_box = QGroupBox("🎯 Top Priorities for Improvement")
            prio_layout = QVBoxLayout(prio_box)
            for p in tech.top_priorities:
                prio_layout.addWidget(QLabel(
                    f"<b>#{p.priority}. {p.element}</b><br>"
                    f"Expected impact: {p.expected_impact}<br>"
                    f"Drill: <i>{p.drill}</i>"
                ))
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet("color: #353b48;")
                prio_layout.addWidget(line)
            v.addWidget(prio_box)

        # Detailed elements by category
        categories = {}
        for elem in tech.elements:
            cat = elem.category.replace("_", " ").title()
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(elem)

        for cat_name, elements in categories.items():
            cat_box = QGroupBox(cat_name)
            cat_layout = QVBoxLayout(cat_box)
            for elem in elements:
                elem_label = QLabel(
                    f"{_badge(elem.rating)} <b>{elem.element}</b><br>"
                    f"<i>Observation:</i> {elem.observation}<br>"
                    f"<i>Recommendation:</i> {elem.recommendation}"
                )
                if elem.elite_reference:
                    elem_label.setText(
                        elem_label.text() +
                        f"<br><i>Elite reference:</i> 🏅 {elem.elite_reference}"
                    )
                if elem.timestamp_hint:
                    elem_label.setText(
                        elem_label.text() +
                        f"<br><i>Timestamp:</i> ⏱ {elem.timestamp_hint}"
                    )
                elem_label.setWordWrap(True)
                cat_layout.addWidget(elem_label)
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet("color: #353b48;")
                cat_layout.addWidget(line)
            v.addWidget(cat_box)

        # Video quality notes
        if tech.video_quality_notes:
            v.addWidget(QLabel(
                f"<b>📹 Video quality notes:</b> {tech.video_quality_notes}"
            ))

        v.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_elite_tab(self, ai: AIAnalysisResult) -> QWidget:
        """Elite Comparison — user vs Olympic athlete."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        v = QVBoxLayout(content)

        comp = ai.elite_comparison
        if not comp:
            v.addWidget(QLabel("No elite comparison available."))
            scroll.setWidget(content)
            return scroll

        # Header with similarity score
        v.addWidget(QLabel(
            f"<h3>Comparison with 🏅 {comp.comparison_athlete}</h3>"
            f"<p>Overall technique similarity: <b>{comp.overall_similarity_pct:.0f}%</b></p>"
        ))

        # Radar chart
        metrics_data = comp.metrics_comparison
        if metrics_data:
            fig = Figure(figsize=(5, 4), tight_layout=True)
            fig.patch.set_facecolor("#2f3640")
            ax = fig.add_subplot(111, polar=True)
            ax.set_facecolor("#353b48")

            categories_list = list(metrics_data.keys())
            if categories_list:
                # Simple bar-style visualization of metrics comparison
                ax_flat = fig.add_subplot(111)
                ax_flat.set_facecolor("#353b48")
                ax_flat.tick_params(colors="#f5f6fa")
                for spine in ax_flat.spines.values():
                    spine.set_color("#4a5568")

                metrics_box = QGroupBox("📊 Metrics Comparison")
                m_layout = QVBoxLayout(metrics_box)
                for key, value in metrics_data.items():
                    m_layout.addWidget(QLabel(
                        f"<b>{key.replace('_', ' ').title()}:</b> {value}"
                    ))
                v.addWidget(metrics_box)

        # Strengths matching elite
        if comp.strengths:
            str_box = QGroupBox("✅ What You're Doing Well (Matches Elite)")
            str_layout = QVBoxLayout(str_box)
            for s in comp.strengths:
                str_layout.addWidget(QLabel(
                    f"🟢 <b>{s.element}</b> — {s.similarity}<br>"
                    f"<i>{s.detail}</i>"
                ))
            v.addWidget(str_box)

        # Key differences
        if comp.differences:
            diff_box = QGroupBox("🔄 Key Differences from Elite")
            diff_layout = QVBoxLayout(diff_box)
            for d in comp.differences:
                color = GAP_COLORS.get(d.gap_severity, "#7f8fa6")
                addr = "✅ Addressable" if d.addressable else "⚠️ Physical limitation"
                diff_layout.addWidget(QLabel(
                    f'<span style="color:{color};">●</span> '
                    f"<b>{d.element}</b> — Gap: {d.gap_severity.upper()}<br>"
                    f"<i>You:</i> {d.user_observation}<br>"
                    f"<i>Elite:</i> {d.elite_model}<br>"
                    f"<i>Fix:</i> {d.how_to_close_gap}<br>"
                    f"<small>{addr}</small>"
                ))
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet("color: #353b48;")
                diff_layout.addWidget(line)
            v.addWidget(diff_box)

        # Realistic targets
        if comp.realistic_targets:
            target_box = QGroupBox("🎯 Realistic Targets")
            target_layout = QVBoxLayout(target_box)
            table = QTableWidget()
            table.setRowCount(len(comp.realistic_targets))
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(["Metric", "Current", "Target", "Timeframe"])
            for i, t in enumerate(comp.realistic_targets):
                table.setItem(i, 0, QTableWidgetItem(t.metric))
                table.setItem(i, 1, QTableWidgetItem(t.current))
                table.setItem(i, 2, QTableWidgetItem(t.target))
                table.setItem(i, 3, QTableWidgetItem(t.timeframe))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            target_layout.addWidget(table)
            v.addWidget(target_box)

        v.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_improvement_tab(self, ai: AIAnalysisResult) -> QWidget:
        """Improvement Plan — multi-phase training program."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        v = QVBoxLayout(content)

        plan = ai.improvement_plan
        if not plan:
            v.addWidget(QLabel("No improvement plan available."))
            scroll.setWidget(content)
            return scroll

        # Summary
        v.addWidget(QLabel(f"<h3>📈 Personalized Improvement Plan</h3><p>{plan.summary}</p>"))

        # Phase 1
        if plan.phase_1:
            p1 = plan.phase_1
            p1_box = QGroupBox(
                f"Phase 1: Immediate ({p1.get('duration_weeks', '2-4')} weeks) — {p1.get('focus', '')}"
            )
            p1_layout = QVBoxLayout(p1_box)

            drills = p1.get("drills", [])
            for drill in drills:
                if isinstance(drill, dict):
                    p1_layout.addWidget(QLabel(
                        f"🏊 <b>{drill.get('name', '')}</b><br>"
                        f"Purpose: {drill.get('purpose', '')}<br>"
                        f"How: {drill.get('description', '')}<br>"
                        f"Volume: {drill.get('sets_reps', '')} | "
                        f"Frequency: {drill.get('frequency', '')}"
                    ))
                    line = QFrame()
                    line.setFrameShape(QFrame.Shape.HLine)
                    line.setStyleSheet("color: #353b48;")
                    p1_layout.addWidget(line)

            targets = p1.get("target_metrics", {})
            if targets:
                p1_layout.addWidget(QLabel(
                    "<b>Target Metrics:</b><br>" +
                    "<br>".join(f"• {k.replace('_', ' ').title()}: {v}"
                               for k, v in targets.items())
                ))
            v.addWidget(p1_box)

        # Phase 2
        if plan.phase_2:
            p2 = plan.phase_2
            p2_box = QGroupBox(
                f"Phase 2: Development ({p2.get('duration_weeks', '4-8')} weeks) — {p2.get('focus', '')}"
            )
            p2_layout = QVBoxLayout(p2_box)

            drills = p2.get("drills", [])
            for drill in drills:
                if isinstance(drill, dict):
                    p2_layout.addWidget(QLabel(
                        f"🏊 <b>{drill.get('name', '')}</b><br>"
                        f"Purpose: {drill.get('purpose', '')}<br>"
                        f"How: {drill.get('description', '')}<br>"
                        f"Volume: {drill.get('sets_reps', '')} | "
                        f"Frequency: {drill.get('frequency', '')}"
                    ))

            if p2.get("race_simulation"):
                p2_layout.addWidget(QLabel(
                    f"<br><b>Race Simulation:</b> {p2['race_simulation']}"
                ))
            v.addWidget(p2_box)

        # Phase 3
        if plan.phase_3:
            p3 = plan.phase_3
            p3_box = QGroupBox(
                f"Phase 3: Race Prep ({p3.get('duration_weeks', '2-4')} weeks) — {p3.get('focus', '')}"
            )
            p3_layout = QVBoxLayout(p3_box)

            key_sets = p3.get("key_sets", [])
            for s in key_sets:
                if isinstance(s, dict):
                    p3_layout.addWidget(QLabel(
                        f"⚡ <b>{s.get('name', '')}</b><br>"
                        f"{s.get('description', '')}<br>"
                        f"<i>Purpose: {s.get('purpose', '')}</i>"
                    ))

            cues = p3.get("mental_cues", [])
            if cues:
                p3_layout.addWidget(QLabel(
                    "<br><b>🧠 Race Mental Cues:</b><br>" +
                    "<br>".join(f"• {c}" for c in cues)
                ))
            v.addWidget(p3_box)

        # Expected improvement
        if plan.expected_improvement:
            ei = plan.expected_improvement
            ei_box = QGroupBox("🎯 Expected Improvement")
            ei_layout = QVBoxLayout(ei_box)
            ei_layout.addWidget(QLabel(
                f"<b>Time reduction estimate:</b> {ei.get('time_reduction_estimate', 'N/A')}<br>"
                f"<b>Primary gains from:</b> {ei.get('primary_gains_from', 'N/A')}<br>"
                f"<i>Caveat: {ei.get('caveat', 'Individual results may vary.')}</i>"
            ))
            v.addWidget(ei_box)

        v.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_strategy_tab(self, ai: AIAnalysisResult) -> QWidget:
        """Race Strategy — pacing analysis and tactical feedback."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        v = QVBoxLayout(content)

        strat = ai.race_strategy
        if not strat:
            v.addWidget(QLabel("No race strategy analysis available."))
            scroll.setWidget(content)
            return scroll

        # Split pattern
        pattern_icons = {
            "negative": "⬇️ Negative Split (faster second half)",
            "positive": "⬆️ Positive Split (faster first half)",
            "even": "➡️ Even Split",
        }
        pattern_text = pattern_icons.get(strat.split_pattern, strat.split_pattern)
        v.addWidget(QLabel(
            f"<h3>🏊 Race Strategy Analysis</h3>"
            f"<p>Split pattern: <b>{pattern_text}</b></p>"
            f"<p>{strat.split_analysis}</p>"
        ))

        # Elite pacing comparison
        if strat.elite_pacing_comparison:
            epc = strat.elite_pacing_comparison
            epc_box = QGroupBox("🏅 How Elites Pace This Event")
            epc_layout = QVBoxLayout(epc_box)
            epc_layout.addWidget(QLabel(
                f"<b>Event norm:</b> {epc.get('event_norm', 'N/A')}<br>"
                f"<b>Your pacing vs norm:</b> {epc.get('user_vs_norm', 'N/A')}<br>"
                f"<b>Example:</b> {epc.get('specific_example', 'N/A')}"
            ))
            v.addWidget(epc_box)

        # Speed loss zones
        if strat.speed_loss_zones:
            slz_box = QGroupBox("⚠️ Speed Loss Zones")
            slz_layout = QVBoxLayout(slz_box)
            for zone in strat.speed_loss_zones:
                slz_layout.addWidget(QLabel(
                    f"<b>{zone.get('zone', '')}</b>: "
                    f"Speed drop {zone.get('speed_drop_pct', 'N/A')}%<br>"
                    f"Likely cause: {zone.get('likely_cause', '')}<br>"
                    f"Fix: {zone.get('fix', '')}"
                ))
            v.addWidget(slz_box)

        # Stroke rate analysis
        if strat.stroke_rate_analysis:
            sra = strat.stroke_rate_analysis
            sra_box = QGroupBox("📊 Stroke Rate Analysis")
            sra_layout = QVBoxLayout(sra_box)
            sra_layout.addWidget(QLabel(
                f"<b>Pattern:</b> {sra.get('pattern', 'N/A')}<br>"
                f"<b>Optimal adjustment:</b> {sra.get('optimal_adjustment', 'N/A')}"
            ))
            v.addWidget(sra_box)

        # Recommended race plan
        if strat.recommended_race_plan:
            rrp = strat.recommended_race_plan
            rrp_box = QGroupBox("🎯 Recommended Race Plan")
            rrp_layout = QVBoxLayout(rrp_box)
            text_parts = []
            if rrp.get("target_first_split"):
                text_parts.append(f"<b>Target 1st split:</b> {rrp['target_first_split']}")
            if rrp.get("target_second_split"):
                text_parts.append(f"<b>Target 2nd split:</b> {rrp['target_second_split']}")
            if rrp.get("pacing_strategy"):
                text_parts.append(f"<b>Strategy:</b> {rrp['pacing_strategy']}")
            focus = rrp.get("key_focus_points", [])
            if focus:
                text_parts.append("<b>Key focus points:</b><br>" +
                                  "<br>".join(f"• {f}" for f in focus))
            rrp_layout.addWidget(QLabel("<br>".join(text_parts)))
            v.addWidget(rrp_box)

        v.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_video_comparison_tab(self, ai: AIAnalysisResult) -> QWidget:
        """Video comparison — user vs reference video analysis."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        v = QVBoxLayout(content)

        vc = ai.video_comparison
        if not vc:
            v.addWidget(QLabel("No video comparison available."))
            scroll.setWidget(content)
            return scroll

        v.addWidget(QLabel(
            f"<h3>🎥 Reference Video Comparison</h3>"
            f"<p>{vc.comparison_summary}</p>"
            f"<p>Overall technique gap: <b>{vc.overall_technique_gap.upper()}</b></p>"
            f"<p>🎯 <b>Top priority change:</b> {vc.top_priority_change}</p>"
        ))

        # What user does well
        if vc.what_user_does_well:
            well_box = QGroupBox("✅ What You Do Well")
            well_layout = QVBoxLayout(well_box)
            for item in vc.what_user_does_well:
                well_layout.addWidget(QLabel(f"🟢 {item}"))
            v.addWidget(well_box)

        # Technique differences
        if vc.technique_differences:
            diff_box = QGroupBox("🔄 Technique Differences")
            diff_layout = QVBoxLayout(diff_box)
            for d in vc.technique_differences:
                diff_color = {
                    "easy": "#27ae60", "moderate": "#f39c12", "hard": "#e74c3c"
                }.get(d.difficulty, "#7f8fa6")
                diff_layout.addWidget(QLabel(
                    f"<b>{d.element}</b> "
                    f'(Difficulty: <span style="color:{diff_color}">{d.difficulty}</span>)<br>'
                    f"<i>You:</i> {d.user_technique}<br>"
                    f"<i>Reference:</i> {d.reference_technique}<br>"
                    f"<i>Impact:</i> {d.impact}<br>"
                    f"<i>Drill:</i> 🏊 {d.drill_to_fix}"
                ))
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet("color: #353b48;")
                diff_layout.addWidget(line)
            v.addWidget(diff_box)

        v.addStretch(1)
        scroll.setWidget(content)
        return scroll

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _go_back(self) -> None:
        """Return to the setup page."""
        stack = self.parentWidget()
        if stack:
            stack.setCurrentIndex(0)

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV report", "race_report.csv", "CSV files (*.csv)")
        if path:
            export_csv(self.result, path)
