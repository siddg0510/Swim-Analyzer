"""Main window — streamlined UI, swimmer tracking target selection, and AI configuration."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QMessageBox, QProgressBar, QGroupBox,
    QFormLayout, QStackedWidget, QCheckBox, QDialog, QDialogButtonBox,
    QScrollArea, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from ..config import resource_path, EVENTS, POOL_TYPES, GENDERS, STROKES
from ..pipeline import AnalysisConfig, AnalysisWorker, AnalysisResult
from .calibration_widget import CalibrationDialog
from .dashboard import DashboardWidget
from .swimmer_selection import SwimmerSelectionDialog


class AISettingsDialog(QDialog):
    """Dialog for entering the Gemini API key and selecting the AI model."""

    def __init__(self, parent=None, current_key: str = "", current_model: str = ""):
        super().__init__(parent)
        self.setWindowTitle("AI Settings")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info_label = QLabel(
            "<b>Google Gemini API Key</b><br>"
            "Get a key at <a href='https://aistudio.google.com/apikey' style='color:#00a8ff;'>"
            "aistudio.google.com/apikey</a><br>"
            "<span style='color:#7f8fa6; font-size:9pt;'>Your key is stored locally on this machine only.</span>"
        )
        info_label.setOpenExternalLinks(True)
        layout.addWidget(info_label)

        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("Enter AIza... key")
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        if current_key:
            self.key_edit.setText(current_key)
        layout.addWidget(self.key_edit)

        self.show_check = QCheckBox("Show key")
        self.show_check.toggled.connect(
            lambda checked: self.key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        layout.addWidget(self.show_check)

        layout.addWidget(QLabel("<b>AI Model</b>"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ])
        if current_model:
            self.model_combo.setCurrentText(current_model)
        layout.addWidget(self.model_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_key(self) -> str:
        return self.key_edit.text().strip()

    def get_model(self) -> str:
        return self.model_combo.currentText().strip()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swim Race Analyzer")
        icon_path = resource_path("assets", "icon.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setMinimumSize(700, 600)
        self.resize(820, 720)

        self.video_path: str | None = None
        self.reference_video_path: str | None = None
        self.calibration_keyframes: list = []
        self.worker: AnalysisWorker | None = None
        self._gemini_key: str | None = None
        self._gemini_model: str | None = None

        # Swimmer tracking target state
        self._target_id: int | None = None
        self._target_lane: int | None = None

        # Load saved API key and model if available
        try:
            from ..ai.gemini_config import resolve_api_key, resolve_model
            self._gemini_key = resolve_api_key()
            self._gemini_model = resolve_model()
        except Exception:
            pass

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.setup_page = self._build_setup_page()
        self.stack.addWidget(self.setup_page)

    # ------------------------------------------------------------------
    def _build_setup_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(16, 16, 16, 16)
        page_layout.setSpacing(12)

        # Scrollable area so configuration never overflows or clips
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 8, 4)
        content_layout.setSpacing(14)

        # -- 1. Video file --
        upload_box = QGroupBox("1. Video")
        upload_layout = QHBoxLayout(upload_box)
        upload_layout.setContentsMargins(14, 16, 14, 14)
        self.video_label = QLabel("No video selected.")
        self.video_label.setWordWrap(True)
        self.video_label.setStyleSheet("color: #7f8fa6;")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(110)
        browse_btn.clicked.connect(self._browse_video)
        upload_layout.addWidget(self.video_label, stretch=1)
        upload_layout.addWidget(browse_btn)
        content_layout.addWidget(upload_box)

        # -- 2. Swimmer Selection (Primary Tracking Target) --
        swimmer_box = QGroupBox("2. Target Swimmer")
        swimmer_layout = QHBoxLayout(swimmer_box)
        swimmer_layout.setContentsMargins(14, 16, 14, 14)
        self.swimmer_status_label = QLabel("Select a video first to identify swimmers.")
        self.swimmer_status_label.setStyleSheet("color: #7f8fa6;")
        self.select_swimmer_btn = QPushButton("🎯 Select Swimmer ID…")
        self.select_swimmer_btn.setFixedWidth(180)
        self.select_swimmer_btn.setEnabled(False)
        self.select_swimmer_btn.clicked.connect(self._open_swimmer_selection)
        swimmer_layout.addWidget(self.swimmer_status_label, stretch=1)
        swimmer_layout.addWidget(self.select_swimmer_btn)
        content_layout.addWidget(swimmer_box)

        # -- 3. Race setup (Pool, distance, stroke, gender) --
        inputs_box = QGroupBox("3. Race Setup")
        form = QFormLayout(inputs_box)
        form.setContentsMargins(14, 16, 14, 14)
        form.setSpacing(10)

        self.pool_length_combo = QComboBox()
        self.pool_length_combo.addItems(POOL_TYPES)
        self.pool_length_combo.setCurrentText("short course (25m)")
        form.addRow("Pool length:", self.pool_length_combo)

        self.event_combo = QComboBox()
        self.event_combo.addItems(EVENTS)
        self.event_combo.setCurrentText("100m")
        form.addRow("Event distance:", self.event_combo)

        self.stroke_combo = QComboBox()
        self.stroke_combo.addItems(["auto-detect"] + STROKES)
        form.addRow("Stroke:", self.stroke_combo)

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(GENDERS)
        form.addRow("Gender:", self.gender_combo)

        content_layout.addWidget(inputs_box)

        # -- 4. Pool Calibration --
        calib_box = QGroupBox("4. Pool Calibration")
        calib_layout = QHBoxLayout(calib_box)
        calib_layout.setContentsMargins(14, 16, 14, 14)
        self.calib_status = QLabel("Not calibrated yet.")
        self.calib_status.setStyleSheet("color: #7f8fa6;")
        calib_btn = QPushButton("Open Calibration…")
        calib_btn.setFixedWidth(160)
        calib_btn.clicked.connect(self._open_calibration)
        calib_layout.addWidget(self.calib_status, stretch=1)
        calib_layout.addWidget(calib_btn)
        content_layout.addWidget(calib_box)

        # -- 5. AI Coaching & Video Recovery (Gemini) --
        ai_box = QGroupBox("5. AI Coaching & Tracking Recovery (Gemini)")
        ai_layout = QVBoxLayout(ai_box)
        ai_layout.setContentsMargins(14, 16, 14, 14)
        ai_layout.setSpacing(10)

        ai_top_row = QHBoxLayout()
        self.ai_enabled_check = QCheckBox("Enable AI technique analysis & splash recovery")
        self.ai_enabled_check.setChecked(bool(self._gemini_key))
        self.ai_enabled_check.toggled.connect(self._on_ai_toggle)
        ai_top_row.addWidget(self.ai_enabled_check, stretch=1)

        self.api_key_status = QLabel(
            "✅ Ready" if self._gemini_key else "❌ No API key"
        )
        self.api_key_status.setStyleSheet(
            "color: #27ae60; font-weight: bold;" if self._gemini_key else "color: #e74c3c;"
        )
        ai_settings_btn = QPushButton("AI Settings…")
        ai_settings_btn.setFixedWidth(120)
        ai_settings_btn.clicked.connect(self._configure_api_key)
        ai_top_row.addWidget(self.api_key_status)
        ai_top_row.addWidget(ai_settings_btn)
        ai_layout.addLayout(ai_top_row)

        # Elite comparison
        compare_row = QHBoxLayout()
        compare_row.addWidget(QLabel("Compare with elite benchmark:"))
        self.elite_combo = QComboBox()
        self.elite_combo.addItem("Auto-select best match", "auto")
        try:
            from ..analysis.benchmarks import get_all_profile_names
            for key, name, flag in get_all_profile_names():
                self.elite_combo.addItem(f"{flag} {name}", key)
        except Exception:
            pass
        compare_row.addWidget(self.elite_combo, stretch=1)
        ai_layout.addLayout(compare_row)

        content_layout.addWidget(ai_box)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        page_layout.addWidget(scroll, stretch=1)

        # -- Sticky bottom action footer --
        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background-color: #242933;
                border: 1px solid #353b48;
                border-radius: 6px;
            }
        """)
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(12, 10, 12, 10)
        footer_layout.setSpacing(6)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedHeight(10)
        self.progress.setTextVisible(False)
        footer_layout.addWidget(self.progress)

        action_row = QHBoxLayout()
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #7f8fa6; font-size: 10pt;")
        action_row.addWidget(self.progress_label, stretch=1)

        self.run_btn = QPushButton("🏊 Run Analysis")
        self.run_btn.setMinimumHeight(42)
        self.run_btn.setMinimumWidth(160)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #00a8ff;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px 24px;
            }
            QPushButton:hover {
                background-color: #0097e6;
            }
            QPushButton:disabled {
                background-color: #353b48;
                color: #7f8fa6;
            }
        """)
        self.run_btn.clicked.connect(self._run_analysis)
        action_row.addWidget(self.run_btn)

        footer_layout.addLayout(action_row)
        page_layout.addWidget(footer)

        return page

    # ------------------------------------------------------------------
    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select race video", "", "Video files (*.mp4 *.mov *.avi *.mkv)"
        )
        if path:
            self.video_path = path
            self.video_label.setText(path)
            self.video_label.setStyleSheet("color: #f5f6fa; font-weight: 500;")
            self.calibration_keyframes = []
            self._target_id = None
            self._target_lane = None
            self.select_swimmer_btn.setEnabled(True)
            self.swimmer_status_label.setText("⚠️ Click 'Select Swimmer ID' to identify and choose target.")
            self.swimmer_status_label.setStyleSheet("color: #f39c12; font-weight: 500;")
            self._update_calibration_status()

            # Immediately prompt swimmer selection
            self._open_swimmer_selection()

    def _open_swimmer_selection(self) -> None:
        if not self.video_path:
            QMessageBox.warning(self, "No video", "Select a video first.")
            return

        dialog = SwimmerSelectionDialog(self.video_path, parent=self)
        if dialog.exec():
            self._target_id = dialog.selected_id
            self._target_lane = dialog.selected_lane
            if self._target_id is not None:
                self.swimmer_status_label.setText(f"🎯 Target Locked: Swimmer ID {self._target_id}")
                self.swimmer_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                self.select_swimmer_btn.setText(f"🎯 Change Swimmer (ID {self._target_id})…")
            elif self._target_lane is not None:
                self.swimmer_status_label.setText(f"🎯 Target Locked: Lane {self._target_lane}")
                self.swimmer_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                self.select_swimmer_btn.setText(f"🎯 Change Lane ({self._target_lane})…")
            else:
                self.swimmer_status_label.setText("⚠️ Swimmer will be auto-selected at race start.")
                self.swimmer_status_label.setStyleSheet("color: #f39c12;")

    def _open_calibration(self) -> None:
        if not self.video_path:
            QMessageBox.warning(self, "No video", "Select a video first.")
            return

        dialog = CalibrationDialog(self.video_path, parent=self)
        if dialog.exec():
            self.calibration_keyframes = dialog.get_calibration_keyframes()
            self._update_calibration_status()

    def _configure_api_key(self) -> None:
        dialog = AISettingsDialog(self, current_key=self._gemini_key or "", current_model=self._gemini_model or "")
        if dialog.exec():
            key = dialog.get_key()
            model = dialog.get_model()
            if key:
                self._gemini_key = key
                self._gemini_model = model
                try:
                    from ..ai.gemini_config import save_api_key, save_model
                    save_api_key(key)
                    save_model(model)
                except Exception:
                    pass
                self.api_key_status.setText("✅ Ready")
                self.api_key_status.setStyleSheet("color: #27ae60; font-weight: bold;")
                self.ai_enabled_check.setChecked(True)
                self._update_calibration_status()
            else:
                self._gemini_key = None
                self.api_key_status.setText("❌ No API key")
                self.api_key_status.setStyleSheet("color: #e74c3c;")
                self.ai_enabled_check.setChecked(False)
                self._update_calibration_status()

    def _on_ai_toggle(self, checked: bool) -> None:
        if checked and not self._gemini_key:
            self._configure_api_key()
            if not self._gemini_key:
                self.ai_enabled_check.setChecked(False)
        self._update_calibration_status()

    def _update_calibration_status(self) -> None:
        if self.calibration_keyframes:
            self.calib_status.setText(f"Calibrated ({len(self.calibration_keyframes)} keyframes saved)")
            self.calib_status.setStyleSheet("color: #27ae60; font-weight: bold;")
        elif self.ai_enabled_check.isChecked() and self._gemini_key:
            self.calib_status.setText("🤖 AI Auto-Calibration will run at analysis time")
            self.calib_status.setStyleSheet("color: #00a8ff;")
        else:
            self.calib_status.setText("Not calibrated yet.")
            self.calib_status.setStyleSheet("color: #7f8fa6;")

    def _get_event_distance(self) -> int:
        text = self.event_combo.currentText()
        return int(text.replace("m", ""))

    def _get_stroke_override(self) -> str | None:
        text = self.stroke_combo.currentText()
        return None if text == "auto-detect" else text

    def _get_elite_key(self) -> str | None:
        key = self.elite_combo.currentData()
        if key == "auto":
            try:
                from ..analysis.benchmarks import get_profiles_for_event
                stroke = self._get_stroke_override() or "freestyle"
                distance = self._get_event_distance()
                profiles = get_profiles_for_event(stroke, distance)
                if profiles:
                    from ..analysis.benchmarks import ELITE_PROFILES
                    for k, p in ELITE_PROFILES.items():
                        if p.name == profiles[0].name:
                            return k
            except Exception:
                pass
            return None
        return key

    def _run_analysis(self) -> None:
        if not self.video_path:
            QMessageBox.warning(self, "No video", "Select a video first.")
            return

        # Ensure a swimmer ID is chosen
        if self._target_id is None and self._target_lane is None:
            self._open_swimmer_selection()
            if self._target_id is None and self._target_lane is None:
                QMessageBox.warning(self, "Swimmer Required", "Please select a swimmer to track before running analysis.")
                return

        ai_enabled = self.ai_enabled_check.isChecked() and bool(self._gemini_key)

        if not self.calibration_keyframes:
            if not ai_enabled:
                QMessageBox.warning(self, "Not calibrated", "Complete calibration first (or enable AI Auto-Calibration).")
                return
            else:
                self.calibration_keyframes = []
        else:
            has_valid_polygon = any(len(kf.lane_polygon_px) >= 3 for kf in self.calibration_keyframes)
            if not has_valid_polygon and not ai_enabled:
                QMessageBox.warning(self, "Warning", "Lane boundaries not drawn — tracking may pick up adjacent swimmers.")

        pool_type_text = self.pool_length_combo.currentText()
        if "50m" in pool_type_text:
            pool_length_m = 50.0
        elif "25m" in pool_type_text:
            pool_length_m = 25.0
        elif "25y" in pool_type_text:
            pool_length_m = 22.86
        else:
            pool_length_m = 25.0

        cfg = AnalysisConfig(
            video_path=self.video_path,
            pool_length_m=pool_length_m,
            calibration_keyframes=self.calibration_keyframes,
            cap_color=None,
            wall_position_m=pool_length_m,
            # AI options
            enable_ai=ai_enabled,
            gemini_api_key=self._gemini_key if ai_enabled else None,
            gemini_model=self._gemini_model if ai_enabled else None,
            stroke_override=self._get_stroke_override(),
            event_distance_m=self._get_event_distance(),
            pool_type=pool_type_text.split(" (")[0],
            swimmer_sex=self.gender_combo.currentText(),
            elite_compare_key=self._get_elite_key() if ai_enabled else None,
            reference_video_path=None,
            user_goal="Improve race time and technique efficiency",
            # Primary swimmer tracking
            enable_multi_swimmer=True,
            target_id=self._target_id,
        )

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        self.worker = AnalysisWorker(cfg)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_error.connect(self._on_finished_error)
        self.worker.start()

    def _on_progress(self, pct: int, message: str) -> None:
        self.progress.setValue(pct)
        self.progress_label.setText(message)

    def _on_finished_ok(self, result: AnalysisResult) -> None:
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.progress_label.setText("")
        dashboard = DashboardWidget(
            result, self.worker.cfg.pool_length_m,
            event_distance=self._get_event_distance(),
            stroke=self._get_stroke_override(),
            swimmer_sex=self.gender_combo.currentText(),
        )
        self.stack.addWidget(dashboard)
        self.stack.setCurrentWidget(dashboard)

    def _on_finished_error(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.progress_label.setText("")
        QMessageBox.critical(self, "Analysis failed", message)
