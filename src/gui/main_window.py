"""Main window — UI, initial calibration, and AI configuration."""
from __future__ import annotations

import cv2
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QGroupBox, QFormLayout, QStackedWidget, QCheckBox, QDialog,
    QDialogButtonBox, QTextEdit,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from ..vision.calibration import PoolCalibrator
from ..config import resource_path, EVENTS, POOL_TYPES, GENDERS, STROKES
from ..pipeline import AnalysisConfig, AnalysisWorker, AnalysisResult
from .calibration_widget import CalibrationDialog, ClickableImageLabel, _cv_frame_to_qpixmap
from .dashboard import DashboardWidget

CAP_COLOR_PRESETS = [
    "red", "orange", "yellow", "green", "neon green", "olive", "dark green", "cyan", "blue",
    "navy", "purple", "pink", "white", "black", "silver", "gray", "Custom hex…",
]


class AISettingsDialog(QDialog):
    """Dialog for entering the Gemini API key and selecting the AI model."""

    def __init__(self, parent=None, current_key: str = "", current_model: str = ""):
        super().__init__(parent)
        self.setWindowTitle("AI Settings")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "<b>Enter your Google Gemini API key</b><br>"
            "Get a free key at <a href='https://aistudio.google.com/apikey'>"
            "aistudio.google.com/apikey</a><br><br>"
            "The key is stored locally on your machine only."
        ))

        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("AIza...")
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

        layout.addWidget(QLabel("<b>Select AI Model</b><br>gemini-1.5-pro is recommended for video analysis."))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-pro",
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


class ColorPickerDialog(QDialog):
    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pick Cap Color")
        self.resize(1000, 700)
        
        self.color_hex = None
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            QMessageBox.warning(self, "Error", "Could not read video.")
            self.reject()
            return
            
        self.frame = frame
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Click on the swimmer's cap to pick the color:"))
        
        from PySide6.QtWidgets import QScrollArea
        
        self.image_label = ClickableImageLabel(self._on_click)
        self.image_label.setPixmap(_cv_frame_to_qpixmap(frame))
        
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, stretch=1)
        
        self.color_preview = QLabel("Selected color: None")
        self.color_preview.setMinimumHeight(30)
        self.color_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.color_preview)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_click(self, x, y):
        h, w = self.frame.shape[:2]
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            b, g, r = self.frame[y, x]
            self.color_hex = f"#{int(r):02X}{int(g):02X}{int(b):02X}"
            self.color_preview.setText(f"Selected color: {self.color_hex}")
            text_color = "black" if (int(r)+int(g)+int(b)) > 380 else "white"
            self.color_preview.setStyleSheet(f"background-color: {self.color_hex}; color: {text_color}; font-weight: bold;")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swim Race Analyzer")
        icon_path = resource_path("assets", "icon.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(800, 700)

        self.video_path: str | None = None
        self.reference_video_path: str | None = None
        self.reference_points = []
        self.lane_polygon = []
        self.worker: AnalysisWorker | None = None
        self._gemini_key: str | None = None
        self._gemini_model: str | None = None

        # Try to load saved API key and model
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
        layout = QVBoxLayout(page)

        # -- Video upload --
        upload_box = QGroupBox("1. Video")
        upload_layout = QHBoxLayout(upload_box)
        self.video_label = QLabel("No video selected.")
        self.video_label.setWordWrap(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_video)
        upload_layout.addWidget(self.video_label, stretch=1)
        upload_layout.addWidget(browse_btn)
        layout.addWidget(upload_box)

        # -- Required inputs --
        inputs_box = QGroupBox("2. Race setup")
        form = QFormLayout(inputs_box)

        self.pool_length_combo = QComboBox()
        self.pool_length_combo.addItems(POOL_TYPES)
        self.pool_length_combo.setCurrentText("short course (25m)")
        form.addRow("Pool length:", self.pool_length_combo)

        self.lane_spin = QSpinBox()
        self.lane_spin.setRange(1, 10)
        self.lane_spin.setValue(4)
        form.addRow("Lane number:", self.lane_spin)

        cap_row = QHBoxLayout()
        self.cap_combo = QComboBox()
        self.cap_combo.addItems(CAP_COLOR_PRESETS)
        self.cap_hex_edit = QLineEdit()
        self.cap_hex_edit.setPlaceholderText("#RRGGBB")
        self.cap_hex_edit.setEnabled(False)
        self.cap_combo.currentTextChanged.connect(
            lambda t: self.cap_hex_edit.setEnabled(t == "Custom hex…")
        )
        self.pick_color_btn = QPushButton("Pick from video…")
        self.pick_color_btn.clicked.connect(self._pick_color_from_video)
        
        cap_row.addWidget(self.cap_combo)
        cap_row.addWidget(self.cap_hex_edit)
        cap_row.addWidget(self.pick_color_btn)
        form.addRow("Swim cap colour:", cap_row)

        # Event details for AI analysis
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



        layout.addWidget(inputs_box)

        # -- Calibration --
        calib_box = QGroupBox("3. Calibration")
        calib_layout = QVBoxLayout(calib_box)
        self.calib_status = QLabel("Not calibrated yet.")
        calib_btn = QPushButton("Open calibration overlay…")
        calib_btn.clicked.connect(self._open_calibration)
        calib_layout.addWidget(self.calib_status)
        calib_layout.addWidget(calib_btn)
        layout.addWidget(calib_box)

        # -- AI Analysis Settings --
        ai_box = QGroupBox("4. 🤖 AI Analysis (Gemini)")
        ai_layout = QVBoxLayout(ai_box)

        self.ai_enabled_check = QCheckBox("Enable AI-powered technique analysis")
        self.ai_enabled_check.setChecked(bool(self._gemini_key))
        self.ai_enabled_check.toggled.connect(self._on_ai_toggle)
        ai_layout.addWidget(self.ai_enabled_check)

        key_row = QHBoxLayout()
        self.api_key_status = QLabel(
            "✅ API key configured" if self._gemini_key else "❌ No API key"
        )
        api_key_btn = QPushButton("AI Settings…")
        api_key_btn.clicked.connect(self._configure_api_key)
        key_row.addWidget(self.api_key_status)
        key_row.addStretch(1)
        key_row.addWidget(api_key_btn)
        ai_layout.addLayout(key_row)

        # Elite comparison
        compare_row = QHBoxLayout()
        compare_row.addWidget(QLabel("Compare with:"))
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

        # Reference video
        ref_row = QHBoxLayout()
        self.ref_video_label = QLabel("No reference video (optional)")
        self.ref_video_label.setWordWrap(True)
        ref_btn = QPushButton("Add reference video…")
        ref_btn.clicked.connect(self._browse_reference_video)
        ref_clear_btn = QPushButton("Clear")
        ref_clear_btn.clicked.connect(self._clear_reference_video)
        ref_row.addWidget(self.ref_video_label, stretch=1)
        ref_row.addWidget(ref_btn)
        ref_row.addWidget(ref_clear_btn)
        ai_layout.addLayout(ref_row)

        # Goal
        goal_row = QHBoxLayout()
        goal_row.addWidget(QLabel("Your goal:"))
        self.goal_edit = QLineEdit()
        self.goal_edit.setPlaceholderText("e.g. Break 55 seconds in 100m freestyle")
        goal_row.addWidget(self.goal_edit, stretch=1)
        ai_layout.addLayout(goal_row)

        layout.addWidget(ai_box)

        # -- Run --
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("🏊 Run Analysis")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._run_analysis)
        run_row.addStretch(1)
        run_row.addWidget(self.run_btn)
        layout.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------
    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select race video", "", "Video files (*.mp4 *.mov *.avi *.mkv)"
        )
        if path:
            self.video_path = path
            self.video_label.setText(path)
            self.calibration_keyframes = []
            self.calib_status.setText("Not calibrated yet.")

    def _browse_reference_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select reference video (e.g. Olympic footage)", "",
            "Video files (*.mp4 *.mov *.avi *.mkv)"
        )
        if path:
            self.reference_video_path = path
            self.ref_video_label.setText(f"📹 {path.split('/')[-1].split(chr(92))[-1]}")

    def _clear_reference_video(self) -> None:
        self.reference_video_path = None
        self.ref_video_label.setText("No reference video (optional)")

    def _open_calibration(self) -> None:
        if not self.video_path:
            QMessageBox.warning(self, "No video", "Select a video first.")
            return

        dialog = CalibrationDialog(self.video_path, parent=self)
        if dialog.exec():
            self.calibration_keyframes = dialog.get_calibration_keyframes()
            self.calib_status.setText(
                f"Calibrated: {len(self.calibration_keyframes)} keyframes saved."
            )

    def _pick_color_from_video(self) -> None:
        if not self.video_path:
            QMessageBox.warning(self, "No video", "Select a video first.")
            return

        dialog = ColorPickerDialog(self.video_path, parent=self)
        if dialog.exec():
            if dialog.color_hex:
                self.cap_combo.setCurrentText("Custom hex…")
                self.cap_hex_edit.setText(dialog.color_hex)

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
                self.api_key_status.setText(f"✅ AI Ready ({model})")
                self.ai_enabled_check.setChecked(True)
            else:
                self._gemini_key = None
                self.api_key_status.setText("❌ No API key")
                self.ai_enabled_check.setChecked(False)

    def _on_ai_toggle(self, checked: bool) -> None:
        if checked and not self._gemini_key:
            self._configure_api_key()
            if not self._gemini_key:
                self.ai_enabled_check.setChecked(False)

    def _get_cap_color(self) -> str:
        choice = self.cap_combo.currentText()
        if choice == "Custom hex…":
            return self.cap_hex_edit.text().strip() or "#FFFF00"
        return choice

    def _get_event_distance(self) -> int:
        text = self.event_combo.currentText()
        return int(text.replace("m", ""))

    def _get_stroke_override(self) -> str | None:
        text = self.stroke_combo.currentText()
        return None if text == "auto-detect" else text

    def _get_elite_key(self) -> str | None:
        key = self.elite_combo.currentData()
        if key == "auto":
            # Auto-select based on event
            try:
                from ..analysis.benchmarks import get_profiles_for_event
                stroke = self._get_stroke_override() or "freestyle"
                distance = self._get_event_distance()
                profiles = get_profiles_for_event(stroke, distance)
                if profiles:
                    # Find the key for this profile
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
        ai_enabled = self.ai_enabled_check.isChecked() and bool(self._gemini_key)

        if not hasattr(self, 'calibration_keyframes') or not self.calibration_keyframes:
            if not ai_enabled:
                QMessageBox.warning(self, "Not calibrated", "Complete calibration first (or enable AI Auto-Calibration).")
                return
            else:
                self.calibration_keyframes = []

        pool_type_text = self.pool_length_combo.currentText()
        if "50m" in pool_type_text:
            pool_length_m = 50.0
        elif "25m" in pool_type_text:
            pool_length_m = 25.0
        elif "25y" in pool_type_text:
            pool_length_m = 22.86  # 25 yards in meters
        else:
            pool_length_m = 25.0

        cfg = AnalysisConfig(
            video_path=self.video_path,
            pool_length_m=pool_length_m,
            calibration_keyframes=self.calibration_keyframes,
            cap_color=self._get_cap_color(),
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
            reference_video_path=self.reference_video_path if ai_enabled else None,
            user_goal=self.goal_edit.text().strip() or "Improve race time and technique efficiency",
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
