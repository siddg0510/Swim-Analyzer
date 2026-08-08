"""Main window — spec section 1 (UI & initial calibration)."""
from __future__ import annotations

import cv2
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QGroupBox, QFormLayout, QStackedWidget,
)
from PySide6.QtCore import Qt

from ..vision.calibration import PoolCalibrator
from ..pipeline import AnalysisConfig, AnalysisWorker, AnalysisResult
from .calibration_widget import CalibrationDialog
from .dashboard import DashboardWidget

CAP_COLOR_PRESETS = [
    "red", "orange", "yellow", "green", "neon green", "olive", "dark green", "cyan", "blue",
    "navy", "purple", "pink", "white", "black", "silver", "gray", "Custom hex…",
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swim Race Analyzer")
        self.resize(720, 560)

        self.video_path: str | None = None
        self.reference_points = []
        self.lane_polygon = []
        self.worker: AnalysisWorker | None = None

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
        inputs_box = QGroupBox("2. Race setup (pool length, lane, cap colour)")
        form = QFormLayout(inputs_box)

        self.pool_length_spin = QDoubleSpinBox()
        self.pool_length_spin.setRange(10, 100)
        self.pool_length_spin.setValue(25.0)
        self.pool_length_spin.setSuffix(" m")
        form.addRow("Pool length:", self.pool_length_spin)

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
        cap_row.addWidget(self.cap_combo)
        cap_row.addWidget(self.cap_hex_edit)
        form.addRow("Swim cap colour:", cap_row)

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

        # -- Run --
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Analysis")
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

    def _get_cap_color(self) -> str:
        choice = self.cap_combo.currentText()
        if choice == "Custom hex…":
            return self.cap_hex_edit.text().strip() or "#FFFF00"
        return choice

    def _run_analysis(self) -> None:
        if not self.video_path:
            QMessageBox.warning(self, "No video", "Select a video first.")
            return
        if not hasattr(self, 'calibration_keyframes') or not self.calibration_keyframes:
            QMessageBox.warning(self, "Not calibrated", "Complete calibration first (at least 1 keyframe).")
            return

        cfg = AnalysisConfig(
            video_path=self.video_path,
            pool_length_m=self.pool_length_spin.value(),
            calibration_keyframes=self.calibration_keyframes,
            cap_color=self._get_cap_color(),
            wall_position_m=self.pool_length_spin.value(),
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
        dashboard = DashboardWidget(result, self.pool_length_spin.value())
        self.stack.addWidget(dashboard)
        self.stack.setCurrentWidget(dashboard)

    def _on_finished_error(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Analysis failed", message)
