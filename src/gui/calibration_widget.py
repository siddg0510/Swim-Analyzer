from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QDoubleSpinBox, QMessageBox, QRadioButton, QButtonGroup, QListWidgetItem,
    QScrollArea, QSlider
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt, QPoint

from src.vision.calibration import CalibrationKeyframe

def _cv_frame_to_qpixmap(frame_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    qimg = qimg.copy()
    qimg.setDevicePixelRatio(1.0)
    return QPixmap.fromImage(qimg)

class ClickableImageLabel(QLabel):
    def __init__(self, on_click):
        super().__init__()
        self._on_click = on_click
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click(event.position().x(), event.position().y())

class CalibrationDialog(QDialog):
    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multi-Frame Pool Calibration")
        self.resize(1100, 750)
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.total_frames = 1000 # Fallback
            
        self.current_frame_idx = 0
        self.display_scale = 1.0
        
        self.keyframes: list[CalibrationKeyframe] = []
        
        # State for current frame
        self.reference_points: list[tuple[tuple[float, float], float]] = []
        self.lane_polygon: list[tuple[float, float]] = []
        self.mode = "reference"

        layout = QHBoxLayout(self)

        # -- Left side: image + slider --
        left_layout = QVBoxLayout()
        
        self.image_label = ClickableImageLabel(self._handle_click)
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(scroll, stretch=1)
        
        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.valueChanged.connect(self._on_slider_change)
        
        self.frame_label = QLabel("Frame 0")
        slider_row.addWidget(self.slider)
        slider_row.addWidget(self.frame_label)
        left_layout.addLayout(slider_row)
        
        layout.addLayout(left_layout, stretch=1)

        # -- Right side: panel --
        panel_widget = QWidget()
        panel_widget.setFixedWidth(350)
        panel = QVBoxLayout(panel_widget)
        panel.setContentsMargins(0, 0, 0, 0)
        
        step1_label = QLabel(
            "<b>Step 1 — Frame Selection.</b> Scrub to a frame where landmarks are visible. "
            "Draw your reference points and lane boundaries. "
            "Then click <b>Save Keyframe</b>."
        )
        step1_label.setWordWrap(True)
        panel.addWidget(step1_label)

        mode_row = QHBoxLayout()
        self.rb_reference = QRadioButton("Reference points")
        self.rb_reference.setChecked(True)
        self.rb_lane = QRadioButton("Lane boundary")
        group = QButtonGroup(self)
        group.addButton(self.rb_reference)
        group.addButton(self.rb_lane)
        self.rb_reference.toggled.connect(self._mode_changed)
        mode_row.addWidget(self.rb_reference)
        mode_row.addWidget(self.rb_lane)
        panel.addLayout(mode_row)

        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(0, 200)
        self.distance_spin.setSuffix(" m")
        self.distance_spin.setDecimals(2)
        panel.addWidget(QLabel("Distance for next point:"))
        panel.addWidget(self.distance_spin)

        self.points_list = QListWidget()
        panel.addWidget(QLabel("Current Frame Points:"))
        panel.addWidget(self.points_list, stretch=1)

        btn_row = QHBoxLayout()
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self._undo_last)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(undo_btn)
        btn_row.addWidget(clear_btn)
        panel.addLayout(btn_row)

        save_kf_btn = QPushButton("Save Keyframe")
        save_kf_btn.setStyleSheet("background-color: #2E8B57; color: white; font-weight: bold; padding: 5px;")
        save_kf_btn.clicked.connect(self._save_keyframe)
        panel.addWidget(save_kf_btn)
        
        self.kf_list = QListWidget()
        panel.addWidget(QLabel("Saved Keyframes:"))
        panel.addWidget(self.kf_list, stretch=1)
        
        kf_btn_row = QHBoxLayout()
        remove_kf_btn = QPushButton("Remove Selected")
        remove_kf_btn.clicked.connect(self._remove_keyframe)
        kf_btn_row.addWidget(remove_kf_btn)
        panel.addLayout(kf_btn_row)

        done_btn = QPushButton("Done")
        done_btn.clicked.connect(self._on_done)
        panel.addWidget(done_btn)

        layout.addWidget(panel_widget)
        
        # Load initial frame
        self._load_frame(0)

    def _load_frame(self, frame_idx: int) -> None:
        self.current_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return
            
        self.frame = frame
        pixmap = _cv_frame_to_qpixmap(self.frame)
        max_w = 760
        if pixmap.width() > max_w:
            self.display_scale = max_w / pixmap.width()
            pixmap = pixmap.scaledToWidth(max_w, Qt.TransformationMode.SmoothTransformation)
        else:
            self.display_scale = 1.0
            
        self.base_pixmap = pixmap
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())
        
        self.frame_label.setText(f"Frame {frame_idx}")
        self._redraw()
        
    def _on_slider_change(self, value: int) -> None:
        self._load_frame(value)
        
    def _save_keyframe(self) -> None:
        if len(self.reference_points) < 2:
            QMessageBox.warning(self, "Not enough points", "Add at least 2 reference points for a keyframe.")
            return
            
        kf = CalibrationKeyframe(
            frame_idx=self.current_frame_idx,
            reference_points=list(self.reference_points),
            lane_polygon_px=list(self.lane_polygon)
        )
        
        # If there's already a keyframe for this exact frame, replace it
        replaced = False
        for i, existing in enumerate(self.keyframes):
            if existing.frame_idx == self.current_frame_idx:
                self.keyframes[i] = kf
                replaced = True
                break
                
        if not replaced:
            self.keyframes.append(kf)
            
        self.keyframes.sort(key=lambda k: k.frame_idx)
        
        self.kf_list.clear()
        for k in self.keyframes:
            self.kf_list.addItem(f"Frame {k.frame_idx}: {len(k.reference_points)} pts, {len(k.lane_polygon_px)} lane pts")
            
        self._clear_all()
        
    def _remove_keyframe(self) -> None:
        row = self.kf_list.currentRow()
        if row >= 0:
            self.kf_list.takeItem(row)
            self.keyframes.pop(row)

    def _mode_changed(self, checked: bool) -> None:
        self.mode = "reference" if self.rb_reference.isChecked() else "lane"

    def _to_original_coords(self, x_display: float, y_display: float) -> tuple[float, float]:
        return x_display / self.display_scale, y_display / self.display_scale

    def _handle_click(self, x_display: float, y_display: float) -> None:
        if not hasattr(self, "base_pixmap"):
            return
        x_display = max(0.0, min(x_display, self.base_pixmap.width() - 1))
        y_display = max(0.0, min(y_display, self.base_pixmap.height() - 1))

        x_orig, y_orig = self._to_original_coords(x_display, y_display)
        if self.mode == "reference":
            dist = self.distance_spin.value()
            self.reference_points.append(((x_orig, y_orig), dist))
            self.points_list.addItem(
                QListWidgetItem(f"Ref: ({x_orig:.0f}, {y_orig:.0f}) -> {dist:.2f} m")
            )
        else:
            self.lane_polygon.append((x_orig, y_orig))
            self.points_list.addItem(
                QListWidgetItem(f"Lane: ({x_orig:.0f}, {y_orig:.0f})")
            )
        self._redraw()

    def _undo_last(self) -> None:
        if self.mode == "reference" and self.reference_points:
            self.reference_points.pop()
        elif self.mode == "lane" and self.lane_polygon:
            self.lane_polygon.pop()
        if self.points_list.count():
            self.points_list.takeItem(self.points_list.count() - 1)
        self._redraw()

    def _clear_all(self) -> None:
        self.reference_points.clear()
        self.lane_polygon.clear()
        self.points_list.clear()
        self._redraw()

    def _redraw(self) -> None:
        if not hasattr(self, "base_pixmap"):
            return
        pixmap = self.base_pixmap.copy()
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor("yellow"), 3))
        for (x, y), _ in self.reference_points:
            xs, ys = x * self.display_scale, y * self.display_scale
            painter.drawEllipse(QPoint(int(xs), int(ys)), 5, 5)
        painter.setPen(QPen(QColor("cyan"), 2))
        pts = [(x * self.display_scale, y * self.display_scale) for x, y in self.lane_polygon]
        for i in range(len(pts)):
            painter.drawEllipse(QPoint(int(pts[i][0]), int(pts[i][1])), 5, 5)
            if i > 0:
                painter.drawLine(
                    int(pts[i - 1][0]), int(pts[i - 1][1]), int(pts[i][0]), int(pts[i][1])
                )
        if len(pts) > 2:
            painter.drawLine(int(pts[-1][0]), int(pts[-1][1]), int(pts[0][0]), int(pts[0][1]))
            
        # Draw Instructions
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.setPen(QColor("yellow"))
        if not self.reference_points:
            painter.drawText(20, 40, "Step 1: Click a reference point")
        else:
            painter.drawText(20, 40, f"{len(self.reference_points)} reference points placed.")
            
        if self.mode == "lane" and len(self.lane_polygon) < 4:
            painter.setPen(QColor("cyan"))
            painter.drawText(20, 70, f"Step 2: Click lane corner {len(self.lane_polygon) + 1}/4 (clockwise)")
            
        painter.end()
        self.image_label.setPixmap(pixmap)

    def _on_done(self) -> None:
        if not self.keyframes:
            QMessageBox.warning(self, "No keyframes", "You must save at least one keyframe before finishing.")
            return
        self.cap.release()
        self.accept()

    def get_calibration_keyframes(self) -> list[CalibrationKeyframe]:
        return self.keyframes
