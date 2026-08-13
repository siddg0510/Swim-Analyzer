from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QDoubleSpinBox, QMessageBox, QRadioButton, QButtonGroup, QListWidgetItem,
    QScrollArea, QSlider, QComboBox, QLineEdit, QGroupBox, QSpinBox, QCheckBox
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt, QPoint

from src.vision.calibration import CalibrationKeyframe
from src.vision.cap_tracker import CapTracker
from src.config import CAP_COLOR_PRESETS

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
        self.mode = "swimmer"
        self._picking_color: bool = False
        self._click_history: list[str] = []

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
            "<b>Step 1 — Swimmer Selection:</b> Pick the cap color (click on cap or select preset), "
            "then draw the lane boundary (4 corners, clockwise). Use 'Preview' to verify tracking.<br>"
            "<b>Step 2 — Reference Points:</b> Click landmarks with known distances.<br>"
            "<b>Step 3 — Save Keyframe:</b> Save and repeat if needed."
        )
        step1_label.setWordWrap(True)
        panel.addWidget(step1_label)

        mode_row = QHBoxLayout()
        self.rb_swimmer = QRadioButton("🏊 Swimmer")
        self.rb_swimmer.setChecked(True)
        self.rb_reference = QRadioButton("📏 Reference points")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_swimmer)
        self.mode_group.addButton(self.rb_reference)
        self.rb_swimmer.toggled.connect(self._mode_changed)
        self.rb_reference.toggled.connect(self._mode_changed)
        mode_row.addWidget(self.rb_swimmer)
        mode_row.addWidget(self.rb_reference)
        panel.addLayout(mode_row)
        
        swim_box = QGroupBox("Swimmer details")
        swim_layout = QVBoxLayout(swim_box)
        
        self.cap_combo = QComboBox()
        self.cap_combo.addItems(CAP_COLOR_PRESETS)
        self.cap_hex_edit = QLineEdit()
        self.cap_hex_edit.setPlaceholderText("#RRGGBB")
        self.cap_hex_edit.setEnabled(False)
        self.cap_combo.currentTextChanged.connect(
            lambda t: self.cap_hex_edit.setEnabled(t == "Custom hex…")
        )
        cap_row = QHBoxLayout()
        cap_row.addWidget(self.cap_combo)
        cap_row.addWidget(self.cap_hex_edit)
        swim_layout.addWidget(QLabel("Cap Color:"))
        swim_layout.addLayout(cap_row)

        self.btn_pick_color = QPushButton("🎨 Pick from image")
        self.btn_pick_color.setCheckable(True)
        self.btn_pick_color.toggled.connect(self._on_pick_color_toggled)
        swim_layout.addWidget(self.btn_pick_color)

        self.lane_spin = QSpinBox()
        self.lane_spin.setRange(1, 10)
        self.lane_spin.setValue(4)
        lane_row = QHBoxLayout()
        lane_row.addWidget(QLabel("Lane number:"))
        lane_row.addWidget(self.lane_spin)
        swim_layout.addLayout(lane_row)
        
        self.preview_check = QCheckBox("Preview Tracking")
        self.preview_check.toggled.connect(self._on_preview_toggled)
        swim_layout.addWidget(self.preview_check)
        
        panel.addWidget(swim_box)

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

    def reject(self) -> None:
        self.cap.release()
        super().reject()

    def _on_pick_color_toggled(self, checked: bool) -> None:
        self._picking_color = checked
        self._redraw()

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
        
    def set_default_cap_color(self, color: str) -> None:
        if color.startswith("#"):
            self.cap_combo.setCurrentText("Custom hex…")
            self.cap_hex_edit.setText(color)
        else:
            idx = self.cap_combo.findText(color)
            if idx >= 0:
                self.cap_combo.setCurrentIndex(idx)

    def _get_cap_color(self) -> str:
        choice = self.cap_combo.currentText()
        if choice == "Custom hex…":
            return self.cap_hex_edit.text().strip() or "#FFFF00"
        return choice

    def _save_keyframe(self) -> None:
        if len(self.reference_points) < 2:
            QMessageBox.warning(self, "Not enough points", "Add at least 2 reference points for a keyframe.")
            return
            
        if len(self.lane_polygon) < 3:
            reply = QMessageBox.question(
                self, "No Lane Boundary",
                "No lane boundary drawn. Tracking may pick up swimmers from adjacent lanes. Save anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        color = self._get_cap_color()
        lane_num = self.lane_spin.value()
            
        kf = CalibrationKeyframe(
            frame_idx=self.current_frame_idx,
            reference_points=list(self.reference_points),
            lane_polygon_px=list(self.lane_polygon),
            cap_color=color,
            lane_number=lane_num
        )
        
        # If preview is on, validate tracking
        if self.preview_check.isChecked() and len(self.lane_polygon) >= 3:
            tracker = CapTracker(color, self.lane_polygon)
            pos = tracker.detect(self.frame)
            if not pos:
                reply = QMessageBox.question(
                    self, "No Swimmer Detected", 
                    "No swimmer detected with this cap color in this lane. Save anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return

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
        self.mode = "swimmer" if self.rb_swimmer.isChecked() else "reference"
        self._redraw()

    def _on_preview_toggled(self, checked: bool) -> None:
        if checked and len(self.lane_polygon) < 3:
            QMessageBox.information(self, "Preview", "Please draw at least 3 lane boundary corners (Step 1) before previewing tracking.")
            self.preview_check.blockSignals(True)
            self.preview_check.setChecked(False)
            self.preview_check.blockSignals(False)
            return
        self._redraw()

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
            self._click_history.append("ref")
            self.points_list.addItem(
                QListWidgetItem(f"Ref: ({x_orig:.0f}, {y_orig:.0f}) -> {dist:.2f} m")
            )
        else:
            if self._picking_color:
                h, w = self.frame.shape[:2]
                xi, yi = int(x_orig), int(y_orig)
                if 0 <= xi < w and 0 <= yi < h:
                    b, g, r = self.frame[yi, xi]
                    hex_color = f"#{int(r):02X}{int(g):02X}{int(b):02X}"
                    self.cap_combo.setCurrentText("Custom hex…")
                    self.cap_hex_edit.setText(hex_color)
                self.btn_pick_color.setChecked(False)
                self._picking_color = False
                self._click_history.append("cap_pick")
            elif len(self.lane_polygon) < 4:
                self.lane_polygon.append((x_orig, y_orig))
                self._click_history.append("lane")
                self.points_list.addItem(
                    QListWidgetItem(f"Lane: ({x_orig:.0f}, {y_orig:.0f})")
                )
        self._redraw()

    def _undo_last(self) -> None:
        if not self._click_history:
            return
            
        last_type = self._click_history.pop()
        if last_type == "ref" and self.reference_points:
            self.reference_points.pop()
            if self.points_list.count():
                self.points_list.takeItem(self.points_list.count() - 1)
        elif last_type == "lane" and self.lane_polygon:
            self.lane_polygon.pop()
            if self.points_list.count():
                self.points_list.takeItem(self.points_list.count() - 1)
        elif last_type == "cap_pick":
            # Just undone the cap pick action, no list item to remove
            pass
            
        self._redraw()

    def _clear_all(self) -> None:
        self.reference_points.clear()
        self.lane_polygon.clear()
        self.points_list.clear()
        self._click_history.clear()
        self._picking_color = False
        if hasattr(self, 'btn_pick_color'):
            self.btn_pick_color.setChecked(False)
        self._redraw()

    def _redraw(self) -> None:
        if not hasattr(self, "base_pixmap"):
            return
            
        if self.preview_check.isChecked() and len(self.lane_polygon) >= 3:
            color = self._get_cap_color()
            tracker = CapTracker(color, self.lane_polygon)
            overlay = self.frame.copy()
            
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (low, high) in tracker.hsv_ranges:
                mask_total |= cv2.inRange(hsv, low, high)
            
            lane_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv2.fillPoly(lane_mask, [np.array(self.lane_polygon, dtype=np.int32)], 255)
            
            green_mask = cv2.bitwise_and(mask_total, lane_mask)
            red_mask = cv2.bitwise_and(mask_total, cv2.bitwise_not(lane_mask))
            
            overlay[green_mask > 0] = [0, 255, 0]
            overlay[red_mask > 0] = [0, 0, 255]
            
            blended = cv2.addWeighted(self.frame, 0.7, overlay, 0.3, 0)
            
            pos = tracker.detect(self.frame)
            if pos:
                cx, cy = int(pos[0]), int(pos[1])
                cv2.drawMarker(blended, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                
            pixmap = _cv_frame_to_qpixmap(blended)
            if pixmap.width() > 760:
                pixmap = pixmap.scaledToWidth(760, Qt.TransformationMode.SmoothTransformation)
        else:
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
            
        painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        painter.setPen(QColor("yellow"))
        if self.mode == "reference":
            if not self.reference_points:
                painter.drawText(20, 40, "Step 2: Click a reference point")
            else:
                painter.drawText(20, 40, f"{len(self.reference_points)} reference points placed.")
        elif self.mode == "swimmer":
            if self._picking_color:
                painter.drawText(20, 40, "Step 1: Click the swimmer's cap to pick color")
            elif len(self.lane_polygon) < 4:
                painter.setPen(QColor("cyan"))
                painter.drawText(20, 40, f"Step 1: Click lane corner {len(self.lane_polygon) + 1}/4 (clockwise)")
            else:
                painter.setPen(QColor("green"))
                painter.drawText(20, 40, "Lane defined ✓ (Switch to Reference points mode)")
            
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
