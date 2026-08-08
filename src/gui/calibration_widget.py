"""
Calibration dialog (spec section 3: "quick manual calibration overlay").

Two click modes on the same first-frame image:
  1. Reference points: click a spot, type the real-world distance (in
     metres, measured from the start wall along the lane) it corresponds
     to — lane rope knots, backstroke flags, T-marks, or simply both end
     walls if nothing else is visible. 4+ points (not all in a line) let
     PoolCalibrator build a full perspective-corrected homography; 2-3
     points fall back to a simpler linear along-lane mapping.
  2. Lane polygon: click the corners of the swimmer's lane so cap
     tracking can ignore neighbouring lanes.

Bug fixed here: clicked points were landing at the wrong spot on the
image. Root cause was two separate issues that both shift click
coordinates away from the image:
  1. The image QLabel sat in a QHBoxLayout with a stretch factor, which
     lets Qt grow the label larger than the actual pixmap inside it
     (e.g. on window resize). event.position() is measured against the
     *label*, not the pixmap, so once the two sizes diverge, converting
     click position -> image pixel using only a single scale factor
     produces an increasingly wrong answer.
  2. On any display with OS-level UI scaling above 100% (the Windows
     default on most laptops, and the Mac Retina default), Qt's
     QImage/QPixmap can apply an implicit device-pixel-ratio scale on
     top of whatever this code already did, double-scaling the
     coordinates.
Fix: pin the label to the pixmap's exact size (so layout stretch can't
touch it), force top-left alignment explicitly instead of relying on
QLabel's default, and force devicePixelRatio to 1.0 on the QImage so Qt
doesn't apply a second, invisible scale on top of this code's own.
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QDoubleSpinBox, QMessageBox, QRadioButton, QButtonGroup, QListWidgetItem,
    QScrollArea
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt, QPoint


def _cv_frame_to_qpixmap(frame_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    qimg = qimg.copy()
    # Force 1:1 — without this, Qt can silently apply the screen's DPI
    # scale factor on top of ours, so a click at logical position (x, y)
    # no longer corresponds to image pixel (x, y) even after our own
    # scale-factor math. This was the main cause of clicks landing on
    # the wrong spot.
    qimg.setDevicePixelRatio(1.0)
    return QPixmap.fromImage(qimg)


class ClickableImageLabel(QLabel):
    def __init__(self, on_click):
        super().__init__()
        self._on_click = on_click
        self.setMouseTracking(True)
        # Top-left alignment, explicitly — QLabel's default can vertically
        # centre a pixmap smaller than the label, which silently shifts
        # every click's y-coordinate if the label is ever taller than the
        # image (e.g. from layout stretch or the side panel forcing a
        # minimum dialog height).
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click(event.position().x(), event.position().y())


class CalibrationDialog(QDialog):
    def __init__(self, first_frame_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pool Calibration")
        self.resize(1000, 700)

        self.frame = first_frame_bgr
        self.display_scale = 1.0
        self.reference_points: list[tuple[tuple[float, float], float]] = []
        self.lane_polygon: list[tuple[float, float]] = []
        self.mode = "reference"  # or "lane"

        layout = QHBoxLayout(self)

        # -- image + click handling --
        self.image_label = ClickableImageLabel(self._handle_click)
        pixmap = _cv_frame_to_qpixmap(self.frame)
        max_w = 760
        if pixmap.width() > max_w:
            self.display_scale = max_w / pixmap.width()
            pixmap = pixmap.scaledToWidth(max_w, Qt.TransformationMode.SmoothTransformation)
        self.base_pixmap = pixmap
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())
        
        # Wrap image in a scroll area to handle small laptop screens
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(scroll, stretch=1)

        # -- side panel --
        panel_widget = QWidget()
        panel_widget.setFixedWidth(350)
        panel = QVBoxLayout(panel_widget)
        panel.setContentsMargins(0, 0, 0, 0)
        
        step1_label = QLabel(
            "<b>Step 1 — Reference points.</b> Click a visible landmark "
            "(lane-rope knot, flag, T-mark, wall) then enter the distance "
            "in metres from the start wall along the lane. Add 4+ "
            "non-collinear points for the best accuracy."
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
        panel.addWidget(QLabel("Distance for next clicked point:"))
        panel.addWidget(self.distance_spin)

        self.points_list = QListWidget()
        panel.addWidget(self.points_list, stretch=1)

        btn_row = QHBoxLayout()
        undo_btn = QPushButton("Undo last point")
        undo_btn.clicked.connect(self._undo_last)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(undo_btn)
        btn_row.addWidget(clear_btn)
        panel.addLayout(btn_row)

        step2_label = QLabel(
            "<b>Step 2 — Lane boundary.</b> Switch to 'Lane boundary', "
            "click the 4 corners of the target lane (clockwise), then "
            "press Done."
        )
        step2_label.setWordWrap(True)
        panel.addWidget(step2_label)

        done_btn = QPushButton("Done")
        done_btn.clicked.connect(self._on_done)
        panel.addWidget(done_btn)

        layout.addWidget(panel_widget)

    def _mode_changed(self, checked: bool) -> None:
        self.mode = "reference" if self.rb_reference.isChecked() else "lane"

    def _to_original_coords(self, x_display: float, y_display: float) -> tuple[float, float]:
        return x_display / self.display_scale, y_display / self.display_scale

    def _handle_click(self, x_display: float, y_display: float) -> None:
        # Defensive clamp: with the label now fixed-size, this should
        # never fire outside [0, pixmap size), but clamp anyway rather
        # than silently accept an out-of-frame coordinate if some
        # platform still delivers a stray click during a resize.
        x_display = max(0.0, min(x_display, self.base_pixmap.width() - 1))
        y_display = max(0.0, min(y_display, self.base_pixmap.height() - 1))

        x_orig, y_orig = self._to_original_coords(x_display, y_display)
        if self.mode == "reference":
            dist = self.distance_spin.value()
            self.reference_points.append(((x_orig, y_orig), dist))
            self.points_list.addItem(
                QListWidgetItem(f"Reference: ({x_orig:.0f}, {y_orig:.0f}) -> {dist:.2f} m")
            )
        else:
            self.lane_polygon.append((x_orig, y_orig))
            self.points_list.addItem(
                QListWidgetItem(f"Lane corner: ({x_orig:.0f}, {y_orig:.0f})")
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
            painter.drawText(20, 40, "Step 1: Click the start wall (set distance to 0m)")
        elif len(self.reference_points) == 1:
            painter.drawText(20, 40, "Step 1: Click the finish wall (set distance to e.g. 50m)")
        else:
            painter.drawText(20, 40, f"{len(self.reference_points)} reference points placed.")
            
        if self.mode == "lane" and len(self.lane_polygon) < 4:
            painter.setPen(QColor("cyan"))
            painter.drawText(20, 70, f"Step 2: Click lane corner {len(self.lane_polygon) + 1}/4 (clockwise)")
            
        painter.end()
        self.image_label.setPixmap(pixmap)

    def _on_done(self) -> None:
        if len(self.reference_points) < 2:
            QMessageBox.warning(self, "Not enough points",
                                 "Add at least 2 reference points (4+ recommended).")
            return
        self.accept()

    def get_calibration_data(self):
        return self.reference_points, self.lane_polygon
