"""
SwimmerSelectionDialog — Target acquisition UI for swimmer tracking.

The user opens a video → the dialog detects all swimmers in the frame →
draws labelled bounding boxes over the frame → user clicks a box or chooses
their swimmer ID from the dropdown or lane input.
"""
from __future__ import annotations

import logging
import cv2
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QComboBox, QGroupBox, QScrollArea, QSizePolicy, QDialogButtonBox,
    QMessageBox,
)

from ..detection.detector import Track

logger = logging.getLogger(__name__)


def _cv_to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR frame to a QPixmap."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


class _ClickableFrameLabel(QLabel):
    """
    A QLabel that displays a video frame with overlaid bounding boxes and
    emits the clicked pixel coordinate to a callback.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self._tracks: list[dict] = []
        self._selected_id: int | None = None
        self._pixmap_base: QPixmap | None = None
        self._scale: float = 1.0
        self._offset_x: int = 0
        self._offset_y: int = 0
        self._on_click = None

    def set_frame(self, frame_bgr: np.ndarray, tracks: list[Track], selected_id: int | None = None) -> None:
        self._tracks = [{'id': t.id, 'bbox': t.bbox, 'conf': t.conf} for t in tracks]
        self._selected_id = selected_id
        self._pixmap_base = _cv_to_pixmap(frame_bgr)
        self._orig_size = QSize(frame_bgr.shape[1], frame_bgr.shape[0])
        self._update_display()

    def set_selected_id(self, selected_id: int | None) -> None:
        self._selected_id = selected_id
        self._update_display()

    def _update_display(self) -> None:
        if self._pixmap_base is None:
            return
        lw, lh = self.width() or 640, self.height() or 480
        pm = self._pixmap_base.scaled(
            lw, lh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self._scale = pm.width() / self._orig_size.width()
        self._offset_x = (lw - pm.width()) // 2
        self._offset_y = (lh - pm.height()) // 2

        # Draw boxes and IDs
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colors = [
            QColor("#00a8ff"), QColor("#e84118"), QColor("#4cd137"),
            QColor("#fbc531"), QColor("#9c88ff"), QColor("#fd9644"),
        ]
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)

        for i, t in enumerate(self._tracks):
            is_selected = (t['id'] == self._selected_id)
            color = QColor("#2ecc71") if is_selected else colors[i % len(colors)]
            thickness = 4 if is_selected else 2

            x, y, w, h = [v * self._scale for v in t['bbox']]
            pen = QPen(color, thickness)
            painter.setPen(pen)
            painter.drawRect(int(x), int(y), int(w), int(h))

            # ID label badge
            badge_w = max(int(w), 68)
            label_rect = QRect(int(x), int(y) - 22, badge_w, 22)
            painter.fillRect(label_rect, color)
            painter.setPen(QColor("white" if not is_selected else "#1e1e24"))
            label_text = f"★ ID {t['id']}" if is_selected else f"ID {t['id']}"
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label_text)

        painter.end()

        # Canvas container
        canvas = QPixmap(lw, lh)
        canvas.fill(QColor("#1e1e24"))
        cp = QPainter(canvas)
        cp.drawPixmap(self._offset_x, self._offset_y, pm)
        cp.end()

        self.setPixmap(canvas)

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self._on_click is None or self._pixmap_base is None:
            return
        lx = event.position().x() - self._offset_x
        ly = event.position().y() - self._offset_y
        if self._scale > 0:
            img_x = lx / self._scale
            img_y = ly / self._scale
            self._on_click(lx, ly, img_x, img_y)
        super().mousePressEvent(event)


class SwimmerSelectionDialog(QDialog):
    """
    Modal dialog that displays the video frame with detected swimmers.
    The user selects their swimmer by clicking a bounding box, picking
    from the Swimmer ID dropdown, or specifying a lane number.
    """

    def __init__(self, video_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Target Swimmer")
        self.setMinimumSize(850, 620)
        self.video_path = video_path
        self.selected_id: int | None = None
        self.selected_lane: int | None = None

        self._tracks: list[Track] = []
        self._frame: np.ndarray | None = None
        self._frame_w: int = 1920

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Header instructions
        instr = QLabel(
            "<b>Select Target Swimmer</b><br>"
            "<span style='color:#7f8fa6;'>Each detected swimmer is identified with a numbered box. "
            "Click directly on a swimmer or choose their ID from the selector below.</span>"
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)

        # Video frame view
        self._frame_label = _ClickableFrameLabel()
        self._frame_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._frame_label._on_click = self._on_frame_click
        layout.addWidget(self._frame_label, stretch=1)

        # Status text
        self._status = QLabel("Detecting swimmers in video…")
        self._status.setStyleSheet("color: #00a8ff; font-weight: 500;")
        layout.addWidget(self._status)

        # Swimmer & Lane selection control bar
        controls_group = QGroupBox("Choose Target Swimmer")
        ctrl_layout = QHBoxLayout(controls_group)
        ctrl_layout.setContentsMargins(12, 10, 12, 10)
        ctrl_layout.setSpacing(16)

        # Dropdown selection
        ctrl_layout.addWidget(QLabel("<b>Swimmer ID:</b>"))
        self._id_combo = QComboBox()
        self._id_combo.setMinimumWidth(180)
        self._id_combo.currentIndexChanged.connect(self._on_combo_changed)
        ctrl_layout.addWidget(self._id_combo)

        ctrl_layout.addSpacing(20)

        # Lane selection
        ctrl_layout.addWidget(QLabel("<b>Or by Lane:</b>"))
        self._lane_spin = QSpinBox()
        self._lane_spin.setRange(1, 10)
        self._lane_spin.setValue(4)
        ctrl_layout.addWidget(self._lane_spin)

        lane_btn = QPushButton("Select by Lane")
        lane_btn.clicked.connect(self._on_select_by_lane)
        ctrl_layout.addWidget(lane_btn)

        ctrl_layout.addStretch(1)
        layout.addWidget(controls_group)

        # Dialog buttons
        btn_row = QHBoxLayout()
        reload_btn = QPushButton("🔄 Re-detect")
        reload_btn.clicked.connect(self._load_frame)
        btn_row.addWidget(reload_btn)

        btn_row.addStretch(1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._ok_btn = QPushButton("✅ Confirm Swimmer Selection")
        self._ok_btn.setEnabled(False)
        self._ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 8px 18px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:disabled { background-color: #353b48; color: #7f8fa6; }
        """)
        self._ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._ok_btn)

        layout.addLayout(btn_row)

        self._load_frame()

    def _load_frame(self) -> None:
        """Scan video frames to identify swimmers."""
        try:
            from ..detection.detector import build_detector
        except ImportError:
            self._status.setText("⚠️ Detection module unavailable — select lane manually.")
            self._ok_btn.setEnabled(True)
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self._status.setText("❌ Could not open video file.")
            return

        detector = build_detector()
        frame = None
        tracks: list[Track] = []

        # Scan up to 120 frames to find a frame with detected swimmers
        for fi in range(120):
            ok, f = cap.read()
            if not ok:
                break
            t = detector.detect(f)
            if t:
                frame = f
                tracks = t
                break
            if fi == 0:
                frame = f

        cap.release()

        if frame is None:
            self._status.setText("❌ Could not read video frames.")
            return

        self._frame = frame
        self._frame_w = frame.shape[1]
        self._tracks = tracks

        self._id_combo.blockSignals(True)
        self._id_combo.clear()

        if tracks:
            for t in tracks:
                self._id_combo.addItem(f"Swimmer ID {t.id} (conf: {t.conf:.0%})", t.id)
            self._status.setText(
                f"✅ Detected <b>{len(tracks)} swimmer(s)</b>. Click a box or choose an ID below."
            )
            # Select first by default
            self.selected_id = tracks[0].id
            self._id_combo.setCurrentIndex(0)
            self._ok_btn.setEnabled(True)
            self._ok_btn.setText(f"✅ Confirm Swimmer (ID {self.selected_id})")
        else:
            self._status.setText(
                "⚠️ No swimmers auto-detected in first frame. Click on your swimmer to set their target box."
            )
            self._id_combo.addItem("No auto-detections", None)
            self._ok_btn.setEnabled(False)

        self._id_combo.blockSignals(False)
        self._frame_label.set_frame(frame, tracks, self.selected_id)

    def _on_combo_changed(self, idx: int) -> None:
        swimmer_id = self._id_combo.currentData()
        if swimmer_id is not None:
            self.selected_id = int(swimmer_id)
            self.selected_lane = None
            self._status.setText(f"✅ <b>Selected: Swimmer ID {self.selected_id}</b>")
            self._ok_btn.setEnabled(True)
            self._ok_btn.setText(f"✅ Confirm Swimmer (ID {self.selected_id})")
            self._frame_label.set_selected_id(self.selected_id)

    def _on_frame_click(self, _lx, _ly, img_x: float, img_y: float) -> None:
        from ..detection.selector import TargetSelector
        sel = TargetSelector()
        resolved = sel.select_by_click(img_x, img_y, self._tracks)

        if resolved is not None:
            self.selected_id = resolved
            self.selected_lane = None
            self._status.setText(f"✅ <b>Selected: Swimmer ID {resolved}</b> (click Confirm to proceed)")
        else:
            # If user clicked on a swimmer that was not boxed, create a target track
            new_id = (max([t.id for t in self._tracks], default=0)) + 1
            bw, bh = 70.0, 50.0
            bx = max(0.0, img_x - bw / 2)
            by = max(0.0, img_y - bh / 2)
            user_track = Track(id=new_id, bbox=(bx, by, bw, bh), conf=0.95)
            self._tracks.append(user_track)
            self.selected_id = new_id
            self._status.setText(f"✅ <b>Target locked: Swimmer ID {new_id}</b> at clicked point")

            self._id_combo.blockSignals(True)
            self._id_combo.addItem(f"Swimmer ID {new_id} (manual pick)", new_id)
            self._id_combo.blockSignals(False)

        # Sync combo
        self._id_combo.blockSignals(True)
        for i in range(self._id_combo.count()):
            if self._id_combo.itemData(i) == self.selected_id:
                self._id_combo.setCurrentIndex(i)
                break
        self._id_combo.blockSignals(False)

        self._ok_btn.setEnabled(True)
        self._ok_btn.setText(f"✅ Confirm Swimmer (ID {self.selected_id})")
        if self._frame is not None:
            self._frame_label.set_frame(self._frame, self._tracks, self.selected_id)

    def _on_select_by_lane(self) -> None:
        from ..detection.selector import TargetSelector
        lane = self._lane_spin.value()
        sel = TargetSelector()
        resolved = sel.select_by_lane(lane, self._tracks, self._frame_w)

        if resolved is not None:
            self.selected_id = resolved
            self._status.setText(f"✅ <b>Lane {lane} selected → Swimmer ID {resolved}</b>")
            self._id_combo.blockSignals(True)
            for i in range(self._id_combo.count()):
                if self._id_combo.itemData(i) == resolved:
                    self._id_combo.setCurrentIndex(i)
                    break
            self._id_combo.blockSignals(False)
        else:
            self.selected_id = None
            self._status.setText(f"✅ <b>Lane {lane} selected</b> (target resolved at start of race)")

        self.selected_lane = lane
        self._ok_btn.setEnabled(True)
        self._ok_btn.setText(f"✅ Confirm Lane {lane}")
        if self._frame is not None:
            self._frame_label.set_frame(self._frame, self._tracks, self.selected_id)
