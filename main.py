#!/usr/bin/env python3
"""
Swim Race Analyzer — entry point.

Run directly:      python main.py
Build to .exe/.app: see build.spec and README.md
"""
import sys
import os

_SCRIPTS_PATH = r"C:\Users\SiddG\AppData\Local\Python\pythoncore-3.14-64\Scripts"
if _SCRIPTS_PATH not in sys.path:
    sys.path.append(_SCRIPTS_PATH)
if _SCRIPTS_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + _SCRIPTS_PATH

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from src.gui.main_window import MainWindow
from src.config import resource_path


def main() -> int:
    from pathlib import Path
    models_dir = Path(__file__).parent.resolve() / "models"
    model_path = models_dir / "pose_landmarker.task"
    if not model_path.exists() or model_path.stat().st_size < 1000000:
        print("First run: downloading required pose model. This may take a minute...")
        try:
            from setup_models import download_pose_model
            download_pose_model()
        except Exception as e:
            print(f"Warning: could not download model automatically: {e}")

    app = QApplication(sys.argv)
    app.setApplicationName("Swim Race Analyzer")
    
    # Modern dark theme stylesheet
    app.setStyleSheet("""
        QWidget {
            background-color: #1e1e24;
            color: #f5f6fa;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 14px;
        }
        QPushButton {
            background-color: #00a8ff;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #0097e6;
        }
        QPushButton:disabled {
            background-color: #353b48;
            color: #7f8fa6;
        }
        QGroupBox {
            border: 1px solid #353b48;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 24px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 4px;
            left: 10px;
        }
        QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
            background-color: #2f3640;
            border: 1px solid #353b48;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QTabWidget::pane {
            border: 1px solid #353b48;
            background: #2f3640;
        }
        QTabBar::tab {
            background: #353b48;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #2f3640;
            color: #00a8ff;
            font-weight: bold;
        }
        QTableWidget {
            background-color: #2f3640;
            alternate-background-color: #353b48;
            gridline-color: #1e1e24;
        }
        QHeaderView::section {
            background-color: #1e1e24;
            padding: 4px;
            border: 1px solid #353b48;
            font-weight: bold;
        }
    """)
    
    icon_path = resource_path("assets", "icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
        
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
