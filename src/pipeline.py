"""
Qt-specific wrapper for the shared analysis pipeline.

This module exists solely to provide the QThread-based AnalysisWorker that
the desktop GUI uses. All actual analysis logic lives in src.core.pipeline;
this module just adapts it to Qt's signal/slot mechanism.

The data classes (AnalysisConfig, AnalysisResult, etc.) are re-exported
from src.core.models for backward compatibility with existing GUI code.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

# Re-export data classes for backward compatibility with GUI imports
from .core.models import AnalysisConfig, AnalysisResult, AIAnalysisResult  # noqa: F401
from .core.pipeline import run_analysis


class AnalysisWorker(QThread):
    """Qt thread wrapper around the shared analysis pipeline."""
    progress = Signal(int, str)          # percent, status message
    finished_ok = Signal(object)          # AnalysisResult
    finished_error = Signal(str)

    def __init__(self, cfg: AnalysisConfig):
        super().__init__()
        self.cfg = cfg
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            result = run_analysis(
                self.cfg,
                progress_callback=self._emit_progress,
                cancel_check=lambda: self._cancelled,
            )
            if not self._cancelled:
                self.finished_ok.emit(result)
        except Exception as exc:
            self.finished_error.emit(f"{type(exc).__name__}: {exc}")

    def _emit_progress(self, pct: int, msg: str) -> None:
        self.progress.emit(pct, msg)
