"""
Shared analysis core — no GUI dependencies.

This package contains the analysis pipeline logic shared by both the desktop
app (PySide6) and the web backend (FastAPI). Everything in this package must
be importable without PySide6 installed.

Public API:
    from src.core.models import AnalysisConfig, AnalysisResult, AIAnalysisResult
    from src.core.pipeline import run_analysis
    from src.core.labels import METRIC_LABELS
"""
from .models import AnalysisConfig, AnalysisResult, AIAnalysisResult
from .pipeline import run_analysis
