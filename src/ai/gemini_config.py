"""
Gemini API configuration, key management, and client initialization.

The API key is resolved in this order:
  1. Environment variable  GEMINI_API_KEY
  2. Local config file     ~/.swim_analyzer/gemini_key.txt
  3. Passed explicitly at runtime (e.g. via the GUI settings dialog)

The key is NEVER committed to source control. The local file approach is
a convenience for desktop users who don't want to set env vars; the file
is plaintext because this is a local desktop app, not a server — the
threat model is the same as any other credential on disk.
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
RETRY_DELAY_S = 2.0
# Gemini charges per-token; keeping clips short saves cost and latency.
MAX_CLIP_DURATION_S = 30
# Auto-expiry on Gemini Files API is 48 h; we clean up after analysis anyway.
FILE_EXPIRY_HOURS = 48


@dataclass
class GeminiConfig:
    """Runtime-immutable configuration for Gemini API access."""
    api_key: str
    model: str = DEFAULT_MODEL
    max_retries: int = MAX_RETRIES
    retry_delay_s: float = RETRY_DELAY_S
    max_clip_duration_s: int = MAX_CLIP_DURATION_S


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path.home() / ".swim_analyzer"
_KEY_FILE = _CONFIG_DIR / "gemini_key.txt"


def _read_key_file() -> Optional[str]:
    if _KEY_FILE.exists():
        text = _KEY_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    return None


def save_api_key(key: str) -> None:
    """Persist the API key to the local config file."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _KEY_FILE.write_text(key.strip(), encoding="utf-8")
    logger.info("API key saved to %s", _KEY_FILE)


def resolve_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Return the best available API key, or None if none is configured."""
    if explicit_key:
        return explicit_key.strip()
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    return _read_key_file()


def has_api_key() -> bool:
    return resolve_api_key() is not None


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------
def get_gemini_client(api_key: Optional[str] = None):
    """Return a configured ``google.genai.Client`` instance.

    Raises ``RuntimeError`` if no API key is available, and
    ``ImportError`` if ``google-genai`` is not installed.
    """
    key = resolve_api_key(api_key)
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Set the GEMINI_API_KEY environment "
            "variable, save a key via the app settings, or pass one explicitly."
        )
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai package not installed. Run: pip install google-genai"
        )
    return genai.Client(api_key=key)


def build_config(api_key: Optional[str] = None, model: Optional[str] = None) -> GeminiConfig:
    """Build a ``GeminiConfig`` from the best available sources."""
    key = resolve_api_key(api_key)
    if not key:
        raise RuntimeError("No Gemini API key available.")
    return GeminiConfig(
        api_key=key,
        model=model or DEFAULT_MODEL,
    )
