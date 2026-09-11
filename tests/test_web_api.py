"""
Tests for FastAPI Web Application endpoints.
"""
import io
import json
import pytest
from fastapi.testclient import TestClient

import web.app as web_app
from web.app import app, _cors_config
from web.models import JobStatus

client = TestClient(app)

# A minimal, valid manual calibration (2 reference points) so a request can pass
# the "usable calibration required" guard and exercise later logic (e.g. the
# size cap) without depending on whether a Gemini key exists in the environment.
_VALID_CALIBRATION = json.dumps(
    {"keyframes": [{"reference_points": [[[100, 240], 0.0], [[500, 240], 50.0]]}]}
)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "swim_analyzer"


def test_benchmarks_meta_endpoint():
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "profiles" in data
    assert len(data["profiles"]) >= 8
    assert data["total_benchmarks"] >= 20


def test_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Swim Analyzer" in response.text


def test_job_not_found():
    response = client.get("/api/jobs/non-existent-uuid-123")
    assert response.status_code == 404


def test_upload_invalid_extension():
    # Create fake text file
    fake_file = io.BytesIO(b"dummy text content")
    response = client.post(
        "/api/jobs",
        files={"video": ("test.txt", fake_file, "text/plain")},
        data={"pool_length_m": "50.0"},
    )
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]


def test_csv_not_ready():
    response = client.get("/api/jobs/dummy-id/csv")
    assert response.status_code == 404


def test_upload_too_large_returns_413(monkeypatch):
    """An upload beyond the size cap is rejected with HTTP 413 and leaves no
    trace (job dir cleaned up, no lingering in-memory record)."""
    # Shrink the cap so we don't have to stream 500 MB in a unit test.
    monkeypatch.setattr(web_app.storage, "max_upload_bytes", 1024)  # 1 KiB
    jobs_before = set(web_app.worker.jobs.keys())

    oversized = io.BytesIO(b"\x00" * (1024 * 8))  # 8 KiB > 1 KiB cap
    response = client.post(
        "/api/jobs",
        files={"video": ("big.mp4", oversized, "video/mp4")},
        data={"pool_length_m": "50.0", "calibration_json": _VALID_CALIBRATION},
    )

    assert response.status_code == 413
    assert "MB" in response.json()["detail"]
    # Rolled back: no new job record survived the rejection.
    assert set(web_app.worker.jobs.keys()) == jobs_before


def test_missing_calibration_with_ai_disabled_returns_400():
    """AI off + no manual calibration = a doomed job. It must be rejected early
    with a clear, actionable message (product decision: graceful fail)."""
    fake = io.BytesIO(b"\x00" * 512)
    response = client.post(
        "/api/jobs",
        files={"video": ("race.mp4", fake, "video/mp4")},
        data={"pool_length_m": "50.0", "enable_ai": "false"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "calibration" in detail
    assert "reference point" in detail


def test_manual_calibration_bypasses_ai_requirement(monkeypatch):
    """AI off but valid manual calibration supplied → the calibration guard must
    pass (the job proceeds far enough to be accepted / fail later on content)."""
    # Force "no key" so we know it's the manual calibration, not AI, satisfying
    # the guard.
    monkeypatch.setattr(web_app, "resolve_api_key", lambda: None)
    tiny = io.BytesIO(b"\x00" * 256)
    response = client.post(
        "/api/jobs",
        files={"video": ("race.mp4", tiny, "video/mp4")},
        data={
            "pool_length_m": "50.0",
            "enable_ai": "false",
            "calibration_json": _VALID_CALIBRATION,
        },
    )
    # Accepted (201) — the guard passed. (The background job may later fail on the
    # bogus video, but that's a processing concern, not an intake rejection.)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == JobStatus.QUEUED
    # Clean up the job we just created so we don't leak background work.
    client.delete(f"/api/jobs/{body['job_id']}")


def test_invalid_calibration_json_returns_400():
    """Malformed calibration JSON is a 400 client error, never a 500 crash."""
    fake = io.BytesIO(b"\x00" * 256)
    response = client.post(
        "/api/jobs",
        files={"video": ("race.mp4", fake, "video/mp4")},
        data={"pool_length_m": "50.0", "calibration_json": "{not valid json"},
    )
    assert response.status_code == 400
    assert "calibration" in response.json()["detail"].lower()


def test_cors_config_default_disables_credentials(monkeypatch):
    """Default (no env): wildcard origin with credentials DISABLED — the only
    valid wildcard CORS combination (wildcard + credentials is spec-invalid)."""
    monkeypatch.delenv("SWIM_ANALYZER_CORS_ORIGINS", raising=False)
    origins, credentials = _cors_config()
    assert origins == ["*"]
    assert credentials is False


def test_cors_config_explicit_origins_enable_credentials(monkeypatch):
    """Explicit origins: echoed back with credentials enabled."""
    monkeypatch.setenv(
        "SWIM_ANALYZER_CORS_ORIGINS", "https://a.example, https://b.example"
    )
    origins, credentials = _cors_config()
    assert origins == ["https://a.example", "https://b.example"]
    assert credentials is True
