"""
Tests for FastAPI Web Application endpoints.
"""
import io
import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.models import JobStatus

client = TestClient(app)


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
