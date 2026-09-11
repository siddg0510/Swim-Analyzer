"""
Storage manager for uploaded videos, job artifacts, and CSV exports.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Default upload directory
UPLOAD_DIR = Path(os.environ.get("SWIM_ANALYZER_UPLOAD_DIR", Path.cwd() / "data" / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Default TTL: 24 hours (86400 seconds)
DEFAULT_RETENTION_SECONDS = int(os.environ.get("SWIM_ANALYZER_RETENTION_HOURS", 24)) * 3600

# Maximum accepted upload size. Race videos are large but not unbounded; a hard
# cap protects a multi-user backend from a single huge (or malicious) upload
# filling the disk. Env-configurable; default 500 MB (product decision).
DEFAULT_MAX_UPLOAD_BYTES = int(os.environ.get("SWIM_ANALYZER_MAX_UPLOAD_MB", 500)) * 1024 * 1024

# Chunk size for streaming an upload to disk (8 MiB).
_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024


class UploadTooLargeError(Exception):
    """Raised when an upload exceeds the configured size cap.

    Carries the limit and the number of bytes seen before aborting so the
    caller can build a precise HTTP 413 message.
    """

    def __init__(self, limit_bytes: int, seen_bytes: int):
        self.limit_bytes = limit_bytes
        self.seen_bytes = seen_bytes
        super().__init__(
            f"Upload exceeds the {limit_bytes / (1024 * 1024):.0f} MB limit."
        )


class StorageManager:
    """Manages file storage and periodic cleanup for web jobs."""

    def __init__(
        self,
        base_dir: Path = UPLOAD_DIR,
        ttl_seconds: int = DEFAULT_RETENTION_SECONDS,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ):
        self.base_dir = base_dir
        self.ttl_seconds = ttl_seconds
        self.max_upload_bytes = max_upload_bytes
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_job_dir(self, job_id: str) -> Path:
        """Get or create directory for a specific job."""
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def save_upload(self, job_id: str, filename: str, file_obj) -> Path:
        """Save an uploaded file to the job directory, enforcing the size cap.

        Streams the upload to disk in chunks (never buffering the whole file in
        memory) while counting bytes. If the cap is exceeded the partial file is
        deleted and :class:`UploadTooLargeError` is raised — this is the
        authoritative enforcement point, robust even when the client lies about
        or omits its Content-Length header.
        """
        job_dir = self.get_job_dir(job_id)
        suffix = Path(filename).suffix.lower()
        if not suffix:
            suffix = ".mp4"
        dest_path = job_dir / f"video{suffix}"

        written = 0
        try:
            with open(dest_path, "wb") as buffer:
                while True:
                    chunk = file_obj.read(_UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > self.max_upload_bytes:
                        buffer.close()
                        dest_path.unlink(missing_ok=True)
                        raise UploadTooLargeError(self.max_upload_bytes, written)
                    buffer.write(chunk)
        except UploadTooLargeError:
            raise
        logger.info("Saved video upload for job %s to %s (%d bytes)", job_id, dest_path, written)
        return dest_path

    def get_video_path(self, job_id: str) -> Path | None:
        """Find video file in job directory."""
        job_dir = self.base_dir / job_id
        if not job_dir.exists():
            return None
        for file in job_dir.iterdir():
            if file.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                return file
        return None

    def get_csv_path(self, job_id: str) -> Path:
        """Path for the job's CSV report."""
        job_dir = self.get_job_dir(job_id)
        return job_dir / "report.csv"

    def delete_job_files(self, job_id: str) -> bool:
        """Delete all files associated with a job."""
        job_dir = self.base_dir / job_id
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                logger.info("Cleaned up files for job %s", job_id)
                return True
            except OSError as e:
                logger.warning("Error deleting files for job %s: %s", job_id, e)
                return False
        return False

    def cleanup_expired(self) -> int:
        """Remove jobs older than the retention TTL."""
        now = time.time()
        deleted = 0
        if not self.base_dir.exists():
            return 0
        for item in self.base_dir.iterdir():
            if item.is_dir():
                mtime = item.stat().st_mtime
                if now - mtime > self.ttl_seconds:
                    try:
                        shutil.rmtree(item)
                        deleted += 1
                        logger.info("Auto-cleaned expired job folder: %s", item.name)
                    except OSError as e:
                        logger.warning("Failed to auto-clean %s: %s", item.name, e)
        return deleted
