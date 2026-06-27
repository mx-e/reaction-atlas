"""Pluggable checkpoint backends for the kinetics solver loop.

Each backend implements two operations:
- ``scan_last_count()``: returns the highest compound count for which a
  prior checkpoint exists. Used on API restart so the in-memory counter
  resumes from the real filesystem state instead of re-aligning to the
  current milestone.
- ``trigger_checkpoint(n_compounds)``: kick off (or perform) a backup
  identifying this milestone. May be async (cloud export) or sync
  (local ``pg_dump``).

Selected via the ``CHECKPOINT_BACKEND`` env var: ``gcs`` | ``local`` |
``noop`` (default). The ``gcs`` and ``local`` backends each take their
own configuration env vars; misconfiguration falls back to noop with a
loud log so deployments don't silently lose backups.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from loguru import logger


class CheckpointBackend(ABC):
    """Interface every checkpoint storage backend implements."""

    name: str

    @abstractmethod
    def scan_last_count(self) -> int:
        """Highest compound count seen across existing checkpoint artifacts."""

    @abstractmethod
    def trigger_checkpoint(self, n_compounds: int) -> tuple[bool, str]:
        """Kick off a checkpoint for the given compound count.

        Returns ``(success, status_message)``. The status message is used
        for logging and Telegram notifications.
        """


class NoopCheckpointBackend(CheckpointBackend):
    """Disable checkpoints entirely. Selected when CHECKPOINT_BACKEND is
    unset or set to 'noop'. Useful for dev / on-prem deployments that
    rely on external Postgres backups (pg_basebackup / WAL shipping)."""

    name = "noop"

    def scan_last_count(self) -> int:
        return 0

    def trigger_checkpoint(self, n_compounds: int) -> tuple[bool, str]:
        return True, "noop"


class GcsCheckpointBackend(CheckpointBackend):
    """Triggers Cloud SQL exports to a GCS bucket via the Cloud SQL Admin
    REST API. Requires ``google-auth`` plus ambient GCP credentials
    (Cloud Run metadata server / sa-key.json / gcloud-cli)."""

    name = "gcs"

    def __init__(
        self,
        bucket: str,
        project_id: str,
        sql_instance: str,
        db_name: str,
    ):
        self.bucket = bucket
        self.project_id = project_id
        self.sql_instance = sql_instance
        self.db_name = db_name

    def _bearer_token(self) -> Optional[str]:
        try:
            import google.auth
            import google.auth.transport.requests
            credentials, _ = google.auth.default()
            credentials.refresh(google.auth.transport.requests.Request())
            return credentials.token
        except Exception as e:
            logger.warning(f"checkpoint(gcs): cannot obtain auth token: {e}")
            return None

    def scan_last_count(self) -> int:
        import requests as _requests
        token = self._bearer_token()
        if token is None:
            return 0

        url = f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o"
        max_count = 0
        page_token: Optional[str] = None
        try:
            while True:
                params = {"prefix": "checkpoint_", "fields": "items(name),nextPageToken"}
                if page_token:
                    params["pageToken"] = page_token
                resp = _requests.get(
                    url, params=params,
                    headers={"Authorization": f"Bearer {token}"}, timeout=15,
                )
                if resp.status_code != 200:
                    logger.warning(
                        f"checkpoint(gcs): list HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    return 0
                data = resp.json()
                for item in data.get("items", []):
                    m = re.match(r"checkpoint_(\d+)\.sql(?:\.gz)?$", item.get("name", ""))
                    if m:
                        max_count = max(max_count, int(m.group(1)))
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
        except Exception as e:
            logger.warning(f"checkpoint(gcs): scan error: {e}")
            return 0
        return max_count

    def trigger_checkpoint(self, n_compounds: int) -> tuple[bool, str]:
        import requests as _requests
        token = self._bearer_token()
        if token is None:
            return False, "checkpoint(gcs): auth token unavailable"

        uri = f"gs://{self.bucket}/checkpoint_{n_compounds}.sql"
        api_url = (
            f"https://sqladmin.googleapis.com/v1/projects/{self.project_id}"
            f"/instances/{self.sql_instance}/export"
        )
        body = {
            "exportContext": {
                "fileType": "SQL",
                "uri": uri,
                "databases": [self.db_name],
            }
        }
        try:
            resp = _requests.post(
                api_url, json=body,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
        except Exception as e:
            return False, f"checkpoint(gcs): error: {e}"

        if resp.status_code in (200, 202):
            return True, f"checkpoint(gcs): {uri}"
        return False, (
            f"checkpoint(gcs): HTTP {resp.status_code}: {resp.text[:300]}"
        )


class LocalPgDumpBackend(CheckpointBackend):
    """Runs ``pg_dump`` against ``DATABASE_URL`` and writes
    ``checkpoint_<n>.sql.gz`` into ``CHECKPOINT_LOCAL_DIR``. Synchronous —
    blocks the kinetics loop poll for the duration of the dump (typically
    seconds for small DBs, minutes for tens-of-GB). Acceptable because
    checkpointing already runs at most once per N compounds added.

    If ``pg_dump`` is not on PATH (slim API images often omit it), this
    backend logs once and degrades to noop. Install ``postgresql-client``
    in the API image to enable."""

    name = "local"

    def __init__(self, output_dir: str, database_url: str):
        self.output_dir = Path(output_dir).resolve()
        self.database_url = database_url
        self._pg_dump_missing_logged = False

    def _have_pg_dump(self) -> bool:
        try:
            subprocess.run(
                ["pg_dump", "--version"], check=True,
                capture_output=True, timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.SubprocessError):
            if not self._pg_dump_missing_logged:
                logger.error(
                    "checkpoint(local): pg_dump not on PATH — install "
                    "postgresql-client in the API image to enable backups"
                )
                self._pg_dump_missing_logged = True
            return False

    def scan_last_count(self) -> int:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"checkpoint(local): cannot create {self.output_dir}: {e}")
            return 0
        max_count = 0
        for p in self.output_dir.iterdir():
            m = re.match(r"checkpoint_(\d+)\.sql(?:\.gz)?$", p.name)
            if m:
                max_count = max(max_count, int(m.group(1)))
        return max_count

    def trigger_checkpoint(self, n_compounds: int) -> tuple[bool, str]:
        if not self._have_pg_dump():
            return False, "checkpoint(local): pg_dump not available"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"checkpoint_{n_compounds}.sql.gz"
        tmp_path = self.output_dir / f".checkpoint_{n_compounds}.sql.gz.partial"

        # pg_dump | gzip > tmp, then rename atomically. Avoids the
        # next-restart scanner picking up a half-written file.
        cmd = (
            f"pg_dump --no-owner --no-acl {shlex.quote(self.database_url)} "
            f"| gzip -1 > {shlex.quote(str(tmp_path))}"
        )
        t0 = time.time()
        try:
            subprocess.run(
                ["bash", "-c", cmd], check=True,
                capture_output=True, timeout=3600,
            )
        except subprocess.CalledProcessError as e:
            tmp_path.unlink(missing_ok=True)
            err = (e.stderr or b"").decode("utf-8", errors="replace")[:500]
            return False, f"checkpoint(local): pg_dump failed: {err}"
        except subprocess.TimeoutExpired:
            tmp_path.unlink(missing_ok=True)
            return False, "checkpoint(local): pg_dump exceeded 1h timeout"

        tmp_path.rename(out_path)
        elapsed = time.time() - t0
        size_mb = out_path.stat().st_size / (1024 * 1024)
        return True, (
            f"checkpoint(local): {out_path} ({size_mb:.1f} MB, {elapsed:.0f}s)"
        )


def select_backend() -> CheckpointBackend:
    """Build the configured backend from env vars. Logs the choice."""
    kind = os.environ.get("CHECKPOINT_BACKEND", "noop").strip().lower()

    if kind == "gcs":
        bucket = os.environ.get("CHECKPOINT_GCS_BUCKET", "")
        project = os.environ.get("GCP_PROJECT_ID", "")
        sql_instance = os.environ.get("CHECKPOINT_SQL_INSTANCE", "")
        db_name = os.environ.get("CHECKPOINT_DB_NAME", "")
        if not all((bucket, project, sql_instance, db_name)):
            logger.error(
                "checkpoint: CHECKPOINT_BACKEND=gcs but one of "
                "CHECKPOINT_GCS_BUCKET / GCP_PROJECT_ID / "
                "CHECKPOINT_SQL_INSTANCE / CHECKPOINT_DB_NAME is unset — "
                "falling back to noop"
            )
            return NoopCheckpointBackend()
        backend = GcsCheckpointBackend(bucket, project, sql_instance, db_name)
        logger.info(
            f"checkpoint: backend=gcs bucket={bucket} instance={sql_instance}"
        )
        return backend

    if kind == "local":
        out_dir = os.environ.get("CHECKPOINT_LOCAL_DIR", "")
        db_url = os.environ.get("DATABASE_URL", "")
        if not out_dir:
            logger.error(
                "checkpoint: CHECKPOINT_BACKEND=local but "
                "CHECKPOINT_LOCAL_DIR is unset — falling back to noop"
            )
            return NoopCheckpointBackend()
        if not db_url:
            logger.error(
                "checkpoint: CHECKPOINT_BACKEND=local but DATABASE_URL is "
                "unset — falling back to noop"
            )
            return NoopCheckpointBackend()
        backend = LocalPgDumpBackend(out_dir, db_url)
        logger.info(f"checkpoint: backend=local dir={out_dir}")
        return backend

    if kind != "noop":
        logger.warning(
            f"checkpoint: unknown CHECKPOINT_BACKEND={kind!r}; using noop"
        )
    logger.info("checkpoint: backend=noop (no DB checkpoints)")
    return NoopCheckpointBackend()
