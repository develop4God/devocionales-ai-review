"""
OpenAI-compatible batch API client: upload → submit → poll → download.

Lifted near-verbatim from GEP/batch_client.py, with the GEP-specific
`from cloud_client import _load_config` coupling replaced by an injected
BatchProviderConfig. Behavior against the wire is unchanged — same multipart
upload shape, same endpoints, same terminal-status handling — so results
collected by either project stay comparable.

Usage:
    client = BatchClient(cfg)
    file_id  = client.upload(Path("batch_input.jsonl"))
    batch_id = client.submit(file_id)
    out_fid  = client.poll(batch_id)
    path     = client.download(out_fid, Path("results.jsonl"))
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from batch_common.config import BatchAPIError, BatchProviderConfig, api_key_from_env

_TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})


class BatchClient:
    """
    OpenAI-compatible batch client.

    All configuration arrives through the BatchProviderConfig — this class is
    open for extension (a new provider is a new config) and closed for
    modification (no per-provider branch lives here).
    """

    def __init__(self, cfg: BatchProviderConfig) -> None:
        self._cfg = cfg
        self._key = api_key_from_env(cfg)
        self._base = cfg.base_url.rstrip("/")

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def model(self) -> str:
        return self._cfg.model

    @property
    def provider_id(self) -> str:
        return self._cfg.provider_id

    # ── Batch operations ──────────────────────────────────────────────────

    def upload(self, file_path: Path) -> str:
        """
        Upload a JSONL file for batch inference via multipart/form-data.
        Includes the required 'purpose: batch' field. Returns file_id.
        """
        boundary = "BatchCommon01"
        file_bytes = Path(file_path).read_bytes()
        body = (
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="purpose"\r\n\r\n'
                f"batch\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; '
                f'filename="{Path(file_path).name}"\r\n'
                f"Content-Type: application/jsonl\r\n\r\n"
            ).encode()
            + file_bytes
            + f"\r\n--{boundary}--\r\n".encode()
        )

        req = urllib.request.Request(
            f"{self._base}/files",
            data=body,
            method="POST",
        )
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        req.add_header("Authorization", f"Bearer {self._key}")

        resp = self._do(req)
        file_id = resp.get("id") or resp.get("file_id")
        if not file_id:
            raise BatchAPIError(f"Upload succeeded but no file_id in response: {resp}")
        return file_id

    def submit(self, file_id: str) -> str:
        """
        Create a batch job via the OpenAI-compatible POST /batches endpoint.
        Returns batch_id.
        """
        payload = {
            "input_file_id": file_id,
            "endpoint": self._cfg.endpoint,
            "completion_window": self._cfg.completion_window,
        }
        resp = self._post_json(f"{self._base}/batches", payload)
        batch_id = resp.get("id") or resp.get("batch_id")
        if not batch_id:
            raise BatchAPIError(f"Submit succeeded but no batch_id in response: {resp}")
        return batch_id

    def poll(
        self,
        batch_id: str,
        interval: int = 30,
        timeout: int = 86_400,
    ) -> str:
        """
        Poll GET /batches/{batch_id} until status is terminal.
        Returns output_file_id when status == 'completed'.
        Raises BatchAPIError on failed/expired/cancelled.
        Raises TimeoutError if timeout is exceeded.
        """
        url = f"{self._base}/batches/{batch_id}"
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            data = self._get_json(url)
            status = data.get("status", "unknown")
            counts = data.get("request_counts", {})
            print(
                f"    [{status}]  {counts.get('completed', '?')}/"
                f"{counts.get('total', '?')} completed",
                flush=True,
            )

            if status == "completed":
                fid = data.get("output_file_id")
                if not fid:
                    raise BatchAPIError(
                        f"Batch completed but output_file_id is missing: {data}"
                    )
                return fid

            if status in _TERMINAL_STATUSES:
                raise BatchAPIError(f"Batch ended with status='{status}': {data}")

            time.sleep(interval)

        raise TimeoutError(f"Batch polling timed out after {timeout}s")

    def download(self, file_id: str, dest: Path) -> Path:
        """Download file content to dest path. Returns dest."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            f"{self._base}/files/{file_id}/content",
            headers={"Authorization": f"Bearer {self._key}"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            dest.write_bytes(resp.read())
        return dest

    # ── HTTP internals ────────────────────────────────────────────────────

    def _post_json(self, url: str, payload: dict) -> dict:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
            method="POST",
        )
        return self._do(req)

    def _get_json(self, url: str) -> dict:
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self._key}"},
        )
        return self._do(req)

    def _do(self, req: urllib.request.Request) -> dict:
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise BatchAPIError(f"HTTP {e.code}: {body[:600]}") from e
