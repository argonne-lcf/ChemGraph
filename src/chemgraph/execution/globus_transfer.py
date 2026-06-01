"""Globus Transfer file-staging manager.

Transfers files between a local Globus collection and a remote HPC
collection using the `Globus Transfer API
<https://docs.globus.org/api/transfer/>`_.  This avoids encoding large
input files (e.g. atomic structures) inside Globus Compute function
payloads.

**Prerequisites**

1. Install ``globus_sdk`` (already a core dependency).
2. Have *Globus Connect Personal* running on the submitting machine
   **or** use a managed Globus endpoint.
3. Configure endpoint IDs and base path in ``config.toml``::

       [execution.globus_transfer]
       source_endpoint_id = "<local-collection-uuid>"
       destination_endpoint_id = "<hpc-collection-uuid>"
       destination_base_path = "/eagle/projects/MyProject/staging"
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Globus Transfer API scope
TRANSFER_SCOPE = "urn:globus:auth:scope:transfer.api.globus.org:all"

# Default Globus native-app client ID (Globus Tutorial client).
# Projects should register their own app at https://app.globus.org.
_DEFAULT_CLIENT_ID = "61338d24-54d5-408f-a10d-66c06b59f6d2"


@dataclass
class TransferResult:
    """Metadata returned after submitting a Globus Transfer task."""

    task_id: str
    source_endpoint_id: str
    destination_endpoint_id: str
    file_mapping: dict[str, str]  # local_path -> remote_path
    remote_directory: str
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    label: str = ""


class GlobusTransferManager:
    """Manage file transfers between local and remote Globus collections.

    Parameters
    ----------
    source_endpoint_id : str
        UUID of the Globus collection on the submitting machine.
    destination_endpoint_id : str
        UUID of the Globus collection on the HPC system.
    destination_base_path : str
        Root directory on the destination where staged files are placed.
        Each transfer batch creates a subdirectory underneath.
    source_base_path : str, optional
        If provided, local paths are resolved relative to this directory.
    client_id : str, optional
        Globus app client ID for OAuth.  Defaults to the Globus Tutorial
        client.
    """

    def __init__(
        self,
        source_endpoint_id: str,
        destination_endpoint_id: str,
        destination_base_path: str,
        source_base_path: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> None:
        self.source_endpoint_id = source_endpoint_id
        self.destination_endpoint_id = destination_endpoint_id
        self.destination_base_path = destination_base_path.rstrip("/")
        self.source_base_path = source_base_path
        self._client_id = client_id or _DEFAULT_CLIENT_ID
        self._transfer_client = None

    # ── authentication ──────────────────────────────────────────────────

    def _get_transfer_client(self):
        """Lazily create an authenticated ``TransferClient``."""
        if self._transfer_client is not None:
            return self._transfer_client

        try:
            import globus_sdk
        except ImportError as exc:
            raise ImportError(
                "globus_sdk is required for Globus Transfer. "
                "Install it with: pip install globus-sdk"
            ) from exc

        client = globus_sdk.NativeAppAuthClient(self._client_id)
        client.oauth2_start_flow(
            requested_scopes=TRANSFER_SCOPE,
            refresh_tokens=True,
        )

        # Try loading cached tokens first
        token_file = (
            Path.home() / ".globus" / "chemgraph_transfer_tokens.json"
        )
        tokens = self._load_tokens(token_file)

        if tokens is None:
            # Interactive login required
            authorize_url = client.oauth2_get_authorize_url()
            logger.info(
                "Globus Transfer authentication required.\n"
                "Go to this URL and login:\n  %s",
                authorize_url,
            )
            print(
                "\nGlobus Transfer authentication required.\n"
                f"Go to this URL and login:\n  {authorize_url}\n"
            )
            auth_code = input("Enter the authorization code: ").strip()
            token_response = client.oauth2_exchange_code_for_tokens(auth_code)
            tokens = token_response.by_resource_server["transfer.api.globus.org"]
            self._save_tokens(token_file, tokens)
        else:
            # Refresh if expired
            if tokens.get("expires_at_seconds", 0) < time.time():
                try:
                    token_response = client.oauth2_refresh_tokens(
                        globus_sdk.RefreshTokenAuthorizer(
                            tokens["refresh_token"], client
                        )
                    )
                    tokens = token_response.by_resource_server[
                        "transfer.api.globus.org"
                    ]
                    self._save_tokens(token_file, tokens)
                except Exception:
                    logger.warning(
                        "Token refresh failed, falling back to existing token."
                    )

        authorizer = globus_sdk.AccessTokenAuthorizer(tokens["access_token"])
        self._transfer_client = globus_sdk.TransferClient(authorizer=authorizer)
        return self._transfer_client

    @staticmethod
    def _load_tokens(path: Path) -> Optional[dict]:
        if not path.is_file():
            return None
        import json

        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def _save_tokens(path: Path, tokens: dict) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(tokens), f, indent=2)
        path.chmod(0o600)

    # ── transfers ───────────────────────────────────────────────────────

    def transfer_files(
        self,
        local_paths: list[str],
        remote_subdir: Optional[str] = None,
        label: Optional[str] = None,
    ) -> TransferResult:
        """Submit a Globus Transfer task to stage files on the remote endpoint.

        Parameters
        ----------
        local_paths : list[str]
            Absolute paths to local files to transfer.
        remote_subdir : str, optional
            Subdirectory name under ``destination_base_path``.  A UUID-based
            name is generated if omitted.
        label : str, optional
            Human-readable label for the transfer task.

        Returns
        -------
        TransferResult
            Metadata including the Globus task ID and local-to-remote
            path mapping.
        """
        import globus_sdk

        tc = self._get_transfer_client()

        if remote_subdir is None:
            remote_subdir = f"batch_{uuid.uuid4().hex[:12]}"

        remote_dir = f"{self.destination_base_path}/{remote_subdir}"
        transfer_label = label or f"ChemGraph file staging ({remote_subdir})"

        tdata = globus_sdk.TransferData(
            tc,
            self.source_endpoint_id,
            self.destination_endpoint_id,
            label=transfer_label,
            sync_level="checksum",
        )

        file_mapping: dict[str, str] = {}
        for local_path in local_paths:
            p = Path(local_path).resolve()
            remote_path = f"{remote_dir}/{p.name}"
            tdata.add_item(str(p), remote_path)
            file_mapping[str(p)] = remote_path

        result = tc.submit_transfer(tdata)
        task_id = result["task_id"]

        logger.info(
            "Globus Transfer submitted: task_id=%s, %d files -> %s",
            task_id,
            len(local_paths),
            remote_dir,
        )

        return TransferResult(
            task_id=task_id,
            source_endpoint_id=self.source_endpoint_id,
            destination_endpoint_id=self.destination_endpoint_id,
            file_mapping=file_mapping,
            remote_directory=remote_dir,
            label=transfer_label,
        )

    def check_transfer_status(self, task_id: str) -> dict[str, Any]:
        """Check the status of a Globus Transfer task.

        Returns
        -------
        dict
            Keys: ``task_id``, ``status``, ``nice_status``, ``bytes_transferred``,
            ``files``, ``files_transferred``.
        """
        tc = self._get_transfer_client()
        task = tc.get_task(task_id)
        return {
            "task_id": task_id,
            "status": task["status"],
            "nice_status": task.get("nice_status", ""),
            "bytes_transferred": task.get("bytes_transferred", 0),
            "files": task.get("files", 0),
            "files_transferred": task.get("files_transferred", 0),
        }

    def wait_for_transfer(
        self,
        task_id: str,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> dict[str, Any]:
        """Block until a transfer completes, fails, or times out.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait (default 300).
        poll_interval : float
            Seconds between status checks (default 5).

        Returns
        -------
        dict
            Final transfer status.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.check_transfer_status(task_id)
            if status["status"] in ("SUCCEEDED", "FAILED"):
                return status
            time.sleep(poll_interval)

        status = self.check_transfer_status(task_id)
        status["timed_out"] = True
        return status

    def list_remote_directory(self, path: str) -> list[dict[str, Any]]:
        """List files in a directory on the destination endpoint.

        Returns
        -------
        list[dict]
            Each dict has ``name``, ``type`` ("file" or "dir"), and ``size``.
        """
        tc = self._get_transfer_client()
        entries = []
        for entry in tc.operation_ls(self.destination_endpoint_id, path=path):
            entries.append(
                {
                    "name": entry["name"],
                    "type": entry["type"],
                    "size": entry.get("size", 0),
                }
            )
        return entries

    def get_remote_path(
        self,
        local_path: str,
        remote_subdir: Optional[str] = None,
    ) -> str:
        """Compute the remote path for a local file."""
        filename = Path(local_path).name
        if remote_subdir:
            return f"{self.destination_base_path}/{remote_subdir}/{filename}"
        return f"{self.destination_base_path}/{filename}"
