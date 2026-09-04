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
       destination_base_path = "/eagle/MyProject/staging"
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Globus Transfer API scope
TRANSFER_SCOPE = "urn:globus:auth:scope:transfer.api.globus.org:all"
TRANSFER_RESOURCE_SERVER = "transfer.api.globus.org"

# Default Globus native-app client ID (Globus Tutorial client).
# Projects should register their own app at https://app.globus.org.
_DEFAULT_CLIENT_ID = "61338d24-54d5-408f-a10d-66c06b59f6d2"


@dataclass
class TransferResult:
    """Metadata returned after submitting a Globus Transfer task."""

    task_id: str
    source_endpoint_id: str
    destination_endpoint_id: str
    file_mapping: dict[str, str]  # local_path -> collection-visible path
    remote_directory: str  # collection-visible path (backward compatible name)
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    label: str = ""
    compute_directory: str = ""  # path visible to destination compute workers
    compute_file_mapping: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Preserve compatibility for callers that construct results using the
        # original fields, where both namespaces were assumed to be identical.
        if not self.compute_directory:
            self.compute_directory = self.remote_directory
        if not self.compute_file_mapping:
            self.compute_file_mapping = dict(self.file_mapping)


class GlobusTransferManager:
    """Manage file transfers between local and remote Globus collections.

    Parameters
    ----------
    source_endpoint_id : str
        UUID of the Globus collection on the submitting machine.
    destination_endpoint_id : str
        UUID of the Globus collection on the HPC system.
    destination_base_path : str
        Collection-visible root directory where staged files are placed. Each
        transfer batch creates a subdirectory underneath.
    destination_compute_base_path : str, optional
        Path to the same directory as seen by compute workers. Defaults to
        ``destination_base_path`` when both namespaces are identical.
    source_base_path : str, optional
        If provided, local paths are resolved relative to this directory.
    client_id : str, optional
        Globus app client ID for OAuth.  Defaults to the Globus Tutorial
        client.
    allow_interactive_auth : bool, optional
        Whether a missing or unusable token cache may trigger an interactive
        Native App login.  MCP servers must disable this so OAuth prompts do
        not consume their protocol stream.
    system : str, optional
        Facility system name associated with this destination. Used for
        agent-visible discovery metadata only.
    """

    def __init__(
        self,
        source_endpoint_id: str,
        destination_endpoint_id: str,
        destination_base_path: str,
        source_base_path: Optional[str] = None,
        client_id: Optional[str] = None,
        allow_interactive_auth: bool = True,
        destination_compute_base_path: Optional[str] = None,
        system: Optional[str] = None,
    ) -> None:
        self.source_endpoint_id = source_endpoint_id
        self.destination_endpoint_id = destination_endpoint_id
        self.destination_base_path = _normalize_base_path(destination_base_path)
        self.destination_compute_base_path = _normalize_base_path(
            destination_compute_base_path or destination_base_path
        )
        self.source_base_path = source_base_path
        self._client_id = client_id or _DEFAULT_CLIENT_ID
        self.allow_interactive_auth = allow_interactive_auth
        self.system = system.strip().lower() if system else None
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
        token_file = Path.home() / ".globus" / "chemgraph_transfer_tokens.json"
        tokens = self._load_tokens(token_file)

        if tokens is None:
            tokens = self._interactive_login(client, token_file)

        authorizer = self._make_refresh_authorizer(
            globus_sdk,
            client,
            tokens,
            token_file,
        )
        try:
            # RefreshTokenAuthorizer is lazy.  Resolve a header now so an
            # authentication preflight really refreshes an expired access
            # token before an MCP subprocess starts.
            authorizer.get_authorization_header()
        except Exception as exc:
            if not self.allow_interactive_auth:
                raise RuntimeError(
                    "Globus Transfer authentication is unavailable in this "
                    "non-interactive process. Run an interactive ChemGraph "
                    "Transfer authentication preflight first."
                ) from exc
            logger.warning(
                "Cached Globus Transfer credentials could not be refreshed; "
                "starting a new interactive login."
            )
            tokens = self._interactive_login(client, token_file)
            authorizer = self._make_refresh_authorizer(
                globus_sdk,
                client,
                tokens,
                token_file,
            )
            authorizer.get_authorization_header()

        self._transfer_client = globus_sdk.TransferClient(authorizer=authorizer)
        return self._transfer_client

    def authenticate(self) -> None:
        """Create or refresh the Transfer credentials for later operations.

        Call this from an interactive parent process before starting an MCP
        server.  Credentials remain in the normal on-disk token cache and are
        never returned to the caller.
        """
        self._get_transfer_client()

    def _interactive_login(self, client, token_file: Path) -> dict:
        """Run the Native App login flow, or fail before touching stdio."""
        if not self.allow_interactive_auth:
            raise RuntimeError(
                "Globus Transfer authentication is required, but interactive "
                "authentication is disabled for this process. Run an "
                "interactive ChemGraph Transfer authentication preflight first."
            )

        client.oauth2_start_flow(
            requested_scopes=TRANSFER_SCOPE,
            refresh_tokens=True,
        )
        authorize_url = client.oauth2_get_authorize_url()
        logger.info("Globus Transfer interactive authentication required.")
        print(
            "\nGlobus Transfer authentication required.\n"
            f"Go to this URL and login:\n  {authorize_url}\n"
        )
        auth_code = input("Enter the authorization code: ").strip()
        token_response = client.oauth2_exchange_code_for_tokens(auth_code)
        tokens = dict(token_response.by_resource_server[TRANSFER_RESOURCE_SERVER])
        self._save_tokens(token_file, tokens)
        return tokens

    def _make_refresh_authorizer(
        self,
        globus_sdk,
        client,
        tokens: dict,
        token_file: Path,
    ):
        """Build an auto-refreshing authorizer backed by the token cache."""
        refresh_token = tokens.get("refresh_token")
        if not refresh_token:
            if self.allow_interactive_auth:
                tokens = self._interactive_login(client, token_file)
                refresh_token = tokens["refresh_token"]
            else:
                raise RuntimeError(
                    "The Globus Transfer token cache has no refresh token and "
                    "interactive authentication is disabled. Run an interactive "
                    "ChemGraph Transfer authentication preflight first."
                )

        def save_refreshed_tokens(token_response) -> None:
            refreshed = dict(
                token_response.by_resource_server[TRANSFER_RESOURCE_SERVER]
            )
            refreshed.setdefault("refresh_token", refresh_token)
            self._save_tokens(token_file, refreshed)

        authorizer_kwargs: dict[str, Any] = {
            "on_refresh": save_refreshed_tokens,
        }
        access_token = tokens.get("access_token")
        expires_at = tokens.get("expires_at_seconds")
        if access_token and expires_at is not None:
            authorizer_kwargs.update(
                access_token=access_token,
                expires_at=int(expires_at),
            )
        return globus_sdk.RefreshTokenAuthorizer(
            refresh_token,
            client,
            **authorizer_kwargs,
        )

    @staticmethod
    def _load_tokens(path: Path) -> Optional[dict]:
        if not path.is_file():
            return None

        try:
            with open(path) as f:
                tokens = json.load(f)
            return tokens if isinstance(tokens, dict) else None
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _save_tokens(path: Path, tokens: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as f:
                tmp_path = Path(f.name)
                os.chmod(tmp_path, 0o600)
                json.dump(dict(tokens), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            path.chmod(0o600)
        except Exception:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

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

        remote_dir = _join_remote_path(self.destination_base_path, remote_subdir)
        compute_dir = _join_remote_path(
            self.destination_compute_base_path,
            remote_subdir,
        )
        transfer_label = label or f"ChemGraph file staging ({remote_subdir})"

        tdata = globus_sdk.TransferData(
            self.source_endpoint_id,
            self.destination_endpoint_id,
            label=transfer_label,
            sync_level="checksum",
        )

        # Disambiguate same-basename inputs (e.g. /a/in.cif and /b/in.cif)
        # by suffixing duplicates with _1, _2, ...  Without this the
        # second add_item silently overwrites the first on the
        # destination collection.
        file_mapping: dict[str, str] = {}
        compute_file_mapping: dict[str, str] = {}
        used_names: dict[str, int] = {}
        for local_path in local_paths:
            p = Path(local_path).resolve()
            base = p.name
            count = used_names.get(base, 0)
            if count == 0:
                remote_name = base
            else:
                stem, dot, suffix = base.partition(".")
                remote_name = (
                    f"{stem}_{count}.{suffix}" if dot else f"{stem}_{count}"
                )
            used_names[base] = count + 1
            remote_path = f"{remote_dir}/{remote_name}"
            compute_path = f"{compute_dir}/{remote_name}"
            tdata.add_item(str(p), remote_path)
            file_mapping[str(p)] = remote_path
            compute_file_mapping[str(p)] = compute_path

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
            compute_directory=compute_dir,
            compute_file_mapping=compute_file_mapping,
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
        """Compute the collection-visible path for a local file."""
        filename = Path(local_path).name
        if remote_subdir:
            remote_directory = _join_remote_path(
                self.destination_base_path,
                remote_subdir,
            )
            return _join_remote_path(remote_directory, filename)
        return _join_remote_path(self.destination_base_path, filename)


def _normalize_base_path(path: str) -> str:
    """Strip trailing separators without turning the root into an empty path."""
    return path.rstrip("/") or "/"


def _join_remote_path(base_path: str, child: str) -> str:
    """Join a normalized remote base path and a relative child path."""
    return f"{base_path.rstrip('/')}/{child.lstrip('/')}"
