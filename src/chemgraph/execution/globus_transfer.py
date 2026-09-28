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


class TransferAuthenticationRequired(RuntimeError):
    """Authentication must be completed outside an agent tool invocation."""

# Globus Transfer API scope
TRANSFER_SCOPE = "urn:globus:auth:scope:transfer.api.globus.org:all"

# Default Globus native-app client ID (Globus Tutorial client).
# Projects should register their own app at https://app.globus.org.
_DEFAULT_CLIENT_ID = "61338d24-54d5-408f-a10d-66c06b59f6d2"


def _token_file(client_id: str) -> Path:
    import hashlib

    name = "chemgraph_transfer_tokens.json"
    if client_id != _DEFAULT_CLIENT_ID:
        identity = hashlib.sha256(client_id.encode()).hexdigest()
        name = f"chemgraph_transfer_tokens.{identity}.json"
    return Path.home() / ".globus" / name


def _login_command(client_id: str) -> str:
    import shlex

    command = "python -m chemgraph.execution.globus_transfer"
    if client_id != _DEFAULT_CLIENT_ID:
        command += f" --client-id {shlex.quote(client_id)}"
    return command


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


@dataclass(frozen=True, repr=False)
class _PreparedTransfer:
    """In-memory submission only; never store this object in run evidence."""

    client: Any = field(repr=False)
    payload: Any = field(repr=False)


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
        token_file = _token_file(self._client_id)
        tokens = self._load_tokens(token_file)
        login = _login_command(self._client_id)

        if tokens is None:
            raise TransferAuthenticationRequired(
                f"Authenticate Globus Transfer before running tools: {login}"
            )
        # Untagged legacy tokens are supported only at the default-client path.
        if tokens.get("client_id", _DEFAULT_CLIENT_ID) != self._client_id:
            raise TransferAuthenticationRequired(
                f"Globus Transfer tokens do not match the configured client. Run {login}"
            )
        if not tokens.get("refresh_token"):
            raise TransferAuthenticationRequired(
                f"Globus Transfer needs a new refresh-token login. Run {login}"
            )

        def on_refresh(response):
            fresh = dict(response.by_resource_server["transfer.api.globus.org"])
            fresh.setdefault("refresh_token", tokens["refresh_token"])
            fresh["client_id"] = self._client_id
            self._save_tokens(token_file, fresh)
            tokens.update(fresh)

        authorizer = globus_sdk.RefreshTokenAuthorizer(
            tokens["refresh_token"], client,
            access_token=tokens.get("access_token"),
            expires_at=tokens.get("expires_at_seconds", 0),
            on_refresh=on_refresh,
        )
        self._transfer_client = globus_sdk.TransferClient(authorizer=authorizer)
        return self._transfer_client

    @staticmethod
    def _load_tokens(path: Path) -> Optional[dict]:
        if not path.is_file():
            return None
        import json

        try:
            with open(path) as f:
                tokens = json.load(f)
                return tokens if isinstance(tokens, dict) else None
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def _save_tokens(path: Path, tokens: dict) -> None:
        import json
        import os
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".transfer-tokens-")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(dict(tokens), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    # ── transfers ───────────────────────────────────────────────────────

    def transfer_mapping(
        self, file_mapping: dict[str, str], *, reverse: bool = False,
        label: str = "ChemGraph HPC staging",
    ) -> str:
        """Transfer explicit collection paths without flattening or renaming.

        The caller validates roots and overwrite policy. Mapping keys always
        identify sources; reverse swaps collection IDs, not mapping direction.
        """
        return self.submit_prepared(
            self.prepare_mapping(file_mapping, reverse=reverse, label=label)
        )

    def prepare_mapping(
        self, file_mapping: dict[str, str], *, reverse: bool = False,
        label: str = "ChemGraph HPC staging",
    ) -> _PreparedTransfer:
        """Validate, authenticate and build a payload without submitting it."""
        import globus_sdk

        file_mapping = dict(file_mapping)
        if not file_mapping or len(set(file_mapping.values())) != len(file_mapping):
            raise ValueError("A nonempty mapping with unique destinations is required.")
        source, destination = self.source_endpoint_id, self.destination_endpoint_id
        if reverse:
            source, destination = destination, source
        tc = self._get_transfer_client()
        try:
            # Refresh now, before the caller records a submission attempt.
            tc.authorizer.get_authorization_header()
        except Exception:
            self._transfer_client = None
            raise TransferAuthenticationRequired(
                "Globus Transfer credential refresh failed. Run "
                + _login_command(self._client_id)
            ) from None
        data = globus_sdk.TransferData(
            source_endpoint=source, destination_endpoint=destination,
            label=label, sync_level="checksum",
            verify_checksum=True,
        )
        for source_path, destination_path in file_mapping.items():
            data.add_item(source_path, destination_path)
        return _PreparedTransfer(tc, data)

    def submit_prepared(self, prepared: _PreparedTransfer) -> str:
        """Invoke SDK submission once; uncertain outcomes must not be retried."""
        return str(prepared.client.submit_transfer(prepared.payload)["task_id"])

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
            source_endpoint=self.source_endpoint_id,
            destination_endpoint=self.destination_endpoint_id,
            label=transfer_label,
            sync_level="checksum",
        )

        # Disambiguate same-basename inputs (e.g. /a/in.cif and /b/in.cif)
        # by suffixing duplicates with _1, _2, ...  Without this the
        # second add_item silently overwrites the first on the
        # destination collection.
        file_mapping: dict[str, str] = {}
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


def authenticate(collections=(), *, client_id=None) -> None:
    """Explicit terminal login, never called by tools."""
    import globus_sdk

    client_id = client_id or _DEFAULT_CLIENT_ID
    client = globus_sdk.NativeAppAuthClient(client_id)
    from globus_sdk.scopes import Scope
    scope = Scope(TRANSFER_SCOPE, dependencies=tuple(
        Scope(f"https://auth.globus.org/scopes/{collection}/data_access")
        for collection in collections
    ))
    client.oauth2_start_flow(requested_scopes=str(scope), refresh_tokens=True)
    print(client.oauth2_get_authorize_url())
    response = client.oauth2_exchange_code_for_tokens(input("Authorization code: ").strip())
    GlobusTransferManager._save_tokens(
        _token_file(client_id),
        {**response.by_resource_server["transfer.api.globus.org"], "client_id": client_id},
    )


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Authenticate ChemGraph Globus Transfer")
    parser.add_argument("--collection", action="append", default=[],
                        help="Managed collection needing data_access consent (repeatable)")
    parser.add_argument("--client-id", default=None,
                        help="OAuth client ID used by the configured transfer manager")
    args = parser.parse_args(argv)
    authenticate(args.collection, client_id=args.client_id)


if __name__ == "__main__":
    main()
