"""Conversation-local FAISS persistence without pickled Python objects."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from chemgraph.execution.scoped import scoped_path


class DocumentStoreError(RuntimeError):
    """A saved document cannot be restored without explicit user action."""


def _path(context, *parts):
    root = context.workspace / ".chemgraph-rag"
    path = root.joinpath(*parts)
    if not path.is_relative_to(root) or ".." in path.parts:
        raise ValueError("Invalid document record path")
    # Internal records must not be redirected, even to other workspace files.
    for candidate in (path, *path.parents):
        if candidate == context.workspace:
            break
        if candidate.is_symlink():
            raise ValueError("Document records cannot be symlinks")
    return path


def _sync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _write(path, content):
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _publish(context, manifest):
    destination = _path(context, "manifest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=destination.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(manifest, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _empty(context):
    return {
        "version": 1,
        "owner": context.owner,
        "thread_id": context.thread_id,
        "documents": {},
        "latest": None,
    }


def manifest(context):
    try:
        path = _path(context, "manifest.json")
        if not path.exists():
            return _empty(context)
        result = json.loads(path.read_text())
        if any(
            result.get(key) != value
            for key, value in (
                ("version", 1),
                ("owner", context.owner),
                ("thread_id", context.thread_id),
            )
        ) or not isinstance(result.get("documents"), dict):
            raise ValueError("Incompatible document manifest or conversation identity")
        for relative, record in result["documents"].items():
            source = Path(relative)
            if source.is_absolute() or ".." in source.parts:
                raise ValueError("Invalid document source path")
            scoped_path(source)
            version = record["version"]
            if len(version) != 32 or any(c not in "0123456789abcdef" for c in version):
                raise ValueError("Invalid document version")
        if result["latest"] is not None and result["latest"] not in result["documents"]:
            raise ValueError("Missing latest document entry")
        return result
    except Exception as exc:
        raise DocumentStoreError(
            "Cannot read saved document state. Restore the conversation's .chemgraph-rag "
            "records or explicitly clear and reload its documents."
        ) from exc


def save_document(context, source, store, embedding, settings):
    """Publish a complete version before making it visible to later queries."""
    import faiss

    saved = manifest(context)
    relative = Path(source).relative_to(context.workspace).as_posix()
    version = uuid4().hex
    directory = _path(context, version)
    directory.mkdir(parents=True)
    documents = {
        "embedding": embedding,
        "settings": settings,
        "documents": [
            {
                "index": index,
                "id": identifier,
                "document": store.docstore.search(identifier).model_dump(),
            }
            for index, identifier in store.index_to_docstore_id.items()
        ],
    }
    payloads = {
        "index.faiss": faiss.serialize_index(store.index).tobytes(),
        "documents.json": json.dumps(documents, allow_nan=False).encode(),
    }
    for name, content in payloads.items():
        _write(_path(context, version, name), content)
    _sync_directory(directory)
    _sync_directory(directory.parent)
    saved["documents"][relative] = {
        "version": version,
        "hashes": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in payloads.items()
        },
    }
    saved["latest"] = relative
    _publish(context, saved)


def restore_document(context, source, embeddings_factory):
    """Restore original vectors and chunks; only query embedding uses the provider."""
    saved = manifest(context)
    relative = Path(source).relative_to(context.workspace).as_posix()
    record = saved["documents"].get(relative)
    if record is None:
        return None
    try:
        import faiss
        import numpy as np
        from langchain_core.documents import Document
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        payloads = {}
        for name in ("index.faiss", "documents.json"):
            content = _path(context, record["version"], name).read_bytes()
            if hashlib.sha256(content).hexdigest() != record["hashes"][name]:
                raise ValueError("Saved document checksum mismatch")
            payloads[name] = content
        metadata = json.loads(payloads["documents.json"])
        index = faiss.deserialize_index(
            np.frombuffer(payloads["index.faiss"], dtype=np.uint8)
        )
        rows = metadata["documents"]
        mapping = {row["index"]: row["id"] for row in rows}
        documents = {row["id"]: Document(**row["document"]) for row in rows}
        if set(mapping) != set(range(index.ntotal)) or len(documents) != index.ntotal:
            raise ValueError("Invalid saved document mapping")
        embeddings = embeddings_factory(metadata["embedding"])
        return FAISS(embeddings, index, InMemoryDocstore(documents), mapping)
    except Exception as exc:
        raise DocumentStoreError(
            f"Cannot restore document {relative!r}. Check the saved index, install the rag "
            "extra, and restore its original embedding model/credentials; otherwise "
            "explicitly reload the document."
        ) from exc


def clear_documents(context):
    _publish(context, _empty(context))
