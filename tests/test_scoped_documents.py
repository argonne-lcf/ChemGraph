"""Durable retrieval contracts with deterministic embeddings and no network."""

import json
import importlib.util
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from chemgraph.execution import documents
from chemgraph.execution.scoped import execution_context, execute_scoped_tool
from chemgraph.registry.tools import ToolSpec
from chemgraph.tools import rag_tools
from tests.test_scoped_execution import Controller, scope


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


@pytest.fixture
def retrieval(monkeypatch):
    stats = {"builds": 0, "restores": 0}

    class Index:
        def __init__(self, vectors):
            self.vectors = vectors
            self.ntotal = len(vectors)

    class Docstore:
        def __init__(self, documents):
            self.documents = documents

        def search(self, identifier):
            return self.documents[identifier]

    class Store:
        def __init__(self, embeddings, index, docstore, mapping):
            self.embedding_function = embeddings
            self.index, self.docstore, self.index_to_docstore_id = (
                index,
                docstore,
                mapping,
            )

        @classmethod
        def from_documents(cls, chunks, embeddings):
            stats["builds"] += 1
            return cls(
                embeddings,
                Index(embeddings.embed_documents([d.page_content for d in chunks])),
                Docstore({str(i): d for i, d in enumerate(chunks)}),
                {i: str(i) for i in range(len(chunks))},
            )

        def similarity_search(self, query, k):
            self.embedding_function.embed_query(query)
            return [
                self.docstore.search(identifier)
                for identifier in list(self.index_to_docstore_id.values())[:k]
            ]

    class Splitter:
        def __init__(self, **kwargs):
            pass

        def create_documents(self, texts, metadatas):
            return [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(texts, metadatas)
            ]

    modules = {
        name: ModuleType(name)
        for name in (
            "faiss",
            "langchain_community",
            "langchain_community.vectorstores",
            "langchain_community.docstore",
            "langchain_community.docstore.in_memory",
            "langchain_text_splitters",
        )
    }
    modules["faiss"].serialize_index = lambda index: np.frombuffer(
        json.dumps(index.vectors).encode(),
        dtype=np.uint8,
    )
    modules["faiss"].deserialize_index = lambda data: Index(json.loads(data.tobytes()))
    modules["langchain_community.vectorstores"].FAISS = Store
    modules["langchain_community.docstore.in_memory"].InMemoryDocstore = Docstore
    modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = Splitter
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        rag_tools, "_get_embeddings", lambda **kwargs: DeterministicEmbeddings()
    )
    monkeypatch.setattr(
        rag_tools,
        "_embedding_configuration",
        lambda value: {
            "provider": "test",
            "model": "deterministic",
        },
    )

    def restore(configuration):
        assert configuration == {"provider": "test", "model": "deterministic"}
        stats["restores"] += 1
        return DeterministicEmbeddings()

    monkeypatch.setattr(rag_tools, "_restore_embeddings", restore)
    return stats


def load(context, text="original document", name="document.txt"):
    (context.workspace / name).write_text(text)
    with execution_context(context):
        result = rag_tools.load_document.invoke({"file_path": name})
    assert result["ok"]


def query(context, name=None):
    with execution_context(context):
        return rag_tools.query_knowledge_base.invoke(
            {"query": "document", "file_path": name}
        )


@pytest.mark.asyncio
async def test_restart_and_completed_load_replay_restore_original_index(
    tmp_path, retrieval
):
    context = scope(tmp_path)
    (context.workspace / "document.txt").write_text("original document")
    request = SimpleNamespace(
        tool_call={
            "type": "tool_call",
            "name": "load_document",
            "id": "load",
            "args": {"file_path": "document.txt"},
        }
    )

    async def handler(request):
        return await rag_tools.load_document.ainvoke(request.tool_call)

    with execution_context(context):
        first = await execute_scoped_tool(
            ToolSpec("load_document", "", None), request, handler
        )
    assert query(context)["ok"]
    journal = Controller()
    journal.saved = json.loads(json.dumps(context.controller.saved))
    restarted = scope(tmp_path, controller=journal)
    (context.workspace / "document.txt").unlink()
    with execution_context(restarted):
        replay = await execute_scoped_tool(
            ToolSpec("load_document", "", None), request, handler
        )
        assert replay.content == first.content
        assert rag_tools.get_loaded_documents() == [
            str(context.workspace / "document.txt")
        ]
    assert query(restarted)["results"][0]["content"] == "original document"
    assert query(restarted, "document.txt")["ok"]
    assert retrieval == {"builds": 1, "restores": 1}


def test_document_replacement_latest_and_conversation_isolation(tmp_path, retrieval):
    alice, bob = scope(tmp_path), scope(tmp_path, "bob")
    load(alice, "alice original")
    load(bob, "bob original")
    load(alice, "second document", "second.txt")
    load(alice, "alice replacement")
    assert query(scope(tmp_path))["results"][0]["content"] == "alice replacement"
    assert (
        query(scope(tmp_path), "second.txt")["results"][0]["content"]
        == "second document"
    )
    assert query(scope(tmp_path, "bob"))["results"][0]["content"] == "bob original"
    with execution_context(alice):
        rag_tools.clear_vector_stores()
    assert not query(scope(tmp_path))["ok"]
    assert query(scope(tmp_path, "bob"))["ok"]


@pytest.mark.parametrize(
    "damage", ["missing", "corrupt", "manifest", "provider", "identity", "symlink"]
)
def test_restoration_failures_do_not_reembed(tmp_path, monkeypatch, retrieval, damage):
    context = scope(tmp_path)
    load(context)
    with execution_context(context):
        record = documents.manifest(context)["documents"]["document.txt"]
    index = context.workspace / ".chemgraph-rag" / record["version"] / "index.faiss"
    if damage == "missing":
        index.unlink()
    elif damage == "corrupt":
        index.write_text("corrupt")
    elif damage == "manifest":
        (context.workspace / ".chemgraph-rag" / "manifest.json").write_text("corrupt")
    elif damage == "provider":

        def missing(configuration):
            raise ImportError("original model unavailable")

        monkeypatch.setattr(rag_tools, "_restore_embeddings", missing)
    elif damage == "symlink":
        outside = tmp_path / "outside-index"
        outside.write_bytes(index.read_bytes())
        index.unlink()
        index.symlink_to(outside)
    restarted = scope(tmp_path)
    if damage == "identity":
        restarted.thread_id = "different-thread"
    result = query(restarted)
    assert not result["ok"]
    assert "reload" in result["error"]
    assert retrieval["builds"] == 1


def test_interrupted_publication_preserves_previous_index(
    tmp_path, monkeypatch, retrieval
):
    context = scope(tmp_path)
    load(context)

    def fail(source, target):
        raise OSError("interrupted manifest write")

    with monkeypatch.context() as patch:
        patch.setattr(documents.os, "replace", fail)
        with pytest.raises(OSError, match="interrupted"):
            load(context, "replacement")
    assert query(context)["results"][0]["content"] == "original document"
    assert query(scope(tmp_path))["results"][0]["content"] == "original document"


def test_persistence_rejects_redirected_record_directory(tmp_path, retrieval):
    context = scope(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (context.workspace / ".chemgraph-rag").symlink_to(outside, target_is_directory=True)
    with pytest.raises(documents.DocumentStoreError):
        load(context)
    assert list(outside.iterdir()) == []


def test_saved_openai_provider_never_falls_back_or_saves_credentials(monkeypatch):
    from langchain_openai import OpenAIEmbeddings

    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key="test-secret")
    configuration = rag_tools._embedding_configuration(embedding)
    assert "test-secret" not in json.dumps(configuration)
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(KeyError):
        rag_tools._restore_embeddings(configuration)


def test_real_faiss_round_trip_without_document_reembedding(tmp_path):
    if any(
        importlib.util.find_spec(name) is None
        for name in ("faiss", "langchain_community")
    ):
        pytest.skip("requires the optional rag dependencies")
    # Isolate FAISS from the incompatible OpenMP runtime in some torch wheels.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent('''
        import sys
        from pathlib import Path
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
        from langchain_community.vectorstores import FAISS
        from chemgraph.execution.scoped import ExecutionContext, execution_context
        from chemgraph.execution.documents import save_document, restore_document

        class Deterministic(Embeddings):
            built = False
            def embed_documents(self, texts):
                assert not self.built, "Restoration re-embedded documents"
                self.built = True
                return [[float(len(text)), 1.0] for text in texts]
            def embed_query(self, text):
                return [float(len(text)), 1.0]

        context = ExecutionContext("owner", "thread", "task", Path(sys.argv[1]), None)
        embeddings = Deterministic()
        store = FAISS.from_documents([Document(page_content="retained text")], embeddings)
        source = str(context.workspace / "doc.txt")
        with execution_context(context):
            save_document(context, source, store, {"provider": "test"}, {})
            restored = restore_document(context, source, lambda configuration: embeddings)
        assert restored.similarity_search("query", k=1)[0].page_content == "retained text"
    '''),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
