"""RAG (Retrieval-Augmented Generation) tools for ChemGraph.

Provides tools to load documents (.txt and .pdf) into a FAISS vector
store and query them for relevant context. Supports OpenAI and
HuggingFace embeddings with automatic fallback.
"""

import os
import logging
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level vector store registry
# ---------------------------------------------------------------------------
# Maps a document identifier (file path or user-provided name) to a
# FAISS vector store instance so that documents loaded during a session
# remain queryable across multiple tool calls.
_vector_stores: dict = {}


# ---------------------------------------------------------------------------
# Pydantic schemas for tool inputs
# ---------------------------------------------------------------------------
class LoadDocumentInput(BaseModel):
    """Input schema for the load_document tool."""

    file_path: str = Field(
        description="Absolute or relative path to a .txt or .pdf file to ingest."
    )
    chunk_size: int = Field(
        default=1000,
        description="Maximum number of characters per text chunk.",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Number of overlapping characters between consecutive chunks.",
    )
    embedding_provider: str = Field(
        default="openai",
        description=(
            "Embedding provider to use: 'openai' (requires OPENAI_API_KEY) "
            "or 'huggingface' (local, no API key needed). "
            "Falls back to huggingface if openai is unavailable."
        ),
    )


class QueryKnowledgeBaseInput(BaseModel):
    """Input schema for the query_knowledge_base tool."""

    query: str = Field(description="The question or search query.")
    file_path: Optional[str] = Field(
        default=None,
        description=(
            "Path of a previously loaded document to search. "
            "If None, searches the most recently loaded document."
        ),
    )
    top_k: int = Field(
        default=5,
        description="Number of most relevant chunks to retrieve.",
    )


# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------
_SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def _extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file using PyMuPDF.

    Parameters
    ----------
    file_path : str
        Absolute path to the PDF file.

    Returns
    -------
    str
        Concatenated text from all pages, separated by newlines.

    Raises
    ------
    ImportError
        If PyMuPDF (``fitz``) is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF support. "
            "Install the 'rag' extra: pip install chemgraphagent[rag]"
        ) from exc

    pages: list[str] = []
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                pages.append(page_text)
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _get_embeddings(provider: str = "openai"):
    """Return an embeddings instance for the requested provider.

    Supports OpenAI-compatible custom endpoints via OPENAI_BASE_URL.
    Falls back to HuggingFace if OpenAI embeddings are unavailable.
    """
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings

            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL") 

            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set")

            kwargs = {
                "model": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
                "api_key": api_key,
                "check_embedding_ctx_length":False,

            }

            if base_url:
                kwargs["base_url"] = base_url

            return OpenAIEmbeddings(**kwargs)

        except Exception as exc:
            logger.warning(
                "OpenAI embeddings unavailable (%s); falling back to HuggingFace.",
                exc,
            )
            provider = "huggingface"

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    except ImportError as exc:
        raise ImportError(
            "Neither langchain-openai nor langchain-huggingface is installed. "
            "Install the 'rag' extra: pip install chemgraphagent[rag]"
        ) from exc


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool(args_schema=LoadDocumentInput)
def load_document(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_provider: str = "openai",
) -> dict:
    """Load a document (.txt or .pdf), split it into chunks, and index it in a FAISS vector store.

    The document remains available for querying via ``query_knowledge_base``
    for the duration of the session.

    Parameters
    ----------
    file_path : str
        Path to the ``.txt`` or ``.pdf`` file to ingest.
    chunk_size : int, optional
        Max characters per chunk, by default 1000.
    chunk_overlap : int, optional
        Overlap between consecutive chunks, by default 200.
    embedding_provider : str, optional
        ``"openai"`` or ``"huggingface"``, by default ``"openai"``.

    Returns
    -------
    dict
        Status information including the number of chunks created.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    resolved_path = os.path.abspath(file_path)
    if not os.path.isfile(resolved_path):
        return {"ok": False, "error": f"File not found: {resolved_path}"}

    _, ext = os.path.splitext(resolved_path)
    ext = ext.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        return {
            "ok": False,
            "error": (f"Unsupported file type '{ext}'. Supported formats: {supported}"),
        }

    # ----- Extract text based on file type -----
    if ext == ".pdf":
        try:
            text = _extract_text_from_pdf(resolved_path)
        except ImportError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Failed to extract text from PDF: {exc}",
            }
    else:
        # .txt
        with open(resolved_path, "r", encoding="utf-8") as fh:
            text = fh.read()

    if not text.strip():
        return {"ok": False, "error": "File is empty or contains no extractable text."}

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents(
        [text],
        metadatas=[{"source": resolved_path, "file_type": ext}],
    )

    # Build FAISS index
    embeddings = _get_embeddings(provider=embedding_provider)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Register in module-level store
    _vector_stores[resolved_path] = vector_store
    # Also track the most-recently loaded path for convenience
    _vector_stores["__latest__"] = resolved_path

    logger.info(
        "Loaded '%s' (%s) into FAISS vector store (%d chunks, chunk_size=%d, overlap=%d).",
        resolved_path,
        ext,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )

    return {
        "ok": True,
        "file_path": resolved_path,
        "file_type": ext,
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_provider": embedding_provider,
    }


@tool(args_schema=QueryKnowledgeBaseInput)
def query_knowledge_base(
    query: str,
    file_path: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """Search a previously loaded document for passages relevant to a query.

    Parameters
    ----------
    query : str
        The natural-language question or search query.
    file_path : str, optional
        Path of a previously loaded document. If ``None``, the most
        recently loaded document is searched.
    top_k : int, optional
        Number of top-matching chunks to return, by default 5.

    Returns
    -------
    dict
        A dict with ``"ok"``, ``"query"``, ``"num_results"``, and
        ``"results"`` (list of dicts with ``"content"`` and ``"metadata"``).
    """
    # Resolve which vector store to query
    if file_path is not None:
        resolved_path = os.path.abspath(file_path)
    else:
        resolved_path = _vector_stores.get("__latest__")

    if resolved_path is None or resolved_path not in _vector_stores:
        available = [k for k in _vector_stores if k != "__latest__"]
        return {
            "ok": False,
            "error": (
                "No document loaded yet. Use the load_document tool first."
                if not available
                else f"Document '{file_path}' not found. Available: {available}"
            ),
        }

    vector_store = _vector_stores[resolved_path]
    docs = vector_store.similarity_search(query, k=top_k)

    results = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

    return {
        "ok": True,
        "query": query,
        "num_results": len(results),
        "results": results,
    }


def get_loaded_documents() -> list[str]:
    """Return a list of file paths currently loaded in the vector store.

    This is a plain helper (not a tool) for programmatic access.
    """
    return [k for k in _vector_stores if k != "__latest__"]


def clear_vector_stores() -> None:
    """Remove all loaded vector stores. Useful for testing and cleanup."""
    _vector_stores.clear()
