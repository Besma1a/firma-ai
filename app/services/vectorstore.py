"""
Vector store wrapper around Chroma + a multilingual embedding model.

This module exposes:
  - get_vector_store(): returns a singleton Chroma instance for queries
  - build_vector_store(): one-shot index construction (run by build script)
  - search(): convenience wrapper used by rag.py

The embedding model and Chroma client are lazily initialized so that import
of this module is cheap (matters for unit tests and FastAPI cold starts).
"""
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


_embeddings: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[Chroma] = None
COLLECTION = "nabta_crops"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy singleton for the embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> Chroma:
    """Lazy singleton for the Chroma vector store (read-only access)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            persist_directory=settings.vector_store_dir,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION,
        )
    return _vector_store


def build_vector_store(kb_dir: str | None = None) -> int:
    """Read every .md chunk in kb_dir, embed each, persist to disk.

    Returns the number of chunks indexed. Run this once after build_kb().
    """
    kb_dir = kb_dir or settings.kb_dir
    if not Path(kb_dir).exists():
        raise FileNotFoundError(f"KB directory not found: {kb_dir}. Run build_kb first.")

    loader = DirectoryLoader(
        kb_dir, glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    # Tag each doc with crop/topic/lang so we can filter queries later.
    for d in docs:
        name = Path(d.metadata["source"]).stem  # "<crop>__<topic>__<lang>"
        try:
            crop_id, topic, lang = name.split("__")
        except ValueError:
            crop_id, topic, lang = "unknown", "unknown", "en"
        d.metadata.update({"crop_id": crop_id, "topic": topic, "lang": lang})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80, length_function=len)
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=settings.vector_store_dir,
        collection_name=COLLECTION,
    ).persist()
    return len(chunks)


def search(query: str, k: int | None = None,
           lang_filter: str | None = None,
           crop_filter: str | None = None) -> list:
    """Return the k most similar chunks to query, with optional filters.

    Filter syntax follows Chroma's metadata `where` clause. Multiple filters
    are AND-ed together.
    """
    k = k or settings.top_k
    where = {}
    if lang_filter:
        where["lang"] = lang_filter
    if crop_filter:
        where["crop_id"] = crop_filter
    # Chroma rejects empty where dict; pass None instead.
    where = where if where else None
    return get_vector_store().similarity_search_with_score(query, k=k, filter=where)


def count_chunks() -> int:
    """How many chunks are currently indexed (used by /healthz)."""
    try:
        return get_vector_store()._collection.count()
    except Exception:
        return 0
