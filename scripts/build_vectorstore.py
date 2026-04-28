"""Embed the KB chunks and build the Chroma vector store.

Usage:
    python scripts/build_vectorstore.py

First run downloads the embedding model (~118 MB). Subsequent runs use cache.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.vectorstore import build_vector_store
from app.config import settings


def main():
    print(f"Building vector store from {settings.kb_dir} -> {settings.vector_store_dir}")
    print(f"Embedding model: {settings.embed_model}")
    print("(first run downloads ~118 MB; subsequent runs are cached)")
    n = build_vector_store()
    print(f"OK — {n} chunks indexed in {settings.vector_store_dir}")


if __name__ == "__main__":
    main()
