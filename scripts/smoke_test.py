"""Smoke-test the retrieval layer (no Claude calls, free).

If this script returns the right crop chunks for cross-language queries,
your RAG core works. Always run this BEFORE testing with Claude.

Usage:
    python scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.vectorstore import search


def show(query: str, **kwargs):
    print(f"\n=== Query: {query} ===")
    results = search(query, **kwargs)
    if not results:
        print("  (no results)")
        return
    for i, (doc, score) in enumerate(results, 1):
        m = doc.metadata
        print(f"  [{i}] score={score:.4f}  {m.get('crop_id'):>10s} / "
              f"{m.get('topic'):>10s} / {m.get('lang')}")
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"      {snippet}...")


def main():
    # Same meaning across 4 language/script combos — should retrieve tomato.
    show("When should I plant tomatoes in Algeria?")
    show("Quand semer la tomate en Algérie?")
    show("متى أزرع الطماطم في الجزائر؟")
    show("kifech nezraɛ tomatic f Lzayer?")

    # Specific topic — should retrieve date_palm/diseases
    show("What are the main diseases of date palm?", k=3)

    # Filter test: only Arabic chunks about wheat watering
    show("ري القمح", k=3, lang_filter="ar")

    # Filter test: only tomato chunks
    show("flowering stage", k=3, crop_filter="tomato")


if __name__ == "__main__":
    main()
