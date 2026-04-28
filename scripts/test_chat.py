"""End-to-end test of the full RAG pipeline (uses Claude API or mock).

Usage:
    python scripts/test_chat.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag import answer
from app.config import settings


QUESTIONS = [
    "Quand dois-je semer le blé dans la région de Sétif ?",
    "كيفاش نقاوم البياض الزغبي على الطماطم؟",
    "What pests should I worry about for date palm in Biskra?",
    "kifech nesqi z-zitoun f sif?",
    "Comment fertiliser la pomme de terre en arrière-saison ?",
]


def main():
    print(f"Mock mode: {settings.mock_claude}")
    print(f"Model: {settings.claude_model}")
    print(f"Top-k: {settings.top_k}")

    for q in QUESTIONS:
        print(f"\n{'='*70}\nQ: {q}")
        result = answer(q)
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {[s['crop']+'/'+s['topic']+'/'+s['lang'] for s in result['sources']]}")


if __name__ == "__main__":
    main()
