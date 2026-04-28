"""Generate the markdown knowledge base from the seed JSON.

Usage:
    python scripts/build_kb.py
"""
import sys
from pathlib import Path

# Make `app.*` importable regardless of where the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.kb_builder import build_kb
from app.config import settings


def main():
    n = build_kb()
    print(f"OK — wrote {n} knowledge-base chunks to {settings.kb_dir}")


if __name__ == "__main__":
    main()
