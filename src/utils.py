import os
from pathlib import Path


def ensure_project_structure():
    """Create expected folders if they are missing."""
    folders = ["data", "chroma_db", "src"]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)


def get_doc_stats(text):
    """Return basic text statistics."""
    words = len(text.split())
    pages = max(1, (words // 500) + 1)
    return {"word_count": words, "pages": pages}
