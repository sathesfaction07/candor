from pathlib import Path

from .base import DocumentReader
from .pdf import PDFReader
from .text import TextReader

READERS: dict[str, type[DocumentReader]] = {
    ".pdf": PDFReader,
    ".txt": TextReader,
    ".md": TextReader,
}


def get_reader(path: Path) -> DocumentReader:
    ext = path.suffix.lower()
    cls = READERS.get(ext)
    if not cls:
        available = ", ".join(sorted(READERS.keys()))
        raise ValueError(f"Unsupported file type: {ext}. Supported: {available}")
    return cls()


__all__ = [
    "DocumentReader",
    "PDFReader",
    "TextReader",
    "get_reader",
]
