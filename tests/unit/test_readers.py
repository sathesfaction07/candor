from pathlib import Path

import pytest

from sme_bridge.ingest import get_reader


def test_get_reader_unknown_extension():
    fake_path = Path("document.xyz")

    with pytest.raises(ValueError) as exc_info:
        get_reader(fake_path)

    error_msg = str(exc_info.value)
    assert "Unsupported file type" in error_msg
    assert ".xyz" in error_msg
    assert ".pdf" in error_msg
    assert ".txt" in error_msg
    assert ".md" in error_msg


def test_get_reader_supported_extensions():
    for ext in [".pdf", ".txt", ".md"]:
        fake_path = Path(f"document{ext}")
        reader = get_reader(fake_path)
        assert reader is not None
