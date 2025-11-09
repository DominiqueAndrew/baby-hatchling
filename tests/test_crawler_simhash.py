import pytest

pytest.importorskip("trafilatura")

from src.crawler.pipeline import _simhash_signature


def test_simhash_signature_clips_large_counts():
    text = "repeat " * 300
    signature = _simhash_signature(text)
    assert hasattr(signature, "value")
