# EDIT: Update import paths to reflect the new directory structure
from pathlib import Path

import pytest

from code_search_mcp.embedder import embed  # Updated for new structure
from code_search_mcp.embedder import is_probably_code


def test_embed_function():
    chunks = [("/path/to/file.py", {"text": "sample code", "metadata": {}})]
    result = embed(chunks, batch_size=1)
    assert "code" in result, "Expected 'code' key in result dict"
    assert "text" in result, "Expected 'text' key in result dict"
    assert len(result["code"]) > 0, "Code embeddings should not be empty"


def test_is_probably_code():
    assert is_probably_code(Path("/path/to/sample.py"), mime_detector=None) == True
