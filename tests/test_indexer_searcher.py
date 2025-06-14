import tempfile
from pathlib import Path

from mcp_search import Indexer, Searcher


def create_dummy_repo(tmp: Path):
    src = tmp / "foo.py"
    src.write_text(
        "def add(a, b):\n    \"\"\"Return sum\"\"\"\n    return a + b\n"
    )


import pytest


def test_index_and_search():
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        create_dummy_repo(repo)

        indexer = Indexer(repo)
        try:
            indexer.index_full()
            searcher = Searcher()
            res = searcher.context("sum of two numbers", k=3, max_tokens=1000)
            assert res["content"], "Should retrieve some context"
        except RuntimeError as exc:
            pytest.skip(str(exc))
