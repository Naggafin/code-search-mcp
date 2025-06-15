import tempfile
from pathlib import Path

from code_search_mcp.mcp_search import Indexer, Searcher


def create_dummy_repo(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)

    src = tmp / "foo.py"
    src.write_text('def add(a, b):\n    """Return sum"""\n    return a + b\n')


def test_index_and_search():
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "code_search_mcp"
        create_dummy_repo(repo)

        indexer = Indexer(repo)
        indexer.index_full()
        searcher = Searcher()
        res = searcher.context("sum of two numbers", k=3, max_tokens=1000)
        assert res["content"], "Should retrieve some context"
