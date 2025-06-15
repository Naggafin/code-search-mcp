import pytest

import code_search_mcp.mcp_stdio as mcp_stdio


class DummySearcher:
    def context(self, query, **kwargs):
        assert query == "demo"
        return {"content": "CTX", "tokens": 10}

    def search(self, query, **kwargs):
        assert query == "demo"
        return {"results": [1, 2, 3]}

    def stream_context(self, query, **kwargs):
        yield 'event: chunk\ndata: {"content": "A"}\n\n'
        yield 'event: end\ndata: {"tokens": 5}\n\n'


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setattr(
        mcp_stdio, "Searcher", lambda project_path=None: DummySearcher()
    )


def _handle(req):
    return list(mcp_stdio._handle_request(req, DummySearcher()))


def test_context_non_stream():
    replies = _handle({"type": "context", "query": "demo"})
    assert replies == [{"status": "ok", "data": {"content": "CTX", "tokens": 10}}]


def test_search():
    replies = _handle({"type": "search", "query": "demo"})
    assert replies == [{"status": "ok", "data": {"results": [1, 2, 3]}}]


def test_stream_context():
    replies = _handle({"type": "context", "query": "demo", "stream": True})
    assert replies[0]["event"] == "chunk"
    assert replies[-1]["event"] == "end"


def test_invalid_type():
    replies = _handle({"type": "unknown"})
    assert replies[0]["status"] == "error"
