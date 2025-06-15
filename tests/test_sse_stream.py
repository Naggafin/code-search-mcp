import httpx
import pytest
from httpx import AsyncClient

import code_search_mcp.mcp_server as mcp_server
from code_search_mcp.mcp_server import stream_context


@pytest.mark.asyncio
async def test_stream_context(monkeypatch):
    """Ensure SSE endpoint yields chunk and end events."""

    def fake_stream_context(*_args, **_kwargs):
        yield 'event: chunk\ndata: {"content": "hello", "metadata": {}}\n\n'
        yield 'event: end\ndata: {"tokens": 5}\n\n'

    # Patch the searcher to avoid heavy dependencies
    monkeypatch.setattr(mcp_server.searcher, "stream_context", fake_stream_context)

    transport = httpx.ASGITransport(app=mcp_server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/mcp/v1/context/stream",
            params={"query": "dummy"},
            headers={"X-API-Key": "test"},
        )

    assert response.status_code == 200
    body = response.text
    assert "event: chunk" in body
    assert "event: end" in body
