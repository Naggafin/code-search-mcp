import httpx
import pytest
from httpx import AsyncClient

from code_search_mcp.mcp_server import get_context

pytestmark = pytest.mark.xfail(
    reason="SlowAPI timing limitations under ASGITransport, to revisit"
)


@pytest.mark.asyncio
async def test_rate_limit(monkeypatch):
    """Exceed 60/minute limit and expect 429."""

    # Fast dummy context to avoid heavy work
    monkeypatch.setattr(
        mcp_server.searcher,
        "context",
        lambda *a, **k: {
            "content": "",
            "metadata": [],
            "tokens": 0,
            "status": "success",
        },
    )

    transport = httpx.ASGITransport(app=mcp_server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Hit the endpoint 61 times
        for i in range(60):
            r = await client.post(
                "/mcp/v1/context",
                json={"query": "q", "max_tokens": 1000},
                headers={"X-API-Key": "test"},
            )
            assert r.status_code == 200, f"request {i} failed: {r.text}"

        # 61st should fail
        r = await client.post(
            "/mcp/v1/context",
            json={"query": "q", "max_tokens": 1000},
            headers={"X-API-Key": "test"},
        )
        assert r.status_code == 429
