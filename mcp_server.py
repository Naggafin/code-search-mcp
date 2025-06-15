import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.requests import Request

from auth import verify_api_key
from config import settings
from mcp_search import Indexer, Searcher
from token_counter import count_tokens

BASE_DIR = Path(settings.project_path)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Register default handler
from slowapi.middleware import SlowAPIMiddleware

app = FastAPI(
    title="MCP Code Search Server",
    description="MCP-compliant server for code search and context retrieval.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Context",
            "description": "Endpoints for fetching and searching code context.",
        },
        {"name": "File", "description": "Endpoints for retrieving file contents."},
        {"name": "Reindex", "description": "Endpoints for reindexing the codebase."},
    ],
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


searcher = Searcher()

# ----------------------------- rate limiting
from fastapi.responses import PlainTextResponse

# Register rate-limit exceeded handler


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):  # noqa: D401
    """Return plain-text rate-limit response."""
    return PlainTextResponse(str(exc.detail), status_code=429)


class ContextRequest(BaseModel):
    query: str = Field(
        ..., description="Search query for code context."
    )
    max_tokens: int = Field(
        8000, description="Maximum tokens for the response context."
    )
    metadata_filter: dict | None = Field(
        None,
        description="Optional filter for metadata (e.g., {'type': 'class'} to retrieve only classes).",
        examples=[{"type": "class"}, {"type": "function"}],
    )


class FileRequest(BaseModel):
    path: str = Field(..., description="Relative path to the file (e.g., 'models.py').")


class ContextResponse(BaseModel):
    content: str
    metadata: list[dict]
    tokens: int
    status: str


from starlette.requests import Request


class SearchResponse(BaseModel):
    results: list[dict]
    tokens: int
    status: str


class FileResponse(BaseModel):
    content: str
    metadata: dict
    tokens: int
    status: str


class ReindexResponse(BaseModel):
    status: str


@limiter.limit("60/minute")
@app.post(
    "/mcp/v1/context",
    response_model=ContextResponse,
    tags=["Context"],
    summary="Fetch code context",
    description="Retrieve code context based on a query, with optional metadata filtering.",
)
async def get_context(
    request: Request, payload: ContextRequest, api_key: str = Depends(verify_api_key)
):
    try:
        result = searcher.context(
            payload.query,
            k=5,
            max_tokens=payload.max_tokens,
            metadata_filter=payload.metadata_filter,
        )
        return {
            "content": result["content"],
            "metadata": result["metadata"],
            "tokens": result["tokens"],
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Context fetch failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch context: {e}"
        ) from e


@limiter.limit("60/minute")
@app.post(
    "/mcp/v1/context/search",
    response_model=SearchResponse,
    tags=["Context"],
    summary="Search code context",
    description="Search for code snippets with optional metadata filtering.",
)
async def search_context(
    request: Request, payload: ContextRequest, api_key: str = Depends(verify_api_key)
):
    try:
        result = searcher.context(
            payload.query,
            k=5,
            max_tokens=payload.max_tokens,
            metadata_filter=payload.metadata_filter,
        )
        return {
            "results": [
                {"content": doc, "metadata": meta}
                for doc, meta in zip(
                    result["content"].split("\n\n") if result["content"] else [],
                    result["metadata"],
                    strict=False,
                )
            ],
            "tokens": result["tokens"],
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")

        raise HTTPException(status_code=500, detail=f"Search failed: {e}") from e


@limiter.limit("60/minute")
@app.get(
    "/mcp/v1/context/stream",
    tags=["Context"],
    summary="Stream code context via Server-Sent Events",
    description="SSE endpoint that streams context results in real-time.",
)
async def stream_context(
    request: Request,
    query: str,
    max_tokens: int = 8000,
    metadata_filter: str | None = None,
    api_key: str = Depends(verify_api_key),
):
    """Return a *text/event-stream* response.

    Emits two events by default:
    1. **result** – full search payload
    2. **end** – terminator so clients know the stream is finished
    """

    try:
        meta_dict: dict | None = None
        if metadata_filter:
            import json

            meta_dict = json.loads(metadata_filter)
        gen = searcher.stream_context(
            query,
            k=5,
            max_tokens=max_tokens,
            metadata_filter=meta_dict,
        )

        async def event_generator():
            for event_str in gen:
                yield event_str
                if await request.is_disconnected():
                    break

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Stream context failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Stream context failed: {e}"
        ) from e

        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}") from e


@limiter.limit("60/minute")
@app.post(
    "/mcp/v1/context/file",
    response_model=FileResponse,
    tags=["File"],
    summary="Retrieve file content",
    description="Fetch the content of a specific file by its path.",
)
async def get_file(request: FileRequest, api_key: str = Depends(verify_api_key)):
    try:
        file_path = BASE_DIR / request.path
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        content = file_path.read_text()
        tokens = count_tokens(content)
        return {
            "content": content,
            "metadata": {"path": request.path, "source": "codebase"},
            "tokens": tokens,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"File fetch failed: {e}")
        raise HTTPException(status_code=404, detail=f"File fetch failed: {e}") from e


@app.post(
    "/mcp/v1/context/reindex",
    response_model=ReindexResponse,
    tags=["Reindex"],
    summary="Reindex codebase",
    description="Trigger incremental reindexing of the codebase.",
)
async def reindex(api_key: str = Depends(verify_api_key)):
    try:
        Indexer(BASE_DIR).index_incremental()
        return {"status": "reindexed"}
    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}") from e
