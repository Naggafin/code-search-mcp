import logging
from pathlib import Path

from auth import verify_api_key
from config import settings
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from search_engine import index_project_incremental, search_code_hybrid
from token_counter import count_tokens

BASE_DIR = Path(settings.project_path)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP Code Search Server",
    description="MCP-compliant server for code search and context retrieval, supporting OpenHands for Django projects.",
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


class ContextRequest(BaseModel):
    query: str = Field(
        ..., description="Search query for code context (e.g., 'Django user model')."
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


@app.post(
    "/mcp/v1/context",
    response_model=ContextResponse,
    tags=["Context"],
    summary="Fetch code context",
    description="Retrieve code context based on a query, with optional metadata filtering.",
)
async def get_context(request: ContextRequest, api_key: str = Depends(verify_api_key)):
    try:
        result = search_code_hybrid(
            request.query,
            k=5,
            max_tokens=request.max_tokens,
            metadata_filter=request.metadata_filter,
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


@app.post(
    "/mcp/v1/context/search",
    response_model=SearchResponse,
    tags=["Context"],
    summary="Search code context",
    description="Search for code snippets with optional metadata filtering.",
)
async def search_context(
    request: ContextRequest, api_key: str = Depends(verify_api_key)
):
    try:
        result = search_code_hybrid(
            request.query,
            k=5,
            max_tokens=request.max_tokens,
            metadata_filter=request.metadata_filter,
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
        index_project_incremental(BASE_DIR)
        return {"status": "reindexed"}
    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}") from e
