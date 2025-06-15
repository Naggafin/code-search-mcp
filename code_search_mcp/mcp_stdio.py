"""StdIO adapter for Code Search MCP.

Run with::

    python -m mcp_stdio [--project PATH]

It reads JSON-lines from *stdin* and writes JSON-lines replies to *stdout*.

Protocol (MVP)
--------------
Input – one JSON object per line::

    {"type": "context", "query": "paginate queryset", "max_tokens": 8000}

Optional keys:
* "stream": true -> stream chunk / end events.
* "metadata_filter": {"type": "class"}

Output – newline-delimited JSON:
1. Non-stream requests → single line::

       {"status": "ok", "data": { … }}

2. Streaming requests → one line per event::

       {"event": "chunk", "data": {...}}
       {"event": "end", "data": {...}}

Errors are returned as::

       {"status": "error", "message": "…"}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from code_search_mcp.config import settings  # Example based on actual imports
from code_search_mcp.mcp_search import Searcher  # Updated for new structure

# --------------------------------------------------------------------------- helpers


def _parse_sse_event(sse: str) -> Dict[str, Any]:
    """Convert an ``event: …\ndata: …`` string to a JSON dict."""
    lines = [ln for ln in sse.split("\n") if ln]
    if (
        len(lines) < 2
        or not lines[0].startswith("event:")
        or not lines[1].startswith("data:")
    ):
        return {"event": "raw", "data": sse}
    event = lines[0].split(":", 1)[1].strip()
    data_part = lines[1].split(":", 1)[1].strip()
    try:
        data = json.loads(data_part)
    except json.JSONDecodeError:
        data = data_part
    return {"event": event, "data": data}


def _handle_request(
    req: Dict[str, Any], searcher: Searcher
) -> Iterable[Dict[str, Any]]:
    """Process a single protocol request and yield reply dicts."""

    req_type = req.get("type")
    if req_type not in {"context", "search"}:
        yield {"status": "error", "message": f"Unknown request type '{req_type}'"}
        return

    query = req.get("query")
    if not isinstance(query, str):
        yield {"status": "error", "message": "Missing or invalid 'query' field"}
        return

    k = int(req.get("k", 5))
    max_tokens = int(req.get("max_tokens", 8000))
    metadata_filter = req.get("metadata_filter")

    if req_type == "search":
        data = searcher.search(
            query, k=k, max_tokens=max_tokens, metadata_filter=metadata_filter
        )
        yield {"status": "ok", "data": data}
        return

    # context request – decide streaming vs aggregate
    if req.get("stream"):
        # convert SSE strings to JSON events
        for sse in searcher.stream_context(
            query,
            k=k,
            max_tokens=max_tokens,
            metadata_filter=metadata_filter,
        ):
            yield _parse_sse_event(sse)
    else:
        data = searcher.context(
            query, k=k, max_tokens=max_tokens, metadata_filter=metadata_filter
        )
        yield {"status": "ok", "data": data}


# --------------------------------------------------------------------------- main loop


def _main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mcp-stdio", description="Run MCP StdIO adapter"
    )
    parser.add_argument(
        "--project", type=Path, default=".", help="Project path for Searcher"
    )
    args = parser.parse_args(argv)

    searcher = Searcher(project_path=args.project)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            json.dump(
                {"status": "error", "message": f"JSON decode error: {exc}"}, sys.stdout
            )
            sys.stdout.write("\n")
            sys.stdout.flush()
            continue

        for reply in _handle_request(request, searcher):
            json.dump(reply, sys.stdout, ensure_ascii=False)
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    _main()
