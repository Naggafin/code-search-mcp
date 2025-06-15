"""Utility helpers for MCP search package."""

from __future__ import annotations

import json
from typing import Any

__all__ = ["sse_event"]


def sse_event(event: str, data: Any) -> str:  # noqa: D401
    """Return a formatted Server-Sent Event string.

    Parameters
    ----------
    event:
        Name of the SSE event (e.g. ``"chunk"`` or ``"end"``).
    data:
        JSON-serialisable payload. Will be serialized via :pymod:`json.dumps`.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
