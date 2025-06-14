# Code Search MCP
<!-- ruff: noqa -->

Code Search MCP ("Memory Context Provider") is a lightweight service that indexes a project folder into vector embeddings and serves relevant code context for LLMs.

* **Self-contained** – runs fully offline, no external DB.
* **Fast** – CPU-only inference with small (≤50 MB) models.
* **Extensible** – pluggable vector store, embedders & chunkers.

The FastAPI adapter included here is optional – the core library (`Indexer`, `Searcher`) may be imported directly once the library-first refactor (road-map phase 0.9) lands.

---

## Quick start (server)
```bash
python -m pip install -r requirements.txt
python mcp_server.py  # starts on http://0.0.0.0:8000
```

### Endpoints (excerpt)
| Method | Path                        | Purpose                                  |
|--------|----------------------------|------------------------------------------|
| POST   | `/mcp/v1/context`          | Return aggregated code context for query |
| POST   | `/mcp/v1/context/search`   | Raw search results (top-k chunks)        |
| GET    | `/mcp/v1/context/stream`   | **SSE** stream of chunks followed by end |
| POST   | `/mcp/v1/context/file`     | Retrieve full file contents              |

All endpoints are protected by an `X-API-Key` header. The key can be configured via the `MCP_API_KEY` environment variable (empty permits any key).

---

## Examples
Examples below assume the server is running on `localhost:8000` and the API key is `demo`.

### 1 · Streaming context via Server-Sent Events
```bash
curl -N -H "Accept: text/event-stream" \
     -H "X-API-Key: demo" \
     "http://localhost:8000/mcp/v1/context/stream?query=pydantic settings"
```
Typical response:
```
event: chunk
data: {"content": "class Settings(BaseSettings): …", "metadata": {"path": "config.py"}}

event: chunk
data: {…}

event: end
data: {"tokens": 752}
```
Each `chunk` carries a JSON object with `content` and `metadata`. The final `end` event contains token accounting.

### 2 · Handling rate limits (HTTP 429)
By default the public endpoints are limited to **60 requests per minute** per IP. When the limit is exceeded the service replies:
```bash
HTTP/1.1 429 Too Many Requests
Retry-After: 60

Rate limit exceeded: 60 per 1 minute"
```
The `Retry-After` header indicates when it is safe to retry.

---

See [SPECIFICATIONS.md](SPECIFICATIONS.md) for full technical details and the project road-map.
