import json
import logging
import sqlite3
from hashlib import md5
from pathlib import Path
from sqlite3 import OperationalError

import magic
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

logger = logging.getLogger(__name__)

# Common code-related MIME type prefixes
CODE_MIME_PREFIXES = ("text/x-", "application/x-")

# Known file extensions for source code (extend as needed)
KNOWN_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".rb",
    ".go",
    ".php",
    ".rs",
    ".swift",
    ".kt",
    ".kts",
    ".sh",
    ".pl",
    ".r",
    ".lua",
    ".sql",
    ".scala",
    ".html",
    ".css",
    ".json",
    ".xml",
    ".yml",
    ".yaml",
}

DB_PATH = Path(".embed_cache/code_embeddings.db")
DB_PATH.parent.mkdir(exist_ok=True)

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    CODE_TOKENIZER = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    CODE_MODEL = AutoModel.from_pretrained("microsoft/codebert-base")
    CODE_MODEL.eval()
    TEXT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    CODE_TOKENIZER, CODE_MODEL, TEXT_MODEL = None, None, None
    logger.info(
        "Models unavailable. Install `transformers`, `sentence-transformers`, `torch`."
    )


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                text TEXT,
                embedding TEXT,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def get_cache_key(text: str, model: str) -> str:
    return md5(f"{text}-{model}".encode()).hexdigest()


def load_from_cache(key: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute("SELECT embedding FROM embeddings WHERE id = ?", (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
    except OperationalError:
        init_db()
    return None


def save_to_cache(key: str, text: str, embedding, model):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (id, text, embedding, model) VALUES (?, ?, ?, ?)",
            (key, text, json.dumps(embedding), model),
        )


def batch_generator(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def is_probably_code(file_path: Path, mime_detector) -> bool:
    try:
        mime_type = mime_detector.from_file(str(file_path))

        # 1. MIME type check
        if mime_type.startswith(CODE_MIME_PREFIXES):
            return True

        # 2. Extension-based fallback
        if file_path.suffix.lower() in KNOWN_CODE_EXTENSIONS:
            return True

        # 3. Pygments lexer-based detection from file content
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            guess_lexer(text)
            return True
        except UnicodeDecodeError as e:
            logger.error(f"Can't decode {file_path}: {e}")
            raise
        except ClassNotFound:
            return False

    except Exception:
        return False


def embed(chunks, batch_size=32):
    if not MODEL_AVAILABLE:
        raise RuntimeError("Models unavailable. Install required packages.")

    mime = magic.Magic(mime=True)
    results = []

    for batch in batch_generator(chunks, batch_size):
        for file_path, chunk_data in batch:
            model_name = (
                "codebert-base"
                if is_probably_code(Path(file_path), mime)
                else "all-MiniLM-L6-v2"
            )

            text = chunk_data["text"]
            metadata = chunk_data["metadata"]
            metadata["model"] = model_name

            key = get_cache_key(text, model_name)
            cached = load_from_cache(key)
            if cached:
                results.append(cached)
                continue

            if model_name == "codebert-base":
                tokens = CODE_TOKENIZER(
                    text, return_tensors="pt", truncation=True, padding=True
                )
                with torch.no_grad():
                    output = CODE_MODEL(**tokens)
                    embedding = output.last_hidden_state.mean(dim=1).squeeze().tolist()
            else:
                embedding = TEXT_MODEL.encode(text, convert_to_tensor=False).tolist()

            save_to_cache(key, text, embedding, model_name)
            results.append(embedding)

    return results
