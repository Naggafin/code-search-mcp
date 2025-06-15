from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import libcst as cst
import magic
import pathspec
from libcst import metadata as _cst_meta
from tree_sitter_languages import get_parser

logger = logging.getLogger(__name__)


# Helper to get mime type if libmagic available
def _get_mime(path: Path) -> str:
    try:
        return magic.from_file(str(path))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Python (libcst) extraction -------------------------------------------------


class _PyVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (_cst_meta.PositionProvider,)

    def __init__(self, lines: List[str]):
        self._lines = lines
        self.chunks: list[dict] = []
        self._class_depth = 0

    def _add(self, node: cst.CSTNode, type: str, name: str):
        pos = self.get_metadata(_cst_meta.PositionProvider, node)
        start, end = pos.start.line, pos.end.line
        text = "\n".join(self._lines[start - 1 : end])
        self.chunks.append(
            {
                "text": text,
                "metadata": {
                    "type": type,
                    "name": name,
                    "start_line": start,
                    "end_line": end,
                },
            }
        )

    def visit_ClassDef(self, node: cst.ClassDef):
        self._add(node, "class", node.name.value)
        self._class_depth += 1

    def leave_ClassDef(self, _node: cst.ClassDef):
        self._class_depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef):
        if self._class_depth:
            return False  # skip methods; included in class chunk
        self._add(node, "function", node.name.value)


def _extract_python(code: str, file_path: Path) -> list[dict]:
    """Chunk python *code* using libcst."""
    try:
        module = cst.parse_module(code)
        wrapper = cst.MetadataWrapper(module)
        visitor = _PyVisitor(code.splitlines())
        wrapper.visit(visitor)
        return visitor.chunks
    except Exception as exc:  # pragma: no cover
        logger.warning("libcst parse failed for %s: %s", file_path, exc)
        return []


# ---------------------------------------------------------------------------
# Tree-sitter extraction -----------------------------------------------------

# TODO: Does tree-sitter support more languages than just these? Let's add them if so.
SUPPORTED_LANGS = {
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".java": "java",
}

# TODO: If you add more languages to SUPPORTED_LANGS, update LANG_NODE_TYPES as well.
LANG_NODE_TYPES = {
    "javascript": ["function_declaration", "method_definition", "class_declaration"],
    "typescript": ["function_declaration", "method_definition", "class_declaration"],
    "go": ["function_declaration", "method_declaration"],
    "java": ["method_declaration", "class_declaration"],
}


def _extract_tree_sitter(code: str, file_path: Path) -> list[dict]:
    lang = SUPPORTED_LANGS.get(file_path.suffix.lower())
    if not lang:
        return []

    try:
        parser = get_parser(lang)
        tree = parser.parse(code.encode())
    except Exception as exc:  # pragma: no cover
        logger.warning("tree-sitter parse failed for %s: %s", file_path, exc)
        return []

    chunks: list[dict] = []
    for node in tree.root_node.walk():
        if node.type not in LANG_NODE_TYPES[lang]:
            continue
        start_line, end_line = node.start_point[0] + 1, node.end_point[0] + 1
        text = code[node.start_byte : node.end_byte]
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode() if name_node else "<anon>"
        chunks.append(
            {
                "text": text,
                "metadata": {
                    "type": node.type,
                    "name": name,
                    "start_line": start_line,
                    "end_line": end_line,
                },
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------


def extract_code_chunks(code: str, file_path: Path) -> list[dict]:
    """Return list of code chunks extracted from *file_path* contents."""
    chunks: list[dict] = []
    if file_path.suffix == ".py":
        chunks = _extract_python(code, file_path)
    if not chunks:
        chunks = _extract_tree_sitter(code, file_path)
    if not chunks:
        lines = code.splitlines()
        chunks = [
            {
                "text": code,
                "metadata": {
                    "type": "file",
                    "name": file_path.name,
                    "start_line": 1,
                    "end_line": len(lines),
                },
            }
        ]
    return chunks


def load_gitignore_patterns(project_path: Path) -> pathspec.PathSpec:
    """Load .gitignore patterns from the project directory."""
    gitignore_path = project_path / ".gitignore"
    patterns = []

    if gitignore_path.exists():
        try:
            with gitignore_path.open("r", encoding="utf-8", errors="ignore") as f:
                patterns = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
        except Exception as e:
            logger.warning(f"Failed to read .gitignore: {e}")

    # Add default patterns as fallback
    default_patterns = [
        "venv/",
        "__pycache__/",
        "migrations/",
        ".git/",
        ".pytest_cache/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".env",
        "dist/",
        "build/",
    ]

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns + default_patterns)


def is_text_file(file_path: Path) -> bool:
    """Heuristic: use libmagic when available else fallback to null-byte sniff."""
    mime_type = _get_mime(file_path)
    if mime_type.startswith("text/") or mime_type.startswith(
        ("application/json", "application/xml")
    ):
        return True

    try:
        with file_path.open("rb") as f:
            sample = f.read(2048)
            if b"\x00" in sample:
                return False
    except Exception as e:
        logger.debug("binary check failed on %s: %s", file_path, e)
        return False

    return True


def scan_project(
    path: Path,
    suffixes: Optional[Union[str, Iterable[str]]] = None,
    suppress_errors: bool = True,
) -> Iterator[Tuple[Path, List[Dict[str, Any]]]]:
    if suffixes is None:
        suffixes = [""]  # Match all files
    gitignore_spec = load_gitignore_patterns(path)

    for suffix in suffixes:
        for file in path.rglob(f"*{suffix}"):
            if not file.is_file():
                continue

            relative_path = file.relative_to(path).as_posix()
            if gitignore_spec.match_file(relative_path):
                continue

            if not is_text_file(file):
                logger.debug(f"Skipping binary file: {file}")
                continue

            try:
                with file.open("rb") as f:
                    content_bytes = f.read()
                mime_type = magic.from_buffer(content_bytes)
                if mime_type.startswith("text/") or mime_type in (
                    "application/json",
                    "application/xml",
                ):
                    code_str = content_bytes.decode("utf-8", errors="replace")
                    yield file, extract_code_chunks(code_str, file)
                else:
                    logger.debug(f"Skipping non-text file based on mime: {file}")
            except UnicodeDecodeError as e:
                logger.error(f"Can't decode {file}: {e}")
                if not suppress_errors:
                    raise
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                if not suppress_errors:
                    raise
