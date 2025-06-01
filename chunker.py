import logging
from pathlib import Path
from typing import Iterable, Optional, Union

import libcst as cst
import magic
import pathspec

logger = logging.getLogger(__name__)

MIME = magic.Magic(mime=True)


def extract_code_chunks(code, file_path):
    class CodeVisitor(cst.CSTVisitor):
        def __init__(self):
            self.chunks = []
            self.current_line = 1

        def visit_FunctionDef(self, node):
            chunk = node.code
            name = node.name.value
            start_line = node.get_metadata(
                cst.MetadataWrapper(node).position
            ).start.line
            self.chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "type": "function",
                        "name": name,
                        "start_line": start_line,
                        "end_line": start_line + len(chunk.splitlines()),
                    },
                }
            )

        def visit_ClassDef(self, node):
            chunk = node.code
            name = node.name.value
            start_line = node.get_metadata(
                cst.MetadataWrapper(node).position
            ).start.line
            self.chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "type": "class",
                        "name": name,
                        "start_line": start_line,
                        "end_line": start_line + len(chunk.splitlines()),
                    },
                }
            )

    try:
        tree = cst.parse_module(code)
        visitor = CodeVisitor()
        tree.visit(visitor)
        chunks = visitor.chunks
    except Exception:
        chunks = []

    # Fallback for non-Python or unparsable files
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
    """Determine if a file is likely to be a text file using MIME and content heuristics."""
    try:
        mime_type = MIME.from_file(str(file_path))
        if mime_type.startswith("text/"):
            return True
        # Accept some source code MIME types
        if mime_type.startswith(("application/json", "application/xml")):
            return True
    except Exception as e:
        logger.warning(f"Failed to get MIME type for {file_path}: {e}")

    # Fallback: check for null bytes
    try:
        with file_path.open("rb") as f:
            sample = f.read(2048)
            if b"\x00" in sample:
                return False
    except Exception as e:
        logger.warning(f"Failed to read file {file_path} for binary check: {e}")
        return False

    return True


def scan_project(
    path: Path,
    suffixes: Optional[Union[str, Iterable[str]]] = None,
    suppress_errors: bool = True,
):
    if suffixes is None:
        suffixes = [""]  # Match all files
    elif isinstance(suffixes, str):
        suffixes = [suffixes]

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
                yield file, extract_code_chunks(file.read_text(), file)
            except UnicodeDecodeError as e:
                logger.error(f"Can't decode {file}: {e}")
                if not suppress_errors:
                    raise
