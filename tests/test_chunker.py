import pytest
from pathlib import Path

import chunker as ch


def test_python_extraction():
    code = (
        "class Foo:\n"  # class
        "    def bar(self):\n"  # method
        "        return 1\n"  # body
        "\n"  # empty line
        "def baz(x, y):\n"  # function
        "    return x + y\n"
    )
    chunks = ch.extract_code_chunks(code, Path("fake.py"))
    # Should produce two chunks: class Foo and function baz
    names = sorted(c["metadata"]["name"] for c in chunks)
    assert names == ["Foo", "baz"], chunks


@pytest.mark.skipif(not ch.TS_AVAILABLE, reason="tree-sitter not available")
def test_tree_sitter_extraction_js():
    js_code = (
        "function add(a, b) {\n"  # function
        "  return a + b;\n"  # body
        "}\n"
    )
    chunks = ch.extract_code_chunks(js_code, Path("util.js"))
    # Expect at least one chunk with name 'add'
    assert any(c["metadata"]["name"] == "add" for c in chunks)
