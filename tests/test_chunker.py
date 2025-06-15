from pathlib import Path

from code_search_mcp.chunker import extract_code_chunks


def test_python_extraction():
    code = (
        "class Foo:\n"  # class
        "    def bar(self):\n"  # method
        "        return 1\n"  # body
        "\n"  # empty line
        "def baz(x, y):\n"  # function
        "    return x + y\n"
    )
    chunks = extract_code_chunks(code, Path("fake.py"))
    # Should produce two chunks: class Foo and function baz
    names = sorted(c["metadata"]["name"] for c in chunks)
    assert names == ["Foo", "baz"], chunks


def test_tree_sitter_extraction_js():
    js_code = (
        "function add(a, b) {\n"  # function
        "  return a + b;\n"  # body
        "}\n"
    )
    chunks = extract_code_chunks(js_code, Path("util.js"))
    # Expect at least one chunk with name 'add'
    assert any(c["metadata"]["name"] == "add" for c in chunks)
