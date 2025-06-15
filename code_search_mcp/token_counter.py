import logging

from code_search_mcp import tiktoken

logger = logging.getLogger(__name__)

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))
