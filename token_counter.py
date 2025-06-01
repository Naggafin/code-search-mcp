import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken

    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TOKENIZER = None
    logger.error("Install `tiktoken` for token counting.")


def count_tokens(text: str) -> int:
    if TOKENIZER is None:
        # Fallback: estimate tokens as 4 chars per token
        return len(text) // 4
    return len(TOKENIZER.encode(text))
