import logging

logger = logging.getLogger(__name__)

# TODO: let's not make this optional; failure to import should be a runtime error
try:
    import tiktoken

    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TOKENIZER = None
    logger.error("Install `tiktoken` for token counting.")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))
