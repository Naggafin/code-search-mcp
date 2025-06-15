from config import settings
from fastapi import Header, HTTPException

# EDIT: Update import paths to reflect the new directory structure
# Assuming this file has imports like from mcp_server or others, update to use code_search_mcp prefix
from code_search_mcp.mcp_server import (  # Example; replace with actual imports if needed
    some_function,
)


def verify_api_key(x_api_key: str = Header(...)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
