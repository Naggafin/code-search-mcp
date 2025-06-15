import sys
from pathlib import Path

# EDIT: Update import paths to reflect the new directory structure
from code_search_mcp.mcp_search import (  # Updated for new structure, adjust based on actual imports
    Searcher,
)

# Ensure project root is importable when running `pytest` from repository root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
