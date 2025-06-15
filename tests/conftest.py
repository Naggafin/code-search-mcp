import sys
from pathlib import Path

# EDIT: Update import paths to reflect the new directory structure

# Ensure project root is importable when running `pytest` from repository root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
