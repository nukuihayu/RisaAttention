"""Load the repository's RISA Python package for source-based tests."""

import sys
from pathlib import Path

try:
    from risa_attention import attention as risa
except ModuleNotFoundError as error:
    if error.name != "risa_attention":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from risa_attention import attention as risa

__all__ = ["risa"]
