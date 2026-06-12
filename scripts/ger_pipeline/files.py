"""Small filesystem checks shared by GER pipeline steps."""
from __future__ import annotations

from pathlib import Path


def file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def require_file(path: Path) -> None:
    if not file_ok(path):
        raise FileNotFoundError(path)

