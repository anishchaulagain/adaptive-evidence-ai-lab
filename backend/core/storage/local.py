"""Local filesystem object storage, for development and tests.

Keys are treated as opaque paths under a root directory. They are validated
rather than trusted: a key that escapes the root is rejected, so a crafted
document name cannot write outside the store.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from core.errors import StorageError


class LocalObjectStorage:
    """Implements `ObjectStorage` against a directory tree."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _resolve(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root):
            raise StorageError(f"Object key escapes the storage root: {key!r}")
        return path

    async def put(self, key: str, data: BinaryIO, *, content_type: str) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            while chunk := data.read(1024 * 1024):
                handle.write(chunk)
        return key

    async def get(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise StorageError(f"Object not found: {key!r}") from exc

    async def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)

    async def signed_url(self, key: str, *, expires_in: int) -> str:
        """Local storage has no signed URLs; the API streams the bytes instead."""
        raise NotImplementedError(
            "Local storage cannot sign URLs. Use the document download endpoint."
        )
