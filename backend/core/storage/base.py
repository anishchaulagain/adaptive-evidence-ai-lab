"""Object storage abstraction (spec section 6).

Implementations: local filesystem (development), S3, Azure Blob. Credentials
stay server-side; the browser receives a signed URL at most.
"""

from __future__ import annotations

from typing import BinaryIO, Protocol


class ObjectStorage(Protocol):
    async def put(self, key: str, data: BinaryIO, *, content_type: str) -> str: ...

    async def get(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...

    async def signed_url(self, key: str, *, expires_in: int) -> str: ...
