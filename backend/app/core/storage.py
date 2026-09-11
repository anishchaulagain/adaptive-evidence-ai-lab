"""Object storage selection — the composition root for storage.

Lives in `app` rather than `core` because it reads `Settings`. Callers depend
only on the `ObjectStorage` protocol (spec section 6), so swapping local
storage for S3 touches this function alone.
"""

from __future__ import annotations

from app.core.config import ObjectStorageBackend, Settings
from core.storage.base import ObjectStorage
from core.storage.local import LocalObjectStorage


def get_object_storage(settings: Settings) -> ObjectStorage:
    match settings.OBJECT_STORAGE_BACKEND:
        case ObjectStorageBackend.LOCAL:
            return LocalObjectStorage(settings.OBJECT_STORAGE_LOCAL_PATH)
        case backend:
            raise NotImplementedError(f"Object storage backend {backend!r} is not implemented.")
