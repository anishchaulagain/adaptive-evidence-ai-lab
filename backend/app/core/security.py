"""Authentication, authorization and tenant isolation primitives.

Spec section 42/43: project and source isolation must be enforced here, and
provider credentials must never leave the backend. Deliberately thin — the
concrete scheme is chosen in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller. Every tenant-scoped query filters on this."""

    user_id: UUID
    organization_id: UUID | None = None
    scopes: frozenset[str] = frozenset()

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


def hash_password(plain: str) -> str:
    """Hash a password with a memory-hard KDF (argon2/bcrypt)."""
    raise NotImplementedError


def verify_password(plain: str, hashed: str) -> bool:
    raise NotImplementedError


def create_access_token(principal: Principal, *, expires_in: int) -> str:
    raise NotImplementedError


def decode_access_token(token: str) -> Principal:
    """Decode and validate a token. Raises `UnauthenticatedError` when invalid."""
    raise NotImplementedError
