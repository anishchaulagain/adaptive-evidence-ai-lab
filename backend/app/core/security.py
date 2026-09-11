"""Authentication, authorization and tenant isolation primitives.

Spec section 42/43: project and source isolation are enforced through the
`Principal`, and provider credentials never leave the backend.

Phase 1 runs with `AUTH_MODE=disabled` and a fixed development principal, so
that tenancy columns are populated from day one. When real authentication
lands, only `app.api.deps.get_current_principal` changes — every caller and
every query already filters on the principal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller. Every tenant-scoped query filters on this."""

    user_id: UUID
    organization_id: UUID
    scopes: frozenset[str] = field(default_factory=frozenset)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


# Fixed identifiers so local data survives restarts and so tests are
# deterministic. Seeded by `ensure_dev_principal()` at startup.
DEV_ORGANIZATION_ID = UUID("00000000-0000-0000-0000-000000000001")
DEV_USER_ID = UUID("00000000-0000-0000-0000-000000000002")
DEV_USER_EMAIL = "dev@localhost"

DEV_PRINCIPAL = Principal(
    user_id=DEV_USER_ID,
    organization_id=DEV_ORGANIZATION_ID,
    scopes=frozenset({"*"}),
)


def hash_password(plain: str) -> str:
    """Hash a password with a memory-hard KDF (argon2)."""
    raise NotImplementedError


def verify_password(plain: str, hashed: str) -> bool:
    raise NotImplementedError


def create_access_token(principal: Principal, *, expires_in: int) -> str:
    raise NotImplementedError


def decode_access_token(token: str) -> Principal:
    """Decode and validate a token. Raises `UnauthenticatedError` when invalid."""
    raise NotImplementedError
