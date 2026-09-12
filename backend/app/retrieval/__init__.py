"""Retriever implementations.

The `Retriever` protocol lives in `core.retrieval.base`; the implementations
here are database-backed and therefore belong in the application layer, beside
the ORM. Phase 4 adds a keyword retriever alongside this one, and Phase 5
composes them.
"""
