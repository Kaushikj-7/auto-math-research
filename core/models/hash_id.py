from __future__ import annotations

from .ast_canonicalize import canonical_hash


def ast_hash(ast_node):
    return canonical_hash(ast_node)
