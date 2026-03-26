from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

COMMUTATIVE_OPS = {"add", "hadamard", "mul"}
ASSOCIATIVE_OPS = {"add", "hadamard", "mul"}

AST = Any


def _stable_json(node: AST) -> str:
    return json.dumps(node, separators=(",", ":"), sort_keys=False)


def _node_hash(node: AST) -> str:
    return sha256(_stable_json(node).encode("utf-8")).hexdigest()


def canonicalize_ast(node: AST) -> AST:
    if not isinstance(node, list):
        return node

    op = node[0]
    children = [canonicalize_ast(child) for child in node[1:]]

    if op in ASSOCIATIVE_OPS:
        flattened = []
        for child in children:
            if isinstance(child, list) and child and child[0] == op:
                flattened.extend(child[1:])
            else:
                flattened.append(child)
        children = flattened

    if op in COMMUTATIVE_OPS:
        children = sorted(children, key=_node_hash)

    return [op, *children]


def canonical_json(node: AST) -> str:
    canonical = canonicalize_ast(node)
    return _stable_json(canonical)


def canonical_hash(node: AST) -> str:
    return sha256(canonical_json(node).encode("utf-8")).hexdigest()
