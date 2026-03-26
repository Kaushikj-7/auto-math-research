from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


AST = Any

ALLOWED_VARIABLES = {"W_t", "grad_L", "M_t", "V_t", "alpha", "epsilon"}
UNARY_OPS = {
    "transpose",
    "trace",
    "diag",
    "exp",
    "sign",
    "cayley",
    "skew",
    "symm",
    "tanh",
    "norm",
    "fft",
    "ifft",
    "random_normal",  # cuRAND
}
BINARY_OPS = {"add", "sub", "hadamard", "matmul", "mul"}
NARY_ASSOCIATIVE_OPS = {"add", "hadamard", "mul"}


class ASTValidationError(ValueError):
    pass


@dataclass(frozen=True)
class NodeType:
    kind: str  # scalar | matrix


def _is_number(node: AST) -> bool:
    return isinstance(node, (int, float))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ASTValidationError(message)


def validate_ast(node: AST) -> None:
    if isinstance(node, str):
        _assert(node in ALLOWED_VARIABLES, f"Unknown variable: {node}")
        return

    if _is_number(node):
        return

    _assert(
        isinstance(node, list), f"Node must be list/string/number. Got {type(node)!r}"
    )
    _assert(len(node) >= 2, "Operator nodes require at least op + one operand")

    op = node[0]
    _assert(isinstance(op, str), "Operator must be a string")
    _assert(op in UNARY_OPS or op in BINARY_OPS, f"Unsupported operator: {op}")

    if op in UNARY_OPS:
        _assert(len(node) == 2, f"Unary operator '{op}' must have exactly one operand")
    elif op in BINARY_OPS:
        if op in NARY_ASSOCIATIVE_OPS:
            _assert(
                len(node) >= 3,
                f"Associative operator '{op}' needs at least two operands",
            )
        else:
            _assert(
                len(node) == 3, f"Binary operator '{op}' must have exactly two operands"
            )

    for child in node[1:]:
        validate_ast(child)


def infer_type(node: AST) -> NodeType:
    if isinstance(node, str):
        if node in {"alpha", "epsilon"}:
            return NodeType("scalar")
        return NodeType("matrix")

    if _is_number(node):
        return NodeType("scalar")

    op = node[0]
    children = [infer_type(child) for child in node[1:]]

    if op in {"trace", "norm"}:
        return NodeType("scalar")

    if op in {"transpose", "diag", "exp", "sign", "cayley", "skew", "symm"}:
        _assert(children[0].kind == "matrix", f"{op} requires matrix input")
        return NodeType("matrix")

    if op == "tanh":
        return children[0]

    if op in {"matmul", "hadamard", "add", "sub", "mul"}:
        if op == "mul":
            if all(child.kind == "scalar" for child in children):
                return NodeType("scalar")
            if all(child.kind == "matrix" for child in children):
                return NodeType("matrix")
            if len(children) == 2 and {children[0].kind, children[1].kind} == {
                "scalar",
                "matrix",
            }:
                return NodeType("matrix")
            raise ASTValidationError(
                "mul supports scalar-scalar, matrix-matrix, or scalar-matrix"
            )

        _assert(
            all(child.kind == "matrix" for child in children),
            f"{op} requires matrix operands",
        )
        return NodeType("matrix")

    raise ASTValidationError(f"Cannot infer type for operator: {op}")


def ensure_matrix_update(node: AST) -> None:
    node_type = infer_type(node)
    if node_type.kind != "matrix":
        raise ASTValidationError("Final update expression must produce a matrix")


def validate_allowed_ops_subset(node: AST, allowed_ops: Iterable[str]) -> None:
    allowed = set(allowed_ops)

    def walk(current: AST) -> None:
        if not isinstance(current, list):
            return
        op = current[0]
        if op not in allowed:
            raise ASTValidationError(f"Operator '{op}' not allowed in this run profile")
        for child in current[1:]:
            walk(child)

    walk(node)
