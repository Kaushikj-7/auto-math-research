from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from .ast_validation import ASTValidationError, ensure_matrix_update, validate_ast

AST = Any
Shape = Tuple[int, int]


@dataclass(frozen=True)
class Inferred:
    kind: str  # scalar | matrix
    shape: Shape | None


class SymbolicSanitizationError(ASTValidationError):
    pass


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SymbolicSanitizationError(message)


def _infer(node: AST, matrix_shape: Shape) -> Inferred:
    if isinstance(node, (int, float)):
        return Inferred("scalar", None)

    if isinstance(node, str):
        if node in {"alpha", "epsilon"}:
            return Inferred("scalar", None)
        return Inferred("matrix", matrix_shape)

    op = node[0]
    args = [_infer(child, matrix_shape) for child in node[1:]]

    if op in {"norm", "trace"}:
        _assert(args[0].kind == "matrix", f"{op} requires matrix input")
        rows, cols = args[0].shape or (0, 1)
        _assert(rows == cols, f"{op} requires square matrix")
        return Inferred("scalar", None)

    if op in {
        "transpose",
        "exp",
        "sign",
        "skew",
        "symm",
        "cayley",
        "diag",
        "fft",
        "ifft",
    }:
        _assert(args[0].kind == "matrix", f"{op} requires matrix input")
        rows, cols = args[0].shape or (0, 1)
        if op in {"exp", "cayley", "skew", "symm"}:
            _assert(rows == cols, f"{op} requires square matrix")
        if op == "transpose":
            return Inferred("matrix", (cols, rows))
        if op in {"fft", "ifft"}:
            return Inferred("matrix", args[0].shape)
        return Inferred("matrix", args[0].shape)

    if op == "random_normal":
        if not args:
            # Maybe takes 0 args? But infer checks children.
            pass
        return Inferred("matrix", matrix_shape)  # assumes standard shape

    if op == "tanh":
        return args[0]

    if op == "sub":
        _assert(
            args[0].kind == "matrix" and args[1].kind == "matrix",
            "sub requires matrix operands",
        )
        _assert(args[0].shape == args[1].shape, "sub requires equal shapes")
        return Inferred("matrix", args[0].shape)

    if op in {"add", "hadamard"}:
        _assert(
            all(arg.kind == "matrix" for arg in args), f"{op} requires matrix operands"
        )
        first_shape = args[0].shape
        _assert(
            all(arg.shape == first_shape for arg in args), f"{op} requires equal shapes"
        )
        return Inferred("matrix", first_shape)

    if op == "matmul":
        _assert(
            args[0].kind == "matrix" and args[1].kind == "matrix",
            "matmul requires matrix operands",
        )
        left = args[0].shape
        right = args[1].shape
        _assert(
            left is not None and right is not None and left[1] == right[0],
            "matmul dimension mismatch",
        )
        return Inferred("matrix", (left[0], right[1]))

    if op == "mul":
        if all(arg.kind == "scalar" for arg in args):
            return Inferred("scalar", None)

        if len(args) == 2 and {args[0].kind, args[1].kind} == {"scalar", "matrix"}:
            matrix_shape_local = (
                args[0].shape if args[0].kind == "matrix" else args[1].shape
            )
            return Inferred("matrix", matrix_shape_local)

        _assert(
            all(arg.kind == "matrix" for arg in args),
            "mul supports matrix-matrix or scalar-matrix",
        )
        first_shape = args[0].shape
        _assert(
            all(arg.shape == first_shape for arg in args),
            "mul matrix operands must have equal shape",
        )
        return Inferred("matrix", first_shape)

    raise SymbolicSanitizationError(f"Unsupported operation in sanitizer: {op}")


def sanitize_ast(ast_node: AST, expected_matrix_shape: Shape = (128, 128)) -> None:
    try:
        validate_ast(ast_node)
        ensure_matrix_update(ast_node)
        final = _infer(ast_node, expected_matrix_shape)
        _assert(final.kind == "matrix", "Update must resolve to matrix")
        _assert(
            final.shape == expected_matrix_shape,
            "Update shape must match parameter shape",
        )
    except ASTValidationError as err:
        raise SymbolicSanitizationError(str(err)) from err
