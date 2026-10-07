"""Restricted arithmetic expression evaluation for custom BSA indices.

Only numeric constants, supplied variable names, parentheses, and arithmetic
operators are accepted.  In particular, function calls, imports, attribute
access, indexing, comprehensions, and Python builtins are never available.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from typing import Any


class UnsafeExpressionError(ValueError):
    """Raised when a custom-index expression uses unsupported syntax."""


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPRESSION_LENGTH = 1_000
_MAX_AST_NODES = 128
_MAX_INTEGER_BITS = 256
_MAX_LITERAL_EXPONENT = 10_000


def evaluate_arithmetic_expression(
    expression: str,
    variables: Mapping[str, Any],
) -> Any:
    """Evaluate a deliberately small arithmetic language.

    Values may be scalars or NumPy arrays.  NumPy's ordinary broadcasting,
    dtype, division-by-zero, NaN, and infinity behavior is therefore retained.
    """

    if not isinstance(expression, str):
        raise UnsafeExpressionError("The formula must be text.")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError("The formula is too long.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError("The formula is not valid arithmetic.") from exc

    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise UnsafeExpressionError("The formula is too complex.")

    return _evaluate_node(tree.body, variables)


def referenced_variables(
    expression: str, allowed: tuple[str, ...] = ("Var1", "Var2", "Var3")
) -> frozenset[str]:
    """Validate an editor formula without evaluating it and return its variables."""

    if not isinstance(expression, str):
        raise UnsafeExpressionError("The formula must be text.")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError("The formula is too long.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError("The formula is not valid arithmetic.") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise UnsafeExpressionError("The formula is too complex.")

    names: set[str] = set()

    def inspect(node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise UnsafeExpressionError("Only numeric constants are supported.")
            if isinstance(value, int) and value.bit_length() > _MAX_INTEGER_BITS:
                raise UnsafeExpressionError("The numeric constant is too large.")
            return
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                raise UnsafeExpressionError(f"Unknown band reference: {node.id}")
            names.add(node.id)
            return
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            inspect(node.operand)
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            inspect(node.left)
            inspect(node.right)
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
                right = node.right.value
                if isinstance(right, (int, float)) and abs(right) > _MAX_LITERAL_EXPONENT:
                    raise UnsafeExpressionError("The exponent is too large.")
            return
        raise UnsafeExpressionError(
            f"Unsupported expression element: {type(node).__name__}"
        )

    inspect(tree.body)
    return frozenset(names)


def _evaluate_node(node: ast.AST, variables: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsafeExpressionError("Only numeric constants are supported.")
        if isinstance(value, int) and value.bit_length() > _MAX_INTEGER_BITS:
            raise UnsafeExpressionError("The numeric constant is too large.")
        return value

    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise UnsafeExpressionError(f"Unknown band reference: {node.id}")
        return variables[node.id]

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand, variables))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left, variables)
        right = _evaluate_node(node.right, variables)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)):
            if abs(right) > _MAX_LITERAL_EXPONENT:
                raise UnsafeExpressionError("The exponent is too large.")
        return _BINARY_OPERATORS[type(node.op)](left, right)

    raise UnsafeExpressionError(
        f"Unsupported expression element: {type(node).__name__}"
    )
