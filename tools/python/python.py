from __future__ import annotations

from dataclasses import dataclass
import ast
import math


@dataclass
class PythonToolResult:
    success: bool
    value: object = None
    error: str | None = None


class PythonTool:
    """
    Intentionally restricted expression evaluator.

    This is not a replacement for a sandboxed Python runtime.
    """

    _allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "pow": pow,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
    }

    def execute(self, expression: str) -> PythonToolResult:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            return PythonToolResult(False, error=str(exc))

        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom,
                    ast.Attribute,
                    ast.Lambda,
                    ast.NamedExpr,
                    ast.ListComp,
                    ast.DictComp,
                    ast.SetComp,
                    ast.GeneratorExp,
                ),
            ):
                return PythonToolResult(
                    False,
                    error="Expression contains a forbidden construct.",
                )

            if isinstance(node, ast.Name) and node.id not in self._allowed_names:
                return PythonToolResult(
                    False,
                    error=f"Name '{node.id}' is not allowed.",
                )

        try:
            value = eval(
                compile(tree, "<numi-python>", "eval"),
                {"__builtins__": {}},
                self._allowed_names,
            )
        except Exception as exc:
            return PythonToolResult(False, error=str(exc))

        return PythonToolResult(True, value=value)
