"""Report missing docstrings in the public Python API."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def public_missing_docstrings(path: Path) -> list[tuple[int, str]]:
    """Return public module-level and class-level definitions without docs."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing = []

    if not ast.get_docstring(tree):
        missing.append((1, f"{path}: module"))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_") and not ast.get_docstring(node):
                missing.append((node.lineno, f"{path}: {node.name}"))
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                for member in node.body:
                    if (isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and not member.name.startswith("_")
                            and not ast.get_docstring(member)):
                        missing.append((member.lineno, f"{path}: {node.name}.{member.name}"))
    return missing


def main() -> int:
    """Run the audit for the supplied Python modules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return status 1 when any public object lacks a docstring",
    )
    args = parser.parse_args()

    missing = [item for path in args.paths for item in public_missing_docstrings(path)]
    if missing:
        for line, name in missing:
            print(f"{name}:{line}: missing docstring")
    else:
        print("No missing public docstrings found.")
    return int(bool(missing) and args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
