"""A module's ``if __name__ == "__main__":`` block must be its LAST statement.

Python runs a script top to bottom, so a ``main()`` called from a guard placed
mid-file runs before the definitions below it exist. Importing the module (what
every other test does) defines everything first and hides the problem; only
``python -m`` hits it. ``make seed`` failed exactly this way: the guard sat above
``realign_identity_sequences`` and the CLI died with ``NameError``.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "tekijin"


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    left = node.test.left
    return isinstance(left, ast.Name) and left.id == "__name__"


def _modules_with_a_main_guard() -> list[tuple[Path, ast.Module]]:
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(_is_main_guard(node) for node in tree.body):
            found.append((path, tree))
    return found


def test_the_scan_finds_the_cli_modules() -> None:
    # Guards against the check below passing vacuously over an empty list.
    names = {path.name for path, _ in _modules_with_a_main_guard()}
    assert {"seed.py", "migrate.py"} <= names


def test_main_guard_is_the_last_statement() -> None:
    for path, tree in _modules_with_a_main_guard():
        assert _is_main_guard(tree.body[-1]), (
            f"{path.relative_to(SRC)}: the __main__ guard must come after every "
            "definition, or `python -m` calls main() before they exist"
        )
