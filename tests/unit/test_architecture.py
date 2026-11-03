"""Architecture test: the layer rule from docs/03 is enforced, not asked for politely.

A dependency rule that only lives in a diagram is a suggestion. This test parses every module and fails on
an import that points backwards, so the rule survives contact with a deadline.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "aletheia"

ALLOWED: dict[str, set[str]] = {
    "core": {"core"},
    "ml": {"ml", "core"},
    "db": {"db", "core"},
    "services": {"services", "core", "ml", "db", "schemas"},
    "schemas": {"schemas", "core"},
    "reporting": {"reporting", "core", "ml", "db", "services"},
    "api": {"api", "core", "ml", "db", "services", "schemas", "reporting"},
}


def _modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _layer(path: Path) -> str | None:
    relative = path.relative_to(SRC)
    return relative.parts[0] if len(relative.parts) > 1 else None


def _imported_modules(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name)
    return found


@pytest.mark.unit
def test_every_module_respects_the_layer_rule() -> None:
    violations: list[str] = []
    for path in _modules():
        layer = _layer(path)
        if layer is None or layer not in ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _imported_modules(tree):
            if not module.startswith("aletheia."):
                continue
            target = module.split(".")[1]
            if target in ALLOWED[layer]:
                continue
            violations.append(f"{path.relative_to(SRC)}: {layer} must not import {target}")
    assert violations == []


def test_sources_actually_exist() -> None:
    assert _modules(), "no Python modules found; the layer test would pass vacuously"


def test_ml_has_no_framework_imports() -> None:
    forbidden = {"fastapi", "starlette", "sqlalchemy", "streamlit"}
    offenders: list[str] = []
    for path in (SRC / "ml").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _imported_modules(tree):
            if module.split(".")[0] in forbidden:
                offenders.append(f"{path.name}: {module}")
    assert offenders == []
