#!/usr/bin/env python3
"""Verify that the frozen Level 6 runtime already contains v1524 micro."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path


EXPECTED_LEVEL6_SHA = "76354ff908b47f8c400e2cf279a3af5f49e98a3da6afb53018143b579f28c6da"
EXPECTED_V1524_SHA = "8c1c48126e444ecf604a7ac79989c5c9e8dd06eaca31a5e8786b0e5f16ae4e8c"
EXPECTED_DIFFERENCES = {"_select_attack_compatible_enemy_unit"}
REQUIRED_MICRO_METHODS = {
    "_micro_get_combat_tags",
    "_micro_basic_hooligan_attack",
    "micro_update_health_snapshot",
    "micro_dodge",
    "micro_special_abilities",
    "micro_fight",
    "_flush_micro_trace",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def methods(path: Path) -> dict[str, str]:
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    script_player = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ScriptPlayer"
    )
    return {
        node.name: hashlib.sha256(
            "".join(lines[node.lineno - 1 : node.end_lineno]).encode()
        ).hexdigest()
        for node in script_player.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_v1524_compat.py LEVEL6_SCRIPT V1524_SCRIPT")

    level6_path, v1524_path = map(Path, sys.argv[1:])
    assert sha256(level6_path) == EXPECTED_LEVEL6_SHA
    assert sha256(v1524_path) == EXPECTED_V1524_SHA

    level6 = methods(level6_path)
    v1524 = methods(v1524_path)
    assert set(level6) == set(v1524)
    differences = {name for name in level6 if level6[name] != v1524[name]}
    assert differences == EXPECTED_DIFFERENCES, differences
    assert REQUIRED_MICRO_METHODS <= set(level6)
    assert all(level6[name] == v1524[name] for name in REQUIRED_MICRO_METHODS)

    source = level6_path.read_text()
    assert 'SC2_ENABLE_NEWEXEC_MICRO", "0"' in source
    assert "await self.micro_dodge(iteration)" in source
    assert "await self.micro_special_abilities(iteration)" in source
    assert "await self.micro_fight(iteration)" in source
    print("V1524_MICRO_COMPAT_STATUS=0")
    print("METHOD_COUNT=72")
    print("ONLY_DIFFERING_METHOD=_select_attack_compatible_enemy_unit")


if __name__ == "__main__":
    main()
