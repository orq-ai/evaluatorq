"""Compile-check every example under examples/ so docs snippets stay valid.

ponytail: py_compile parses + byte-compiles without importing, so no example
side effects run and no third-party deps are needed beyond stdlib. Upgrade to
real execution only if a broken-at-runtime example slips past a clean parse.
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def main() -> int:
    files = sorted(EXAMPLES.rglob("*.py"))
    if not files:
        print(f"no example .py files under {EXAMPLES}", file=sys.stderr)
        return 1
    failed = 0
    for f in files:
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as exc:
            failed += 1
            print(f"FAIL {f.relative_to(EXAMPLES.parent)}: {exc.msg}", file=sys.stderr)
    print(f"compiled {len(files) - failed}/{len(files)} examples")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
