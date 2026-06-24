#!/usr/bin/env python3
"""Validate mermaid blocks in markdown for STRICT renderers (GitHub, VS Code preview).

mkdocs build --strict does NOT catch this: mermaid syntax errors are client-side.
GitHub and VS Code use a stricter parser than mmdc — bare ``/ : , ( ) -`` in a node
or edge label, or a literal ``\\n``, throws "Syntax error in text" for the reader.

This is a non-mutating CHECK. It reuses the exact tokenizer from the
``mermaid-markdown`` skill's harden script: a label is a defect if hardening would
change it (quote it / convert ``\\n`` -> ``<br/>``). To auto-fix, run:

    python3 ~/.claude/skills/mermaid-markdown/harden_mermaid.py docs/**/*.md

Usage:
    python scripts/validate_mermaid.py [paths...]   # default: docs/**/*.md
Exit 0 = clean, 1 = violations (or unbalanced fences).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# shape delimiter pairs, longest-open first so ([ / [( / (( / {{ win over [ / ( / {
PAIRS = [("([", "])"), ("[(", ")]"), ("((", "))"), ("{{", "}}"), ("[", "]"), ("{", "}")]


def quote_inner(inner: str) -> str:
    s = inner.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        # already quoted — still normalise literal \n inside to <br/>
        return inner.replace("\\n", "<br/>")
    s = s.replace("\\n", "<br/>")
    s = s.replace('"', "&quot;")
    return f'"{s}"'


def transform_node_labels(line: str) -> str:
    out = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        # never treat brackets inside an existing quoted span as shape delimiters
        if ch == '"':
            j = line.find('"', i + 1)
            if j == -1:
                out.append(line[i:])
                break
            out.append(line[i : j + 1].replace("\\n", "<br/>"))
            i = j + 1
            continue
        matched = False
        # a label opens only right after an identifier char (node id) or 'subgraph ID'
        for op, cl in PAIRS:
            if line.startswith(op, i) and i > 0 and (line[i - 1].isalnum() or line[i - 1] in "_-"):
                j = line.find(cl, i + len(op))
                if j == -1:
                    continue
                inner = line[i + len(op) : j]
                if '"' in inner:  # quoted-label boundary fell inside; let quote pass handle it
                    continue
                out.append(op)
                out.append(quote_inner(inner))
                out.append(cl)
                i = j + len(cl)
                matched = True
                break
        if not matched:
            out.append(ch)
            i += 1
    return "".join(out)


def transform_edge_labels(line: str) -> str:
    # pipe labels: -->|text|  ==> -->|"text"|
    def pipe(m: re.Match[str]) -> str:
        return "|" + quote_inner(m.group(1)) + "|"

    line = re.sub(r"\|([^|]*)\|", pipe, line)

    # dotted/standard inline labels: -. text .->  or  -. text .-
    def dotted(m: re.Match[str]) -> str:
        return m.group(1) + quote_inner(m.group(2)) + m.group(3)

    line = re.sub(r"(-\.\s*)([^.\"|][^.]*?)(\s*\.-?->?)", dotted, line)
    return line


_DIAGRAM_KEYWORDS = ("flowchart", "graph", "sequenceDiagram", "classDiagram", "erDiagram")


def harden_line(ln: str) -> str:
    stripped = ln.strip()
    if not stripped or stripped.startswith("%%") or stripped.startswith(_DIAGRAM_KEYWORDS):
        return ln
    return transform_node_labels(transform_edge_labels(ln))


def check_file(path: Path) -> list[str]:
    """Return a list of violation message lines for one markdown file."""
    txt = path.read_text()
    problems: list[str] = []

    # fence balance: odd count means a truncated / unclosed block
    if txt.count("```") % 2 != 0:
        problems.append(f"{path}: unbalanced ``` fences (odd count) — a code block is unclosed")

    for m in re.finditer(r"```mermaid\n(.*?)```", txt, flags=re.S):
        block_start_line = txt.count("\n", 0, m.start(1)) + 1  # 1-based file line of block body
        for offset, ln in enumerate(m.group(1).split("\n")):
            fixed = harden_line(ln)
            if fixed != ln:
                lineno = block_start_line + offset
                problems.append(f"{path}:{lineno}: mermaid label needs quoting / <br/>")
                problems.append(f"    - {ln}")
                problems.append(f"    + {fixed}")
    return problems


def iter_paths(args: list[str]) -> list[Path]:
    if args:
        return [Path(a) for a in args]
    return sorted(Path("docs").rglob("*.md"))


def main(argv: list[str]) -> int:
    paths = iter_paths(argv)
    all_problems: list[str] = []
    for p in paths:
        if not p.exists():
            print(f"warning: {p} does not exist, skipping", file=sys.stderr)
            continue
        all_problems.extend(check_file(p))

    if all_problems:
        print("\n".join(all_problems))
        n = sum(1 for line in all_problems if ": mermaid label" in line or "unbalanced" in line)
        print(f"\n✗ {n} mermaid issue(s) across {len(paths)} file(s).")
        print(
            "  Fix: quote every label (A[\"text\"], -->|\"label\"|) and use <br/> not \\n.\n"
            "  Auto-fix: python3 ~/.claude/skills/mermaid-markdown/harden_mermaid.py docs/**/*.md"
        )
        return 1

    print(f"✓ mermaid OK — {len(paths)} file(s) checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
