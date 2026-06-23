"""Generate API-reference pages and ingest out-of-tree markdown.

Run by the mkdocs-gen-files plugin during build. Writes VIRTUAL files
into the docs tree (nothing written to disk under docs/).
"""

import importlib
import re
from pathlib import Path
from typing import Any

import mkdocs_gen_files

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "evaluatorq"
EXAMPLES = REPO / "examples"
BLOB = "https://github.com/orq-ai/evaluatorq/blob/main"

# --- Public API packages driven by each package's __all__ ----------------
# Public surface is each package's __init__.py __all__ re-exports; griffe /
# mkdocstrings will document exactly those members. This auto-excludes
# internal modules (processings, send_results, table_display, fetch_data,
# progress, job_helper) that are not in any package's public symbols.
# Note: symbols (e.g. `job` decorator) re-exported through `evaluatorq`
# __all__ appear there and are documented once. Empty-__init__ subpackages
# (redteam/backends, redteam/frameworks, redteam/runtime) have no __all__
# and are covered by prose guides (Task 4), not API pages.
API_PACKAGES = [
    "evaluatorq",
    "evaluatorq.redteam",
    "evaluatorq.simulation",
    "evaluatorq.openresponses",
    "evaluatorq.tracing",
    "evaluatorq.integrations",
]


def _rewrite_relative_links(text: str, source_dir: Path) -> str:
    """Rewrite relative Markdown links to absolute GitHub blob URLs.

    Links that are already absolute (http/https/mailto) or anchor-only
    (#section) are left unchanged; all others are resolved relative to
    source_dir and converted to a GitHub blob URL so they remain valid
    when the file is served from a different path inside the docs tree.

    Args:
        text: Markdown source text.
        source_dir: Directory of the source file (used to resolve relative
            paths before turning them into BLOB anchors).

    Returns:
        The Markdown text with relative links replaced by absolute GitHub
        blob URLs.
    """

    def repl(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            return match.group(0)
        resolved = (source_dir / target).resolve()
        try:
            rel = resolved.relative_to(REPO)
        except ValueError:
            return match.group(0)
        return f"[{label}]({BLOB}/{rel.as_posix()})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl, text)


# (out-of-tree source, destination page in docs tree)
INGEST = [
    (REPO / "CONTRIBUTING.md", "contributing.md"),
    (REPO / "CHANGELOG.md", "changelog.md"),
    (REPO / "ROADMAP.md", "roadmap.md"),
]


def ingest_markdown() -> None:
    for source, dest in INGEST:
        if not source.exists():
            raise FileNotFoundError(f"ingest source missing: {source}")  # fail loud, not silent empty page
        text = source.read_text(encoding="utf-8")
        text = _rewrite_relative_links(text, source.parent)
        with mkdocs_gen_files.open(dest, "w") as fd:
            fd.write(text)
        mkdocs_gen_files.set_edit_path(dest, source.relative_to(REPO).as_posix())


def _canonical_owner(module_name: str) -> str | None:
    """Return the longest API package that owns where a symbol is defined."""
    cands = [p for p in API_PACKAGES if module_name == p or module_name.startswith(p + ".")]
    return max(cands, key=len) if cands else None


def _safe_getattr(mod: object, name: str, parent_dotted: str) -> object | None:
    """Get a member from a module without triggering recursive __getattr__.

    Some packages (e.g. ``evaluatorq.integrations``) use a lazy-loader
    ``__getattr__`` that does ``from . import <submod>``. On CPython 3.13 this
    causes infinite recursion because ``_handle_fromlist`` calls ``__getattr__``
    again before the attribute is bound. We work around it by:

    1. Checking ``mod.__dict__`` directly (no attribute protocol).
    2. Falling back to ``importlib.import_module`` with the full dotted path so
       the sub-module is loaded without going through the parent's
       ``__getattr__``.
    """
    import sys

    obj = vars(mod).get(name)  # type: ignore[arg-type]
    if obj is not None:
        return obj
    full = f"{parent_dotted}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    try:
        return importlib.import_module(full)
    except (ImportError, ModuleNotFoundError):
        return None


# One-line blurb per API package for the reference landing page.
_PACKAGE_DESC = {
    "evaluatorq": "Core evaluation API — `evaluatorq()`, `DataPoint`, `job`, built-in evaluators.",
    "evaluatorq.redteam": "Adversarial red teaming — `red_team()`, targets, OWASP frameworks.",
    "evaluatorq.openresponses": "OpenAI Responses API integration.",
    "evaluatorq.tracing": "OpenTelemetry tracing helpers.",
    "evaluatorq.integrations": "Third-party agent integrations (LangChain, LangGraph, …).",
}


def write_api_pages() -> None:
    nav_lines = []
    index_rows = []
    for dotted in API_PACKAGES:
        mod = importlib.import_module(dotted)
        names = list(getattr(mod, "__all__", []))
        # Document each symbol once, on its canonical package page. A name whose
        # definition lives in (or under) a DIFFERENT API package is dropped here
        # and documented on that package's page instead (single-location docs).
        members = [
            n
            for n in names
            if _canonical_owner(
                getattr(_safe_getattr(mod, n, dotted), "__module__", "") or ""
            )
            in (dotted, None)
        ]
        page_path = Path("reference", *dotted.split(".")).with_suffix(".md")
        with mkdocs_gen_files.open(page_path, "w") as fd:
            fd.write(f"# `{dotted}`\n\n")
            if 0 < len(members) < len(names):
                # Only some members are canonical here — pin the member list.
                fd.write(f"::: {dotted}\n    options:\n      members:\n")
                for n in members:
                    fd.write(f"        - {n}\n")
            else:
                # All members canonical here (or filter would empty the page) — document all.
                fd.write(f"::: {dotted}\n")
        mkdocs_gen_files.set_edit_path(page_path, "gen_pages.py")
        title = dotted.replace("evaluatorq.integrations.", "integrations/").replace("evaluatorq.", "") or "evaluatorq"
        href = page_path.relative_to("reference").as_posix()
        nav_lines.append(f"- [{title}]({href})\n")
        desc = _PACKAGE_DESC.get(dotted, "")
        index_rows.append(f"- [`{dotted}`]({href}) — {desc}\n")

    # Landing page so /reference/ resolves (not a 404) and section-index has a home.
    with mkdocs_gen_files.open("reference/index.md", "w") as fd:
        fd.write("# API Reference\n\n")
        fd.write("The public API, documented from each package's `__all__`.\n\n")
        fd.writelines(index_rows)
    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as fd:
        fd.write("- [Overview](index.md)\n")
        fd.writelines(nav_lines)


def _pretty(stem: str) -> str:
    """Filename/dir → readable title: drop a leading numeric prefix, title-case."""
    return re.sub(r"^\d+[_-]", "", stem).replace("_", " ").title()


def write_example_pages() -> None:
    """Render each examples/*.py as a doc page (source embedded) + literate nav.

    Pages are virtual; nothing is written under docs/. Source is fenced as-is
    with a GitHub link so readers can copy or open the original.
    """
    files = sorted(
        p
        for p in EXAMPLES.rglob("*.py")
        if p.name != "__init__.py" and "__pycache__" not in p.parts
    )
    tree: dict = {}
    for f in files:
        rel = f.relative_to(EXAMPLES)
        page = Path("examples", rel).with_suffix(".md")
        source = f.read_text(encoding="utf-8")
        gh = f"{BLOB}/{f.relative_to(REPO).as_posix()}"
        with mkdocs_gen_files.open(page, "w") as fd:
            fd.write(f"# {_pretty(f.stem)}\n\n")
            fd.write(f"[View on GitHub]({gh})\n\n")
            fd.write("```python\n")
            fd.write(source if source.endswith("\n") else source + "\n")
            fd.write("```\n")
        mkdocs_gen_files.set_edit_path(page, f.relative_to(REPO).as_posix())
        node = tree
        for part in rel.parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault("__files__", []).append(
            (_pretty(f.stem), page.relative_to("examples").as_posix())
        )

    lines: list[str] = []

    def emit(node: dict, depth: int) -> None:
        indent = "    " * depth
        for name in sorted(k for k in node if k != "__files__"):
            lines.append(f"{indent}- {_pretty(name)}:\n")
            emit(node[name], depth + 1)
        for label, href in node.get("__files__", []):
            lines.append(f"{indent}- [{label}]({href})\n")

    emit(tree, 0)
    # Section landing page (navigation.indexes uses the first SUMMARY entry).
    cats = sorted(_pretty(k) for k in tree if k != "__files__")
    with mkdocs_gen_files.open("examples/index.md", "w") as fd:
        fd.write("# Examples\n\n")
        fd.write(f"{len(files)} runnable scripts, grouped by area:\n\n")
        for c in cats:
            fd.write(f"- {c}\n")
    with mkdocs_gen_files.open("examples/SUMMARY.md", "w") as fd:
        fd.write("- [Overview](index.md)\n")
        fd.writelines(lines)


write_api_pages()
write_example_pages()
ingest_markdown()
