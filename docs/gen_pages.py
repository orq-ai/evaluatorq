"""Generate API-reference pages and ingest out-of-tree markdown.

Run by the mkdocs-gen-files plugin during build. Writes VIRTUAL files
into the docs tree (nothing written to disk under docs/).
"""

import importlib
import re
from pathlib import Path

import mkdocs_gen_files

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "evaluatorq"
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


def write_api_pages() -> None:
    nav_lines = []
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
        nav_lines.append(f"- [{title}]({page_path.relative_to('reference').as_posix()})\n")
    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as fd:
        fd.writelines(nav_lines)


write_api_pages()
ingest_markdown()
