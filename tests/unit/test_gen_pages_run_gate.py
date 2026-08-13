"""``docs/gen_pages.py`` only generates when mkdocs-gen-files runs it.

The generators sit behind ``__name__ in ("<run_path>", "__main__")`` because
mkdocs-gen-files calls ``runpy.run_path(file_name)`` with no ``run_name``, and
CPython then names the module ``<run_path>``. If that ever changes on either
side the gate stops matching, generation silently no-ops, and the whole
reference tree vanishes from the site while ``mkdocs build --strict`` stays
green — the same silent-drop failure the rest of this branch exists to close.

So run it the way mkdocs does and assert pages actually come out.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GEN_PAGES = REPO / "docs" / "gen_pages.py"

mkdocs_config = pytest.importorskip(
    "mkdocs.config", reason="docs dependency group not installed"
)
gen_files_editor = pytest.importorskip("mkdocs_gen_files.editor")
mkdocs_files = pytest.importorskip("mkdocs.structure.files")


def test_run_path_generates_pages(tmp_path: Path) -> None:
    config = mkdocs_config.load_config(str(REPO / "mkdocs.yml"))
    editor = gen_files_editor.FilesEditor(
        mkdocs_files.Files([]), config, directory=str(tmp_path)
    )
    with editor:
        runpy.run_path(str(GEN_PAGES))

    written = {f.src_uri for f in editor.files}
    assert "reference/SUMMARY.md" in written
    assert "reference/evaluatorq.md" in written
    assert "examples/SUMMARY.md" in written


def test_shadowed_entry_points_are_emitted_on_the_package_page(tmp_path: Path) -> None:
    """The filenames alone prove nothing: `evaluatorq()`, `deployment()` and
    `llm_jury()` are the symbols a same-named sub-module hides from griffe, and
    they only reach the site as their own `::: pkg.mod.symbol` blocks. Dropping
    those blocks leaves every page still generated and `--strict` still green,
    so assert on the page body, not the page list.
    """
    config = mkdocs_config.load_config(str(REPO / "mkdocs.yml"))
    editor = gen_files_editor.FilesEditor(
        mkdocs_files.Files([]), config, directory=str(tmp_path)
    )
    with editor:
        runpy.run_path(str(GEN_PAGES))

    page = (tmp_path / "reference" / "evaluatorq.md").read_text()
    for dotted in (
        "evaluatorq.evaluatorq.evaluatorq",
        "evaluatorq.deployment.deployment",
        "evaluatorq.llm_jury.llm_jury",
    ):
        assert f"::: {dotted}\n" in page, f"{dotted} lost its API reference block"


def test_every_exported_symbol_is_documented_exactly_once(tmp_path: Path) -> None:
    """Both failure directions at once, because fixing one caused the other.

    `CrewAITarget` and `PydanticAITarget` were rendered in full on *two* pages:
    `evaluatorq/simulation/` (a lazy re-export the owner check could not
    resolve) and `evaluatorq/integrations/` (which renders each sub-module's
    `__all__`). Dropping names on ownership alone then fixed that by taking
    `ReplayError` and `OrqResponsesTarget` off the site entirely — their owners
    do not export them, so nothing rendered them at all, and `--strict` stayed
    green through both states.

    Generated in a subprocess because `gen_pages._safe_getattr` reads
    ``vars(module)`` first: once another test has touched
    ``evaluatorq.simulation.TokenUsage`` the name is bound on the package, the
    owner resolves where it did not before, and the member lists change. mkdocs
    runs the generator in a fresh interpreter, so that is what to assert against.
    """
    import importlib
    import subprocess
    import sys as _sys

    script = (
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(REPO / 'docs')!r})\n"
        "from mkdocs.config import load_config\n"
        "from mkdocs_gen_files.editor import FilesEditor\n"
        "from mkdocs.structure.files import Files\n"
        f"cfg = load_config({str(REPO / 'mkdocs.yml')!r})\n"
        f"with FilesEditor(Files([]), cfg, directory={str(tmp_path)!r}):\n"
        f"    runpy.run_path({str(GEN_PAGES)!r})\n"
    )
    proc = subprocess.run(
        [_sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr

    _sys.path.insert(0, str(REPO / "docs"))
    try:
        gen_pages = importlib.import_module("gen_pages")
    finally:
        _sys.path.remove(str(REPO / "docs"))

    def block_renders(dotted: str, pinned: list[str]) -> set[str]:
        """What one `::: target` block puts on the page."""
        if pinned:  # an explicit `members:` list wins over the module's __all__
            return set(pinned)
        try:
            return set(getattr(importlib.import_module(dotted), "__all__", []))
        except ModuleNotFoundError:  # `::: pkg.mod.symbol` — one symbol, not a module
            return {dotted.rpartition(".")[2]}

    homes: dict[str, list[str]] = {}
    for page in sorted((tmp_path / "reference").rglob("*.md")):
        rendered: set[str] = set()
        target, pinned = "", []
        for line in [*page.read_text().splitlines(), "::: "]:
            if line.startswith("::: "):
                if target:
                    rendered |= block_renders(target, pinned)
                target, pinned = line.removeprefix("::: ").strip(), []
            elif line.startswith("        - "):
                pinned.append(line.removeprefix("        - ").strip())
        for name in rendered:
            homes.setdefault(name, []).append(page.name)

    # Pre-existing gap, pinned so it cannot grow: these are re-exported from
    # `evaluatorq.contracts`, which `evaluatorq.__all__` does not list, so the
    # owner's page renders nothing for them and the re-exporting page drops them.
    # Restoring them to `evaluatorq/redteam/` makes autorefs prefer that page for
    # every `AgentTarget`/`AgentContext` cross-reference and emit anchors
    # mkdocstrings never wrote (it anchors re-exports under the defining module),
    # which fails --strict. Tracked as RES-1303; not a highlighting fix.
    # Emptying this set is that ticket's acceptance criterion — do not add to it.
    KNOWN_UNDOCUMENTED = {
        "AgentContext",
        "JuryResult",
        "JuryStats",
        "JuryVote",
        "KnowledgeBaseInfo",
        "MemoryStoreInfo",
        "ReasoningOutputItem",
        "ToolInfo",
    }
    undocumented = {
        name
        for dotted in gen_pages.API_PACKAGES
        for name in getattr(importlib.import_module(dotted), "__all__", [])
        if not homes.get(name)
    }
    assert undocumented <= KNOWN_UNDOCUMENTED, (
        "newly undocumented on every reference page: " + ", ".join(sorted(undocumented - KNOWN_UNDOCUMENTED))
    )

    # Duplication is checked by identity, not by name: `redteam.DefaultHooks` and
    # `simulation.DefaultHooks` are different classes that share a name and each
    # rightly gets its own entry. The targets are one class re-exported from two
    # packages, which is the case that actually rendered twice.
    for name in (
        "CallableTarget",
        "CrewAITarget",
        "LangGraphTarget",
        "OpenAIAgentTarget",
        "PydanticAITarget",
        "VercelAISdkTarget",
    ):
        where = sorted(set(homes.get(name, [])))
        assert where == ["integrations.md"], f"{name} is documented on {where}"


def test_plain_import_generates_nothing(tmp_path: Path) -> None:
    """The gate's other half: importing the module for a helper must no-op."""
    config = mkdocs_config.load_config(str(REPO / "mkdocs.yml"))
    editor = gen_files_editor.FilesEditor(
        mkdocs_files.Files([]), config, directory=str(tmp_path)
    )
    with editor:
        runpy.run_path(str(GEN_PAGES), run_name="gen_pages")

    assert not list(tmp_path.rglob("*.md"))
