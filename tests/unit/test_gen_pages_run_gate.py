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


def test_plain_import_generates_nothing(tmp_path: Path) -> None:
    """The gate's other half: importing the module for a helper must no-op."""
    config = mkdocs_config.load_config(str(REPO / "mkdocs.yml"))
    editor = gen_files_editor.FilesEditor(
        mkdocs_files.Files([]), config, directory=str(tmp_path)
    )
    with editor:
        runpy.run_path(str(GEN_PAGES), run_name="gen_pages")

    assert not list(tmp_path.rglob("*.md"))
