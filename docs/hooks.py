"""MkDocs build hooks.

Parity guard (RES-1037 review #1): the `llmstxt.sections` in mkdocs.yml restates the
pages in `nav:`. The plugin silently drops any nav page it doesn't list from both
`llms.txt` and `llms-full.txt`, and does not fail the build. This hook closes that gap:
if a page reachable from `nav` is not covered by some `sections` entry (exact path or
glob), the build fails, so the two can't drift apart unnoticed.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

from mkdocs.exceptions import PluginError

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.nav import Navigation

# Structural pages that are navigation scaffolding, not documentation content, so they
# are not expected in the llms.txt catalog (the section-index landings + literate-nav files).
_PARITY_EXEMPT = ('*/SUMMARY.md', 'reference/index.md', 'examples/index.md')


def _section_patterns(config: MkDocsConfig) -> list[str]:
    """Every path/glob listed under `llmstxt.sections` (each entry is a str or {path: desc})."""
    plugin = config.plugins.get('llmstxt')
    if plugin is None:
        return []
    patterns: list[str] = []
    for entries in plugin.config.sections.values():
        for entry in entries:
            patterns.append(entry if isinstance(entry, str) else next(iter(entry)))
    return patterns


def on_nav(nav: Navigation, config: MkDocsConfig, files: Files) -> Navigation:
    patterns = _section_patterns(config)
    if not patterns:
        return nav
    missing = [
        page.file.src_uri
        for page in nav.pages
        if page.file.src_uri
        and not any(fnmatch.fnmatch(page.file.src_uri, exempt) for exempt in _PARITY_EXEMPT)
        and not any(fnmatch.fnmatch(page.file.src_uri, pat) for pat in patterns)
    ]
    if missing:
        raise PluginError(
            'llms.txt is out of sync with nav: these nav pages are not in `llmstxt.sections` '
            '(mkdocs.yml), so they would be dropped from llms.txt / llms-full.txt: '
            + ', '.join(sorted(missing))
            + '. Add them to `llmstxt.sections`.'
        )
    return nav
