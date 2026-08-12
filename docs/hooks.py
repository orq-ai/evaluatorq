"""MkDocs build hooks.

Parity guard (RES-1037 review #1): the `llmstxt.sections` in mkdocs.yml restates the
pages in `nav:`. The plugin silently drops any nav page it doesn't list from both
`llms.txt` and `llms-full.txt`, and does not fail the build. This hook closes that gap:
if a page reachable from `nav` is not covered by some `sections` entry (exact path or
glob), the build fails, so the two can't drift apart unnoticed.

Highlighting guard: a docstring code sample written as an indented block instead of a
fenced one reaches Pygments with no language and renders as grey text. Nothing warns —
`mkdocs build --strict` stays green. Checked here on the rendered HTML rather than in a
unit test, because the defect has several source spellings (RST `::` literal blocks,
plain indentation, a fence with an unknown language) and only one rendered symptom.
"""

from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING

from mkdocs.exceptions import PluginError

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.nav import Navigation
    from mkdocs.structure.pages import Page

# Structural pages that are navigation scaffolding, not documentation content, so they
# are not expected in the llms.txt catalog (the section-index landings + literate-nav files).
_PARITY_EXEMPT = ('*/SUMMARY.md', 'reference/index.md', 'examples/index.md')


def _section_patterns(config: MkDocsConfig) -> list[str]:
    """Every path/glob listed under `llmstxt.sections` (each entry is a str or {path: desc})."""
    plugin = config.plugins.get('llmstxt')
    if plugin is None:
        return []
    return [
        entry if isinstance(entry, str) else next(iter(entry))
        for entries in plugin.config.sections.values()
        for entry in entries
    ]


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


# A highlighted block carries Pygments spans that have a token CLASS. Testing for
# `<span` alone is not enough and silently passes the exact defect this guards: the
# plain-text lexer still emits one empty `<pre><span></span><code>`. Mermaid renders as
# a bare <pre> too, and legitimately so — it is a diagram, not code.
_PRE = re.compile(r'<pre[^>]*>(.*?)</pre>', re.DOTALL)
_CLASSED_SPAN = re.compile(r'<span class=')
_DIAGRAM = re.compile(r'^\s*(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram)\b')


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str:
    if not page.file.src_uri.startswith('reference/'):
        return output
    unhighlighted = [
        text
        for body in (m.group(1) for m in _PRE.finditer(output))
        if not _CLASSED_SPAN.search(body)
        for text in [re.sub(r'<[^>]+>', '', body).strip()]
        if text and not _DIAGRAM.match(text)
    ]
    if unhighlighted:
        raise PluginError(
            f'{page.file.src_uri}: code block(s) rendered without syntax highlighting — '
            'the docstring almost certainly indents the sample instead of fencing it as '
            '```python. First offender: ' + repr(unhighlighted[0][:120])
        )
    return output
