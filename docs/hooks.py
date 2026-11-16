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

The guard covers every page, not just `reference/`. Scoping it to docstring-generated
pages missed the whole `examples/` tree, where an example whose own docstring fences a
snippet closed the generator's outer fence and rendered its body as prose — caught the
moment the scope widened.

Strategy table: the curated attack strategy table in the red teaming guide is rendered
here from `STRATEGY_REGISTRY` in place of a marker, so it cannot drift from the code. A
hand-written copy needed a test re-implementing the row format to stay in sync.
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
# Match a mermaid *header line*, not merely a keyword: `graph = {}` in an unfenced
# Python sample starts with `graph` and would otherwise buy itself an exemption from
# the very check this hook exists to run.
# MULTILINE so `$` ends the *header line*, not the whole block — without it the
# keyword-only diagrams (`sequenceDiagram`, `erDiagram`, …) never matched at all,
# because a real diagram always has a body on the next line.
_DIAGRAM = re.compile(
    r'^\s*(?:(?:graph|flowchart)\s+(?:TB|TD|BT|RL|LR)\b'
    r'|(?:sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|erDiagram)\s*$)',
    re.MULTILINE,
)


# Pages that legitimately render an unlabelled plain block. Both are prose diagrams,
# not code: the console-output sample on the landing page and the span-hierarchy trees
# in tracing.md. There is no way to tell them from a genuinely unhighlighted code block
# in the rendered HTML — a bare fence and a ```text fence emit identical markup — so the
# exemption is per page and kept to exactly these two. Everything else, `reference/`
# included, is zero-tolerance; do not add a page here to silence a real finding.
_HIGHLIGHT_EXEMPT = ('index.md', 'tracing.md')


_STRATEGY_PAGE = 'guides/red-teaming.md'
_STRATEGY_MARKER = '<!-- curated-strategies -->'


def _strategy_table() -> str:
    from evaluatorq.redteam.adaptive.strategy_registry import STRATEGY_REGISTRY
    from evaluatorq.redteam.contracts import DeliveryMethod

    def codes(values) -> str:
        return ', '.join(f'`{v.value}`' for v in values)

    # The registry also carries every category under an `OWASP-` alias.
    by_category = {c: group for c, group in STRATEGY_REGISTRY.items() if group and not c.startswith('OWASP-')}
    strategies = [s for group in by_category.values() for s in group]
    used = {d for s in strategies for d in s.delivery_methods}
    rows = [
        '| Category | Strategy | What it attempts | Turns | Needs | Technique | Delivery methods |',
        '|---|---|---|---|---|---|---|',
    ]
    for s in strategies:
        capabilities = ' or '.join(f'`{c.value}`' for c in s.required_capabilities)
        needs = ' + '.join(part for part in ('tools' if s.requires_tools else '', capabilities) if part)
        description = s.description.replace('|', r'\|')
        rows.append(
            f'| `{s.category}` | `{s.name}` | {description} | {s.turn_type.value} '
            f'| {needs or "—"} | `{s.attack_technique.value}` | {codes(s.delivery_methods)} |'
        )
    summary = (
        f'{len(strategies)} curated strategies across {len(by_category)} categories, '
        f'using {len(used)} of the {len(DeliveryMethod)} delivery methods.'
    )
    return summary + '\n\n' + '\n'.join(rows)


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    if page.file.src_uri != _STRATEGY_PAGE:
        return markdown
    if _STRATEGY_MARKER not in markdown:
        raise PluginError(f'{_STRATEGY_PAGE}: the {_STRATEGY_MARKER} marker the strategy table renders into is gone')
    return markdown.replace(_STRATEGY_MARKER, _strategy_table())


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str:
    if page.file.src_uri in _HIGHLIGHT_EXEMPT:
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
            'the source almost certainly indents the sample instead of fencing it as '
            '```python, or opens a fence a nested one closes early. First offender: '
            + repr(unhighlighted[0][:120])
        )
    return output
