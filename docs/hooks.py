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

Generated tables: the coverage table, the curated attack strategy table and the full-sweep
attack count in the red teaming guide are rendered here from the registries in place of
markers, so they cannot drift from the code.
"""

from __future__ import annotations

import fnmatch
import functools
import re
from collections import Counter
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


_GENERATED_PAGE = 'guides/red-teaming.md'


def _coverage_table() -> str:
    from evaluatorq.common.reports.md_helpers import md_table
    from evaluatorq.redteam.adaptive.strategy_registry import curated_strategies
    from evaluatorq.redteam.contracts import OWASP_CATEGORY_NAMES
    from evaluatorq.redteam.frameworks.owasp.evaluators import VULNERABILITY_EVALUATOR_REGISTRY
    from evaluatorq.redteam.vulnerability_registry import CATEGORY_TO_VULNERABILITY, FRAMEWORK_TO_VULNERABILITIES

    curated = Counter(s.category for s in curated_strategies())
    codes = sorted(code for framework in FRAMEWORK_TO_VULNERABILITIES.values() for code in framework)
    vulnerabilities = {code: CATEGORY_TO_VULNERABILITY[code] for code in codes}
    rows = [
        [
            f'`{code}`',
            f'{OWASP_CATEGORY_NAMES[code]} (`{vulnerability.value}`)',
            str(curated[code] or 'generated'),
            '✅' if vulnerability in VULNERABILITY_EVALUATOR_REGISTRY else '❌',
        ]
        for code, vulnerability in vulnerabilities.items()
    ]
    summary = f'{len(codes)} framework categories map onto {len(set(vulnerabilities.values()))} vulnerabilities.'
    return summary + '\n\n' + md_table(['Category', 'Vulnerability', 'Curated strategies', 'Judge'], rows)


def _strategy_table() -> str:
    from evaluatorq.common.reports.md_helpers import md_table
    from evaluatorq.redteam.adaptive.strategy_registry import strategy_catalog
    from evaluatorq.redteam.contracts import DeliveryMethod

    catalog = strategy_catalog()
    used = {d for row in catalog for d in row['delivery_methods']}
    headers = [
        'Category', 'Vulnerability', 'Strategy', 'What it attempts', 'Turns', 'Needs', 'Technique', 'Delivery methods'
    ]
    rows = [
        [
            f"`{row['category']}`",
            f"`{row['vulnerability']}`",
            f"`{row['name']}`",
            row['description'],
            row['turn_type'],
            row['needs'] or '—',
            f"`{row['attack_technique']}`",
            ', '.join(f'`{d}`' for d in row['delivery_methods']),
        ]
        for row in catalog
    ]
    summary = (
        f'{len(catalog)} curated strategies across {len({row["category"] for row in catalog})} categories, '
        f'using {len(used)} of the {len(DeliveryMethod)} delivery methods.'
    )
    return summary + '\n\n' + md_table(headers, rows)


def _generated_per_vulnerability() -> int:
    from evaluatorq.redteam.run_config import RedTeamRunConfig

    return RedTeamRunConfig.model_fields['generated_strategy_count'].default


@functools.cache
def _full_sweep(*, multi_agent: bool) -> int:
    from evaluatorq.redteam.adaptive.strategy_planner import full_sweep_attack_ceiling

    per_vulnerability = _generated_per_vulnerability()
    return full_sweep_attack_ceiling(multi_agent=multi_agent, generated_per_vulnerability=per_vulnerability)


def _full_sweep_total() -> str:
    return f'{_full_sweep(multi_agent=False)} ({_full_sweep(multi_agent=True)} multi-agent)'


def _full_sweep_ceiling() -> str:
    from evaluatorq.redteam.adaptive.strategy_registry import curated_strategies

    return (
        f'**{_full_sweep(multi_agent=False)}** for a single-agent target and **{_full_sweep(multi_agent=True)}** '
        f'for a multi-agent one: {len(curated_strategies())} curated strategies plus {_generated_per_vulnerability()} '
        'generated per vulnerability, and the inter-agent vulnerabilities are only planned for multi-agent targets'
    )


_GENERATED = {
    '<!-- coverage-table -->': _coverage_table,
    '<!-- curated-strategies -->': _strategy_table,
    '<!-- full-sweep-ceiling -->': _full_sweep_ceiling,
    '<!-- full-sweep-total -->': _full_sweep_total,
}


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    if page.file.src_uri != _GENERATED_PAGE:
        return markdown
    for marker, render in _GENERATED.items():
        if marker not in markdown:
            raise PluginError(
                f'{_GENERATED_PAGE}: the {marker} marker is missing, so its generated content has nowhere to go'
            )
        markdown = markdown.replace(marker, render())
    return markdown


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
