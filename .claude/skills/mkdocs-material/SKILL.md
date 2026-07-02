---
name: mkdocs-material
description: Use when creating, configuring, fixing, or deploying MkDocs / Material for MkDocs documentation sites - especially when task-list checkboxes render as literal "[ ]" text, bare URLs are not clickable, tables overflow the page width, Mermaid diagrams show as plain code blocks, mkdocs gh-deploy succeeds but GitHub Pages shows errors or a 404, or an internal/private repo's Pages URL is unclear. Covers mkdocs.yml configuration, pymdownx extensions, navigation, strict builds, and GitHub Pages deployment.
license: MIT
metadata:
  author: forcewake
  version: "1.0"
---

# Material for MkDocs

## Overview

Material for MkDocs renders only what its Markdown extensions enable. Most "broken page" reports are a missing one-line entry in `mkdocs.yml`, not broken content. Configure the known extension set up front, verify the **built HTML** (not the Markdown), and validate every change with `mkdocs build --strict` before deploying.

Full config template: [references/mkdocs.yml.template](references/mkdocs.yml.template). Deployment recipes: [references/github-pages.md](references/github-pages.md).

## Quick Start

```bash
pip install mkdocs-material        # use a venv; pin in requirements.txt
mkdocs serve                       # live preview at 127.0.0.1:8000
mkdocs build --strict              # fails on broken links/nav — run before every deploy
mkdocs gh-deploy --force           # build + force-push to gh-pages branch
```

Content lives in `docs/`, with `docs/index.md` as the home page. Cross-page links are relative Markdown paths to `.md` files (`[text](../checklists/cutover.md)`) — `--strict` validates them.

## Rendering Failure Modes (symptom → fix)

These all produce a successful build with a broken-looking page. Check this table before debugging anything else.

| Symptom on the rendered page | Root cause | Fix in `mkdocs.yml` |
|---|---|---|
| `[ ] item` renders as literal bracket text | task lists are not core Markdown | `pymdownx.tasklist` with `custom_checkbox: true` |
| Bare URLs are plain text, not links | autolinking is not core Markdown | `pymdownx.magiclink` |
| Table overflows page width; columns clipped | a cell contains an unbreakable token — URLs without hyphens (e.g. AWS doc URLs like `PostgreSQL.Concepts.General.FeatureSupport.LogicalReplication.html`), long code spans | no config fix — replace bare URLs with short named links `[AWS documentation](url)`; keep long tokens out of cells |
| Mermaid block shows as a code listing | fenced block not routed to Mermaid | `pymdownx.superfences` with the `mermaid` custom fence (see template) |
| Mermaid node labels show literal `\n` | `\n` support varies by renderer | use `<br/>` inside node labels |
| Admonitions (`!!! note`) render as plain paragraphs | extension missing | `admonition` (+ `pymdownx.details` for collapsible) |
| Content tabs (`=== "Tab"`) render as quoted text | extension missing | `pymdownx.tabbed` with `alternate_style: true` |
| No copy button on code blocks | theme feature missing | `content.code.copy` under `theme.features` |

## Verify the Built HTML

`mkdocs build` succeeding does not mean the page renders as intended. After config changes, check the output:

```bash
mkdocs build --strict
grep -c 'class="task-list-item"' site/path/page/index.html   # checkboxes rendered
grep -c 'class="mermaid"'        site/path/page/index.html   # diagrams rendered
# bare (unlinked) URLs left in table cells:
python3 -c "
import re; html = open('site/path/page/index.html').read()
tds = re.findall(r'<td>(.*?)</td>', html, re.S)
print('bare URL cells:', sum(1 for t in tds if 'http' in t and '<a ' not in t))"
```

## GitHub Pages Deployment

`mkdocs gh-deploy --force` builds and force-pushes the site to the `gh-pages` branch. It does **not** enable Pages — that is a one-time repo setting. Critical traps:

- **Private/internal repos**: the site is served at an obfuscated `https://<random>.pages.github.io/` domain behind org SSO — **not** `https://<org>.github.io/<repo>/`. Read the real URL from `gh api repos/<org>/<repo>/pages --jq .html_url` and set `site_url` to it.
- **Pages build can error independently** of a successful deploy. Check and rebuild via API — see [references/github-pages.md](references/github-pages.md).
- `gh-deploy` writes `.nojekyll` automatically; do not add Jekyll config to the branch.

## Common Mistakes

| Mistake | Consequence |
|---|---|
| Assuming GitHub-flavored Markdown features work by default | task lists, autolinks, mermaid all need explicit extensions |
| Verifying the Markdown source instead of `site/` HTML | extension gaps are invisible in source |
| Skipping `--strict` | broken relative links ship silently |
| Putting long bare URLs in table cells | unbreakable tokens blow up the layout |
| Hardcoding `https://<org>.github.io/<repo>/` as `site_url` for a private repo | wrong canonical URLs and sitemap |
| Treating a green `gh-deploy` as "site is live" | Pages may be disabled or its build errored |
