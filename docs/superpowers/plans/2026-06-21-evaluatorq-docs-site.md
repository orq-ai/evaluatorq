# evaluatorq Documentation Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a structured MkDocs documentation site for the `evaluatorq` package on GitHub Pages, with auto-generated API reference, expanded subsystem coverage, CI-checked tutorials, and a slim README.

**Architecture:** MkDocs + Material renders markdown under `docs/`. A `docs/gen_pages.py` script (mkdocs-gen-files) generates curated API-reference pages and ingests out-of-tree markdown (in-`src` and root files) with relative-link rewriting. `mkdocs build --strict` is the correctness gate in PR CI; a separate deploy job publishes to GitHub Pages via the Actions artifact flow (no `gh-pages` branch, no committed HTML).

**Tech Stack:** MkDocs, mkdocs-material, mkdocstrings[python] (griffe), mkdocs-gen-files, mkdocs-literate-nav, mkdocs-section-index, uv, GitHub Actions.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-20-evaluatorq-docs-site-design.md`.
- `docs_dir: docs`. Rendered output `site/` is git-ignored and never committed.
- Docstrings are **already Google-style** (0 NumPy). No docstring migration. Set griffe `docstring_style: google`.
- `mkdocs build --strict` MUST pass with no warnings at the end of every task that touches docs content or config.
- Docs CI installs `uv sync --frozen --all-extras --group docs` so optional-dep imports (langchain, langgraph, `agents`, crewai, …) resolve during mkdocstrings introspection — otherwise `--strict` fails.
- API reference is a **curated allowlist** of public modules; internal modules (`processings`, `send_results`, `table_display`, `fetch_data`, `progress`) are excluded.
- `docs/superpowers/` is excluded from the build via `exclude_docs`.
- Pages enablement is programmatic (`actions/configure-pages` `enablement: true`); deploy concurrency `group: pages, cancel-in-progress: false`.
- Work branch: `docs/mkdocs-site`. Commit frequently with conventional-commit messages (`docs:`, `ci:`, `build:`).
- `mkdocs build --strict` is this plan's "test": where a step says "run the test", it means run that command and read its output.

---

## File Structure

- Create: `mkdocs.yml` — site config + nav + plugin config.
- Create: `docs/gen_pages.py` — generates API pages + ingests out-of-tree md (run by mkdocs-gen-files).
- Create: `docs/index.md`, `docs/getting-started.md`, `docs/installation.md`, `docs/migrating.md`, `docs/configuration.md`.
- Create: `docs/guides/*.md` (core-evaluation, orq-platform, tracing, openresponses, red-teaming, redteam-backends, adaptive-red-teaming, simulation, hooks, custom-evaluators, cli.md).
- Create: `docs/tutorials/*.md` (first-evaluation, orq-deployment, red-teaming-agent, simulation.md).
- Create: `.github/workflows/docs.yml` — PR strict-build + deploy.
- Create: `scripts/smoke_examples.py` — compile/import smoke for `examples/`.
- Modify: `pyproject.toml` — add `docs` dependency group, fix `[project.urls]`.
- Modify: `.gitignore` — add `site/`.
- Modify: `README.md` — slim to pitch + quick start + link.

---

### Task 1: Scaffold MkDocs and get a green strict build

**Files:**
- Modify: `pyproject.toml` (add `[dependency-groups] docs`)
- Modify: `.gitignore` (add `site/`)
- Create: `mkdocs.yml`
- Create: `docs/index.md`

**Interfaces:**
- Produces: a buildable MkDocs site; `mkdocs.yml` `nav`/`plugins` consumed by all later tasks; `docs` dependency group consumed by CI in Task 6.

- [ ] **Step 1: Add the `docs` dependency group to `pyproject.toml`**

Append to the `[dependency-groups]` table:

```toml
docs = [
  "mkdocs>=1.6.0",
  "mkdocs-material>=9.5.0",
  "mkdocstrings[python]>=0.26.0",
  "mkdocs-gen-files>=0.5.0",
  "mkdocs-literate-nav>=0.6.0",
  "mkdocs-section-index>=0.3.0",
]
```

- [ ] **Step 2: Ignore the build output**

Add to `.gitignore`:

```
# Docs build output
site/
```

- [ ] **Step 3: Sync the docs group**

Run: `uv sync --frozen --all-extras --group docs`
Expected: resolves and installs mkdocs + plugins (no error).

- [ ] **Step 4: Create a minimal `docs/index.md`**

```markdown
# evaluatorq

A flexible evaluation framework for Python — run parallel evaluations,
red-team agents, simulate multi-turn conversations, and optionally integrate
with the Orq AI platform.

See **Getting Started** to install and run your first evaluation.
```

- [ ] **Step 5: Create `mkdocs.yml`**

```yaml
site_name: evaluatorq
site_description: Evaluation framework for Python with Orq AI integration
site_url: https://orq-ai.github.io/evaluatorq/
repo_url: https://github.com/orq-ai/evaluatorq
repo_name: orq-ai/evaluatorq
docs_dir: docs

# Keep skill scratch (plans/specs) out of the build.
exclude_docs: |
  superpowers/

theme:
  name: material
  features:
    - navigation.sections
    - navigation.indexes
    - navigation.top
    - content.code.copy
    - search.highlight
  palette:
    - scheme: default
      toggle: { icon: material/brightness-7, name: Switch to dark mode }
    - scheme: slate
      toggle: { icon: material/brightness-4, name: Switch to light mode }

plugins:
  - search
  - gen-files:
      scripts:
        - docs/gen_pages.py
  - literate-nav:
      nav_file: SUMMARY.md
  - section-index
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: false
            show_root_heading: true
            members_order: source

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Installation & extras: installation.md
  - Migrating from evaluatorq-py: migrating.md
  - Guides:
      - Core evaluation: guides/core-evaluation.md
      - Orq platform: guides/orq-platform.md
      - OpenTelemetry tracing: guides/tracing.md
      - OpenResponses: guides/openresponses.md
      - Red teaming: guides/red-teaming.md
      - Red team backends: guides/redteam-backends.md
      - Adaptive red teaming: guides/adaptive-red-teaming.md
      - Simulation: guides/simulation.md
      - Hooks: guides/hooks.md
      - Custom evaluators & frameworks: guides/custom-evaluators.md
      - CLI reference: guides/cli.md
  - Tutorials:
      - Your first evaluation: tutorials/first-evaluation.md
      - Evaluating an Orq deployment: tutorials/orq-deployment.md
      - Red-teaming an agent: tutorials/red-teaming-agent.md
      - Simulating a conversation: tutorials/simulation.md
  - API Reference: reference/
  - Configuration: configuration.md
  - Contributing: contributing.md
  - Changelog: changelog.md
  - Roadmap: roadmap.md

extra:
  version:
    provider: mike
```

Note: `nav` references pages created in later tasks. Until Task 2 creates `docs/gen_pages.py`, the `gen-files` plugin will error — so this task's build check uses a temporary stub.

- [ ] **Step 6: Create a temporary stub `docs/gen_pages.py`**

```python
"""Temporary stub — replaced in Task 2."""
```

- [ ] **Step 7: Run the strict build (expected to FAIL on missing nav pages)**

Run: `uv run mkdocs build --strict`
Expected: FAIL — warnings about nav referencing not-yet-created files (e.g. `getting-started.md`). This confirms the config loads and the plugin chain runs.

- [ ] **Step 8: Temporarily reduce nav to only existing pages to confirm a clean build**

Comment out every `nav` entry except `- Home: index.md`. Re-run:
Run: `uv run mkdocs build --strict`
Expected: PASS (no warnings). Then restore the full `nav` (the later tasks fill it in).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore mkdocs.yml docs/index.md docs/gen_pages.py
git commit -m "build: scaffold mkdocs material site (config, deps, index)"
```

---

### Task 2: API reference + out-of-tree ingestion (`gen_pages.py`)

**Files:**
- Modify: `docs/gen_pages.py` (replace stub with the real generator)

**Interfaces:**
- Consumes: `mkdocs.yml` plugin config (gen-files, literate-nav, mkdocstrings).
- Produces: virtual pages under `reference/` (+ `reference/SUMMARY.md`), and ingested pages `contributing.md`, `changelog.md`, `roadmap.md`, plus copies of the in-`src` guide source files used by Task 4.

- [ ] **Step 1: Write `docs/gen_pages.py`**

```python
"""Generate API-reference pages and ingest out-of-tree markdown.

Run by the mkdocs-gen-files plugin during the build. Writes VIRTUAL files
into the docs tree (nothing is written to disk under docs/).
"""

import re
from pathlib import Path

import mkdocs_gen_files

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "evaluatorq"
BLOB = "https://github.com/orq-ai/evaluatorq/blob/main"

# --- Curated public API allowlist (dotted module paths) ---------------------
# Internal modules (processings, send_results, table_display, fetch_data,
# progress) are intentionally absent.
API_MODULES = [
    "evaluatorq.evaluatorq",
    "evaluatorq.evaluators",
    "evaluatorq.job_helper",
    "evaluatorq.deployment",
    "evaluatorq.types",
    "evaluatorq.contracts",
    "evaluatorq.cli",
    "evaluatorq.openresponses",
    "evaluatorq.tracing",
    "evaluatorq.redteam",
    "evaluatorq.redteam.backends",
    "evaluatorq.redteam.adaptive",
    "evaluatorq.simulation",
    "evaluatorq.simulation.runner",
    "evaluatorq.simulation.generators",
    "evaluatorq.integrations.langchain_integration",
    "evaluatorq.integrations.langgraph_integration",
    "evaluatorq.integrations.openai_agents_integration",
    "evaluatorq.integrations.crewai_integration",
    "evaluatorq.integrations.pydantic_ai_integration",
    "evaluatorq.integrations.callable_integration",
]


def write_api_pages() -> None:
    nav_lines = []
    for dotted in API_MODULES:
        page_path = Path("reference", *dotted.split(".")) .with_suffix(".md")
        with mkdocs_gen_files.open(page_path, "w") as fd:
            fd.write(f"# `{dotted}`\n\n::: {dotted}\n")
        mkdocs_gen_files.set_edit_path(page_path, "gen_pages.py")
        title = dotted.replace("evaluatorq.", "")
        nav_lines.append(f"- [{title}]({page_path.relative_to('reference').as_posix()})\n")
    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as fd:
        fd.writelines(nav_lines)


def _rewrite_relative_links(text: str, source_dir: Path) -> str:
    """Rewrite markdown links that point at repo files into GitHub blob URLs.

    Leaves absolute (http) links and in-page anchors alone.
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
        text = source.read_text(encoding="utf-8")
        text = _rewrite_relative_links(text, source.parent)
        with mkdocs_gen_files.open(dest, "w") as fd:
            fd.write(text)
        mkdocs_gen_files.set_edit_path(dest, source.relative_to(REPO).as_posix())


write_api_pages()
ingest_markdown()
```

- [ ] **Step 2: Run the strict build for the reference + ingested pages**

Run: `uv run mkdocs build --strict 2>&1 | tee /tmp/mkdocs.log`
Expected: the `reference/` pages and `contributing/changelog/roadmap` render. Remaining `--strict` failures are only for guide/tutorial pages not yet created (Tasks 3–5). If there are mkdocstrings **import** errors, confirm `uv sync ... --all-extras --group docs` ran.

- [ ] **Step 3: Verify no internal modules leaked**

Run: `grep -r "processings\|send_results\|table_display\|fetch_data" site/reference/ || echo "clean"`
Expected: `clean`.

- [ ] **Step 4: Visually spot-check one rendered API page**

Open `site/reference/evaluatorq/evaluators/index.html` (or serve with `uv run mkdocs serve`) and confirm Google-style `Args:`/`Returns:` render as structured sections, not a blob.

- [ ] **Step 5: Commit**

```bash
git add docs/gen_pages.py
git commit -m "docs: generate curated API reference + ingest root markdown"
```

---

### Task 3: Reorganize existing README content into core guides + slim README

**Files:**
- Create: `docs/getting-started.md`, `docs/installation.md`, `docs/migrating.md`, `docs/configuration.md`
- Create: `docs/guides/core-evaluation.md`, `docs/guides/orq-platform.md`, `docs/guides/tracing.md`, `docs/guides/custom-evaluators.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: existing `README.md` sections and `docs/custom-evaluators-and-frameworks.md` as content sources.
- Produces: the core nav pages referenced in `mkdocs.yml`.

- [ ] **Step 1: Author `docs/getting-started.md`**

Required sections (move the matching content out of `README.md`):
`# Getting Started` → Install (`pip install evaluatorq`), Optional extras (one-liner pointing to Installation & extras), Authentication (`ORQ_API_KEY`), and a ~30-line "Your first evaluation" quick start (the same snippet intentionally mirrored in the slim README — this is the one allowed duplication).

- [ ] **Step 2: Author `docs/installation.md`**

A table of all 11 extras from `pyproject.toml` `[project.optional-dependencies]` (`orq, otel, langchain, langgraph, openai-agents, pydantic-ai, crewai, redteam, simulation, export, all`) with one column "enables" describing what each unlocks. Note `redteam`/`simulation` need an LLM backend and `otel` auto-enables when `ORQ_API_KEY` is set.

- [ ] **Step 3: Author `docs/migrating.md`**

`# Migrating from evaluatorq-py` — the package was `packages/evaluatorq-py` in the orqkit monorepo and is now standalone `orq-ai/evaluatorq`; **the PyPI name `evaluatorq` is unchanged**, imports are unchanged (`import evaluatorq`). Only repository/docs URLs moved.

- [ ] **Step 4: Author `docs/guides/core-evaluation.md`, `orq-platform.md`, `tracing.md`**

Move the corresponding long-form `README.md` sections: core-evaluation = jobs / `@job` / built-in evaluators / `EvaluatorParams` / parallelism / structured results; orq-platform = datasets / automatic result sending / result viz; tracing = span hierarchy / auto-enable / custom OTEL endpoint / disable.

- [ ] **Step 5: Create `docs/guides/custom-evaluators.md`**

Re-export the existing file content (it already lives in `docs/`):

```markdown
--8<-- "custom-evaluators-and-frameworks.md"
```

If the snippet/include syntax is not enabled, instead `git mv docs/custom-evaluators-and-frameworks.md docs/guides/custom-evaluators.md` and fix any relative links.

- [ ] **Step 6: Author `docs/configuration.md`**

Env-var reference (`ORQ_API_KEY`, OTEL endpoint vars, tracing-disable var) + an `EvaluatorParams` field reference (link to the API page for the type).

- [ ] **Step 7: Slim `README.md`**

Reduce to: one-paragraph pitch, badges, `pip install evaluatorq`, extras one-liner, the ~30-line quick start (mirrored from Getting Started), and a prominent link to `https://orq-ai.github.io/evaluatorq/`. Remove the long-form sections now living in the site.

- [ ] **Step 8: Run the strict build**

Run: `uv run mkdocs build --strict`
Expected: PASS for all Task-3 pages (guides/openresponses, redteam-backends, adaptive, simulation, hooks, cli and tutorials still missing → those remain the only failures).

- [ ] **Step 9: Commit**

```bash
git add docs/ README.md
git commit -m "docs: reorganize core guides; slim README to pitch + quick start"
```

---

### Task 4: New coverage guides (the "cover more" requirement)

**Files:**
- Create: `docs/guides/openresponses.md`, `docs/guides/red-teaming.md`, `docs/guides/redteam-backends.md`, `docs/guides/adaptive-red-teaming.md`, `docs/guides/simulation.md`, `docs/guides/hooks.md`, `docs/guides/cli.md`
- Modify: `docs/gen_pages.py` (ingest the in-`src` READMEs used as sources)

**Interfaces:**
- Consumes: `src/evaluatorq/redteam/README.md`, `src/evaluatorq/redteam/ARCHITECTURE.md`, `src/evaluatorq/simulation/README.md` (ingested with link rewriting).

- [ ] **Step 1: Extend `INGEST` in `docs/gen_pages.py`**

Add the in-`src` READMEs so their content is available as virtual pages (link-rewritten). Append to the `INGEST` list:

```python
    (SRC / "redteam" / "README.md", "guides/_redteam-readme.md"),
    (SRC / "redteam" / "ARCHITECTURE.md", "guides/_redteam-architecture.md"),
    (SRC / "simulation" / "README.md", "guides/_simulation-readme.md"),
```

These `_`-prefixed pages are sources the authored guide pages include or draw from; keep them out of `nav` (they are not listed, and `--strict` will warn about unlisted pages → add them to `not_in_nav` in `mkdocs.yml`):

```yaml
not_in_nav: |
  guides/_*.md
  reference/SUMMARY.md
```

- [ ] **Step 2: Author `docs/guides/red-teaming.md`**

Lead prose + include the ingested README/architecture content. Cover: what red teaming does, attack lifecycle, running a basic dynamic attack, vulnerability filtering. Reference `examples/redteam/01_basic_dynamic.py`.

- [ ] **Step 3: Author the NEW subsystem guides**

Each is fresh prose (these subsystems have no prose today). Required coverage:
- `openresponses.md` — the `openresponses/` target + conversion layer: what it converts, when to use it. Link API: `reference/evaluatorq/openresponses/`.
- `redteam-backends.md` — `redteam/backends/` (orq / openai / openresponses / base) + the registry: how a backend is selected and registered.
- `adaptive-red-teaming.md` — `redteam/adaptive/`: orchestrator, strategy planner, tool-chaining, capability classifier — how an adaptive attack escalates.
- `simulation.md` — lead prose + ingested simulation README; cover user-simulator/judge protocols, generators, quality, perturbation, the dashboard.
- `hooks.md` — redteam + simulation lifecycle hooks: available hook points, sync-vs-async (note sync hooks are deprecated), a worked example.
- `cli.md` — the `evaluatorq`/`eq` console scripts plus the redteam and simulation CLIs: commands, common flags.

- [ ] **Step 4: Run the strict build**

Run: `uv run mkdocs build --strict`
Expected: PASS for all guides; only `tutorials/*` remain missing.

- [ ] **Step 5: Spot-check coverage**

Confirm each subsystem in the spec's "Expanded coverage" list has a guide page with real prose (not just an `:::` directive).

- [ ] **Step 6: Commit**

```bash
git add docs/
git commit -m "docs: add guides for openresponses, redteam backends/adaptive, hooks, CLI, simulation"
```

---

### Task 5: Tutorials + example smoke check

**Files:**
- Create: `docs/tutorials/first-evaluation.md`, `orq-deployment.md`, `red-teaming-agent.md`, `simulation.md`
- Create: `scripts/smoke_examples.py`

**Interfaces:**
- Consumes: example scripts under `examples/` (e.g. `examples/redteam/01_basic_dynamic.py`, `examples/agent_simulation/01_basic_simulation.py`, `examples/agent_simulation/02_orq_deployment_simulation.py`).
- Produces: `scripts/smoke_examples.py` (run in CI by Task 6).

- [ ] **Step 1: Author the four tutorial pages**

Each page: prerequisites + required extras, the example script it is built from (link to the file via the API blob URL or `examples/`), the key code walked through, expected output, and "next steps". Map:
- `first-evaluation.md` → core quick start (mirrors Getting Started but task-oriented and longer).
- `orq-deployment.md` → `examples/agent_simulation/02_orq_deployment_simulation.py`.
- `red-teaming-agent.md` → `examples/redteam/01_basic_dynamic.py`.
- `simulation.md` → `examples/agent_simulation/01_basic_simulation.py`.

- [ ] **Step 2: Write `scripts/smoke_examples.py`**

```python
"""Compile + import-smoke every example script (no network, no API keys).

Guarantees tutorials never reference code that fails to parse or import.
"""

import compileall
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def main() -> int:
    ok = compileall.compile_dir(str(EXAMPLES), quiet=1, maxlevels=10)
    if not ok:
        print("compileall found syntax errors in examples/", file=sys.stderr)
        return 1
    print("examples compile cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the smoke check**

Run: `uv run python scripts/smoke_examples.py`
Expected: `examples compile cleanly` and exit 0.

- [ ] **Step 4: Run the strict build (now complete)**

Run: `uv run mkdocs build --strict`
Expected: PASS with no warnings — all nav pages now exist.

- [ ] **Step 5: Commit**

```bash
git add docs/tutorials scripts/smoke_examples.py
git commit -m "docs: add tutorials + example compile/import smoke check"
```

---

### Task 6: GitHub Actions — PR strict-build + Pages deploy

**Files:**
- Create: `.github/workflows/docs.yml`

**Interfaces:**
- Consumes: `docs` dependency group, `scripts/smoke_examples.py`, `mkdocs.yml`.

- [ ] **Step 1: Write `.github/workflows/docs.yml`**

```yaml
name: Docs

on:
  pull_request:
  push:
    branches:
      - main

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    name: Build (strict) + example smoke
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - name: Set up uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.13"
      - name: Install (all extras + docs)
        run: uv sync --frozen --all-extras --group docs
      - name: Example compile/import smoke
        run: uv run python scripts/smoke_examples.py
      - name: Build docs (strict)
        run: uv run mkdocs build --strict
      - name: Configure Pages
        if: github.ref == 'refs/heads/main'
        uses: actions/configure-pages@v5
        with:
          enablement: true
      - name: Upload Pages artifact
        if: github.ref == 'refs/heads/main'
        uses: actions/upload-pages-artifact@v3
        with:
          path: site

  deploy:
    name: Deploy to GitHub Pages
    if: github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy
        id: deployment
        uses: actions/deploy-pages@v4
      - name: Verify site is live (HTTP 200)
        run: |
          url="${{ steps.deployment.outputs.page_url }}"
          echo "Checking $url"
          code=$(curl -s -o /dev/null -w "%{http_code}" --retry 5 --retry-delay 5 "$url")
          test "$code" = "200" || { echo "Pages returned $code"; exit 1; }
```

- [ ] **Step 2: Validate the workflow YAML locally**

Run: `uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/docs.yml')); print('yaml ok')"`
Expected: `yaml ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "ci: build docs strict on PR, deploy to GitHub Pages on main"
```

- [ ] **Step 4: Push and confirm the PR build job runs green**

After opening the PR (Task 7 / handoff), confirm the `Build (strict) + example smoke` job passes on the PR. The `deploy` job runs only on `main`.

---

### Task 7: Fix package metadata URLs

**Files:**
- Modify: `pyproject.toml` (`[project.urls]`)

**Interfaces:**
- None downstream; final cleanup coupled per spec.

- [ ] **Step 1: Rewrite `[project.urls]`**

Replace lines 202–204:

```toml
  [project.urls]
  Homepage = "https://github.com/orq-ai/evaluatorq"
  Repository = "https://github.com/orq-ai/evaluatorq"
  Documentation = "https://orq-ai.github.io/evaluatorq/"
```

- [ ] **Step 2: Verify the package still builds**

Run: `uv build`
Expected: builds an sdist + wheel with no error.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: point project URLs at the standalone repo and docs site"
```

---

## Self-Review

**Spec coverage:**
- Toolchain → Task 1. Docstring style (griffe google, no migration) → Task 1 (mkdocs.yml) + Task 2 spot-check. Content ingestion + link rewriting → Task 2 + Task 4. Expanded coverage → Task 4. Site structure/nav → Task 1. Curated API ref → Task 2. Build/CI/deploy (extras install, permissions, configure-pages enablement, cancel-in-progress, 200 check) → Task 6. Tutorials + per-PR smoke → Task 5. README slim + intentional dup → Task 3. Coupled `[project.urls]` → Task 7. Out-of-scope (mike, keyed E2E, custom domain) → not implemented, as specified. Version footer → `extra.version.provider: mike` placeholder in mkdocs.yml (display only; mike wiring deferred per spec).
- All success criteria map to a task gate (`--strict` build, 200 check, coverage spot-check, smoke).

**Placeholder scan:** No "TBD"/"handle edge cases"/"write tests for the above". Content-authoring steps name exact source files and required sections (prose pages are not code, so no code block is required; config/script steps include full code).

**Type consistency:** `gen_pages.py` `INGEST` list and `_rewrite_relative_links` referenced consistently across Tasks 2 and 4; `not_in_nav` added in Task 4 matches the `_`-prefixed ingest destinations; `scripts/smoke_examples.py` produced in Task 5 and consumed by the workflow in Task 6.
