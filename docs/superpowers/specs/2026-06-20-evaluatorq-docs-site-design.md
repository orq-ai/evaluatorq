# evaluatorq documentation site — design

**Date:** 2026-06-20
**Repo:** `orq-ai/evaluatorq`
**Branch:** `docs/mkdocs-site`
**Status:** approved (reworked after `/hate` review)

## Goal

Publish a structured, public documentation site for the `evaluatorq` Python
package on GitHub Pages. The site becomes the primary user-facing documentation,
**expands coverage to currently-undocumented subsystems**, adds CI-checked
tutorials, and exposes an auto-generated API reference. The README slims to a
pitch + an intentionally-mirrored quick start + a link to the site.

## Toolchain

- **MkDocs** + **Material** theme — markdown-native; existing content needs no
  format conversion.
- **mkdocstrings[python]** (griffe) for the API reference, rendering from the
  **Google-style** docstrings the codebase already uses (verified: 0 NumPy-style
  docstrings; 216 Google `Args:`/`Returns:` blocks across 55 files).
- **mkdocs-gen-files** generates the API stub pages and pulls out-of-tree
  markdown into the build (see "Content ingestion"); **mkdocs-literate-nav**
  assembles the API Reference nav subtree from a generated `SUMMARY.md`. The
  rest of the nav is explicit in `mkdocs.yml`.
- **mkdocs-section-index** for clickable section landing pages.
- Added as a `docs` dependency group in `pyproject.toml`; config in `mkdocs.yml`
  at the repo root. `docs_dir: docs`.

Rejected: Sphinx (+autodoc / +MyST) — stronger autodoc edge cases, intersphinx,
doctest-in-CI, PDF/man output, but costs markdown→rst conversion and heavier
config. None of those are needed for a markdown-native, type-hinted package.

## Docstring style (NOT a migration)

The `/hate` review corrected the earlier draft: there is **no NumPy→Google
migration to do**. The codebase is already Google-style. The only work is:

- Set mkdocstrings/griffe `docstring_style: google`.
- Run `mkdocs build --strict` and fix the handful of docstrings it actually
  warns about. A stray malformed docstring renders wrong **without** failing
  `--strict`, so the build-strict gate is not sufficient on its own — the API
  reference is spot-checked visually as an acceptance step (see Success
  criteria).

## Content ingestion (the real work)

MkDocs only serves files under `docs_dir`. Several reused sources live elsewhere
and use relative links to siblings/source/examples that would break the build:

- `src/evaluatorq/redteam/README.md`, `src/evaluatorq/redteam/ARCHITECTURE.md`
- `src/evaluatorq/simulation/README.md`
- root `CONTRIBUTING.md`, `CHANGELOG.md`, `ROADMAP.md`
- `docs/custom-evaluators-and-frameworks.md` (already in `docs/`)

A `docs/gen_pages.py` (run by mkdocs-gen-files) copies each out-of-tree file
into the virtual docs tree and **rewrites relative links** during the copy:
links to source files → GitHub blob URLs at the pinned ref; links to `examples/`
→ the corresponding tutorial/example page or a GitHub blob URL. There is **no
`simulation/ARCHITECTURE.md`** (only redteam has one) — the simulation guide is
sourced from its README plus new prose.

## Expanded coverage (the "cover more" requirement)

Reorganising the README is not enough. These subsystems have little or no prose
today and get **new authored guide pages** (not just an API dump):

- `openresponses/` — target + conversion layer
- `redteam/backends/` — orq / openai / openresponses / base + registry
- `redteam/adaptive/` — orchestrator, strategy planner, tool-chaining,
  capability classifier (the attack engine)
- simulation generators / quality / perturbation
- hooks — redteam and simulation lifecycle hooks
- CLIs — `evaluatorq` / `eq` console scripts, plus redteam and simulation CLIs

## Site structure (explicit nav; API Reference subtree via literate-nav)

```
Home                     pitch, badges, install, link into docs
Getting Started          install, extras matrix, auth (ORQ_API_KEY), first eval
Installation & extras    matrix of all 11 extras and what each enables
Migrating from evaluatorq-py   monorepo path -> standalone repo (same PyPI name)
Guides/
  Core evaluation        jobs, @job, evaluators, EvaluatorParams, parallelism,
                         structured results
  Orq platform           datasets, automatic result sending, result viz
  OpenTelemetry tracing  span hierarchy, auto-enable, custom endpoint, disable
  OpenResponses          target + conversion layer (NEW)
  Red teaming            redteam/README.md + ARCHITECTURE.md
  Red team backends      backends + registry (NEW)
  Adaptive red teaming   adaptive engine internals (NEW)
  Simulation             simulation/README.md + new prose
  Hooks                  redteam + simulation lifecycle hooks (NEW)
  Custom evaluators & frameworks   docs/custom-evaluators-and-frameworks.md
  CLI reference          evaluatorq/eq + redteam/simulation CLIs (NEW)
Tutorials/               task-oriented; code lives in examples/, CI-checked
  Your first evaluation
  Evaluating an Orq deployment
  Red-teaming an agent
  Simulating a multi-turn conversation
API Reference/           auto-generated (mkdocstrings), __all__-driven per package
Configuration            env vars, EvaluatorParams reference
Contributing             CONTRIBUTING.md
Changelog                CHANGELOG.md
Roadmap                  ROADMAP.md
```

`docs/superpowers/` (skill scratch: plans, specs) is kept out of the build with
`exclude_docs` (not just explicit nav, which alone still warns).

## API reference (driven by `__all__`)

Public API = whatever each package's `__init__.py` `__all__` re-exports.
`docs/gen_pages.py` emits one `::: evaluatorq.<package>` page per package that
declares `__all__`, and griffe documents exactly those members. `__all__` is the
single source of truth, so internal modules (`processings`, `send_results`,
`table_display`, `fetch_data`, `progress`, `job_helper`) are excluded
automatically — they are not packages, and their public symbols (e.g. the `job`
decorator) surface through the top-level `evaluatorq` `__all__`, documented once.
Empty-`__init__` subpackages (`redteam/backends`, `redteam/frameworks`,
`redteam/runtime`) declare no `__all__` and are covered by prose guides, not API
pages. Integration packages import optional heavy deps, so docs CI installs all
extras (below) to let mkdocstrings introspect them.

## Build / CI / deploy

**Import-failure fix (was a guaranteed `--strict` failure):** integration
modules import optional deps unconditionally (e.g.
`openai_agents_integration/target.py` → `from agents import ...`,
langchain/langgraph targets). With only the `docs` group installed, mkdocstrings
introspection raises `ModuleNotFoundError` and `--strict` fails. Docs CI
therefore runs `uv sync --frozen --all-extras --group docs` (mirrors the main
CI's `--all-extras --all-groups`) so every import resolves.

`.github/workflows/docs.yml`:

- **On pull request:** `mkdocs build --strict` only (no deploy) — gates broken
  refs/nav before merge.
- **On push to `main`:** build + deploy to GitHub Pages.
- `permissions: { contents: read, pages: write, id-token: write }`.
- `environment: github-pages`.
- Steps use `astral-sh/setup-uv`, then `actions/configure-pages@v5` with
  `enablement: true` (programmatically enables Pages — **no manual settings
  toggle**, removing the top silent-failure risk), `actions/upload-pages-artifact`,
  `actions/deploy-pages`.
- `concurrency: { group: pages, cancel-in-progress: false }` — never cancel a
  deploy mid-publish.
- **Post-deploy verification step:** assert the deployed URL returns HTTP 200
  (fail the job otherwise) so "Pages live" is a verified deliverable, not an
  assumption.

## Tutorials (defined + CI-checked, with an honest ceiling)

Each tutorial page documents: prerequisites + extras, the runnable script it is
built from, expected output, and next steps. Tutorial code lives as real scripts
under `examples/` (reuse the existing example scripts and the
`red_teaming_intro` / `agent_simulation_intro` notebooks where they fit).

CI gating, in two tiers:

- **Per-PR (no secrets):** `python -m compileall` over the example scripts
  (compile-only, NOT import — many examples import deps in no extra, e.g.
  `fastapi`), so a tutorial can never reference a script that fails to parse.
  `mkdocs build --strict` additionally guarantees every referenced snippet file
  exists.
- **Full execution needs `ORQ_API_KEY` / LLM access**, so genuine end-to-end
  runs are a separate manual/nightly job, not the per-PR gate.
  *ponytail: per-PR runs would need live API keys; compile+import smoke is the
  honest gate, upgrade to a keyed nightly run if tutorials start breaking.*

## README

Slim to: one-paragraph pitch, badges, install (+ extras one-liner), a ~30-line
quick start, and a prominent link to the docs site. The quick start is
**intentionally mirrored** on the Getting Started page (standard for the
PyPI/GitHub landing page) — this is the one deliberate duplication. All other
long-form content lives in the site only.

## Coupled package-metadata fix

`pyproject.toml` `[project.urls]` still point at the monorepo and must be
rewritten as part of this work (also tracked under RES-949):

- `Homepage = "https://github.com/orq-ai/orqkit"` → `https://github.com/orq-ai/evaluatorq`
- `Repository = ".../orqkit/tree/main/packages/evaluatorq-py"` → repo root
- `Documentation = ".../orqkit/tree/main/packages/evaluatorq-py"` → the Pages URL

## Out of scope (YAGNI — add when)

- **Versioned docs (`mike`)** — add when a v2 with breaking changes ships. No
  version selector/footer until then (the `mike` provider is NOT wired up — it
  would render an empty selector without a mike deployment).
- **Keyed end-to-end tutorial runs in per-PR CI** — add if tutorials drift;
  nightly/manual for now.
- **Custom domain** — default `orq-ai.github.io/evaluatorq` until marketing asks.

## Success criteria

- `mkdocs build --strict` passes with no warnings.
- Site deploys to GitHub Pages on merge to `main`; the post-deploy step confirms
  the URL returns 200; PR builds are gated.
- API reference renders the curated public modules; internal modules are absent.
- **Every subsystem listed under "Expanded coverage" has a prose guide page**,
  not only an API entry (spot-checked).
- Tutorials reference scripts that pass the per-PR compile/import smoke.
- README is slim, links to the site, and duplicates only the quick start.
- `pyproject.toml` URLs point at the new repo / Pages site.
- API reference rendering is visually spot-checked (catches malformed docstrings
  that `--strict` does not).
