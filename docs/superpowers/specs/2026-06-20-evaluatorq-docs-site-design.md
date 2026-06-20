# evaluatorq documentation site — design

**Date:** 2026-06-20
**Repo:** `orq-ai/evaluatorq`
**Branch:** `docs/mkdocs-site`
**Status:** approved

## Goal

Publish a structured, public documentation site for the `evaluatorq` Python
package on GitHub Pages. The site replaces the single 34 KB `README.md` as the
primary user-facing documentation, adds task-oriented tutorials, and exposes an
auto-generated API reference. The README slims down to a pitch + quick start +
link to the site.

## Toolchain

- **MkDocs** with the **Material** theme — markdown-native, so existing content
  needs near-zero conversion.
- **mkdocstrings[python]** (griffe backend) for the API reference, rendering
  from **Google-style** docstrings + type hints.
- **mkdocs-gen-files** + **mkdocs-literate-nav** to generate one API page per
  public module from the package tree.
- **mkdocs-section-index** so section landing pages work.
- All added as a `docs` dependency group in `pyproject.toml`. Site config lives
  in `mkdocs.yml` at the repo root.

Rejected alternatives: Sphinx (+autodoc / +MyST). Sphinx gives stronger
autodoc edge-case handling, intersphinx, doctest-in-CI, and PDF/man output, but
costs markdown→rst conversion and heavier config. None of those Sphinx-only
capabilities are needed for a markdown-native, type-hinted public package. If
executable doctests-in-CI ever become a required quality gate, revisit.

## Docstring style migration (its own phase)

The codebase today is **mostly NumPy-style** docstrings (~2325 `----------`
section underlines) with ~100 Google-style (`Args:`) mixed in. The site
standardises on **Google style** for readability and consistent mkdocstrings
rendering.

- Convert NumPy → Google across `src/evaluatorq/`.
- Auto-converters (e.g. pyment) are unreliable; do it carefully, module by
  module.
- Each converted module is verified by (a) a clean `mkdocs build --strict`
  (mkdocstrings resolves it) and (b) the existing test suite still passing.
- griffe configured with `docstring_style: google`.
- This phase is independent of the site scaffolding and can proceed in
  parallel module batches.

## Site structure (explicit nav)

```
Home                     slim intro, badges, install, link into docs
Getting Started          install, optional extras, auth (ORQ_API_KEY), first eval
Guides/
  Core evaluation        jobs, @job decorator, evaluators, EvaluatorParams,
                         parallelism, structured results
  Orq platform           datasets, automatic result sending, result viz
  OpenTelemetry tracing  span hierarchy, auto-enable, custom endpoint, disable
  Red teaming            from src/evaluatorq/redteam/README.md + ARCHITECTURE.md
  Simulation             from src/evaluatorq/simulation/README.md + dashboard
  Custom evaluators & frameworks   from docs/custom-evaluators-and-frameworks.md
Tutorials/               NEW — task-oriented, runnable end-to-end
  Your first evaluation
  Evaluating an Orq deployment
  Red-teaming an agent
  Simulating a multi-turn conversation
Configuration            env vars reference, EvaluatorParams reference
API Reference/           auto-generated (mkdocstrings), one page per module
Contributing             reuse CONTRIBUTING.md
Changelog                reuse CHANGELOG.md
Roadmap                  reuse ROADMAP.md
```

Existing content sources mapped above are reorganised, not rewritten, except
where splitting the monolithic README requires light editing for standalone
page flow. `docs/superpowers/` (skill scratch: plans, specs) is excluded from
the build via explicit nav + `exclude_docs`.

## API reference

A `docs/gen_ref_pages.py` script run by mkdocs-gen-files walks
`src/evaluatorq/`, and for each public module emits a virtual stub page
containing a `::: evaluatorq.<module>` mkdocstrings directive. mkdocs-literate-nav
assembles the API nav tree from a generated `SUMMARY.md`. Private modules
(leading underscore) and test helpers are skipped.

## Deployment

`.github/workflows/docs.yml`:

- **On pull request:** `mkdocs build --strict` only (no deploy) — catches
  broken refs / nav before merge.
- **On push to `main`:** build + deploy to GitHub Pages.
- Pages source = **GitHub Actions** (not a `gh-pages` branch).
- `concurrency` guard so overlapping deploys cancel cleanly.
- Uses `astral-sh/setup-uv` consistent with the existing CI; installs the
  `docs` dependency group.

Repo settings prerequisite (manual, outside this work): enable GitHub Pages
with source = GitHub Actions. Noted as a handoff item.

## README

Slim `README.md` to: one-paragraph pitch, badges, install (+ optional extras),
a ~30-line quick start, and a prominent link to the docs site. All long-form
content (advanced features, configuration tables, per-domain guides) lives in
the site only — no duplication between README and site.

## Out of scope (YAGNI — add when)

- **Versioned docs (`mike`)** — add when a v2 with breaking changes ships.
- **Doctest-in-CI** — add if docs code samples start drifting from the API.
- **Custom domain** — add when marketing requests one; default
  `orq-ai.github.io/evaluatorq` until then.

## Success criteria

- `mkdocs build --strict` passes with no warnings.
- Site deploys to GitHub Pages on merge to `main`; PR builds are gated.
- API reference renders every public module from Google-style docstrings.
- README is slim and links to the site; no content duplicated.
- Existing test suite remains green after docstring migration.
