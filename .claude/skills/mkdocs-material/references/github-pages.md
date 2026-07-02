# GitHub Pages Deployment Recipes

## Standard flow (public repo)

```bash
mkdocs build --strict          # gate: never deploy a non-strict build
mkdocs gh-deploy --force       # builds + force-pushes site to gh-pages branch
```

`gh-deploy` adds `.nojekyll` automatically. Site appears at
`https://<org>.github.io/<repo>/` once Pages is enabled.

## Enabling Pages (one-time, required — gh-deploy does not do this)

```bash
# Check current state; enable from gh-pages branch if 404:
gh api repos/<org>/<repo>/pages 2>/dev/null \
  || gh api -X POST repos/<org>/<repo>/pages \
       -f "source[branch]=gh-pages" -f "source[path]=/"
```

## Private / internal repos

The site is NOT at `https://<org>.github.io/<repo>/`. GitHub serves it on an
obfuscated private domain behind org SSO. Get the real URL and use it as
`site_url` in mkdocs.yml:

```bash
gh api repos/<org>/<repo>/pages --jq .html_url
# e.g. https://shiny-adventure-e4ky67z.pages.github.io/
```

A `curl` of that URL returns `302` to `https://github.com/pages/auth?...` —
that means it works; auth happens in the browser.

## Verifying the deployment actually built

A successful `gh-deploy` push does not guarantee Pages built the site.
The Pages build runs server-side and can error independently:

```bash
gh api repos/<org>/<repo>/pages/builds/latest --jq '{status, error: .error.message}'
```

- `"status": "built"` — live.
- `"status": "building"` — poll again; large sites can take 1–3 minutes.
- `"status": "errored"` — trigger a rebuild, then re-check:

```bash
gh api -X POST repos/<org>/<repo>/pages/builds
```

A transient `errored` right after first enabling Pages is common; one
rebuild usually fixes it. Persistent errors: check the gh-pages branch has
`index.html` and `.nojekyll` at its root (`git ls-tree --name-only origin/gh-pages`).

## CI auto-deploy (optional)

```yaml
# .github/workflows/docs.yml
name: docs
on:
  push:
    branches: [main]
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.x" }
      - run: pip install mkdocs-material
      - run: mkdocs gh-deploy --force
```
