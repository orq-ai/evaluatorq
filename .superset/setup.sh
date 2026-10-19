#!/usr/bin/env bash
# Superset workspace setup: make a fresh worktree usable without manual steps.
# Idempotent: safe to re-run.
set -euo pipefail

MAIN="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
if [ "$MAIN" = "$PWD" ]; then
  echo "setup: running in the main checkout, nothing to do"
  exit 0
fi

# 1. Credentials. .env is gitignored, so a fresh worktree has none. Orq CLI
#    profiles live in ~/.orq and are shared already.
if [ ! -e .env ] && [ -f "$MAIN/.env" ]; then
  cp "$MAIN/.env" .env
  echo "setup: copied .env from $MAIN"
fi

# 2. Shared run store. Symlink, never copy: previous runs, sim-runs and the
#    dashboard's history stay visible from every workspace.
if [ ! -e .evaluatorq ] && [ ! -L .evaluatorq ]; then
  mkdir -p "$MAIN/.evaluatorq"
  ln -s "$MAIN/.evaluatorq" .evaluatorq
  echo "setup: linked .evaluatorq -> $MAIN/.evaluatorq"
fi

# 3. Dependencies. --all-extras pulls the optional stacks (orq, otel, langchain,
#    langgraph, openai-agents, pydantic-ai, crewai, redteam, simulation,
#    dashboard) so framework comparisons and the dashboard work out of the box.
uv sync --all-extras

echo "setup: done"
