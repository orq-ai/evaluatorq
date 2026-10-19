#!/usr/bin/env bash
# Superset workspace teardown: archive anything that would be lost, then free disk.
# Runs when the workspace is deleted — the worktree directory disappears right after.
set -euo pipefail

MAIN="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
if [ "$MAIN" = "$PWD" ]; then
  echo "teardown: running in the main checkout, refusing to touch it"
  exit 0
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo detached)"
# Archives live outside every checkout, under ~/.superset/archive/<repo>/, so they
# survive deleting the worktree AND never show up in `git status`.
ARCHIVE_ROOT="${SUPERSET_ARCHIVE_ROOT:-$HOME/.superset/archive}"
ARCHIVE_PARENT="$ARCHIVE_ROOT/$(basename "$MAIN")"

# Archive uncommitted work: a patch plus any untracked, non-ignored files.
# Committed work survives in the shared object store via the branch; this covers
# the rest.
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  mkdir -p "$ARCHIVE_PARENT"
  DEST="$(mktemp -d "$ARCHIVE_PARENT/$(basename "$PWD")-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
  {
    echo "branch: $BRANCH"
    echo "head:   $(git rev-parse HEAD 2>/dev/null || echo none)"
    echo "path:   $PWD"
    echo "date:   $(date -Iseconds)"
  } >"$DEST/MANIFEST.txt"
  git status --porcelain >"$DEST/status.txt"

  # Use binary-safe output; a failed write must stop teardown before cleanup.
  git diff --binary HEAD >"$DEST/uncommitted.patch"

  if [ -n "$(git ls-files --others --exclude-standard)" ]; then
    git ls-files --others --exclude-standard -z |
      tar -czf "$DEST/untracked.tar.gz" --null -T -
  fi
  echo "teardown: archived uncommitted work to $DEST"
fi

# Free disk: the venv is ~2 GB per worktree and fully rebuildable from uv.lock.
rm -rf .venv
echo "teardown: done"
