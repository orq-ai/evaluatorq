#!/usr/bin/env bash
# Advisory scope check for docs-autofill step 5. Prints how far the adjacent
# edits are from the budget and in which direction; never fails the run. The
# agent reads it and decides, so exit 0 is deliberate.
#
# Usage: .claude/skills/docs-autofill/scope_check.sh docs/guides/<page>.md [base-ref]
#
# Budget: at most 3 files besides the page, mkdocs.yml and the ledger, and about
# 60 added lines across them.

set -uo pipefail

PAGE="${1:?usage: scope_check.sh <page.md> [base-ref]}"
BASE="${2:-origin/main}"
MAX_FILES=3
MAX_LINES=60
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || { printf 'scope check did not run: cannot enter %s\n' "$ROOT"; exit 0; }

if ! git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null; then
  printf 'scope check did not run: unknown base ref %s\n' "$BASE"
  exit 0
fi

files=0
lines=0
fences=0
src=""
list=""

count() {
  local file="$1" added="$2"
  case "$file" in
    "$PAGE" | mkdocs.yml | .claude/skills/docs-autofill/ledger.md) return ;;
    src/*) src="$src $file" ;;
  esac
  files=$((files + 1))
  lines=$((lines + added))
  list="$list\n  $file (+$added)"
}

while IFS=$'\t' read -r added _ file; do
  [ "$added" = "-" ] && continue
  count "$file" "$added"
done < <(git diff --numstat "$BASE" 2>/dev/null)

while IFS= read -r file; do
  count "$file" "$(wc -l <"$file" | tr -d ' ')"
done < <(git ls-files --others --exclude-standard)

fences=$(git diff -U0 "$BASE" -- . ':!'"$PAGE" 2>/dev/null | grep -cE '^[+-][[:space:]]*```')
while IFS= read -r file; do
  [ "$file" = "$PAGE" ] && continue
  fences=$((fences + $(grep -cE '^[[:space:]]*```' "$file" 2>/dev/null)))
done < <(git ls-files --others --exclude-standard)

printf 'adjacent edits: %s files (budget %s), %s added lines (budget about %s)\n' "$files" "$MAX_FILES" "$lines" "$MAX_LINES"
[ -n "$list" ] && printf '%b\n' "$list"
[ "$files" -gt "$MAX_FILES" ] && printf 'over by %s file(s): move the least related edit to a follow-up\n' "$((files - MAX_FILES))"
[ "$lines" -gt "$MAX_LINES" ] && printf 'over by %s line(s): trim, or move the largest edit to a follow-up\n' "$((lines - MAX_LINES))"
[ "$fences" -gt 0 ] && printf 'note: %s code-fence line(s) added or removed outside the page; adjacent edits must not change code blocks\n' "$fences"
[ -n "$src" ] && printf 'src/ touched:%s — step 5 says revert it\n' "$src"
exit 0
