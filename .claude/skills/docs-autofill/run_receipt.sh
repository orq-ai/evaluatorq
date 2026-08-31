#!/usr/bin/env bash
# Extract every fenced code block from a docs page, run the executable ones in
# ONE persistent shell in document order, and write .context/receipt.txt.
#
# Tracked on purpose: the receipt itself is gitignored, so a runner living in
# .context/ could not be reviewed. A reviewer reads this file to know what
# "exit=0" actually meant.
#
# Usage: .claude/skills/docs-autofill/run_receipt.sh docs/guides/<page>.md
#
# Languages:
#   python        -> uv run python -, in the session's working directory
#   bash / sh     -> run with the project venv on PATH
#   everything else (text, yaml, json, ...) -> recorded as illustrative, never run
#
# Blocks run against a fresh EVALUATORQ_DIR so a page cannot pass by leaning on
# a run store some earlier probe happened to populate.

set -uo pipefail

PAGE="${1:?usage: run_receipt.sh <page.md>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1

RECEIPT="$ROOT/.context/receipt.txt"
WORK="$(mktemp -d)"
mkdir -p "$ROOT/.context"

EVALUATORQ_DIR="$(mktemp -d)"
export EVALUATORQ_DIR
export PATH="$ROOT/.venv/bin:$PATH"

: >"$RECEIPT"
{
  echo "receipt for $PAGE"
  echo "generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "EVALUATORQ_DIR=$EVALUATORQ_DIR (fresh)"
  echo
} >>"$RECEIPT"

# Split the page into blocks: <index>\t<lang> index files land in $WORK/NNN.body
uv run python - "$PAGE" "$WORK" <<'PY'
import pathlib
import sys

page, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lines = page.read_text().splitlines()

blocks, lang, buf, fence = [], None, [], None
for line in lines:
    stripped = line.lstrip()
    if fence is None:
        if stripped.startswith("```"):
            fence = len(line) - len(stripped)
            lang = stripped[3:].strip() or "text"
            buf = []
        continue
    if stripped.startswith("```"):
        blocks.append((lang, "\n".join(buf)))
        fence, lang, buf = None, None, []
        continue
    buf.append(line[fence:] if line[:fence].strip() == "" else line)

index = work / "index.tsv"
with index.open("w") as fh:
    for i, (block_lang, body) in enumerate(blocks, start=1):
        (work / f"{i:03d}.body").write_text(body + "\n")
        fh.write(f"{i:03d}\t{block_lang}\n")
print(f"{len(blocks)} blocks extracted", file=sys.stderr)
PY

STATE="$WORK/state.sh"   # carries exports between bash blocks (one logical shell)
: >"$STATE"
FAILED=0

while IFS=$'\t' read -r idx lang; do
  body="$WORK/$idx.body"
  printf -- '--- block %s (%s)\n' "$idx" "$lang" >>"$RECEIPT"

  case "$lang" in
    python)
      out="$(uv run python - <"$body" 2>&1)"
      code=$?
      cmd="uv run python -  # block $idx"
      ;;
    bash | sh | shell)
      out="$(bash -c "source '$STATE' 2>/dev/null; source '$ROOT/.venv/bin/activate' 2>/dev/null; $(cat "$body"); rc=\$?; export -p >'$STATE'; exit \$rc" 2>&1)"
      code=$?
      cmd="bash  # block $idx"
      ;;
    *)
      printf 'command: —\nexit: — illustrative (%s block, not executable by nature)\n\n' "$lang" >>"$RECEIPT"
      continue
      ;;
  esac

  # Full output goes beside the receipt so the author can check that a sample
  # output quoted in the page matches what the block really printed. The receipt
  # itself stays at 5 lines; this is the evidence behind it, not a second receipt.
  mkdir -p "$ROOT/.context/receipt-full"
  printf '%s\n' "$out" >"$ROOT/.context/receipt-full/$idx.out"

  printf 'command: %s\nexit: %s\n' "$cmd" "$code" >>"$RECEIPT"
  printf 'output (first 5 lines):\n' >>"$RECEIPT"
  printf '%s\n' "$out" | head -5 >>"$RECEIPT"
  printf '\n' >>"$RECEIPT"
  [ "$code" -ne 0 ] && FAILED=1
done <"$WORK/index.tsv"

if [ "$FAILED" -ne 0 ]; then
  echo "RESULT: FAILED — at least one block exited nonzero" >>"$RECEIPT"
else
  echo "RESULT: OK — every executable block exited 0" >>"$RECEIPT"
fi

cat "$RECEIPT"
exit "$FAILED"
