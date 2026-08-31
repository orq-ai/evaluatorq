#!/usr/bin/env bash
# Extract every fenced code block from a docs page, run the executable ones in
# document order, and write .context/receipt.txt.
#
# Tracked on purpose: the receipt itself is gitignored, so a runner living in
# .context/ could not be reviewed. A reviewer reads this file to know what
# "exit=0" actually meant.
#
# Usage: .claude/skills/docs-autofill/run_receipt.sh docs/guides/<page>.md
#
# Languages:
#   bash / sh     -> one logical shell: exports carry forward between blocks
#   python        -> uv run python -, a FRESH process per block (no shared state)
#   everything else (text, yaml, json, console, ...) -> recorded, never run
#
# Blocks run against a fresh EVALUATORQ_DIR so a page cannot pass by leaning on
# a run store some earlier probe happened to populate.
#
# The OK verdict is a counted assertion, never a default: the script refuses to
# claim success unless every extracted block was accounted for. A receipt that
# ran nothing must not be able to say "every block exited 0".

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
trap 'rm -rf "$WORK" "$EVALUATORQ_DIR"' EXIT

: >"$RECEIPT"
{
  echo "receipt for $PAGE"
  echo "generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "EVALUATORQ_DIR=$EVALUATORQ_DIR (fresh)"
  echo
} >>"$RECEIPT"

# Bail out with a receipt that says why, rather than a silent green one.
die() {
  printf 'RESULT: FAILED — %s\n' "$1" >>"$RECEIPT"
  cat "$RECEIPT"
  exit 1
}

# Split the page into blocks: <index>\t<lang> index files land in $WORK/NNN.body
if ! uv run python - "$PAGE" "$WORK" <<'PY'
import pathlib
import sys

page, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lines = page.read_text().splitlines()

# Track the opening fence's backtick count and close only on a run at least that
# long — CommonMark's own rule. A 3-backtick fence inside a 4-backtick block is
# then content, which is how this repo writes docstring samples.
blocks, lang, buf, fence, ticks = [], None, [], None, 0
for line in lines:
    stripped = line.lstrip()
    run = len(stripped) - len(stripped.lstrip("`"))
    if fence is None:
        if run >= 3:
            fence = len(line) - len(stripped)
            ticks = run
            lang = stripped[run:].strip() or "text"
            buf = []
        continue
    if run >= ticks and not stripped[run:].strip():
        blocks.append((lang, "\n".join(buf)))
        fence, lang, buf, ticks = None, None, [], 0
        continue
    buf.append(line[fence:] if line[:fence].strip() == "" else line)

index = work / "index.tsv"
with index.open("w") as fh:
    for i, (block_lang, body) in enumerate(blocks, start=1):
        (work / f"{i:03d}.body").write_text(body + "\n")
        fh.write(f"{i:03d}\t{block_lang}\n")
(work / "count").write_text(str(len(blocks)))
print(f"{len(blocks)} blocks extracted", file=sys.stderr)
PY
then
  die "block extraction failed — no blocks were run"
fi

[ -s "$WORK/count" ] || die "extractor wrote no block count — no blocks were run"
EXTRACTED="$(cat "$WORK/count")"
[ "$EXTRACTED" -gt 0 ] || die "the page contains no fenced code blocks"

STATE="$WORK/state.sh"   # carries exports between bash blocks (one logical shell)
: >"$STATE"
FAILED=0
SEEN=0

while IFS=$'\t' read -r idx lang; do
  body="$WORK/$idx.body"
  SEEN=$((SEEN + 1))
  printf -- '--- block %s (%s)\n' "$idx" "$lang" >>"$RECEIPT"

  case "$lang" in
    python)
      out="$(uv run python - <"$body" 2>&1)"
      code=$?
      ;;
    bash | sh | shell)
      # The epilogue goes in a file, on its own line: appending it to the body
      # inline lets a trailing comment in the block swallow it, which silently
      # drops every export the next block expects.
      {
        cat "$body"
        printf '\nrc=$?\nexport -p >%q\nexit $rc\n' "$STATE"
      } >"$WORK/$idx.run.sh"
      out="$(bash -c "source '$STATE'; source '$ROOT/.venv/bin/activate' 2>/dev/null; exec bash '$WORK/$idx.run.sh'" 2>&1)"
      code=$?
      ;;
    *)
      # "not run by this runner" — never "not executable by nature". Whether a
      # console or yaml block could have been executed is a claim for the
      # reviewer to make, not one this script is in a position to assert.
      printf 'command: —\nexit: — illustrative (%s block, not run by this runner)\n\n' "$lang" >>"$RECEIPT"
      continue
      ;;
  esac

  printf 'command: %s\nexit: %s\n' "$lang" "$code" >>"$RECEIPT"
  printf 'output (first 5 lines):\n' >>"$RECEIPT"
  printf '%s\n' "$out" | head -5 >>"$RECEIPT"
  printf '\n' >>"$RECEIPT"
  [ "$code" -ne 0 ] && FAILED=1
done <"$WORK/index.tsv"

[ "$SEEN" -eq "$EXTRACTED" ] || die "accounted for $SEEN of $EXTRACTED extracted blocks"

if [ "$FAILED" -ne 0 ]; then
  echo "RESULT: FAILED — at least one block exited nonzero" >>"$RECEIPT"
else
  echo "RESULT: OK — $EXTRACTED blocks accounted for, every executable one exited 0" >>"$RECEIPT"
fi

cat "$RECEIPT"
exit "$FAILED"
