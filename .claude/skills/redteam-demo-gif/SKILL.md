---
name: redteam-demo-gif
description: Record, re-record or re-render the red-team demo GIF (`docs/assets/redteam-demo.gif`) and refresh the README "Sample finding" that comes from the same run. Use it for any terminal recording, asciinema cast, agg or gifski work in this repo, and whenever someone says the demo GIF is choppy, colourless, cut off or out of date, even if they never name the file. It covers installing the tools when ffmpeg and a browser are missing, the colour and smoothness traps, and the README claims that must change with each new recording.
---

# redteam-demo-gif

The GIF is the real terminal output of one `run.py` run against the deliberately vulnerable support agent (`examples/redteam/vulnerable_support_agent/`), retimed so it streams, framed as a terminal window, and encoded at a fixed frame rate. The exact commands live in that example's README under "Re-recording"; that section is the single source of truth for flags. This skill holds what the README does not: getting the tools, the traps, and the claims that must move with a new recording.

## 1. Get the tools

Nothing is installed on a fresh box, so download the prebuilt release binaries into a scratch dir. Each tool ships a Linux musl or macOS build.

```bash
K=/tmp/gifkit && mkdir -p $K/fonts && cd $K
gh release download -R asciinema/asciinema -p 'asciinema-x86_64-unknown-linux-musl' -O asciinema
gh release download -R asciinema/agg -p 'agg-x86_64-unknown-linux-musl' -O agg
chmod +x asciinema agg
gh release download -R ImageOptim/gifski -p 'gifski-*.tar.xz' && tar -xJf gifski-*.tar.xz linux/gifski   # binary at $K/linux/gifski
gh release download -R JetBrains/JetBrainsMono -p 'JetBrainsMono-*.zip'
python3 -c "import zipfile,glob;z=zipfile.ZipFile(glob.glob('JetBrainsMono-*.zip')[0]);[open('fonts/'+n.split('/')[-1],'wb').write(z.read(n)) for n in z.namelist() if n.split('/')[-1] in ('JetBrainsMono-Regular.ttf','JetBrainsMono-Bold.ttf','JetBrainsMono-Italic.ttf','JetBrainsMono-BoldItalic.ttf')]"
```

Done when `asciinema --version`, `agg --version` and `linux/gifski --version` all print a version and `fonts/` holds the JetBrains Mono TTFs.

## 2. Record real runs

A live run needs `ORQ_API_KEY` and costs about $0.06. From an orq CLI profile, export it without printing it: `export ORQ_API_KEY="$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.orq/credentials.json')))['profiles']['<profile>']['api_key'])")"`.

Traps, each of which cost a take:

- **Colour.** Agent shells set `NO_COLOR=1` and `TERM=dumb`, and Rich then strips every colour. Record under `env -u NO_COLOR TERM=xterm-256color`. Check the cast has colour codes such as `\x1b[36m`, not only bold (`\x1b[1m`).
- **Width.** Record at a fixed `--window-size 100x32`. The RichHooks tables need about 100 columns; narrower wraps them into garbage.
- **Noise.** `LOGURU_LEVEL=ERROR` hides DEBUG lines. A line that survives it is a real warning from the run: fix its cause in `run.py` instead of cutting it from the cast.
- **Variance.** Runs are dynamic. Record several (they can run in parallel, since result files are reserved atomically) and read the `goal_hijacking: N vulnerable / 3 attacks` lines at the end of each cast. Pick a representative run, not the best one, and note the spread across all takes.

Done when you have chosen one cast and its matching `results/ava_XXX.json`, and you have the per-run counts of every take.

## 3. Render

Follow the README's "Re-recording" block: `recording/retime.py`, then agg, then `recording/frame.py`, then gifski. Run the Python scripts with `uv run --project <repo root> python` from the example dir; it has Pillow.

Why each step exists, so nobody "simplifies" it away:

- `retime.py`: Rich writes each table in one go, so a raw replay is a few frames held for seconds each, and that is what made the first GIF choppy. Retiming keeps the output verbatim.
- agg `--line-height 1.2`: at agg's default of 1.4, box-drawing characters leave gaps between rows.
- `frame.py` and then gifski: gifski needs a fixed frame rate and handles transparency, so the rounded corners stay transparent on light and dark pages. agg's own encoder cannot add the window frame.

Done when the GIF is under 5 MB, and a sampled mid-run frame and the last frame show colour, unbroken tables and the final per-category summary. Open them as images: composite the frames on white to check the corners.

## 4. Keep the claims true

A new recording changes facts stated in several places. Update every one:

- The GIF alt text in `README.md`, `docs/guides/red-teaming.md` and the example README: the goal-hijacking and prompt-injection counts.
- The example README: the counts in the intro, and the spread across takes.
- `examples/redteam/vulnerable_support_agent/config.py`: the comment on `TARGET_MODEL`.
- `sample_output/ava_XXX_findings.json`: regenerate it from the chosen run's report. Keep only the `VULNERABLE` results, and drop token usage and raw judge output.
- The "Sample finding" in `README.md`: quote the judge explanation and the report's recommendation verbatim from that file, and link the file.

Done when a search for the old run's counts and file name finds nothing stale.

## 5. Clean up and verify

- Delete `examples/redteam/vulnerable_support_agent/.venv` if `uv sync` created it. `scripts/check_examples.py` walks into it and reports hundreds of false failures.
- Delete `results/*.json` (it is gitignored, but stale files confuse the next take).
- Run `uv run python scripts/check_examples.py`, and run `ruff check` and `ruff format --check` with the example's `pyproject.toml` as config.
