# Dashboard (`eq dashboard`)

Launches the FastHTML dashboard, the primary UI for browsing saved runs. It is
the only report surface receiving new features; `eq redteam ui` and `eq sim ui`
are the retired Streamlit viewers, still registered but no longer documented.

For what the dashboard shows once it is open — the run index, the red-team and
simulation walkthroughs, and applying recommendations to an agent — see
[Dashboard](../dashboard.md). This page covers the command only.

```bash
eq dashboard [PATHS]... [OPTIONS]
```

| Flag / Argument | Type / Default | Description |
|---|---|---|
| `PATHS` | `Path` (repeatable) | Directories to scan for reports, or a single report file. Omit to scan both default stores. |
| `--host` | `str` / `127.0.0.1` | Host to bind the server to. |
| `--port` | `int` / `8080` | Port for the server. |

## What gets scanned

The argument is what decides which runs appear, and the three forms behave
differently:

- **No path** — scans both default stores together: `.evaluatorq/runs/` (red
  team) and `.evaluatorq/sim-runs/` (simulation). This is the usual invocation.
- **One or more directories** — scans exactly those, together. Useful for
  putting simulation runs from one repo next to red-team runs from another.
- **A file** — scans that file's *parent directory*, so sibling reports still
  resolve, and prints the direct URL for that one report. You land on the report
  rather than the index.

```bash
# every saved run, both stores
eq dashboard

# simulation runs only
eq dashboard .evaluatorq/sim-runs

# two stores from different checkouts, side by side
eq dashboard ~/work/api-agent/.evaluatorq/runs ~/work/support-bot/.evaluatorq/sim-runs

# open one report directly
eq dashboard .evaluatorq/runs/red-team-2026-08-18T09-14-02.json
```

## Binding

`--host` defaults to `127.0.0.1`, so the dashboard is reachable only from the
machine running it. Binding elsewhere exposes every saved run — transcripts,
system prompts, and judge verdicts — to anyone who can reach the port. There is
no authentication layer. Put it behind one before binding to `0.0.0.0`.
