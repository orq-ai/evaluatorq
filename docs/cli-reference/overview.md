# CLI Reference

Both `evaluatorq` and `eq` are aliases for the same entry point:

```toml
# pyproject.toml [project.scripts]
evaluatorq = "evaluatorq.cli:main"
eq         = "evaluatorq.cli:main"
```

Subcommands are registered at startup. `eq redteam` requires the `redteam` extra; `eq sim` requires the `simulation` extra.

Two command groups have their own pages:

- **[Red Teaming](redteam.md)** — adversarial testing (`eq redteam`).
- **[Simulation](simulation.md)** — multi-turn user simulation (`eq sim`; `sim` is shorthand).

---

## Top-level options

`eq --version` prints the installed version (e.g. `evaluatorq 1.3.2`) and exits. Running `eq` with no arguments prints help and exits.

---

## `eq ui`

Launch the FastHTML dashboard showing all red team **and** simulation runs.

```bash
eq ui [PATH] [--host HOST] [--port PORT]
```

`PATH` is optional. Omit it to scan both default run stores (`.evaluatorq/runs/` and `.evaluatorq/sim-runs/`). Pass a directory to restrict the scan; pass a file to scope to that file's parent directory and print a direct report URL.

| Flag / Argument | Type / Default | Description |
|---|---|---|
| `PATH` | `Path \| None` / `None` | Optional path to scan. |
| `--host` | `str` / `127.0.0.1` | Host to bind the dashboard server to. |
| `--port` | `int` / `8080` | Port for the dashboard server. |

Requires `evaluatorq[dashboard]` (`python-fasthtml`, `uvicorn`).

---

## Recipes

```bash
# CI smoke run — one strategy per category, no LLM-generated strategies, quiet
eq redteam run -t agent:my-agent --max-per-category 1 --no-generate-strategies -q

# Save full per-stage artifacts to a directory
eq redteam run -t agent:my-agent --save detail --output-dir ./runs

# Quick simulation — two personas, two scenarios
eq sim run -t agent:my-agent --num-personas 2 --num-scenarios 2
```
