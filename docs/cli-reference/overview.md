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

## Top-level options

`eq --version` prints the installed version (e.g. `evaluatorq 1.3.2`) and exits. Running `eq` with no arguments prints help and exits.

## Recipes

```bash
# CI smoke run — one strategy per category, no LLM-generated strategies, quiet
eq redteam run -t agent:my-agent --max-per-category 1 --no-generate-strategies -q

# Save full per-stage artifacts to a directory
eq redteam run -t agent:my-agent --save detail --artifacts-dir ./runs

# Quick simulation — two personas, two scenarios
eq sim run -t agent:my-agent --num-personas 2 --num-scenarios 2
```

## Where to next

- **[Agent Simulation](../guides/agent-simulation.md)** — the `eq sim` workflow in depth.
- **[Red Teaming](../guides/red-teaming.md)** — the `eq redteam` workflow in depth.
- **[Getting Started](../guides/getting-started.md)** — run your first evaluation end-to-end.
