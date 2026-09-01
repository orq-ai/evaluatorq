# Installation

evaluatorq installs as one package. The base install runs evaluations, red teaming and simulation from Python; each extra adds one thing on top — a dataset loader, the charts in a report, a viewer, a framework adapter.

Read this page when you install for the first time, or when a command tells you an extra is missing. Keys, endpoints and run stores are [Configuration](configuration.md).

## Install

```bash
uv add evaluatorq
```

`uv add` installs into the current project — run `uv init` first if you do not have one. With pip, use `python -m pip install evaluatorq`, which installs into the interpreter you just named rather than whichever `pip` happens to be first on your `PATH`.

evaluatorq needs Python 3.10 or newer. Check what landed:

```bash
uv run eq --version
```

## What the base install already does

No extras, no API key, no account. This scores one row with a built-in evaluator and prints a results table:

```python
import asyncio

from evaluatorq import DataPoint, evaluatorq, job, string_contains_evaluator


@job("uppercase")
async def uppercase(data: DataPoint, _row: int) -> str:
    return str(data.inputs["text"]).upper()


async def main() -> None:
    await evaluatorq(
        "install-check",
        data=[DataPoint(inputs={"text": "hello"}, expected_output="HELLO")],
        jobs=[uppercase],
        evaluators=[string_contains_evaluator()],
    )


if __name__ == "__main__":
    asyncio.run(main())
```

```bash
uv run install_check.py
```

`red_team()` and `simulate()` also import and run from the base install. What the extras add is around them: the datasets they read, the reports they render, and the viewers you browse the results in.

## Extras

```bash
uv add "evaluatorq[redteam]"
uv add "evaluatorq[redteam,dashboard]"     # combine in one install
uv add "evaluatorq[all]"                   # every extra below
```

| Extra | Adds | Add it when |
|---|---|---|
| `redteam` | `huggingface-hub` for dataset loading, chart rendering, the retired Streamlit viewer | You run static or hybrid red teaming, which reads an attack dataset |
| `simulation` | Chart rendering and the retired Streamlit viewer | You want charts in a simulation report |
| `dashboard` | `python-fasthtml` and `uvicorn` | You run `eq dashboard` to browse saved runs |
| `otel` | The OpenTelemetry SDK and its OTLP exporter | You want [traces](tracing.md) — without it initialisation is skipped silently and no span is ever exported |
| `langchain`, `langgraph`, `openai-agents`, `pydantic-ai`, `crewai` | The framework itself | Your agent under test is built on that framework. See [Framework integrations](framework-integrations.md) |
| `orq` | Nothing — `orq-ai-sdk` is already a base dependency | Never needed; it exists so `evaluatorq[orq]` does not fail |
| `all` | Every extra in this table | You are exploring and would rather not decide yet |

`crewai` installs only on Python 3.11 and newer; on 3.10 the extra resolves to nothing.

## When an extra is missing

Most missing extras announce themselves at the point of use, and the message names the install command:

```console
$ eq dashboard
The dashboard requires "dashboard" extra. Install with: uv add "evaluatorq[dashboard]" (or: python -m pip install "evaluatorq[dashboard]")
```

One is quieter. Without `vl-convert-python` — part of both the `redteam` and `simulation` extras — an HTML report still builds and still opens, with every chart omitted and the tables left in place. The only signal is a single log line, `vl-convert-python not installed; charts omitted from reports.`, printed once per process and easy to lose in a long run. The report itself says nothing, so a chartless report means checking your install rather than concluding there was no data.

## Where to next

- [Components](components.md) — which of evaluatorq's surfaces answers your question.
- [Configuration](configuration.md) — the keys and environment variables each surface reads.
- [Getting Started](guides/getting-started.md) — a first evaluation, end to end.
