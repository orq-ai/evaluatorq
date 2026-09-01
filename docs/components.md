# Components

evaluatorq is four things in one package: an evaluation runner, a red teamer, an agent simulator, and a set of LLM judges the first three share. They answer different questions, and picking the wrong one is the most expensive mistake available at the start.

Read this page once, before your first run. It says which component answers your question and what each one needs from you. How to use the one you picked is in its own guide.

## Pick one

| You have | You want to know | Use | Import from | CLI |
|---|---|---|---|---|
| Rows of input, and a notion of a good answer | Does my system get them right, and did the last change help? | `evaluatorq()` | `evaluatorq` | Python only |
| An agent, no test data | How does it hold up over a real multi-turn conversation? | `simulate()`, `generate_and_simulate()` | `evaluatorq.simulation` | `eq sim` |
| An agent you are about to expose to users | How does someone break it? | `red_team()` | `evaluatorq.redteam` | `eq redteam` |
| Outputs and a quality question no string match answers | Is this answer good — and is one answer better than another? | `llm_jury()`, `run_pairwise()` | `evaluatorq` | Python only |

Only the first and last are top-level exports. `from evaluatorq import red_team` raises `ImportError` — red teaming and simulation live in subpackages of the same base install. What some of them need on top of it is in [Installation](installation.md).

A **datapoint** is one row: inputs plus, usually, an `expected_output`. A **target** is the live system under test. A **judge** is an LLM scoring an output against a rubric; several judges voting are a **jury**.

## Evaluation — `evaluatorq()`

The core loop. You bring datapoints and one or more **jobs** — async functions that turn a datapoint into an output — and evaluatorq runs every job over every datapoint in parallel, then applies each evaluator to the results. Two jobs become two columns, which is how you compare a change against what you had before.

It needs data you already have, in one of the three shapes `data=` accepts: rows written inline, an Orq dataset via `DatasetIdInput`, or a past experiment's recorded responses via `ExperimentInput` — see [Data sources](evaluation-reference.md#data-sources). Production traces are none of the three until you turn them into one: collect the rows into an Orq dataset, or read them into `DataPoint`s yourself. Simulation is the surface with a direct trace path, `eq sim from-traces`. If you have no data at all, that is what the other two components are for.

```python
from evaluatorq import DataPoint, evaluatorq, job, string_contains_evaluator
```

It is the one component that needs no API key: a local job scored by a local evaluator runs on the base install alone, as the [runnable example on the Installation page](installation.md#what-the-base-install-already-does) does. Swap either for a model call and you need a key — see [Configuration](configuration.md).

→ [Getting Started](guides/getting-started.md), [Evaluation reference](evaluation-reference.md)

## Simulation — `simulate()`

A user-simulator LLM talks to your agent for several turns, driven by a **persona** (who is talking) and a **scenario** (what they want), while a judge decides per turn whether the goal was met or a rule was broken. `generate_and_simulate()` writes the personas and scenarios for you from a description of the agent, so the missing test data stops being a blocker.

```python
from evaluatorq.simulation import generate_and_simulate, simulate
```

It needs a target and a key. Both the user simulator and the judge are model calls of their own, so a conversation costs more than the turns your agent answers.

Reach for it when the failure you care about only appears over multiple turns — the agent that forgets the constraint stated three turns ago, or re-asks a question it already has the answer to. A single-turn evaluation cannot see either.

→ [Agent simulation](guides/agent-simulation.md)

## Red teaming — `red_team()`

Adversarial attacks against a live target, scored per attempt as resistant or vulnerable and mapped to the OWASP LLM Top 10 and the Agentic Security Initiative categories. The dynamic pipeline writes its own attacks against the target's discovered tools and memory, the static pipeline replays a fixed attack dataset, and hybrid does both.

```python
from evaluatorq.redteam import red_team
```

It needs a target and a key, and it is the most expensive of the four: every attack costs a call to your target and a call to judge the answer, plus one to write the attack when the dynamic pipeline generates it — multiplied across the vulnerability categories you select.

It answers a different question from the other two, and the inversion catches people out: a run is not a score, it is a list of the ways your agent broke. `passed=True` means the attack failed and the agent held.

→ [Red teaming](guides/red-teaming.md), [Vulnerabilities and frameworks](guides/vulnerabilities-and-frameworks.md)

## Judges and juries — `llm_jury()`, `run_pairwise()`

```python
from evaluatorq import llm_jury, run_pairwise
```

Not a runner, and every verdict is a model call of its own — a three-judge panel triples the judging cost of a row. `llm_jury()` builds an evaluator you drop into the `evaluators=[...]` list of `evaluatorq()`, backed by one judge or a panel of several. `run_pairwise()` answers the comparative question instead — given two outputs for the same input, which is better — because models are more reliable at ranking two answers than at scoring one in isolation.

→ [LLM as a jury](llm-as-a-jury.md), [Pairwise judging](pairwise-judging.md)

## How they fit together

- **The target is shared.** An Orq agent, an Orq deployment, a LangGraph or CrewAI agent, or a plain async function is the system under test for both red teaming and simulation, described once. See [Targets](guides/targets.md).
- **A jury plugs into evaluation and red teaming.** The same panel machinery scores an `evaluatorq()` row and an attack response.
- **Simulation feeds evaluation.** `wrap_simulation_agent()` turns a simulation into an `evaluatorq()` job, so a whole multi-turn conversation becomes one scored row next to everything else you measure. See [Simulation in an evaluatorq run](guides/simulation-in-evaluatorq.md).
- **Red teaming and simulation both write runs to disk.** `eq dashboard` reads both stores and shows them side by side. `evaluatorq()` writes nothing there — its results come back in the returned object, and upload to Orq as an Experiment whenever `ORQ_API_KEY` is set. There is no flag to keep a run local while the key is in the environment; unset it for that run instead.

## Python or CLI

`eq redteam` and `eq sim` cover red teaming and simulation end to end — running, listing runs, exporting reports — so a scheduled job or a CI step needs no Python file. Both take a target as a string, `agent:<key>` or `deployment:<key>`, where the key is the one your agent or deployment carries in Orq and `ORQ_API_KEY` is what resolves it — see [Targets](guides/targets.md). Not every combination runs: a red team against a `deployment:` target needs `--mode static`, because the dynamic and hybrid pipelines drive a multi-turn conversation and refuse a deployment with an error rather than attacking the wrong thing. And the string form is why a custom target written in Python cannot be reached from the CLI at all.

Core evaluation has no CLI command. `evaluatorq()`, `llm_jury()` and `run_pairwise()` take your own functions as arguments, and there is no way to pass a function on a command line — so evaluation is a Python script, always.

## Where to next

- [Configuration](configuration.md) — the keys each component reads, and the two-step setup.
- [Getting Started](guides/getting-started.md) — a first evaluation, end to end.
