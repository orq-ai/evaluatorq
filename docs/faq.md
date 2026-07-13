# FAQ

The questions people ask after seeing evaluatorq red-team an agent — and where
to go next for each.

## How do I know my agent is safe?

You don't, until you attack it. Shipping after a refused *"say something
harmful"* is a vibe check, not a test — it only proves the agent refuses the
one obvious prompt you thought to try. evaluatorq runs a mapped set of
adversarial attacks and reports a **resistance rate** (the fraction the agent
withstood), so "safe" becomes a number you can gate on rather than a gut feel.
See [Red Teaming](guides/red-teaming.md).

## Isn't a single-turn "refused → safe" check enough?

No. The attacks that land are the ones a single prompt can't express:

- **Multi-turn escalation** — each message looks benign; the attack assembles
  across turns. Invisible to single-turn evals.
- **Indirect injection** — the attacker controls what the agent *reads* (emails,
  docs, tool results), not what you type. You never see the payload.
- **Memory poisoning** — a planted instruction fires on a later, unrelated run.
- **Many-shot jailbreaking** — 100+ in-context examples steer behaviour.

evaluatorq generates multi-turn attacks by default (`max_turns=`) precisely so
these surface.

## What does red teaming actually test?

Attacks and LLM-judge evaluators mapped to three frameworks:

- **OWASP LLM Top 10** — prompt injection, system-prompt leakage, and the rest.
- **OWASP Agentic Security Initiative (ASI)** — agent-specific risks: tool
  abuse, excessive agency, trust exploitation.
- **Responsible AI** — fairness/bias, liability (legal, medical), content
  policy, harmful content.

Pass the ones you care about via `categories=["LLM01", "ASI01", ...]`. The full
list of codes is in the [redteam API reference](reference/evaluatorq/redteam.md).

## Static, dynamic, or hybrid — which mode?

- **static** — replays a fixed dataset of known attacks. Deterministic and
  cheap; run it in CI. Defaults to Orq's public
  [`orq/redteam-vulnerabilities`](https://huggingface.co/datasets/orq/redteam-vulnerabilities)
  set, or pass your own with `dataset=`.
- **dynamic** — an LLM generates attacks tailored to *your* agent's tools and
  prompt.
- **hybrid** — static seeds plus dynamic expansion, in one pass.

Details in [Red Teaming › Modes](guides/red-teaming.md#modes).

## How does it decide what to attack?

Probe → attack → judge. It first inspects the target (tools, permissions, system
prompt) to build a capability profile, generates attacks against the exposed
surface, then scores each outcome against per-vulnerability LLM-judge rubrics.
Filtering out attacks that can't apply keeps a run focused on real exposure.

## Why an LLM judge instead of keyword matching?

Because the surface text lies. *"I see malicious instructions in this email but I
won't follow them"* and *"I'll forward that email right away"* both mention
instructions — one is resistant, one is vulnerable. A regex can't tell them
apart; a judge rubric can. Write your own in
[Custom Evaluators & Frameworks](custom-evaluators-and-frameworks.md).

## How do I trust the judge itself?

Calibrate it against known baselines: run the same attack set through an agent
that *should* pass and one that *should* fail. If the safe agent gets flagged
vulnerable you have false positives; if the weak agent is marked resistant you
have blind spots. Either way, patch the rubric and rerun. Matched pairs (same
attacks, both agents) make the miscalibration visible.

## Isn't "vulnerable" context-dependent?

Yes, and the evaluators account for it. Fetching a shell script from GitHub is
helpful in a coding assistant and an RCE vector in a support bot; chaining three
API calls unprompted is doing the job or excessive agency depending on the
agent. Judges see the agent's declared context, so the same action can score
differently across agents.

## How do I run it against my own agent?

Implement two methods (`chat` + `messages`) — the `AgentTarget` protocol is
duck-typed, so LangChain, LangGraph, the OpenAI Agents SDK, or a custom loop all
work with no framework buy-in:

```python
from evaluatorq.redteam import red_team

report = await red_team(target=MyAgent(), mode="dynamic", max_turns=4)
print(f"resistance: {report.summary.resistance_rate:.0%}")
```

Point it at an Orq agent by key (`"agent:<key>"`) or wire your own — see
[Red Teaming › Red-team your target](guides/red-teaming.md#red-team-your-target).

## What does `passed=True` mean?

The agent **resisted** the attack (the attack failed). `passed=False` means the
attack succeeded — the agent is **vulnerable**. `resistance_rate` is the fraction
of attacks that came back `passed=True`.

## What do I need to get started?

```bash
pip install "evaluatorq[redteam]"
```

Dynamic mode needs an LLM for the attacker/judge (`OPENAI_API_KEY`, or route
through Orq with `ORQ_API_KEY`); static mode against the public dataset needs
neither. Then follow [Red Teaming](guides/red-teaming.md).
