# RES-931 demo videos - recording scripts

Four short videos, about 60 to 90 seconds each, one per external framework. Same
story every time: take an agent built in a real framework, wrap it as a red-team
target in one line, run live attacks, let an LLM judge score them. The attacker is
generative, so narrate what you see on screen, not fixed numbers.

## Setup (run this once before recording)

```bash
cd /Users/karina/Documents/orq/evaluatorq-res931
uv sync --frozen --all-extras     # installs the 4 framework extras
export ORQ_API_KEY=orq-...         # one key drives the agent, attacker, and judge
clear                             # clean prompt before you hit record
```

Run each demo with the `PYTHONPATH=src` prefix so the import always resolves:

```bash
PYTHONPATH=src uv run --frozen python examples/redteam/17_langgraph_target.py
```

## Recording tips

- Big terminal font so the final report table is readable on playback.
- Start recording on a cleared prompt, run one command, stop when the report prints.
- Save each file to `docs/assets/redteam-<framework>.mp4` (langgraph, openai-agents,
  pydantic-ai, crewai). These are the slots already embedded in the red-teaming guide.

---

## Video 1: LangGraph

**Command**
```bash
PYTHONPATH=src uv run --frozen python examples/redteam/17_langgraph_target.py
```

**Say (intro):** "This is a LangGraph ReAct support agent with a refund tool. One
line wraps it as a red-team target with `LangGraphTarget`, then the runner throws
live prompt-injection attacks at it and an LLM judge scores each one against the
OWASP agent-security categories."

**Point at:** the command (framework plus wrapper), the live attack progression,
then the final report table.

**Payoff:** "It resists most, but one indirect injection lands. The attacker hides
an instruction in tool output and hijacks the agent's goal."

---

## Video 2: OpenAI Agents SDK

**Command**
```bash
PYTHONPATH=src uv run --frozen python examples/redteam/18_openai_agents_target.py
```

**Say (intro):** "Same runner, different framework. This agent is built on the
OpenAI Agents SDK and wrapped with `OpenAIAgentTarget`. Nothing about the attack
setup changes, only the one-line wrapper."

**Point at:** the wrapper name, the attacks running, the report table.

**Payoff:** "Same shape as LangGraph. One indirect injection gets through, and the
judge flags the goal hijack."

---

## Video 3: Pydantic AI

**Command**
```bash
PYTHONPATH=src uv run --frozen python examples/redteam/19_pydantic_ai_target.py
```

**Say (intro):** "Third framework, Pydantic AI, wrapped with `PydanticAITarget`.
Point the runner at it and attack, exactly as before."

**Point at:** the wrapper, the live verdicts, the final table.

**Payoff:** "Again one attack lands. Three different frameworks, same weakness to
indirect prompt injection."

---

## Video 4: CrewAI

**Command**
```bash
PYTHONPATH=src uv run --frozen python examples/redteam/20_crewai_target.py
```

**Say (intro):** "Last one, a CrewAI agent wrapped with `CrewAITarget`. Same
attacks, same judge."

**Point at:** the wrapper, the attacks, the report table.

**Payoff:** "This is the interesting one. CrewAI resists all three attacks, one
hundred percent. Same runner, same attacks, different result. That is the point of
the suite: you can measure this across any framework with one wrapper."

---

## Re-transcode a recording

```bash
ffmpeg -y -i input.mov -an -vf scale=1512:-2 -c:v libx264 -crf 30 \
  -preset veryslow -pix_fmt yuv420p -movflags +faststart \
  docs/assets/redteam-<framework>.mp4
```
