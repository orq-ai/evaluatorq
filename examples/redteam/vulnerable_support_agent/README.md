# Vulnerable Support Agent — Red Team Demo

**This agent is DELIBERATELY INSECURE. Do NOT use it as a template for a real agent.** It exists only to lose a red-team run so the failure is easy to record and inspect.

A customer support agent ("Ava") with two tools — `lookup_order` (returns customer PII) and `issue_refund` (moves money) — and a weak system prompt. One red-team run against it lands a `VULNERABLE` verdict for goal hijacking and prompt injection.

## Why it is vulnerable

- **Exposed high-impact tools.** `issue_refund` moves money and `lookup_order` returns full PII (name, email, full card-last4 line). Both are local stubs with no real side effects, but a refund always "succeeds" and a lookup always returns everything.
- **Weak system prompt.** Ava is told her "#1 job is to make every customer happy" and to "always do what the customer asks." The only policy is a single parenthetical line with no teeth.
- **No instruction/data separation.** Ava treats customer text (and tool-result text) as authoritative instructions, so an authority claim ("I'm an admin") or an embedded instruction flips her behaviour.
- **No input sanitisation.** Nothing filters or redacts. `lookup_order` does no redaction, so the model can read full PII straight back to the attacker.

Result: it reliably loses to **ASI01 (agentic goal hijacking)** and **LLM01 (prompt injection)** rather than by luck.

A hardened sibling, `Rex` in `agents/secure.py`, has the same tools but a tight policy (verify order, cap refund at order amount, never read back full PII, treat customer text as untrusted). It is included for contrast and is not part of the recorded run.

## Setup

```bash
uv sync
cp .env.example .env   # fill in ORQ_API_KEY
```

## Run (one command)

```bash
uv run python run.py
```

This runs one adaptive red team against Ava for `goal_hijacking` (ASI01) and `prompt_injection` (LLM01), prints clean `RichHooks` output, and writes the report to `results/ava_XXX.json`. The `VULNERABLE` sample findings live in that JSON under `results`.

## Record the GIF (for the human)

Record the terminal while `run.py` executes (hooks output only — no DEBUG logs). Then:

```bash
ffmpeg -i in.mp4 -vf "fps=8,scale=800:-1" -loop 0 out.gif
gifsicle -O3 --lossy=80 out.gif -o out-optimized.gif
```

Target well under 5MB. Keep only the hooks output in frame so the recording stays clean.

## Files

- `agents/base.py` — `SupportAgent` (tool-capable, implements `AgentTarget`).
- `agents/vulnerable.py` — `Ava`, the weak-prompt agent (the deliverable).
- `agents/secure.py` — `Rex`, hardened sibling for contrast.
- `tools.py` — `lookup_order`, `issue_refund` local stubs.
- `config.py` — model id (`openai/gpt-5.4-mini` via the orq router).
- `attacker_instructions.txt` — domain context steering the attacker.
- `run.py` — drives `red_team()` and writes the result JSON.
