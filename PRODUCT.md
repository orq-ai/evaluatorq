# Product

## Register

product

## Users

ML/AI engineers and safety researchers who run `evaluatorq` red-team and agent-simulation jobs from the CLI, then open the FastHTML dashboard to inspect the results. They arrive after a run finishes, usually to answer one question: *did my agent hold up, and where did it break?* They are technically fluent, work in dense information, and drill from a summary into individual conversation transcripts.

## Product Purpose

A local, zero-config results dashboard for two evaluation surfaces:
- **Agent simulation** — persona × scenario conversations scored for goal completion.
- **Red team** — OWASP LLM/ASI attacks scored pass (resistant) / fail (vulnerable).

Success = a researcher lands on a report and, within seconds, sees the headline outcome, the shape of the failures (which categories/personas/agents), and can open any single conversation to read what actually happened. The dashboard reports; it does not run jobs.

## Brand Personality

Precise, calm, trustworthy. An instrument, not a marketing surface. Three words: *legible, dense, honest.* Numbers are the hero; chrome recedes. `passed=True` means the attack was resisted — the UI must never let that polarity confuse the reader.

## Anti-references

- Generic SaaS-cream dashboard with hero-metric cards and gradient accents.
- Streamlit-default look (uncomposed widgets stacked on a page).
- Anything that decorates: side-stripe callouts, glassmorphism, gratuitous motion.

## Design Principles

1. **Numbers are the hero.** Chrome, borders, and color recede so data reads first.
2. **Outcome polarity is sacred.** Resistant/vulnerable and goal/no-goal must be unmistakable at a glance and consistent across every surface.
3. **Summary → shape → transcript.** Every report supports the same drill path.
4. **Earned familiarity.** Standard table/tab/filter affordances; the tool disappears.
5. **Honest visualization.** No chart that distorts scale or hides sample size.

## Accessibility & Inclusion

WCAG AA: body text ≥4.5:1, large text ≥3:1. Pass/fail state must never rely on color alone (pair with label/icon). Respect `prefers-reduced-motion`.
