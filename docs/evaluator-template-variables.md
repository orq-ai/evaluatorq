# Evaluator Template Variables

`llm_jury()` and `llm_jury_pairwise()` render their prompt templates through a
Mustache-style substitution (`{{name}}`, dotted paths supported). Both the
built-in default templates and any `prompt=`/`criteria=` override you supply
draw from the same fixed namespace. The `input.*`, `output.*`, and `log.*` families
come from `evaluatorq.common.judge._build_namespace`, the single builder shared by
both jury types; `criteria` and `question` are layered on top by `llm_jury()` /
`llm_jury_pairwise()` themselves.

## Pointwise (`llm_jury`)

| Variable | Meaning | Example |
| --- | --- | --- |
| `{{input.all_messages}}` | Full input message list, as JSON. | `[{"role": "user", "content": "What is the capital of France?"}]` |
| `{{input.expected_output}}` | Expected/reference output text, or empty string if none. | `Paris` |
| `{{input.system_instructions}}` | System instructions passed to the judge, or empty string if none. | _empty_ — pointwise jury does not set this |
| `{{output.response}}` | The assistant's text response being judged. | `The capital of France is Paris.` |
| `{{output.tools_called}}` | Tool calls made while producing the output (name/arguments/result/id), as JSON. | `[{"name": "search", "arguments": {...}, "result": "...", "id": "call_1"}]` |
| `{{output.messages}}` | Structured output transcript (text, reasoning, and tool-call turns), as JSON. | `[{"role": "assistant", "content": "Paris"}]` |
| `{{output.error}}` | Error message when the agent errored, else empty string. | _empty_ on success |
| `{{log.input}}` | Content of the last input message. | `What is the capital of France?` |
| `{{log.output}}` | Same value as `{{output.response}}`. | `The capital of France is Paris.` |
| `{{log.reference}}` | Same value as `{{input.expected_output}}`. | `Paris` |
| `{{log.expected_output}}` | Also the same value as `{{input.expected_output}}` (alias of `{{log.reference}}`). | `Paris` |
| `{{log.messages}}` | Full input message list, as JSON (same content as `{{input.all_messages}}`). | `[{"role": "user", "content": "..."}]` |
| `{{criteria}}` | The evaluation criteria passed to `llm_jury(criteria=...)`. | `The answer is factually correct.` |

The default pointwise template uses `{{criteria}}`, `{{input.all_messages}}`,
`{{output.response}}`, and `{{input.expected_output}}`. Pass `prompt=` to
`llm_jury()` to use a different subset of the table above.

## Pairwise (`llm_jury_pairwise`)

| Variable | Meaning | Example |
| --- | --- | --- |
| `{{question}}` | The question/prompt both responses are answering. | `What is the capital of France?` |
| `{{criteria}}` | The comparison criteria (`criteria=...`, or the built-in default). | `Compare on accuracy, helpfulness, clarity...` |
| `{{response_a.*}}` | Mirrors the full pointwise `input.*`/`output.*`/`log.*` namespace above for side A — e.g. `{{response_a.output.response}}`, `{{response_a.input.all_messages}}`, `{{response_a.output.error}}`. | `{{response_a.output.response}}` → `Paris is the capital of France.` |
| `{{response_b.*}}` | Same mirror as `response_a.*`, for side B. | `{{response_b.output.response}}` → `Paris.` |

A pairwise side carries an answer only, so `response_{a,b}.input.*` carries no data —
`all_messages` renders as `[]` and the other input fields render blank. Only
`response_{a,b}.output.*` is populated.

The default pairwise template uses `{{criteria}}`, `{{question}}`,
`{{response_a.output.response}}`, and `{{response_b.output.response}}`. Pass `prompt=`
to `llm_jury_pairwise()` to override it with any subset of the table above — that is
the only way to reach `output.tools_called` / `output.messages` per side.

## Migration note

!!! warning "Breaking change to custom prompts"

    Bare `{{input}}`, `{{output}}`, `{{log}}`, `{{response_a}}`, and `{{response_b}}`
    have been **removed**. They now render as intact literal text (e.g. the string
    `{{input}}` reaches the judge verbatim) instead of substituting anything, since a
    bare object has no single sensible string form. A warning is logged whenever one is
    left unresolved.

    If you passed a custom `prompt=` using any of them, switch to the dotted paths above
    — for example `{{input.all_messages}}` in place of `{{input}}`, and
    `{{response_a.output.response}}` in place of `{{response_a}}`.

    This shipped in a minor release rather than a major one: the jury template namespace
    is a very recent feature with no known external users of the bare form.

`llm_jury_pairwise()` also now accepts a `prompt=` override, mirroring
`llm_jury(prompt=...)`, to replace the built-in pairwise template entirely.

## Errored targets are not judged

When a target returns an `AgentResponse` carrying an error, neither jury calls its
judges: `llm_jury()` returns an `inconclusive` result with `pass` unset and the error in
the explanation, and `llm_jury_pairwise()` returns `winner='inconclusive'`. Grading an
errored generation would otherwise score "the agent said nothing" as if the agent had
genuinely answered that way. `{{output.error}}` remains available in the namespace for
custom prompts that want to inspect it.

## Where to next

- **[LLM as a Jury](llm-as-a-jury.md)** — panel configuration and verdict modes for `llm_jury()`.
- **[Pairwise Judging](pairwise-judging.md)** — `llm_jury_pairwise()` usage and reading comparison results.
- **[Custom Evaluators & Frameworks](custom-evaluators-and-frameworks.md)** — the equivalent placeholders for Orq-format evaluator prompts.
