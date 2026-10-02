# ATIF fixture sources

- `harbor_linear_history_v18.json`, `harbor_invalid_json_v18.json`: https://github.com/harbor-framework/harbor, `tests/golden/terminus_2/`, Apache-2.0.
- `harbor_atif2otel_pass_v17.json`: https://github.com/harbor-framework/harbor, `packages/harbor-atif2otel/tests/fixtures/trajectory_pass.json`, Apache-2.0.
- `phoenix_v17_embedded_subagents.json`: https://github.com/Arize-ai/phoenix, `packages/phoenix-client` atif fixtures, Apache-2.0 (the client package only; the Phoenix repository root is Elastic License 2.0 and nothing else from it is vendored).

Files older than ATIF-v1.7 are deliberately not vendored (the reader rejects them).
