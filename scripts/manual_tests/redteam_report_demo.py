#!/usr/bin/env python3
"""Generate a red-team HTML report from the deterministic static pipeline.

No API key needed — uses the in-repo deterministic OpenAI-compatible client
and the static E2E fixture dataset, then renders the report via export_html.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import cast

# Reuse the deterministic client + target resolver from the E2E smoke test.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from redteam_static_e2e import DeterministicAsyncOpenAI, _resolve_target  # noqa: E402

from evaluatorq.redteam import red_team  # noqa: E402
from evaluatorq.redteam.contracts import LLMCallConfig, LLMConfig  # noqa: E402
from evaluatorq.redteam.reports import export_html  # noqa: E402


async def main() -> int:
    package_root = Path(__file__).resolve().parent.parent.parent
    dataset = package_root / "tests/redteam/fixtures/static_e2e_dataset.json"

    from openai import AsyncOpenAI

    client = cast(AsyncOpenAI, cast(object, DeterministicAsyncOpenAI()))
    target = _resolve_target("demo-model", client=client)

    report = await red_team(
        target,
        mode="static",
        llm_config=LLMConfig(evaluator=LLMCallConfig(model="demo-evaluator")),
        parallelism=4,
        dataset=str(dataset),
        llm_client=client,
        description="Red-team HTML report demo (deterministic static pipeline)",
    )

    if export_html is None:
        print("export_html unavailable (missing optional dependency)", file=sys.stderr)
        return 1

    html = export_html(report)
    out = Path(__file__).resolve().parent / "redteam_report.html"
    out.write_text(html, encoding="utf-8")
    print(
        f"Wrote {out} "
        f"({report.total_results} attacks, "
        f"{report.summary.vulnerabilities_found} vulnerabilities, "
        f"ASR={report.summary.vulnerability_rate:.0%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
