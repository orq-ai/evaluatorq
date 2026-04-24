"""Pipeline hook system for the evaluatorq red teaming runner.

Provides a ``PipelineHooks`` protocol that separates rendering concerns from
pipeline logic.  Hook implementations are injected via ``red_team(hooks=...)``.

Public API:
    PipelineHooks  — runtime-checkable Protocol
    ConfirmPayload — TypedDict for on_confirm payload
    DefaultHooks   — loguru-based implementation (library callers)
    RichHooks      — Rich terminal implementation (CLI)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

from loguru import logger

from evaluatorq.redteam.contracts import PipelineStage
from evaluatorq.redteam.reports.display import print_report_summary

if TYPE_CHECKING:
    from rich.console import Console as RichConsole

    from evaluatorq.redteam.contracts import RedTeamReport

# ---------------------------------------------------------------------------
# Stage label mapping
# ---------------------------------------------------------------------------

_STAGE_LABELS: dict[str, str] = {
    PipelineStage.CONTEXT_RETRIEVAL: 'Retrieving Agent Context',
    PipelineStage.DATAPOINT_GENERATION: 'Generating Attack Datapoints',
    PipelineStage.ATTACK_EXECUTION: 'Executing Attacks',
    PipelineStage.REPORT_GENERATION: 'Generating Report',
    PipelineStage.CLEANUP: 'Cleaning Up',
    PipelineStage.TARGET_START: 'Starting Target',
    PipelineStage.TARGET_COMPLETE: 'Target Complete',
}

# ---------------------------------------------------------------------------
# ConfirmPayload TypedDict
# ---------------------------------------------------------------------------


class ConfirmPayload(TypedDict, total=False):
    """Payload passed to ``on_confirm`` before pipeline execution begins."""

    agent_context: dict[str, Any] | None
    """Serialized agent context for the first target (None for static mode)."""

    agent_contexts: dict[str, dict[str, Any]] | None
    """Per-target agent contexts keyed by target string (multi-target runs)."""

    num_datapoints: int
    """Total number of attack datapoints to be executed."""

    num_dynamic: int | None
    """Number of dynamic datapoints (hybrid mode only)."""

    num_static: int | None
    """Number of static datapoints (hybrid mode only)."""

    categories: list[str]
    """OWASP categories being tested."""

    attack_model: str
    """Model used for adversarial prompt generation."""

    evaluator_model: str
    """Model used for OWASP evaluation scoring."""

    max_turns: int
    """Maximum conversation turns per attack."""

    parallelism: int
    """Maximum concurrent evaluatorq jobs."""

    filtering_metadata: dict[str, Any] | None
    """Strategy filtering metadata from datapoint generation."""

    strategy_breakdown: dict[str, Any] | None
    """Per-category breakdown of template/generated/filtered strategies for the confirm table."""

    mode: str
    """Execution mode: 'dynamic', 'static', or 'hybrid'."""

    target: str
    """Target identifier string."""

    dataset_path: str | None
    """Path to static dataset (static/hybrid modes)."""

    vulnerabilities: list[str] | None
    """Vulnerability labels loaded from dataset (static mode)."""


# ---------------------------------------------------------------------------
# PipelineHooks Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PipelineHooks(Protocol):
    """Protocol for pipeline lifecycle hooks.

    Implementations are injected into ``red_team()`` via ``hooks=...``.
    All methods are synchronous.  If a hook raises, the pipeline breaks.
    """

    def on_stage_start(self, stage: PipelineStage | str, meta: dict[str, Any]) -> None:
        """Called when a pipeline stage begins.

        Args:
            stage: Stage identifier (e.g. ``PipelineStage.CONTEXT_RETRIEVAL``).
            meta:  Stage-specific metadata dict.
        """
        ...

    def on_stage_end(self, stage: PipelineStage | str, meta: dict[str, Any]) -> None:
        """Called when a pipeline stage completes.

        Args:
            stage: Stage identifier matching the corresponding ``on_stage_start``.
            meta:  Stage-specific result metadata.
        """
        ...

    def on_confirm(self, payload: ConfirmPayload) -> bool:
        """Called before execution begins to confirm the run plan.

        Args:
            payload: Summary of the planned run.

        Returns:
            ``True`` to proceed, ``False`` to cancel (raises ``CancelledError``).
        """
        ...

    def on_complete(self, report: RedTeamReport, *, output_dir: str | None = None, auto_save_path: str | None = None) -> None:
        """Called once with the final merged report after all targets complete.

        Args:
            report: The final ``RedTeamReport``.
            output_dir: Directory where the report JSON was saved (if any).
        """
        ...


# ---------------------------------------------------------------------------
# DefaultHooks — loguru output
# ---------------------------------------------------------------------------


class DefaultHooks:
    """Default hook implementation that logs via loguru.

    Used by library callers who do not supply a ``hooks`` argument to
    ``red_team()``.  ``on_confirm`` always returns ``True`` (no interactive
    prompt).
    """

    def on_stage_start(self, stage: PipelineStage | str, meta: dict[str, Any]) -> None:
        label = _STAGE_LABELS.get(stage, str(stage))
        logger.info(f"[redteam] Stage started: {stage} — {label} | meta={meta}")

    def on_stage_end(self, stage: PipelineStage | str, meta: dict[str, Any]) -> None:
        label = _STAGE_LABELS.get(stage, str(stage))
        logger.info(f"[redteam] Stage complete: {stage} — {label} | meta={meta}")

    def on_confirm(self, payload: ConfirmPayload) -> bool:
        """Log plan details and always approve."""
        num_dp = payload.get("num_datapoints", "?")
        categories = payload.get("categories") or []
        attack_model = payload.get("attack_model", "?")
        evaluator_model = payload.get("evaluator_model", "?")
        target = payload.get("target", "?")
        mode = payload.get("mode", "?")
        logger.info(
            f"[redteam] Run plan: target={target!r} mode={mode!r} "
            f"datapoints={num_dp} categories={len(categories)} "
            f"attack_model={attack_model!r} evaluator_model={evaluator_model!r}"
        )
        return True

    def on_complete(self, report: RedTeamReport, *, output_dir: str | None = None, auto_save_path: str | None = None) -> None:
        """Log a brief summary and UI hint."""
        for warning in report.pipeline_warnings:
            logger.warning(f'[redteam] PIPELINE WARNING: {warning}')

        summary = report.summary
        logger.info(
            f"[redteam] Run complete — resistance_rate={summary.resistance_rate:.0%} "
            f"vulnerabilities={summary.vulnerabilities_found} "
            f"attacks={summary.total_attacks}"
        )
        if output_dir:
            try:
                report_path = str(Path(output_dir, "03_summary_report.json").relative_to(Path.cwd()))
            except ValueError:
                report_path = f"{output_dir}/03_summary_report.json"
            logger.info(
                f'[redteam] Tip: visualise results with  "evaluatorq redteam ui {report_path}"'
            )
        elif auto_save_path:
            try:
                report_path = str(Path(auto_save_path).relative_to(Path.cwd()))
            except ValueError:
                report_path = auto_save_path
            logger.info(
                f'[redteam] Tip: visualise results with  "evaluatorq redteam ui {report_path}"'
            )
        else:
            logger.info(
                '[redteam] Tip: visualise results with  "evaluatorq redteam ui <report.json>"'
            )


# ---------------------------------------------------------------------------
# RichHooks — Rich terminal output
# ---------------------------------------------------------------------------


class RichHooks:
    """Rich terminal hook implementation for the evaluatorq CLI.

    Renders stage banners, a detailed confirmation table, and delegates the
    final report summary to :func:`~evaluatorq.redteam.reports.display.print_report_summary`.

    Args:
        console:      A :class:`rich.console.Console` instance.  A new one is
                      created when ``None`` is passed (default).
        skip_confirm: When ``True``, renders the plan but skips the interactive
                      ``typer.confirm`` prompt and returns ``True`` automatically.
    """

    def __init__(self, *, console: RichConsole | None = None, skip_confirm: bool = False) -> None:
        if console is None:
            from rich.console import Console

            console = Console()
        self._console = console
        self._skip_confirm = skip_confirm

    # ------------------------------------------------------------------
    # on_stage_start
    # ------------------------------------------------------------------

    def on_stage_start(self, stage: PipelineStage | str, meta: dict[str, Any]) -> None:
        label = _STAGE_LABELS.get(stage, str(stage).replace('_', ' ').title())
        self._console.rule(f"[bold cyan]{label}[/bold cyan]")

    # ------------------------------------------------------------------
    # on_stage_end
    # ------------------------------------------------------------------

    def on_stage_end(self, stage: PipelineStage | str, meta: dict[str, Any]) -> None:
        if stage == PipelineStage.CONTEXT_RETRIEVAL:
            self._render_context_summary(meta)

        elapsed = meta.get("elapsed_s")
        if elapsed is not None:
            self._console.print(f"[dim]  completed in {elapsed:.1f}s[/dim]")

    def _render_context_summary(self, meta: dict[str, Any]) -> None:
        """Print a one-line per-target summary after agent context retrieval."""
        target = meta.get("target", "?")
        num_tools = meta.get("num_tools", 0)
        num_memory = meta.get("num_memory_stores", 0)
        num_kb = meta.get("num_knowledge_bases", 0)

        parts: list[str] = []
        parts.append(f"{num_tools} tool{'s' if num_tools != 1 else ''}")
        if num_memory:
            parts.append(f"{num_memory} memory store{'s' if num_memory != 1 else ''}")
        if num_kb:
            parts.append(f"{num_kb} knowledge base{'s' if num_kb != 1 else ''}")

        self._console.print(f"  [cyan]{target}[/cyan]: {', '.join(parts)}")

    # ------------------------------------------------------------------
    # on_confirm
    # ------------------------------------------------------------------

    def on_confirm(self, payload: ConfirmPayload) -> bool:
        """Render a detailed run plan and prompt for confirmation."""
        from rich import box
        from rich.table import Table

        # ── Run parameters table ────────────────────────────────────────
        table = Table(title="Run Plan", show_header=True, header_style="bold", box=box.ROUNDED)
        table.add_column("Parameter", style="white", min_width=20)
        table.add_column("Value", style="cyan")

        target = payload.get("target", "?")
        mode = payload.get("mode", "?")
        num_dp = payload.get("num_datapoints")
        categories = payload.get("categories") or []
        attack_model = payload.get("attack_model", "?")
        evaluator_model = payload.get("evaluator_model", "?")
        max_turns = payload.get("max_turns", "?")
        parallelism = payload.get("parallelism", "?")
        dataset_path = payload.get("dataset_path")
        num_dynamic = payload.get("num_dynamic")
        num_static = payload.get("num_static")

        table.add_row("Target", str(target))
        table.add_row("Mode", str(mode))

        # Compute target count from agent_contexts or comma-separated target string
        agent_contexts_raw = payload.get("agent_contexts") or {}
        num_targets = len(agent_contexts_raw) or 1

        if num_dp is not None:
            table.add_row("Datapoints", str(num_dp))
            if num_targets > 1:
                total_attacks = num_dp * num_targets
                table.add_row(
                    "Total Attacks",
                    f'{total_attacks} ({num_dp} datapoints × {num_targets} targets)',  # noqa: RUF001
                )

        table.add_row("Categories", str(len(categories)))
        table.add_row("Attack Model", str(attack_model))
        table.add_row("Evaluator Model", str(evaluator_model))
        table.add_row("Max Turns", str(max_turns))
        table.add_row("Parallelism", str(parallelism))

        if dataset_path:
            table.add_row("Dataset", str(dataset_path))

        self._console.print(table)

        # ── Agent capabilities panel(s) ────────────────────────────────
        agent_contexts = payload.get("agent_contexts")
        if agent_contexts and len(agent_contexts) > 1:
            for target_str, ctx in agent_contexts.items():
                self._render_agent_capabilities(ctx, title=f"Agent Capabilities — {target_str}")
        elif agent_contexts:
            ctx = next(iter(agent_contexts.values()))
            self._render_agent_capabilities(ctx)
        else:
            agent_context = payload.get("agent_context")
            if agent_context is not None:
                self._render_agent_capabilities(agent_context)

        # ── Filtering metadata ──────────────────────────────────────────
        filtering_metadata = payload.get("filtering_metadata")
        if filtering_metadata:
            self._render_filtering_metadata(filtering_metadata)

        # ── Strategy breakdown ─────────────────────────────────────────
        strategy_breakdown = payload.get("strategy_breakdown")
        if strategy_breakdown:
            self._render_strategy_breakdown(strategy_breakdown, num_dynamic=num_dynamic, num_static=num_static)

        # ── Prompt or auto-approve ──────────────────────────────────────
        if self._skip_confirm:
            return True

        import typer

        return typer.confirm("Proceed with this run?", default=True)

    def _render_agent_capabilities(self, agent_context: dict[str, Any], title: str = "Agent Capabilities") -> None:
        """Render an agent capabilities panel."""
        tools = agent_context.get("tools") or []
        memory_stores = agent_context.get("memory_stores") or []
        knowledge_bases = agent_context.get("knowledge_bases") or []

        lines: list[str] = []

        if tools:
            tool_names = ", ".join(
                t.get("name", str(t)) if isinstance(t, dict) else str(t) for t in tools[:10]
            )
            suffix = f" (+{len(tools) - 10} more)" if len(tools) > 10 else ""
            lines.append(f"Tools ({len(tools)}): {tool_names}{suffix}")
        else:
            lines.append("Tools: none")

        if memory_stores:
            store_ids = ", ".join(
                m.get("key") or m.get("id", str(m)) if isinstance(m, dict) else str(m)
                for m in memory_stores[:5]
            )
            lines.append(f"Memory Stores ({len(memory_stores)}): {store_ids}")
        else:
            lines.append("Memory: none")

        if knowledge_bases:
            kb_ids = ", ".join(
                kb.get("name") or kb.get("key") or kb.get("id", str(kb)) if isinstance(kb, dict) else str(kb)
                for kb in knowledge_bases[:5]
            )
            lines.append(f"Knowledge Bases ({len(knowledge_bases)}): {kb_ids}")
        else:
            lines.append("Knowledge Bases: none")

        self._console.print(
            self._make_panel("\n".join(lines), title=title)
        )

    def _render_filtering_metadata(self, filtering_metadata: dict[str, Any]) -> None:
        """Render strategy filtering summary per category."""
        from rich import box
        from rich.table import Table

        # filtering_metadata is keyed by category, each value is a dict with
        # count fields plus full strategy objects — only show the counts.
        _COUNT_KEYS = (
            "all_hardcoded_count",
            "applicable_count",
            "generated_count",
            "filtered_count",
            "total_selected",
        )

        table = Table(
            title="Dynamic Strategy Filtering",
            show_header=True,
            header_style="bold",
            box=box.ROUNDED,
        )
        table.add_column("Category", style="white")
        table.add_column("Hardcoded", style="cyan", justify="right")
        table.add_column("Applicable", style="cyan", justify="right")
        table.add_column("Generated", style="cyan", justify="right")
        table.add_column("Filtered", style="yellow", justify="right")
        table.add_column("Selected", style="bold green", justify="right")

        for category, info in filtering_metadata.items():
            if not isinstance(info, dict):
                continue
            table.add_row(
                category.upper(),
                str(info.get("all_hardcoded_count", "?")),
                str(info.get("applicable_count", "?")),
                str(info.get("generated_count", "?")),
                str(info.get("filtered_count", "?")),
                str(info.get("total_selected", "?")),
            )

        self._console.print(table)

    def _render_strategy_breakdown(
        self,
        strategy_breakdown: dict[str, Any],
        num_dynamic: int | None = None,
        num_static: int | None = None,
    ) -> None:
        """Render a per-category table showing template vs generated strategy counts."""
        from rich import box
        from rich.table import Table

        # Detect if any category has a 'capped' key (round-robin allocation was applied)
        has_cap = any(
            isinstance(info, dict) and 'capped' in info
            for info in strategy_breakdown.values()
        )

        table = Table(
            title="Datapoint Breakdown (per target)",
            show_header=True,
            header_style="bold",
            box=box.ROUNDED,
        )
        table.add_column("Category", style="white")
        table.add_column("Hardcoded", style="cyan", justify="right")
        table.add_column("Applicable", style="cyan", justify="right")
        table.add_column("Filtered", style="yellow", justify="right")
        table.add_column("Generated", style="cyan", justify="right")
        table.add_column("Selected", style="bold green", justify="right")
        if has_cap:
            table.add_column("Capped", style="bold yellow", justify="right")

        total_hardcoded = 0
        total_applicable = 0
        total_filtered = 0
        total_generated = 0
        total_selected = 0
        total_capped = 0

        for category, info in strategy_breakdown.items():
            if not isinstance(info, dict):
                continue
            hc = info.get('total_hardcoded', 0)
            ap = info.get('applicable', 0)
            fi = info.get('filtered', 0)
            ge = info.get('generated', 0)
            se = info.get('selected', 0)
            ca = info.get('capped', se)
            total_hardcoded += hc
            total_applicable += ap
            total_filtered += fi
            total_generated += ge
            total_selected += se
            total_capped += ca
            row = [
                category.upper(),
                str(hc),
                str(ap),
                str(fi) if fi else '-',
                str(ge) if ge else '-',
                str(se),
            ]
            if has_cap:
                row.append(str(ca))
            table.add_row(*row)

        table.add_section()
        dynamic_row = [
            '[bold]Dynamic total[/bold]',
            str(total_hardcoded),
            str(total_applicable),
            str(total_filtered) if total_filtered else '-',
            str(total_generated) if total_generated else '-',
            f'[bold]{total_selected}[/bold]',
        ]
        if has_cap:
            dynamic_row.append(f'[bold yellow]{total_capped}[/bold yellow]')
        table.add_row(*dynamic_row)

        if num_static:
            static_row = ['[bold]Static[/bold]', '', '', '', '', f'[bold]{num_static}[/bold]']
            if has_cap:
                static_row.append(f'[bold]{num_static}[/bold]')
            table.add_row(*static_row)

            final = total_capped + num_static if has_cap else total_selected + num_static
            table.add_section()
            total_row = ['[bold]Total datapoints[/bold]', '', '', '', '', '']
            if has_cap:
                total_row.append(f'[bold green]{final}[/bold green]')
            else:
                total_row[-1] = f'[bold green]{final}[/bold green]'
            table.add_row(*total_row)

        self._console.print(table)

    @staticmethod
    def _make_panel(content: str, title: str) -> Any:
        from rich.panel import Panel

        return Panel(content, title=title, expand=False)

    # ------------------------------------------------------------------
    # on_complete
    # ------------------------------------------------------------------

    def on_complete(self, report: RedTeamReport, *, output_dir: str | None = None, auto_save_path: str | None = None) -> None:
        """Render the full report summary and a UI hint."""
        from rich.panel import Panel

        if report.pipeline_warnings:
            warning_text = '\n'.join(f'[bold]WARNING:[/bold] {w}' for w in report.pipeline_warnings)
            self._console.print(
                Panel(warning_text, title='[bold red]Pipeline Warnings[/bold red]', border_style='red', expand=False)
            )

        print_report_summary(report, console=self._console)
        self._console.print()
        self._console.rule("[bold green]Run Complete[/bold green]")
        if output_dir:
            try:
                report_path = str(Path(output_dir, "03_summary_report.json").relative_to(Path.cwd()))
            except ValueError:
                report_path = f"{output_dir}/03_summary_report.json"
        elif auto_save_path:
            try:
                report_path = str(Path(auto_save_path).relative_to(Path.cwd()))
            except ValueError:
                report_path = auto_save_path
        else:
            report_path = None

        if report_path:
            self._console.print(
                f'[dim]Tip:[/dim] Visualise results interactively with '
                f'[bold cyan]"evaluatorq redteam ui {report_path}"[/bold cyan]'
            )
        else:
            self._console.print(
                '[dim]Tip:[/dim] Visualise results interactively with '
                '[bold cyan]"evaluatorq redteam ui <report.json>"[/bold cyan]'
            )
