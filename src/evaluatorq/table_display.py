"""Table display for evaluation results using Rich."""

import shutil
from collections import defaultdict
from typing import TypedDict

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .types import EvaluatorqResult

ScoreValue = float | bool | str
"""Score value type - can be numeric, boolean, or string"""

ScoresByEvaluatorAndJob = dict[str, dict[str, list[ScoreValue]]]
"""Mapping of evaluator_name -> job_name -> list of score values"""


class EvaluatorAverages(TypedDict):
    """Type for evaluator averages calculation result."""

    job_names: list[str]
    evaluator_names: list[str]
    averages: dict[str, dict[str, tuple[str, str]]]


def get_terminal_width() -> int:
    """Get terminal width with fallback"""
    return shutil.get_terminal_size(fallback=(80, 24)).columns


def create_summary_display(results: EvaluatorqResult) -> Table:
    """Create a summary table of evaluation results"""
    total_data_points = len(results)
    failed_data_points = 0
    total_jobs = 0
    failed_jobs = 0

    for r in results:
        if r.error:
            failed_data_points += 1

        if r.job_results:
            total_jobs += len(r.job_results)
            failed_jobs += sum(1 for job_result in r.job_results if job_result.error)

    success_rate = (
        round(((total_jobs - failed_jobs) / total_jobs) * 100) if total_jobs > 0 else 0
    )

    # Create summary table with borders
    table = Table(show_header=True, header_style="bold", box=box.ROUNDED)
    table.add_column("Metric", style="white", width=20)
    table.add_column("Value", width=15)

    table.add_row("Total Data Points", Text(str(total_data_points), style="cyan"))

    failed_dp_style = "red" if failed_data_points > 0 else "green"
    table.add_row(
        "Failed Data Points", Text(str(failed_data_points), style=failed_dp_style)
    )

    table.add_row("Total Jobs", Text(str(total_jobs), style="cyan"))

    failed_jobs_style = "red" if failed_jobs > 0 else "green"
    table.add_row("Failed Jobs", Text(str(failed_jobs), style=failed_jobs_style))

    # Success rate with color coding
    if failed_jobs == 0:
        rate_style = "green"
    elif success_rate >= 80:
        rate_style = "green"
    elif success_rate >= 50:
        rate_style = "yellow"
    else:
        rate_style = "red"

    table.add_row("Success Rate", Text(f"{success_rate}%", style=rate_style))

    return table


def calculate_evaluator_averages(
    results: EvaluatorqResult,
) -> EvaluatorAverages:
    """
    Calculate averages for evaluator scores across all data points.

    Returns:
        EvaluatorAverages with:
        - job_names: sorted list of job names
        - evaluator_names: sorted list of evaluator names
        - averages: dict mapping evaluator_name -> job_name -> (display_value, style)
    """
    all_job_names: set[str] = set()
    all_evaluator_names: set[str] = set()

    # Store all scores per evaluator per job
    scores_by_evaluator_and_job: ScoresByEvaluatorAndJob = defaultdict(
        lambda: defaultdict(list)
    )

    # Collect scores
    for result in results:
        if not result.job_results:
            continue

        for job_result in result.job_results:
            all_job_names.add(job_result.job_name)

            if not job_result.evaluator_scores:
                continue

            for score in job_result.evaluator_scores:
                all_evaluator_names.add(score.evaluator_name)

                if not score.error:
                    scores_by_evaluator_and_job[score.evaluator_name][
                        job_result.job_name
                    ].append(score.score.value)

    job_names = sorted(all_job_names)
    evaluator_names = sorted(all_evaluator_names)

    # Calculate averages
    averages: dict[str, dict[str, tuple[str, str]]] = {}

    for evaluator_name in evaluator_names:
        evaluator_averages: dict[str, tuple[str, str]] = {}

        for job_name in job_names:
            scores = scores_by_evaluator_and_job[evaluator_name].get(job_name, [])

            if not scores:
                evaluator_averages[job_name] = ("-", "dim")
            else:
                first_score = scores[0]

                if isinstance(first_score, bool):
                    # Calculate pass rate for boolean scores
                    pass_count = sum(1 for s in scores if s is True)
                    pass_rate = (pass_count / len(scores)) * 100

                    if pass_rate == 100:
                        style = "green"
                    elif pass_rate >= 50:
                        style = "yellow"
                    else:
                        style = "red"

                    evaluator_averages[job_name] = (f"{pass_rate:.1f}%", style)

                elif isinstance(first_score, (int, float)):
                    # Calculate average for numeric scores
                    avg = sum(float(s) for s in scores) / len(scores)
                    evaluator_averages[job_name] = (f"{avg:.2f}", "yellow")

                else:
                    # For strings, show placeholder
                    evaluator_averages[job_name] = ("[string]", "dim")

        averages[evaluator_name] = evaluator_averages

    return {
        "job_names": job_names,
        "evaluator_names": evaluator_names,
        "averages": averages,
    }


def create_results_display(results: EvaluatorqResult) -> Table:
    """Create the main results table showing evaluator averages per job"""
    if not results:
        return Table()

    data = calculate_evaluator_averages(results)
    job_names = data["job_names"]
    evaluator_names = data["evaluator_names"]
    averages = data["averages"]

    if not job_names or not evaluator_names:
        # Return empty table or message
        table = Table(show_header=False)
        table.add_row(Text("No job results or evaluators found.", style="yellow"))
        return table

    # Create table with dynamic columns and borders
    table = Table(show_header=True, header_style="bold", box=box.ROUNDED)

    # Add evaluator column
    table.add_column("Evaluators", style="white", width=20)

    # Add job columns
    for job_name in job_names:
        table.add_column(job_name, style="white", width=15, justify="center")

    # Add rows for each evaluator
    for evaluator_name in evaluator_names:
        row_data = [Text(evaluator_name)]

        for job_name in job_names:
            avg_value, avg_style = averages[evaluator_name].get(job_name, ("-", "dim"))
            row_data.append(Text(avg_value, style=avg_style))

        table.add_row(*row_data)

    return table


def collect_errors(results: EvaluatorqResult) -> list[str]:
    """Collect and format error messages from results"""
    errors: list[str] = []

    for idx, result in enumerate(results):
        if result.error:
            errors.append(f"• Data point {idx + 1}: {result.error}")

        if result.job_results:
            for job in result.job_results:
                if job.error:
                    errors.append(f'• Job "{job.job_name}": {job.error}')

                if job.evaluator_scores:
                    for score in job.evaluator_scores:
                        if score.error:
                            errors.append(
                                f'• Evaluator "{score.evaluator_name}": {score.error}'
                            )

    return errors


async def display_results_table(results: EvaluatorqResult) -> None:
    """
    Display evaluation results in formatted tables.

    Args:
        results: List of DataPointResult objects to display
    """
    console = Console()

    if not results:
        console.print("\n[yellow]No results to display.[/yellow]\n")
        return

    console.print()

    # Title
    console.print("[bold underline white]EVALUATION RESULTS[/bold underline white]")
    console.print()

    # Summary section
    console.print("[bold white]Summary:[/bold white]")
    summary_table = create_summary_display(results)
    console.print(summary_table)
    console.print()

    # Detailed results section
    console.print("[bold white]Detailed Results:[/bold white]")
    results_table = create_results_display(results)
    console.print(results_table)
    console.print()

    # Show errors if any
    errors = collect_errors(results)
    if errors:
        console.print("[bold red]Errors:[/bold red]")
        for error in errors:
            console.print(f"[red]{error}[/red]")
        console.print()

    # Show tip
    console.print("[dim]💡 Tip: Use print=False to get raw JSON results.[/dim]")
    console.print()

    # Success message (shown after table, matching TypeScript)
    console.print("[green]✓ Evaluation completed successfully[/green]")
    console.print()
