"""``eq models``: compare catalogue models on Artificial Analysis score against price, speed or latency."""

from __future__ import annotations

import asyncio
import math
import sys
from typing import TYPE_CHECKING, Annotated, cast

import typer

from evaluatorq.common import cli_width  # noqa: F401 — import for its non-TTY width side effect
from evaluatorq.common.cli_epilog import examples
from evaluatorq.common.cli_help import CONTEXT_SETTINGS
from evaluatorq.common.cli_json import echo_json
from evaluatorq.common.model_frontier import MIN_SEAT_INDEX

if TYPE_CHECKING:
    from evaluatorq.models import Index, ParetoPoint, ParetoResult, ScoredModel, XAxis

app = typer.Typer(help='Compare catalogue models.', no_args_is_help=True, context_settings=CONTEXT_SETTINGS)

_PARETO_EPILOG = examples(
    'eq models pareto',
    'eq models pareto --dominated',
    'eq models pareto --min-score 0',
    'eq models pareto --index legal --provider openai --provider anthropic',
    'eq models pareto --model gpt-5.6-terra --dominated --json',
    'eq models pareto --effort high --x output_speed --measured-only',
)


@app.callback()
def models() -> None:
    """Compare the models in the Orq catalogue."""


def _print_table(
    result: ParetoResult, points: list[ParetoPoint], *, dominated: bool, left_out_by_provider: int
) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    from evaluatorq.common.model_frontier import X_AXIS_SPECS

    spec = X_AXIS_SPECS[result.x]
    effort = result.effort or 'default effort'
    kind = 'Dominated models' if dominated else 'Pareto frontier'
    table = Table(title=f'{kind}: {result.index} index ({effort}) vs {result.x}', box=box.ROUNDED)
    table.add_column('Model', style='cyan')
    table.add_column('Score', justify='right')
    table.add_column(spec.short_label, justify='right')
    if result.x == 'price':
        table.add_column('Score per $', justify='right')
    table.add_column('Est.', justify='center')
    table.add_column('Dominated by' if dominated else 'Frontier', style='white')
    for point in points:
        row = [point.model.router_id, f'{point.score:.1f}', spec.format(point.x)]
        if result.x == 'price':
            row.append(f'{point.pareto_score:.1f}' if point.pareto_score is not None else '—')
        row.extend((
            'yes' if point.model.is_estimated else '',
            point.dominated_by or ('— (on the frontier)' if dominated else 'yes'),
        ))
        table.add_row(*row)
    console = Console(file=sys.stdout)
    console.print(table)
    hidden = len(result.below_min_score)
    if hidden:
        console.print(
            f'[dim]{hidden} model(s) hidden for scoring below --min-score {result.min_score:g}. Pass --min-score 0 to show every model.[/dim]'
        )
    if missing := len(result.excluded) - hidden:
        console.print(
            f'[dim]{missing} model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.[/dim]'
        )
    if left_out_by_provider:
        console.print(f'[dim]{left_out_by_provider} model(s) from other providers left out by --provider.[/dim]')


@app.command('pareto', epilog=_PARETO_EPILOG)
def pareto_cmd(
    dominated: Annotated[  # noqa: FBT002
        bool, typer.Option('--dominated', help='List dominated models and what beats each, instead of the frontier.')
    ] = False,
    provider: Annotated[
        list[str] | None,
        typer.Option(
            '--provider', help='Compare only models from this provider or developer, e.g. openai. Repeatable.'
        ),
    ] = None,
    model: Annotated[
        list[str] | None,
        typer.Option(
            '--model',
            help='Show only this model, compared against every other model in the comparison, including when --provider '
            'would leave it out. A provider/model id, or a bare id held by exactly one model. Repeatable.',
        ),
    ] = None,
    effort: Annotated[
        str | None,
        typer.Option('--effort', help='Score the intelligence index at this reasoning effort instead of the default.'),
    ] = None,
    index: Annotated[
        str,
        typer.Option('--index', help='The y axis: intelligence, or a domain index such as legal or economics.'),
    ] = 'intelligence',
    x: Annotated[str, typer.Option('--x', help='The x axis: price, output_speed or time_to_first_token.')] = 'price',
    include_disabled: Annotated[  # noqa: FBT002
        bool, typer.Option('--include-disabled', help='Include models switched off in the workspace.')
    ] = False,
    measured_only: Annotated[  # noqa: FBT002
        bool, typer.Option('--measured-only', help='Leave out models whose score is estimated rather than measured.')
    ] = False,
    min_score: Annotated[
        float,
        typer.Option(
            '--min-score',
            min=0.0,
            help='Hide models scoring below this on the chosen index. 0 shows every model.',
        ),
    ] = MIN_SEAT_INDEX,
    json_output: Annotated[  # noqa: FBT002
        bool, typer.Option('--json', help='Emit the analysis as JSON on stdout (machine-readable).')
    ] = False,
) -> None:
    """Show which models nothing beats on score and x axis, or which are beaten and by what."""
    from evaluatorq.common.model_frontier import available_efforts, available_indices, matches_vendor
    from evaluatorq.models import INDICES, X_AXES, find_model, load_models, pareto

    if index not in INDICES:
        raise typer.BadParameter(f'choose one of {", ".join(INDICES)}', param_hint='--index')
    if x not in X_AXES:
        raise typer.BadParameter(f'choose one of {", ".join(X_AXES)}', param_hint='--x')
    if effort is not None and index != 'intelligence':
        raise typer.BadParameter('domain indices exist only at the default effort', param_hint='--effort')

    pool = asyncio.run(load_models(include_disabled=include_disabled))
    indices = available_indices(pool)
    if index not in indices:
        raise typer.BadParameter(
            f'no model in the catalogue has a {index} score; choose one of {", ".join(indices)}', param_hint='--index'
        )
    efforts = available_efforts(pool)
    if effort is not None and effort not in efforts:
        listed = ', '.join(efforts) or 'none'
        raise typer.BadParameter(
            f'no model in the catalogue has a score at effort {effort!r}; the catalogue has: {listed}',
            param_hint='--effort',
        )
    if not math.isfinite(min_score):
        raise typer.BadParameter('must be a finite number', param_hint='--min-score')
    # Resolve --model against the whole catalogue first, so --provider cannot make a named model unknown.
    shown: set[str] | None = None
    named: dict[str, ScoredModel] = {}
    if model:
        shown = set()
        for model_id in model:
            match = find_model(pool, model_id=model_id)
            if match is None:
                raise typer.BadParameter(
                    f'{model_id!r} matches no model, or more than one; pass the provider/model id', param_hint='--model'
                )
            shown.add(match.router_id)
            named[match.router_id] = match
    left_out_by_provider = 0
    if provider:
        kept = [m for m in pool if matches_vendor(m, provider)]
        kept_ids = {m.router_id for m in kept}
        kept.extend(m for router_id, m in named.items() if router_id not in kept_ids)
        left_out_by_provider = len(pool) - len(kept)
        pool = kept

    result = pareto(
        pool,
        index=cast('Index', index),
        effort=effort,
        x=cast('XAxis', x),
        measured_only=measured_only,
        min_score=min_score,
    )
    # A named model is shown whichever side of the frontier it is on; --dominated then only picks the table style.
    points = result.dominated if dominated else result.frontier
    if shown is not None:
        points = [p for p in result.points if p.model.router_id in shown]
        dominated = dominated or any(not p.on_frontier for p in points)
        for gone in (e for e in result.excluded if e.router_id in shown):
            typer.echo(f'{gone.router_id} is not plotted: {gone.reason}.', err=True)

    if json_output:
        echo_json({
            'index': result.index,
            'effort': result.effort,
            'x': result.x,
            'min_score': result.min_score,
            'points': [p.summary() for p in points],
            'excluded': [e.model_dump() for e in result.excluded],
            'left_out_by_provider': left_out_by_provider,
        })
        return
    if not points:
        what = 'dominated' if dominated else 'on the frontier'
        typer.echo(f'No models {what} for these filters.')
        return
    _print_table(result, points, dominated=dominated, left_out_by_provider=left_out_by_provider)
