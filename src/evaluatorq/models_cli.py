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
    from evaluatorq.models import Index, ParetoPoint, ParetoResult, PriceMix, ScoredModel, XAxis

app = typer.Typer(help='Compare catalogue models.', no_args_is_help=True, context_settings=CONTEXT_SETTINGS)

_PARETO_EPILOG = examples(
    'eq models pareto',
    'eq models pareto --dominated',
    'eq models pareto --mix list',
    'eq models pareto --mix 7:1:1:1',
    'eq models pareto --min-score 0',
    'eq models pareto --index legal --provider openai --provider anthropic',
    'eq models pareto --model gpt-5.6-terra --dominated --json',
    'eq models pareto --effort high --x output_speed --no-estimate',
)


@app.callback()
def models() -> None:
    """Compare the models in the Orq catalogue."""


# How the table marks a cache rate that is not the host's own.
_SOURCE_MARK = {'host': '', 'artificial_analysis': ' (AA)', 'input': ' (input)'}


def _print_table(
    result: ParetoResult, points: list[ParetoPoint], *, dominated: bool, left_out_by_provider: int
) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    spec = result.x_spec
    mix = result.mix
    show_read, show_write = bool(mix and mix.read), bool(mix and mix.write)
    effort = result.effort or 'default effort'
    kind = 'Dominated models' if dominated else 'Pareto frontier'
    table = Table(title=f'{kind}: {result.index} index ({effort}) vs {result.x}', box=box.ROUNDED)
    table.add_column('Model', style='cyan')
    table.add_column('Score', justify='right')
    table.add_column(spec.short_label, justify='right')
    if result.x == 'price':
        table.add_column('Score per $', justify='right')
    if show_read:
        table.add_column('Cache read $/1M', justify='right')
    if show_write:
        table.add_column('Cache write $/1M', justify='right')
    table.add_column('Est.', justify='center')
    table.add_column('Dominated by' if dominated else 'Frontier', style='white')
    for point in points:
        row = [point.model.router_id, f'{point.score:.1f}', spec.format(point.x)]
        if result.x == 'price':
            row.append(f'{point.pareto_score:.1f}' if point.pareto_score is not None else '—')
        if show_read:
            rate, source = point.model.cache_read_rate()
            row.append(f'{rate:.3f}{_SOURCE_MARK[source]}')
        if show_write:
            rate, source = point.model.cache_write_rate()
            row.append(f'{rate:.3f}{_SOURCE_MARK[source]}')
        row.extend((
            'yes' if point.model.is_estimated else '',
            point.dominated_by or ('— (on the frontier)' if dominated else 'yes'),
        ))
        table.add_row(*row)
    console = Console(file=sys.stdout)
    console.print(table)
    if mix is not None and mix.name != 'list':
        console.print(
            f'[dim]Price blends the rates per 1M tokens as {mix.ratio} (cache read : cache write : fresh input : '
            'output). '
            'Pass --mix list for the 3:1 input/output list blend.[/dim]'
        )
    if fallbacks := [p for p in points if p.price_fallback]:
        console.print(
            f'[dim]{len(fallbacks)} shown model(s) price a cache rate their host does not list: (AA) is the Artificial '
            'Analysis price on the same card, (input) bills those tokens at the input rate.[/dim]'
        )
    hidden = len(result.below_min_score)
    if hidden:
        console.print(
            f'[dim]{hidden} model(s) hidden for scoring below --min-score {result.min_score:g}. Pass --min-score 0 to show every model.[/dim]'
        )
    if contributors := len(result.contributors):
        console.print(
            f'[dim]{contributors} contributor listing(s) not compared: the vendor trains on your prompts in exchange for the discount.[/dim]'
        )
    if missing := len(result.excluded) - hidden - contributors:
        console.print(
            f'[dim]{missing} model(s) not compared: no value on one of the axes, or filtered out. Use --json for the reason per model.[/dim]'
        )
    if left_out_by_provider:
        console.print(f'[dim]{left_out_by_provider} model(s) from other providers left out by --provider.[/dim]')


def _checked_options(*, index: str, x: str, effort: str | None, mix: str | None) -> PriceMix:
    """Refuse an unknown index, x axis or mix, and an option the chosen axes do not take; return the price mix."""
    from evaluatorq.models import INDICES, X_AXES, parse_mix

    if index not in INDICES:
        raise typer.BadParameter(f'choose one of {", ".join(INDICES)}', param_hint='--index')
    if x not in X_AXES:
        raise typer.BadParameter(f'choose one of {", ".join(X_AXES)}', param_hint='--x')
    if effort is not None and index != 'intelligence':
        raise typer.BadParameter(
            'domain indices exist only at the one effort Artificial Analysis benchmarked', param_hint='--effort'
        )
    if mix is not None and x != 'price':
        raise typer.BadParameter('a price mix applies only to --x price', param_hint='--mix')
    try:
        return parse_mix(mix or 'agentic')
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint='--mix') from None


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
            'would leave it out. The full router id, or the id after the provider or its last part when exactly one '
            'model has it. Repeatable.',
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
    mix: Annotated[
        str | None,
        typer.Option(
            '--mix',
            help='How --x price blends token prices: agentic (default; 7 cache read : 2 fresh input : 1 output, the '
            'Artificial Analysis blend), list (3 input : 1 output), or a READ:WRITE:FRESH:OUTPUT ratio such as 7:1:1:1.',
        ),
    ] = None,
    include_disabled: Annotated[  # noqa: FBT002
        bool, typer.Option('--include-disabled', help='Include models switched off in the workspace.')
    ] = False,
    exclude_estimated: Annotated[  # noqa: FBT002
        bool, typer.Option('--no-estimate', help='Leave out models whose score is estimated rather than measured.')
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
    from evaluatorq.models import find_model, load_models, pareto

    price_mix = _checked_options(index=index, x=x, effort=effort, mix=mix)

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
                    f'{model_id!r} matches no model, or more than one; pass the full router id', param_hint='--model'
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
        mix=price_mix,
        exclude_estimated=exclude_estimated,
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
            'x_label': result.x_spec.label,
            'mix': result.mix.model_dump() if result.mix else None,
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
