"""Side-by-side comparison of two simulation runs.

The single-run sim views live in ``sim_views.py``; this module adds a compare
surface on top of the same ``sim`` adapter/library. Nothing here touches the sim
runner or data model — it only reads two already-stored ``SimulationRun`` objects.

Routes (registered by ``register_sim_compare_routes``):

    GET  /compare/sim?a={ridA}&b={ridB}[&<filter params>][&cmp_view=&cmp_sort=…]
        Full page: hero with a run/match status line, "what changed" headline
        rows, number tiles with A → B deltas, outcome and termination bars, a
        scores panel, a persona x scenario change heatmap, the conversation
        table, and the sim filter rail. Bars, legends, heatmap cells and the
        headline scenario name are filter links.

    POST /compare/sim/filter?a={ridA}&b={ridB}
    GET  /compare/sim/filter?a={ridA}&b={ridB}&<filter params>
        Filter-rail swap: re-renders the body and the rail from the filtered
        conversations of both runs. Sets ``HX-Push-Url`` to the canonical GET
        URL of the resulting view, so the address bar is shareable.

    GET  /compare/sim/conversations?a={ridA}&b={ridB}&cmp_view=&cmp_sort=…
        The conversation table fragment (segment, sort, page, size), with the
        current filter state pulled in via ``hx-include``. Also sets ``HX-Push-Url``.

    GET  /compare/sim/transcript?a={ridA}&b={ridB}&ia={idxA}&ib={idxB}
        Paired transcript for one matched conversation, loaded into the drawer.

Conversations are matched by ``(persona, scenario)``: runs that share a dataset
line up; anything unmatched is listed separately (with a low-overlap warning) so
a mismatched dataset is visible rather than silently dropped.

The filter decides which rows survive: a matched pair stays when either of its
conversations passes, an unmatched conversation stays when it passes. Every
number on the page is computed over each run's side of the surviving rows, so
both runs are always described by the same conversations the table lists.

Everything renders as server-side HTML and CSS — no chart library.
"""

from evaluatorq.dashboard.sim_compare.routes import error_page, register_sim_compare_routes, resolve_run

__all__ = ['error_page', 'register_sim_compare_routes', 'resolve_run']
