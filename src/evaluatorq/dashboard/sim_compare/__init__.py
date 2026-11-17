"""Side-by-side comparison of two simulation runs (``/compare/sim``).

Reads two stored ``SimulationRun`` objects through the sim adapter and renders
server-side HTML. Conversations pair by ``(persona, scenario)``; a pair stays
under a filter when either side passes. ``docs/dashboard.md`` section 9 describes
the page; ``routes.py`` registers the routes.
"""

from evaluatorq.dashboard.sim_compare.routes import error_page, register_sim_compare_routes, resolve_run

__all__ = ['error_page', 'register_sim_compare_routes', 'resolve_run']
