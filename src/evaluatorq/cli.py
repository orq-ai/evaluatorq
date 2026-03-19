"""Top-level CLI for evaluatorq.

Usage:
    evaluatorq redteam run --target agent:my-agent
    evaluatorq redteam ui report.json
"""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point that lazily imports typer to avoid hard dependency."""
    try:
        import typer
    except ImportError:
        print(
            "The evaluatorq CLI requires optional dependencies.\n"
            "Install with: pip install 'evaluatorq[redteam]'",
            file=sys.stderr,
        )
        sys.exit(1)

    app = typer.Typer(
        name="evaluatorq",
        help="Evaluation framework for AI systems.",
        no_args_is_help=True,
    )

    try:
        from evaluatorq.redteam.cli import app as redteam_app

        app.add_typer(redteam_app, name="redteam", help="Red teaming commands.")
    except ImportError:
        pass

    app()


# Allow `python -m evaluatorq.cli` as well as the entry point.
if __name__ == "__main__":
    main()
