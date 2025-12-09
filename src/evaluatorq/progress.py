from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import CoroutineType
from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from .types import EvaluatorqResult


class Phase(str, Enum):
    """Progress phases"""

    INITIALIZING = "initializing"
    PROCESSING = "processing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"


@dataclass
class ProgressState:
    """State tracking for evaluation progress"""

    total_data_points: int = 0
    current_data_point: int = 0
    current_job: str | None = None
    current_evaluator: str | None = None
    phase: Phase = Phase.INITIALIZING


class ProgressService:
    """
    Progress tracking service using Rich for terminal output.

    Manages a spinner with real-time progress updates during evaluation.
    """

    def __init__(self):
        self.state: ProgressState = ProgressState()
        self.console: Console = Console()
        self.live: Live | None = None
        self.spinner: Spinner = Spinner("dots", style="cyan")

    def _format_progress_text(self) -> Text:
        """Format the progress text based on current state"""
        text = Text()

        # Calculate percentage
        percentage = 0
        if self.state.total_data_points > 0:
            percentage = round(
                (self.state.current_data_point / self.state.total_data_points) * 100
            )

        if self.state.phase == Phase.INITIALIZING:
            text = text.append("Initializing evaluation...", style="cyan")

        elif self.state.phase == Phase.PROCESSING:
            text = text.append(
                f"Processing data point {self.state.current_data_point}/{self.state.total_data_points} ({percentage}%)",
                style="cyan",
            )
            if self.state.current_job:
                text = text.append(" - Running job: ", style="dim")
                text = text.append(self.state.current_job, style="white")

        elif self.state.phase == Phase.EVALUATING:
            text = text.append(
                f"Evaluating results {self.state.current_data_point}/{self.state.total_data_points} ({percentage}%)",
                style="cyan",
            )
            if self.state.current_evaluator:
                text = text.append(" - Running evaluator: ", style="dim")
                text = text.append(self.state.current_evaluator, style="white")

        elif self.state.phase == Phase.COMPLETED:
            text = text.append("✓ Evaluation completed", style="green")

        return text

    def _get_renderable(self):
        """Get the renderable content for Live display"""
        if self.state.phase == Phase.COMPLETED:
            # No spinner for completed state
            return self._format_progress_text()

        # Combine spinner with progress text using Columns for horizontal layout
        return Columns([self.spinner, self._format_progress_text()], expand=False)

    async def update_progress(
        self,
        total_data_points: int | None = None,
        current_data_point: int | None = None,
        current_job: str | None = None,
        current_evaluator: str | None = None,
        phase: Phase | None = None,
    ):
        """
        Update the progress state.

        Args:
            total_data_points: Total number of data points to process
            current_data_point: Current data point index
            current_job: Name of the currently running job
            current_evaluator: Name of the currently running evaluator
            phase: Current phase of evaluation
        """
        if total_data_points is not None:
            self.state.total_data_points = total_data_points
        if current_data_point is not None:
            self.state.current_data_point = current_data_point
        if current_job is not None:
            self.state.current_job = current_job
        if current_evaluator is not None:
            self.state.current_evaluator = current_evaluator
        if phase is not None:
            self.state.phase = Phase(phase)

        # Update live display if active
        if self.live:
            self.live.update(self._get_renderable())

    async def start_spinner(self):
        """Start the progress spinner"""
        if not self.live:
            self.live = Live(
                self._get_renderable(),
                console=self.console,
                refresh_per_second=10,
                transient=False,
            )
            self.live.start()

    async def stop_spinner(self):
        """Stop the progress spinner"""
        if self.live:
            self.live.stop()

            # Add newline for spacing
            self.console.print()
            self.live = None

    async def show_message(self, message: str):
        """
        Show an informational message.

        Args:
            message: The message to display
        """
        if self.live:
            # Temporarily stop live, show message, restart
            was_started = self.live.is_started
            if was_started:
                self.live.stop()

            info_text = Text()
            info_text = info_text.append(f"ℹ {message} ", style="blue")

            self.console.print(info_text)

            if was_started:
                self.live.start()
        else:
            self.console.print(message)


async def with_progress(
    coroutine: CoroutineType[Any, Any, EvaluatorqResult],
    progress_service: ProgressService,
    show_progress: bool = True,
):
    """
    Run a coroutine with progress tracking.

    Args:
        coroutine: The async function to run
        progress_service: The progress service instance
        show_progress: Whether to show progress (default: True)

    Returns:
        The result of the coroutine

    Example:
        ```python
        progress = ProgressService()
        result = await with_progress(
            process_evaluation(),
            progress,
            show_progress=True
        )
        ```
    """
    if not show_progress:
        return await coroutine

    try:
        # Start spinner
        await progress_service.start_spinner()

        # Run the coroutine
        result = await coroutine

        # Update to completed state
        await progress_service.update_progress(phase=Phase.COMPLETED)

        # Stop spinner with success
        await progress_service.stop_spinner()

        return result

    except Exception as error:
        # Stop spinner on error
        await progress_service.stop_spinner()
        raise error
