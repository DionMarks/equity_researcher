"""What a run is doing while it runs, so a person who agreed to spend knows how long to wait."""

from __future__ import annotations

from decimal import Decimal
from types import TracebackType
from typing import Final, Protocol

from rich.console import Console
from rich.progress import Progress, ProgressColumn, SpinnerColumn, Task, TaskID, TextColumn
from rich.text import Text

from ..domain.money import aud

__all__ = [
    "ACTIVITIES",
    "PURPOSES",
    "NoProgress",
    "RunProgress",
    "TerminalProgress",
    "duration",
    "usual_range",
]

#: What each model step is for, in the words the line shows while it waits on one.
ACTIVITIES: Final[dict[str, str]] = {
    "identify_bank": "Looking up the bank's financial year on the ASX",
    "discover": "Searching for the filing",
    "discover_classify": "Checking the filing is the right one",
    "map_rows": "Matching the bank's own names for its figures",
    "locate": "Looking for the page with the bank's walk",
    "text_walk": "Reading the bank's walk from its text",
    "chart_read": "Reading the bank's walk from its chart",
    "taxonomy_classify": "Naming the walk's drivers",
    "explain": "Writing the memo in the bank's words",
    "verify": "Checking every cited sentence against its page",
}


#: The same, said once the call is done: what the memo's list of model calls shows.
PURPOSES: Final[dict[str, str]] = {
    "identify_bank": "Looked up the bank's financial year on the ASX",
    "discover": "Searched for the filing",
    "discover_classify": "Checked the filing was the right one",
    "map_rows": "Matched the bank's own names for its figures",
    "locate": "Looked for the page with the bank's walk",
    "text_walk": "Read the bank's walk from its text",
    "chart_read": "Read the bank's walk from its chart",
    "taxonomy_classify": "Named the walk's drivers",
    "explain": "Wrote the memo in the bank's words",
    "verify": "Checked every cited sentence against its page",
}


class RunProgress(Protocol):
    """Told what a run is doing. It only ever watches: nothing it does changes the run."""

    def doing(self, activity: str) -> None: ...

    def calling(self, step: str) -> None: ...

    def called(self, step: str, cost_usd: Decimal) -> None: ...


class NoProgress:
    """Says nothing. What a script, an eval or a test gets."""

    def doing(self, activity: str) -> None:
        return None

    def calling(self, step: str) -> None:
        return None

    def called(self, step: str, cost_usd: Decimal) -> None:
        return None


def duration(seconds: int) -> str:
    """A length of time as a person would say it: 25 seconds, 1 minute, 1.5 minutes."""
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = round(seconds / 30) / 2
    return "1 minute" if minutes == 1 else f"{minutes:g} minutes"


def usual_range(low: int, high: int) -> str:
    """The time a run like this one usually takes: "25 seconds to 1.5 minutes"."""
    return f"{duration(low)} to {duration(high)}"


class _Elapsed(ProgressColumn):
    """Minutes and seconds since the run started: 0:42, 1:05."""

    def render(self, task: Task) -> Text:
        seconds = int(task.elapsed or 0)
        return Text(f"{seconds // 60}:{seconds % 60:02d}", style="cyan")


class TerminalProgress:
    """One line on stderr that keeps moving while a run works, and is gone when it ends.

    Used as a context manager around the run. On a console that isn't a terminal it does
    nothing at all.
    """

    def __init__(self, console: Console, usual_seconds: tuple[int, int] | None = None) -> None:
        self._console = console
        self._usual = usual_range(*usual_seconds) if usual_seconds else ""
        self._calls = 0
        self._spent = Decimal(0)
        self._progress: Progress | None = None
        self._task: TaskID | None = None

    def __enter__(self) -> TerminalProgress:
        if self._console.is_terminal:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                _Elapsed(),
                TextColumn("[dim]{task.fields[context]}"),
                console=self._console,
                transient=True,
            )
            self._progress.start()
            self._task = self._progress.add_task(
                "Working out the question", total=None, context=self.context()
            )
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def doing(self, activity: str) -> None:
        self._show(activity)

    def calling(self, step: str) -> None:
        self._show(ACTIVITIES.get(step, "Waiting on the model"))

    def called(self, step: str, cost_usd: Decimal) -> None:
        self._calls += 1
        self._spent += cost_usd
        self._show(None)

    def context(self) -> str:
        """Everything on the line after the elapsed time."""
        calls = "1 model call" if self._calls == 1 else f"{self._calls} model calls"
        spent = f"{calls}, {aud(self._spent)} so far"
        return f"usually {self._usual}  ·  {spent}" if self._usual else spent

    def _show(self, activity: str | None) -> None:
        if self._progress is None or self._task is None:
            return
        if activity is None:
            self._progress.update(self._task, context=self.context())
        else:
            self._progress.update(self._task, description=activity, context=self.context())
