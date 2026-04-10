"""Rich terminal reporter for HPF analysis results."""

from __future__ import annotations

import math
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from hpf import __version__
from hpf.models import (
    AnalysisResult,
    HPFReport,
    ParameterSuggestion,
    ParameterType,
    SuggestionAction,
)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_ACTION_STYLES: dict[SuggestionAction, str] = {
    SuggestionAction.EXPAND: "bold yellow",
    SuggestionAction.NARROW: "bold green",
    SuggestionAction.KEEP: "dim",
    SuggestionAction.LOG_SCALE: "bold cyan",
    SuggestionAction.REMOVE: "bold red",
}

_CONFIDENCE_DOTS: dict[str, str] = {
    "high": "[green]●[/green]",
    "medium": "[yellow]●[/yellow]",
    "low": "[red]●[/red]",
}

_PARAM_TYPE_LABELS: dict[ParameterType, str] = {
    ParameterType.FLOAT: "FLOAT",
    ParameterType.INT: "INT",
    ParameterType.CATEGORICAL: "CATEGORICAL",
}

_REASONING_MAX_CHARS = 60


class Reporter:
    """Print a rich, human-readable HPF report to the terminal.

    Parameters
    ----------
    console:
        A :class:`rich.console.Console` instance.  When *None* a default
        console is constructed (outputs to stdout with auto-detected width
        and colour support).
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print_report(self, report: HPFReport) -> None:
        """Print the full analysis report to the terminal."""
        self.print_welcome()
        self._print_study_summary(report.analysis)
        self._print_parameter_table(report.suggestions)

        if report.llm_explanation is not None:
            self._print_llm_explanation(report.llm_explanation)
        else:
            self._print_next_steps(report.suggestions)

        self._print_optuna_code(report.optuna_code)

    def print_welcome(self) -> None:
        """Print the HPF welcome banner."""
        banner = Text()
        banner.append("HPF", style="bold bright_white")
        banner.append(" — ", style="dim")
        banner.append("Hyperparameter Finder", style="bold cyan")
        banner.append(f"  v{__version__}", style="dim")

        self.console.print()
        self.console.print(
            Panel(
                banner,
                border_style="cyan",
                padding=(0, 2),
                expand=False,
            )
        )
        self.console.print()

    # ------------------------------------------------------------------
    # Private section printers
    # ------------------------------------------------------------------

    def _print_study_summary(self, analysis: AnalysisResult) -> None:
        """Print a panel summarising the loaded study."""
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold", justify="right")
        grid.add_column()

        def _row(label: str, value: str) -> None:
            grid.add_row(label, value)

        _row("Study:", analysis.study_name)
        _row("Trials:", f"{analysis.n_complete_trials} complete / {analysis.n_trials} total")
        _row(
            "Best value:",
            f"{analysis.best_value:.6g}" if not math.isnan(analysis.best_value) else "n/a",
        )
        _row("Direction:", analysis.direction.capitalize())
        _row(
            "Top-K analysed:",
            f"{analysis.n_top_trials} trials ({analysis.top_k_percent:.0f}%)",
        )
        if analysis.model_type:
            _row("Model type:", analysis.model_type)

        self.console.print(
            Panel(
                grid,
                title="[bold]Study Summary[/bold]",
                border_style="blue",
                padding=(1, 2),
            )
        )
        self.console.print()

    def _print_parameter_table(self, suggestions: list[ParameterSuggestion]) -> None:
        """Print the per-parameter analysis table."""
        if not suggestions:
            self.console.print("[dim]No parameters to display.[/dim]")
            return

        table = Table(
            title="[bold]Parameter Analysis[/bold]",
            border_style="bright_black",
            header_style="bold bright_white",
            show_lines=True,
            expand=True,
        )

        table.add_column("Parameter", style="bold", no_wrap=True)
        table.add_column("Type", justify="center")
        table.add_column("Original Range", justify="left")
        table.add_column("Suggested Range", justify="left")
        table.add_column("Action", justify="center", no_wrap=True)
        table.add_column("Confidence", justify="center", no_wrap=True)
        table.add_column("Reasoning", justify="left")

        for s in suggestions:
            action_style = _ACTION_STYLES.get(s.action, "")
            action_text = Text(s.action.value.upper(), style=action_style)

            confidence_dot = _CONFIDENCE_DOTS.get(s.confidence, "●")

            # For KEEP, the suggested range is the same — dim it to reduce noise.
            if s.action == SuggestionAction.KEEP:
                suggested_range = Text(s.suggested_range_str, style="dim")
            elif s.action == SuggestionAction.REMOVE:
                suggested_range = Text("—", style="bold red")
            else:
                suggested_range = Text(s.suggested_range_str)

            reasoning = _truncate(s.reasoning, _REASONING_MAX_CHARS)

            table.add_row(
                s.name,
                _PARAM_TYPE_LABELS.get(s.param_type, s.param_type.value),
                s.original_range_str,
                suggested_range,
                action_text,
                confidence_dot,
                reasoning,
            )

        self.console.print(table)
        self.console.print()

    def _print_llm_explanation(self, explanation: str) -> None:
        """Print the LLM-generated markdown explanation in a panel."""
        self.console.print(
            Panel(
                Markdown(explanation),
                title="[bold cyan]AI Explanation[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        self.console.print()

    def _print_optuna_code(self, code: str) -> None:
        """Print the generated Optuna search space code with syntax highlighting."""
        syntax = Syntax(
            code,
            "python",
            theme="monokai",
            line_numbers=False,
            word_wrap=False,
        )
        self.console.print(
            Panel(
                syntax,
                title="[bold green]Updated Search Space[/bold green]",
                border_style="green",
                padding=(1, 1),
            )
        )
        self.console.print()

    def _print_next_steps(self, suggestions: list[ParameterSuggestion]) -> None:
        """Print a prioritised bullet-list of the top actionable changes.

        Shown only when no LLM explanation is available.  Priority order:
        1. EXPAND (potentially missing good regions)
        2. NARROW with high confidence
        3. LOG_SCALE
        """
        actionable: list[ParameterSuggestion] = []

        # Priority 1: EXPAND
        actionable.extend(
            s for s in suggestions if s.action == SuggestionAction.EXPAND
        )
        # Priority 2: NARROW (high confidence first)
        narrow_high = [
            s for s in suggestions
            if s.action == SuggestionAction.NARROW and s.confidence == "high"
        ]
        narrow_other = [
            s for s in suggestions
            if s.action == SuggestionAction.NARROW and s.confidence != "high"
        ]
        actionable.extend(narrow_high)
        actionable.extend(narrow_other)
        # Priority 3: LOG_SCALE
        actionable.extend(
            s for s in suggestions if s.action == SuggestionAction.LOG_SCALE
        )

        if not actionable:
            self.console.print(
                Panel(
                    "[dim]All parameters look well-tuned. Consider increasing trial count "
                    "or exploring new architectures.[/dim]",
                    title="[bold]Next Steps[/bold]",
                    border_style="bright_black",
                    padding=(1, 2),
                )
            )
            self.console.print()
            return

        lines: list[Text] = []
        for s in actionable:
            action_style = _ACTION_STYLES.get(s.action, "")
            bullet = Text()
            bullet.append("  • ", style="bold bright_white")
            bullet.append(f"[{s.action.value.upper()}]", style=action_style)
            bullet.append(f" {s.name}", style="bold")
            bullet.append(f": {_truncate(s.reasoning, 80)}", style="default")
            lines.append(bullet)

        tip = Text()
        tip.append("\n  Tip: ", style="dim bold")
        tip.append(
            "Remove `--no-llm` to get an AI-powered narrative explanation.",
            style="dim",
        )

        group_lines: list[Text] = lines + [tip]

        from rich.console import Group

        self.console.print(
            Panel(
                Group(*group_lines),
                title="[bold]Next Steps[/bold]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        self.console.print()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* characters, appending '…' if needed."""
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)] + "…"
