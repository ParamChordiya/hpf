"""HPF command-line interface.

Entry point: ``hpf`` (registered in pyproject.toml as ``hpf = "hpf.cli:app"``).

Commands
--------
hpf analyze  — load an Optuna study and print HPF recommendations.
hpf setup    — interactive LLM provider configuration wizard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(
    name="hpf",
    help="Find optimal hyperparameter search ranges for your Optuna studies.",
    add_completion=False,
)

_console = Console()
_err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# hpf analyze
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    study_name: str = typer.Option(
        ...,
        "--study",
        "-s",
        help="Optuna study name.",
    ),
    storage: Optional[str] = typer.Option(
        None,
        "--storage",
        help="Optuna storage URL (e.g. sqlite:///study.db). Omit for in-memory.",
    ),
    storage_file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a SQLite .db file (shorthand for sqlite:///absolute_path).",
    ),
    top_k: float = typer.Option(
        20.0,
        "--top-k",
        help="Top K% of trials to analyse (default: 20).",
    ),
    model_type: Optional[str] = typer.Option(
        None,
        "--model-type",
        "-m",
        help="Model type hint (xgboost, lightgbm, neural_net, …).",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip LLM explanation and show a rule-based next-steps list instead.",
    ),
) -> None:
    """Analyse an Optuna study and suggest better hyperparameter ranges."""
    # Deferred imports so the CLI starts fast even when optional deps are absent.
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        _fatal(
            "optuna is not installed.",
            hint="Run: pip install optuna",
        )

    from hpf.analyzer import StudyAnalyzer
    from hpf.formatters import Reporter
    from hpf.formatters.code_gen import generate_optuna_code
    from hpf.models import HPFReport
    from hpf.range_suggester import RangeSuggester

    # ── Step 1: Resolve storage URL ───────────────────────────────────────
    resolved_storage = _resolve_storage(storage, storage_file)

    # ── Step 2: Load study ────────────────────────────────────────────────
    try:
        study = optuna.load_study(study_name=study_name, storage=resolved_storage)
    except KeyError:
        _fatal(
            f"Study [bold]{study_name!r}[/bold] was not found.",
            hint=(
                "Check that the study name is correct and that the storage URL "
                "points to the right database.\n"
                f"  Storage: {resolved_storage or 'in-memory (default)'}"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        _handle_optuna_load_error(exc, resolved_storage)

    # ── Step 3: Analyse ───────────────────────────────────────────────────
    try:
        analysis = StudyAnalyzer(top_k_percent=top_k).analyze(
            study, model_type=model_type  # type: ignore[union-attr]
        )
    except Exception as exc:  # noqa: BLE001
        _fatal(f"Analysis failed: {exc}")

    if analysis.n_complete_trials == 0:  # type: ignore[union-attr]
        _warn(
            f"Study [bold]{study_name!r}[/bold] has no completed trials. "
            "Nothing to analyse."
        )
        raise typer.Exit(code=1)

    # ── Step 4: Suggest ranges ────────────────────────────────────────────
    try:
        suggestions = RangeSuggester().suggest(analysis)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        _fatal(f"Range suggestion failed: {exc}")

    # ── Step 5: Generate Optuna code ──────────────────────────────────────
    optuna_code = generate_optuna_code(suggestions, study_name)  # type: ignore[arg-type]

    # ── Step 6: Optional LLM explanation ─────────────────────────────────
    llm_explanation: Optional[str] = None

    if not no_llm:
        llm_explanation = _try_get_llm_explanation(
            analysis, suggestions, optuna_code  # type: ignore[arg-type]
        )

    # ── Step 7: Build report ──────────────────────────────────────────────
    report = HPFReport(
        analysis=analysis,  # type: ignore[arg-type]
        suggestions=suggestions,  # type: ignore[arg-type]
        llm_explanation=llm_explanation,
        optuna_code=optuna_code,
    )

    # ── Step 8: Print ─────────────────────────────────────────────────────
    Reporter(_console).print_report(report)


# ---------------------------------------------------------------------------
# hpf setup
# ---------------------------------------------------------------------------


@app.command()
def setup() -> None:
    """Configure your LLM provider for AI-powered explanations."""
    try:
        from hpf.llm import SetupWizard  # type: ignore[import-not-found]
    except ImportError:
        _fatal(
            "The LLM module is not available.",
            hint="Ensure the hpf package is fully installed.",
        )

    try:
        SetupWizard().run()  # type: ignore[union-attr]
        _console.print()
        _console.print(
            Panel(
                "[bold green]LLM provider configured successfully.[/bold green]\n"
                "You can now run [bold]hpf analyze[/bold] without [dim]--no-llm[/dim].",
                border_style="green",
                padding=(1, 2),
            )
        )
    except KeyboardInterrupt:
        _console.print("\n[dim]Setup cancelled.[/dim]")
        raise typer.Exit(code=0)
    except Exception as exc:  # noqa: BLE001
        _fatal(f"Setup failed: {exc}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_storage(
    storage: Optional[str],
    storage_file: Optional[Path],
) -> Optional[str]:
    """Build the final Optuna storage URL from the CLI inputs.

    Priority: ``--storage`` > ``--file`` > None (in-memory).
    """
    if storage:
        return storage
    if storage_file is not None:
        abs_path = storage_file.expanduser().resolve()
        if not abs_path.exists():
            _warn(
                f"SQLite file [bold]{abs_path}[/bold] does not exist yet. "
                "Optuna will create it on first use."
            )
        return f"sqlite:///{abs_path}"
    return None


def _try_get_llm_explanation(
    analysis: object,
    suggestions: object,
    optuna_code: str,
) -> Optional[str]:
    """Attempt to fetch an LLM explanation; return None on any failure."""
    try:
        from hpf.llm import SetupWizard  # type: ignore[import-not-found]
    except ImportError:
        _tip(
            "LLM module not found. Run [bold]hpf setup[/bold] to configure "
            "an AI provider."
        )
        return None

    try:
        wizard = SetupWizard()
        client = wizard.get_client()
    except Exception:  # noqa: BLE001
        _tip(
            "No LLM provider is configured. Run [bold]hpf setup[/bold] to "
            "enable AI-powered explanations."
        )
        return None

    if client is None:
        _tip(
            "No LLM provider is configured. Run [bold]hpf setup[/bold] to "
            "enable AI-powered explanations."
        )
        return None

    try:
        explanation: Optional[str] = client.explain(
            analysis=analysis,
            suggestions=suggestions,
            optuna_code=optuna_code,
        )
        return explanation
    except Exception as exc:  # noqa: BLE001
        _warn(f"LLM explanation failed ({exc}). Falling back to rule-based next steps.")
        return None


def _handle_optuna_load_error(exc: Exception, storage: Optional[str]) -> None:
    """Translate common Optuna storage errors into user-friendly messages."""
    exc_str = str(exc).lower()
    hint: str

    if "no module named" in exc_str and (
        "sqlalchemy" in exc_str or "psycopg2" in exc_str or "pymysql" in exc_str
    ):
        hint = (
            "A database driver is missing. For SQLite/PostgreSQL/MySQL support run:\n"
            "  pip install optuna[storages]"
        )
    elif "unable to open database" in exc_str or "no such file" in exc_str:
        hint = (
            f"Cannot open the SQLite database at [bold]{storage}[/bold].\n"
            "Check that the file path is correct and the file is readable."
        )
    elif "connection refused" in exc_str or "could not connect" in exc_str:
        hint = (
            "Cannot connect to the storage backend.\n"
            "Check that the database server is running and the URL is correct."
        )
    else:
        hint = (
            "Verify that the storage URL is correct and all required packages "
            "are installed.\n"
            f"  Storage: {storage or 'in-memory (default)'}"
        )

    _fatal(f"Failed to load study: {exc}", hint=hint)


def _fatal(message: str, hint: Optional[str] = None) -> None:
    """Print a styled error panel and exit with code 1."""
    body = Text.from_markup(f"[bold red]Error:[/bold red] {message}")
    if hint:
        body.append(f"\n\n{hint}")
    _err_console.print()
    _err_console.print(
        Panel(body, border_style="red", padding=(1, 2))
    )
    _err_console.print()
    raise typer.Exit(code=1)


def _warn(message: str) -> None:
    """Print a yellow warning line."""
    _console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def _tip(message: str) -> None:
    """Print a dim informational tip."""
    _console.print(f"[dim]ℹ  {message}[/dim]")
