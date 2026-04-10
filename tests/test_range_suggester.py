"""Tests for hpf.range_suggester.RangeSuggester."""

import pytest

from hpf.models import (
    AnalysisResult,
    ParameterScale,
    ParameterStats,
    ParameterType,
    SuggestionAction,
)
from hpf.range_suggester import RangeSuggester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis(params: list[ParameterStats], direction: str = "minimize") -> AnalysisResult:
    """Build a minimal AnalysisResult wrapping the given parameter stats."""
    return AnalysisResult(
        n_trials=len(params[0].all_values) if params else 0,
        n_complete_trials=len(params[0].all_values) if params else 0,
        n_top_trials=len(params[0].best_values) if params else 0,
        top_k_percent=20.0,
        direction=direction,
        best_value=0.1,
        model_type=None,
        study_name="test_study",
        parameters=params,
    )


def _float_stats(
    name: str,
    low: float,
    high: float,
    best_values: list[float],
    all_values: list[float] | None = None,
    scale: ParameterScale = ParameterScale.LINEAR,
) -> ParameterStats:
    return ParameterStats(
        name=name,
        param_type=ParameterType.FLOAT,
        original_low=low,
        original_high=high,
        original_choices=None,
        original_scale=scale,
        best_values=best_values,
        all_values=all_values if all_values is not None else best_values,
        best_objective_values=[0.1] * len(best_values),
    )


def _cat_stats(
    name: str,
    choices: list,
    best_values: list,
) -> ParameterStats:
    return ParameterStats(
        name=name,
        param_type=ParameterType.CATEGORICAL,
        original_low=None,
        original_high=None,
        original_choices=choices,
        original_scale=ParameterScale.LINEAR,
        best_values=best_values,
        all_values=best_values,
        best_objective_values=[0.1] * len(best_values),
    )


# ---------------------------------------------------------------------------
# 1. Expand lower boundary
# ---------------------------------------------------------------------------

def test_expand_lower_boundary():
    # Best values all very near the lower bound (0.0) of [0.0, 1.0].
    best = [0.01, 0.02, 0.005, 0.015, 0.008]
    stats = _float_stats("x", low=0.0, high=1.0, best_values=best)
    analysis = _make_analysis([stats])

    suggestion = RangeSuggester().suggest(analysis)[0]

    assert suggestion.action == SuggestionAction.EXPAND
    assert suggestion.suggested_low < stats.original_low


# ---------------------------------------------------------------------------
# 2. Expand upper boundary
# ---------------------------------------------------------------------------

def test_expand_upper_boundary():
    # Best values all very near the upper bound (1.0) of [0.0, 1.0].
    best = [0.99, 0.98, 0.995, 0.985, 0.992]
    stats = _float_stats("x", low=0.0, high=1.0, best_values=best)
    analysis = _make_analysis([stats])

    suggestion = RangeSuggester().suggest(analysis)[0]

    assert suggestion.action == SuggestionAction.EXPAND
    assert suggestion.suggested_high > stats.original_high


# ---------------------------------------------------------------------------
# 3. Narrow — best values tightly clustered in a small subregion
# ---------------------------------------------------------------------------

def test_narrow():
    # Original range [0, 100]; best values all in [40, 45] (~5% of range).
    best = [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 40.5, 43.5]
    all_v = list(range(0, 100)) + best
    stats = _float_stats("x", low=0.0, high=100.0, best_values=best, all_values=all_v)
    analysis = _make_analysis([stats])

    suggestion = RangeSuggester().suggest(analysis)[0]

    assert suggestion.action == SuggestionAction.NARROW
    # New range must be a subset of the original
    assert suggestion.suggested_low >= stats.original_low
    assert suggestion.suggested_high <= stats.original_high
    assert suggestion.suggested_low < suggestion.suggested_high


# ---------------------------------------------------------------------------
# 4. Keep — best values spread across most of the original range
# ---------------------------------------------------------------------------

def test_keep():
    # Best values uniformly spread from 0.1 to 0.9 in [0.0, 1.0].
    import numpy as np
    best = list(np.linspace(0.1, 0.9, 20))
    stats = _float_stats("x", low=0.0, high=1.0, best_values=best)
    analysis = _make_analysis([stats])

    suggestion = RangeSuggester().suggest(analysis)[0]

    assert suggestion.action == SuggestionAction.KEEP


# ---------------------------------------------------------------------------
# 5. Log scale — best values spanning >100x ratio
# ---------------------------------------------------------------------------

def test_log_scale():
    # Best values span from 0.001 to 0.5 on a LINEAR scale → >100x ratio
    best = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    stats = _float_stats(
        "lr", low=0.0001, high=1.0, best_values=best,
        scale=ParameterScale.LINEAR,
    )
    analysis = _make_analysis([stats])

    suggestion = RangeSuggester().suggest(analysis)[0]

    assert suggestion.action == SuggestionAction.LOG_SCALE


# ---------------------------------------------------------------------------
# 6. Categorical narrow — some choices never appear in best
# ---------------------------------------------------------------------------

def test_categorical_narrow():
    choices = ["adam", "sgd", "rmsprop", "adagrad"]
    # Only adam and sgd appear in top trials
    best = ["adam", "adam", "sgd", "adam", "sgd"]
    stats = _cat_stats("optimizer", choices=choices, best_values=best)
    analysis = _make_analysis([stats])

    suggestion = RangeSuggester().suggest(analysis)[0]

    assert suggestion.action == SuggestionAction.NARROW


# ---------------------------------------------------------------------------
# 7. Categorical keep — all choices appear in best
# ---------------------------------------------------------------------------

def test_categorical_keep():
    choices = ["adam", "sgd", "rmsprop"]
    # All three appear
    best = ["adam", "sgd", "rmsprop", "adam", "sgd", "rmsprop"]
    stats = _cat_stats("optimizer", choices=choices, best_values=best)
    analysis = _make_analysis([stats])

    suggestion = RangeSuggester().suggest(analysis)[0]

    assert suggestion.action == SuggestionAction.KEEP


# ---------------------------------------------------------------------------
# 8. No best values → KEEP with "medium" confidence
# ---------------------------------------------------------------------------

def test_no_best_values():
    stats = _float_stats("x", low=0.0, high=1.0, best_values=[], all_values=[0.5])
    analysis = _make_analysis([stats])

    suggestion = RangeSuggester().suggest(analysis)[0]

    assert suggestion.action == SuggestionAction.KEEP
    assert suggestion.confidence == "medium"


# ---------------------------------------------------------------------------
# 9. One suggestion per parameter
# ---------------------------------------------------------------------------

def test_suggest_returns_one_per_param():
    params = [
        _float_stats("lr", 1e-4, 1e-1, [0.01, 0.02, 0.03]),
        _float_stats("weight_decay", 1e-5, 1e-2, [1e-4, 2e-4, 5e-4]),
        _cat_stats("optimizer", ["adam", "sgd"], ["adam", "adam", "sgd"]),
    ]
    analysis = _make_analysis(params)

    suggestions = RangeSuggester().suggest(analysis)

    assert len(suggestions) == len(analysis.parameters)
