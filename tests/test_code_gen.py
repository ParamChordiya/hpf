"""Tests for hpf.formatters.code_gen.generate_optuna_code."""

import pytest

from hpf.formatters.code_gen import generate_optuna_code
from hpf.models import (
    ParameterScale,
    ParameterSuggestion,
    ParameterType,
    SuggestionAction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_suggestion(
    name: str = "lr",
    low: float = 1e-4,
    high: float = 1e-1,
    action: SuggestionAction = SuggestionAction.KEEP,
    scale: ParameterScale = ParameterScale.LINEAR,
) -> ParameterSuggestion:
    return ParameterSuggestion(
        name=name,
        param_type=ParameterType.FLOAT,
        action=action,
        original_low=low,
        original_high=high,
        original_scale=scale,
        suggested_low=low,
        suggested_high=high,
        suggested_scale=scale,
        original_choices=None,
        confidence="medium",
        reasoning="Test reasoning.",
    )


def _int_suggestion(
    name: str = "max_depth",
    low: float = 2.0,
    high: float = 10.0,
    action: SuggestionAction = SuggestionAction.KEEP,
) -> ParameterSuggestion:
    return ParameterSuggestion(
        name=name,
        param_type=ParameterType.INT,
        action=action,
        original_low=low,
        original_high=high,
        original_scale=ParameterScale.LINEAR,
        suggested_low=low,
        suggested_high=high,
        suggested_scale=ParameterScale.LINEAR,
        original_choices=None,
        confidence="high",
        reasoning="Test reasoning.",
    )


def _cat_suggestion(
    name: str = "optimizer",
    choices: list | None = None,
    action: SuggestionAction = SuggestionAction.KEEP,
) -> ParameterSuggestion:
    if choices is None:
        choices = ["adam", "sgd", "rmsprop"]
    return ParameterSuggestion(
        name=name,
        param_type=ParameterType.CATEGORICAL,
        action=action,
        original_low=None,
        original_high=None,
        original_scale=ParameterScale.LINEAR,
        suggested_low=None,
        suggested_high=None,
        suggested_scale=ParameterScale.LINEAR,
        original_choices=choices,
        confidence="medium",
        reasoning="Test reasoning.",
    )


# ---------------------------------------------------------------------------
# 1. Float param → trial.suggest_float
# ---------------------------------------------------------------------------

def test_float_param_in_output():
    suggestions = [_float_suggestion(name="lr")]
    code = generate_optuna_code(suggestions, study_name="my_study")
    assert "trial.suggest_float" in code


# ---------------------------------------------------------------------------
# 2. Int param → trial.suggest_int
# ---------------------------------------------------------------------------

def test_int_param_in_output():
    suggestions = [_int_suggestion(name="max_depth")]
    code = generate_optuna_code(suggestions, study_name="my_study")
    assert "trial.suggest_int" in code


# ---------------------------------------------------------------------------
# 3. Categorical param → trial.suggest_categorical
# ---------------------------------------------------------------------------

def test_categorical_param_in_output():
    suggestions = [_cat_suggestion(name="optimizer")]
    code = generate_optuna_code(suggestions, study_name="my_study")
    assert "trial.suggest_categorical" in code


# ---------------------------------------------------------------------------
# 4. Log-scale param → log=True in output
# ---------------------------------------------------------------------------

def test_log_scale_in_output():
    suggestion = _float_suggestion(
        name="lr",
        low=1e-5,
        high=1e-1,
        scale=ParameterScale.LOG,
    )
    # Make suggested_scale LOG too
    suggestion.suggested_scale = ParameterScale.LOG
    code = generate_optuna_code([suggestion], study_name="my_study")
    assert "log=True" in code


# ---------------------------------------------------------------------------
# 5. Study name appears in output (as a comment)
# ---------------------------------------------------------------------------

def test_study_name_in_output():
    suggestions = [_float_suggestion()]
    code = generate_optuna_code(suggestions, study_name="awesome_study_42")
    assert "awesome_study_42" in code


# ---------------------------------------------------------------------------
# 6. REMOVE action → parameter is commented out, not a suggest call
# ---------------------------------------------------------------------------

def test_remove_action_commented_out():
    removed = _float_suggestion(name="useless_param", action=SuggestionAction.REMOVE)
    kept = _int_suggestion(name="max_depth", action=SuggestionAction.KEEP)
    code = generate_optuna_code([removed, kept], study_name="my_study")

    # The removed param should appear as a comment, not a suggest call
    lines = code.splitlines()
    for line in lines:
        if "useless_param" in line:
            assert line.strip().startswith("#"), (
                f"Expected 'useless_param' to be commented out, got: {line!r}"
            )

    # The kept param should have a real suggest call
    assert "trial.suggest_int" in code
