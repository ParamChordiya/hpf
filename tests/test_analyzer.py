"""Tests for hpf.analyzer.StudyAnalyzer."""

import math

import optuna
import pytest

optuna.logging.set_verbosity(optuna.logging.WARNING)

from hpf.analyzer import StudyAnalyzer
from hpf.models import ParameterScale, ParameterType


def make_study(direction="minimize"):
    study = optuna.create_study(direction=direction)
    return study


# ---------------------------------------------------------------------------
# 1. Empty study
# ---------------------------------------------------------------------------

def test_analyze_empty_study():
    study = make_study()
    result = StudyAnalyzer().analyze(study)

    assert result.n_complete_trials == 0
    assert result.parameters == []
    assert math.isnan(result.best_value)


# ---------------------------------------------------------------------------
# 2. Basic float parameter
# ---------------------------------------------------------------------------

def test_analyze_basic_float():
    study = make_study()

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-1)
        n_est = trial.suggest_int("n_estimators", 50, 500)
        return lr * n_est

    study.optimize(objective, n_trials=30)

    result = StudyAnalyzer().analyze(study)

    assert result.n_complete_trials == 30
    param_names = [p.name for p in result.parameters]
    assert "lr" in param_names
    lr_stat = next(p for p in result.parameters if p.name == "lr")
    assert lr_stat.param_type == ParameterType.FLOAT


# ---------------------------------------------------------------------------
# 3. Top-K floored to 3 when study is small
# ---------------------------------------------------------------------------

def test_analyze_top_k_minimum_3():
    study = make_study()

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        return x

    study.optimize(objective, n_trials=5)

    result = StudyAnalyzer(top_k_percent=20.0).analyze(study)
    # 20% of 5 = 1; floor is 3
    assert result.n_top_trials == 3


# ---------------------------------------------------------------------------
# 4. Maximize direction — best trials are highest values
# ---------------------------------------------------------------------------

def test_analyze_direction_maximize():
    study = make_study(direction="maximize")

    def objective(trial):
        x = trial.suggest_float("score", 0.0, 1.0)
        return x

    study.optimize(objective, n_trials=20)

    result = StudyAnalyzer(top_k_percent=20.0).analyze(study)

    assert result.direction == "maximize"
    # best_value should be the maximum
    assert result.best_value == pytest.approx(study.best_value)

    # best_values in top-K stats should be >= median of all values
    score_stat = next(p for p in result.parameters if p.name == "score")
    import statistics
    all_vals = score_stat.all_values
    top_vals = score_stat.best_values
    assert min(top_vals) >= statistics.median(all_vals) - 1e-6  # top vals should be high


# ---------------------------------------------------------------------------
# 5. Categorical parameter
# ---------------------------------------------------------------------------

def test_analyze_categorical_param():
    study = make_study()

    def objective(trial):
        opt = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
        return {"adam": 0.1, "sgd": 0.3, "rmsprop": 0.2}[opt]

    study.optimize(objective, n_trials=15)

    result = StudyAnalyzer().analyze(study)
    opt_stat = next(p for p in result.parameters if p.name == "optimizer")
    assert opt_stat.param_type == ParameterType.CATEGORICAL
    assert opt_stat.original_choices is not None
    assert set(opt_stat.original_choices) == {"adam", "sgd", "rmsprop"}


# ---------------------------------------------------------------------------
# 6. Int parameter
# ---------------------------------------------------------------------------

def test_analyze_int_param():
    study = make_study()

    def objective(trial):
        depth = trial.suggest_int("max_depth", 2, 10)
        return float(depth)

    study.optimize(objective, n_trials=20)

    result = StudyAnalyzer().analyze(study)
    depth_stat = next(p for p in result.parameters if p.name == "max_depth")
    assert depth_stat.param_type == ParameterType.INT
    assert depth_stat.original_low == 2.0
    assert depth_stat.original_high == 10.0


# ---------------------------------------------------------------------------
# 7. Log-scale float parameter
# ---------------------------------------------------------------------------

def test_analyze_log_scale_param():
    study = make_study()

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        return lr

    study.optimize(objective, n_trials=20)

    result = StudyAnalyzer().analyze(study)
    lr_stat = next(p for p in result.parameters if p.name == "lr")
    assert lr_stat.original_scale == ParameterScale.LOG


# ---------------------------------------------------------------------------
# 8. Invalid top_k_percent raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_top_k_percent_zero():
    with pytest.raises(ValueError):
        StudyAnalyzer(top_k_percent=0)


def test_invalid_top_k_percent_over_100():
    with pytest.raises(ValueError):
        StudyAnalyzer(top_k_percent=101)


# ---------------------------------------------------------------------------
# 9. all_values length == n_complete_trials
# ---------------------------------------------------------------------------

def test_analyze_all_values_collected():
    study = make_study()
    n_trials = 25

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        return x

    study.optimize(objective, n_trials=n_trials)

    result = StudyAnalyzer().analyze(study)
    assert result.n_complete_trials == n_trials

    x_stat = next(p for p in result.parameters if p.name == "x")
    assert len(x_stat.all_values) == n_trials
