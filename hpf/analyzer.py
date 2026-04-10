"""Core statistical analysis engine for HPF.

Analyses a completed Optuna study and produces per-parameter statistics
summarised in an AnalysisResult.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from hpf.models import (
    AnalysisResult,
    ParameterScale,
    ParameterStats,
    ParameterType,
)

if TYPE_CHECKING:
    pass  # optuna already imported at runtime


class StudyAnalyzer:
    """Analyse a completed Optuna study and return rich per-parameter stats.

    Parameters
    ----------
    top_k_percent:
        Percentage of the best-performing trials to treat as the "top-K"
        cohort.  Floored to a minimum of 3 trials when the study is small.
    """

    def __init__(self, top_k_percent: float = 20.0) -> None:
        if not (0.0 < top_k_percent <= 100.0):
            raise ValueError(
                f"top_k_percent must be in (0, 100], got {top_k_percent}"
            )
        self.top_k_percent = top_k_percent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        study: optuna.Study,
        model_type: Optional[str] = None,
    ) -> AnalysisResult:
        """Analyse *study* and return an :class:`~hpf.models.AnalysisResult`.

        Parameters
        ----------
        study:
            A finished (or partially finished) Optuna study.  Only
            ``TrialState.COMPLETE`` trials are considered.
        model_type:
            Optional free-form label, e.g. ``"xgboost"`` or ``"lightgbm"``.
        """
        complete_trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        n_trials = len(study.trials)
        n_complete = len(complete_trials)

        if n_complete == 0:
            # Return an empty result — nothing to analyse.
            return AnalysisResult(
                n_trials=n_trials,
                n_complete_trials=0,
                n_top_trials=0,
                top_k_percent=self.top_k_percent,
                direction=study.direction.name.lower(),
                best_value=float("nan"),
                model_type=model_type,
                study_name=study.study_name,
                parameters=[],
            )

        # Sort trials so that the "best" ones come first.
        minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
        sorted_trials = sorted(
            complete_trials,
            key=lambda t: t.value,  # type: ignore[arg-type]
            reverse=not minimize,
        )

        n_top = max(3, math.ceil(n_complete * self.top_k_percent / 100.0))
        n_top = min(n_top, n_complete)  # can't exceed what we have
        top_trials = sorted_trials[:n_top]

        best_value = sorted_trials[0].value  # type: ignore[assignment]

        # Collect the union of all parameter names across complete trials.
        all_param_names: set[str] = set()
        for trial in complete_trials:
            all_param_names.update(trial.params.keys())

        # Build ParameterStats for every discovered parameter.
        param_stats_list: list[ParameterStats] = []
        for name in sorted(all_param_names):
            stats = self._build_param_stats(
                name=name,
                top_trials=top_trials,
                all_complete_trials=complete_trials,
            )
            if stats is not None:
                param_stats_list.append(stats)

        return AnalysisResult(
            n_trials=n_trials,
            n_complete_trials=n_complete,
            n_top_trials=n_top,
            top_k_percent=self.top_k_percent,
            direction=study.direction.name.lower(),
            best_value=best_value,
            model_type=model_type,
            study_name=study.study_name,
            parameters=param_stats_list,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_param_stats(
        self,
        name: str,
        top_trials: list[optuna.trial.FrozenTrial],
        all_complete_trials: list[optuna.trial.FrozenTrial],
    ) -> Optional[ParameterStats]:
        """Build a :class:`~hpf.models.ParameterStats` for *name*.

        Returns ``None`` when no trial (top-K or otherwise) contains the
        parameter — this can happen in dynamic search spaces.
        """
        # Locate the first available distribution for this parameter so we
        # can determine its type and bounds.  Trials that don't use the param
        # are skipped gracefully.
        distribution: Any = None
        for trial in all_complete_trials:
            if name in trial.distributions:
                distribution = trial.distributions[name]
                break

        if distribution is None:
            # Parameter name was seen in params but distributions is somehow
            # missing — skip rather than crash.
            return None

        param_type, original_scale, low, high, choices = (
            self._parse_distribution(distribution)
        )

        # Collect values only from trials that actually sampled this param.
        best_values: list[float] = []
        best_objective_values: list[float] = []
        for trial in top_trials:
            if name in trial.params:
                best_values.append(trial.params[name])
                best_objective_values.append(trial.value)  # type: ignore[arg-type]

        all_values: list[float] = []
        for trial in all_complete_trials:
            if name in trial.params:
                all_values.append(trial.params[name])

        return ParameterStats(
            name=name,
            param_type=param_type,
            original_low=low,
            original_high=high,
            original_choices=choices,
            original_scale=original_scale,
            best_values=best_values,
            all_values=all_values,
            best_objective_values=best_objective_values,
        )

    @staticmethod
    def _parse_distribution(
        distribution: Any,
    ) -> tuple[
        ParameterType,
        ParameterScale,
        Optional[float],
        Optional[float],
        Optional[list[Any]],
    ]:
        """Extract type, scale, and bounds from an Optuna distribution object.

        Returns
        -------
        param_type, original_scale, low, high, choices
        """
        if isinstance(distribution, FloatDistribution):
            scale = (
                ParameterScale.LOG if distribution.log else ParameterScale.LINEAR
            )
            return (
                ParameterType.FLOAT,
                scale,
                float(distribution.low),
                float(distribution.high),
                None,
            )

        if isinstance(distribution, IntDistribution):
            scale = (
                ParameterScale.LOG if distribution.log else ParameterScale.LINEAR
            )
            return (
                ParameterType.INT,
                scale,
                float(distribution.low),
                float(distribution.high),
                None,
            )

        if isinstance(distribution, CategoricalDistribution):
            return (
                ParameterType.CATEGORICAL,
                ParameterScale.LINEAR,  # scale is meaningless for categoricals
                None,
                None,
                list(distribution.choices),
            )

        # Fallback for any future / unknown distribution type.
        low = getattr(distribution, "low", None)
        high = getattr(distribution, "high", None)
        log = getattr(distribution, "log", False)
        scale = ParameterScale.LOG if log else ParameterScale.LINEAR
        return (
            ParameterType.FLOAT,
            scale,
            float(low) if low is not None else None,
            float(high) if high is not None else None,
            None,
        )
