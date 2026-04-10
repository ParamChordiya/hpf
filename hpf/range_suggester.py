"""Range suggester — pure-statistics hyperparameter range recommendations.

Takes an AnalysisResult produced by the HPF analyzer and returns a list of
ParameterSuggestion objects.  No LLM is involved; every decision is rule-based
and backed by numpy statistics.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from hpf.models import (
    AnalysisResult,
    ParameterScale,
    ParameterStats,
    ParameterSuggestion,
    ParameterType,
    SuggestionAction,
)


class RangeSuggester:
    """Produce ParameterSuggestion objects from an AnalysisResult.

    Parameters
    ----------
    boundary_threshold:
        Fraction of the original range within which a best-value mean/median
        is considered "hitting" a boundary, triggering an EXPAND action.
    narrow_percentile_low / narrow_percentile_high:
        Percentile band used to measure how much of the original range the
        best trials actually use.  Default 5th–95th.
    expand_factor:
        When a boundary is hit the boundary is pushed outward by
        ``expand_factor * (high - low)``.
    min_coverage:
        If the 5th–95th percentile band of best values covers less than this
        fraction of the original range the parameter is a candidate for
        narrowing.
    """

    def __init__(
        self,
        boundary_threshold: float = 0.1,
        narrow_percentile_low: float = 5.0,
        narrow_percentile_high: float = 95.0,
        expand_factor: float = 2.0,
        min_coverage: float = 0.3,
    ) -> None:
        self.boundary_threshold = boundary_threshold
        self.narrow_percentile_low = narrow_percentile_low
        self.narrow_percentile_high = narrow_percentile_high
        self.expand_factor = expand_factor
        self.min_coverage = min_coverage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self, analysis: AnalysisResult) -> list[ParameterSuggestion]:
        """Return one ParameterSuggestion per parameter in *analysis*."""
        return [self._suggest_for_param(stats) for stats in analysis.parameters]

    # ------------------------------------------------------------------
    # Per-parameter dispatch
    # ------------------------------------------------------------------

    def _suggest_for_param(self, stats: ParameterStats) -> ParameterSuggestion:
        """Route to the correct handler based on parameter type."""
        if stats.param_type == ParameterType.CATEGORICAL:
            return self._suggest_categorical(stats)
        return self._suggest_numeric(stats)

    # ------------------------------------------------------------------
    # Categorical parameters
    # ------------------------------------------------------------------

    def _suggest_categorical(self, stats: ParameterStats) -> ParameterSuggestion:
        choices: list[Any] = stats.original_choices or []
        best_set: set[Any] = set(stats.best_values)
        unused = [c for c in choices if c not in best_set]

        if unused:
            kept = [c for c in choices if c in best_set]
            unused_repr = ", ".join(repr(u) for u in unused)
            kept_repr = ", ".join(repr(k) for k in kept)
            reasoning = (
                f"Choice(s) {unused_repr} never appeared in the top trials. "
                f"Keeping only: [{kept_repr}] reduces search space "
                f"by {len(unused)}/{len(choices)} choices."
            )
            return ParameterSuggestion(
                name=stats.name,
                param_type=stats.param_type,
                action=SuggestionAction.NARROW,
                original_low=None,
                original_high=None,
                original_scale=stats.original_scale,
                suggested_low=None,
                suggested_high=None,
                suggested_scale=stats.original_scale,
                original_choices=choices,
                confidence="high" if len(unused) / max(len(choices), 1) >= 0.5 else "medium",
                reasoning=reasoning,
            )

        reasoning = (
            f"All {len(choices)} choice(s) appear in top trials. "
            "No change recommended."
        )
        return ParameterSuggestion(
            name=stats.name,
            param_type=stats.param_type,
            action=SuggestionAction.KEEP,
            original_low=None,
            original_high=None,
            original_scale=stats.original_scale,
            suggested_low=None,
            suggested_high=None,
            suggested_scale=stats.original_scale,
            original_choices=choices,
            confidence="medium" if len(stats.best_values) < 5 else "high",
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Numeric parameters (FLOAT / INT)
    # ------------------------------------------------------------------

    def _suggest_numeric(self, stats: ParameterStats) -> ParameterSuggestion:
        """Apply the four-rule decision tree for continuous/integer params."""
        low: float = stats.original_low  # type: ignore[assignment]
        high: float = stats.original_high  # type: ignore[assignment]
        best: list[float] = stats.best_values

        # Degenerate cases — nothing to analyze.
        if not best or low is None or high is None:
            return self._keep(stats, low, high, stats.original_scale,
                              "medium", "No best-trial data available.")

        original_range = high - low
        n_best = len(best)
        arr = np.array(best, dtype=float)

        # ── Rule 1: Log-scale switch ──────────────────────────────────
        if self._should_switch_to_log(best, stats.original_scale):
            ratio = float(np.max(arr) / np.min(arr[arr > 0]))
            reasoning = (
                f"Best values span {ratio:.1f}× (min={float(np.min(arr)):.4g}, "
                f"max={float(np.max(arr)):.4g}) across >2 orders of magnitude "
                f"on a linear scale. Switching to log scale will sample these "
                f"regions proportionally."
            )
            return ParameterSuggestion(
                name=stats.name,
                param_type=stats.param_type,
                action=SuggestionAction.LOG_SCALE,
                original_low=low,
                original_high=high,
                original_scale=stats.original_scale,
                suggested_low=low,
                suggested_high=high,
                suggested_scale=ParameterScale.LOG,
                original_choices=None,
                confidence="high",
                reasoning=reasoning,
            )

        # ── Rule 2: Boundary hit → EXPAND ────────────────────────────
        threshold = self.boundary_threshold * original_range
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))

        hit_lower = (mean_val - low < threshold) or (median_val - low < threshold)
        hit_upper = (high - mean_val < threshold) or (high - median_val < threshold)

        if hit_lower or hit_upper:
            delta = self.expand_factor * original_range
            new_low = (low - delta) if hit_lower else low
            new_high = (high + delta) if hit_upper else high

            # Round integers to stay integral.
            if stats.param_type == ParameterType.INT:
                new_low = math.floor(new_low)
                new_high = math.ceil(new_high)

            near_boundary = sum(
                1 for v in best
                if (v - low < threshold) or (high - v < threshold)
            )
            pct_near = near_boundary / n_best
            confidence = "high" if pct_near > 0.6 else "medium"

            parts: list[str] = []
            if hit_lower:
                parts.append(
                    f"mean ({mean_val:.4g}) / median ({median_val:.4g}) within "
                    f"{threshold:.4g} of lower bound {low:.4g}"
                )
            if hit_upper:
                parts.append(
                    f"mean ({mean_val:.4g}) / median ({median_val:.4g}) within "
                    f"{threshold:.4g} of upper bound {high:.4g}"
                )
            reasoning = (
                f"{' and '.join(parts).capitalize()}. "
                f"{pct_near * 100:.0f}% of best trials are near a boundary. "
                f"Expanding from [{low:.4g}, {high:.4g}] to "
                f"[{new_low:.4g}, {new_high:.4g}]."
            )
            return ParameterSuggestion(
                name=stats.name,
                param_type=stats.param_type,
                action=SuggestionAction.EXPAND,
                original_low=low,
                original_high=high,
                original_scale=stats.original_scale,
                suggested_low=new_low,
                suggested_high=new_high,
                suggested_scale=stats.original_scale,
                original_choices=None,
                confidence=confidence,
                reasoning=reasoning,
            )

        # ── Rule 3: Narrow check ─────────────────────────────────────
        p5 = float(np.percentile(arr, self.narrow_percentile_low))
        p95 = float(np.percentile(arr, self.narrow_percentile_high))
        coverage = (p95 - p5) / original_range if original_range > 0 else 1.0

        if coverage < self.min_coverage:
            buffer = 0.10 * (p95 - p5) if p95 > p5 else 0.05 * original_range
            new_low = max(low, p5 - buffer)
            new_high = min(high, p95 + buffer)

            if stats.param_type == ParameterType.INT:
                new_low = math.floor(new_low)
                new_high = math.ceil(new_high)

            savings_pct = (1.0 - (new_high - new_low) / original_range) * 100
            if coverage < 0.15:
                confidence = "high"
            elif coverage < 0.30:
                confidence = "medium"
            else:
                confidence = "medium"

            reasoning = (
                f"Best trials cluster in [{p5:.4g}, {p95:.4g}] "
                f"(original: [{low:.4g}, {high:.4g}]). "
                f"That covers only {coverage * 100:.1f}% of the original range. "
                f"Narrowing to [{new_low:.4g}, {new_high:.4g}] "
                f"saves ~{savings_pct:.0f}% of the search space."
            )
            return ParameterSuggestion(
                name=stats.name,
                param_type=stats.param_type,
                action=SuggestionAction.NARROW,
                original_low=low,
                original_high=high,
                original_scale=stats.original_scale,
                suggested_low=new_low,
                suggested_high=new_high,
                suggested_scale=stats.original_scale,
                original_choices=None,
                confidence=confidence,
                reasoning=reasoning,
            )

        # ── Rule 4: Keep ─────────────────────────────────────────────
        confidence = "medium" if n_best < 5 else "high"
        reasoning = (
            f"Best values are well distributed across [{low:.4g}, {high:.4g}] "
            f"(coverage {coverage * 100:.1f}%, n={n_best}). "
            "No change recommended."
        )
        return self._keep(stats, low, high, stats.original_scale, confidence, reasoning)

    # ------------------------------------------------------------------
    # Log-scale heuristic
    # ------------------------------------------------------------------

    def _should_switch_to_log(
        self, values: list[float], original_scale: ParameterScale
    ) -> bool:
        """Return True if *values* span >2 orders of magnitude on a linear scale.

        Conditions:
        - original scale must be LINEAR (no point switching if already log).
        - all positive values (log of non-positive is undefined).
        - max / min > 100  (i.e. more than 2 decades).
        """
        if original_scale != ParameterScale.LINEAR:
            return False
        if not values:
            return False
        positive = [v for v in values if v > 0]
        if len(positive) < len(values):
            # Non-positive values present; log scale is not applicable.
            return False
        min_v = min(positive)
        max_v = max(positive)
        if min_v == 0:
            return False
        return (max_v / min_v) > 100

    # ------------------------------------------------------------------
    # Optuna code generation
    # ------------------------------------------------------------------

    def _generate_optuna_code(
        self, suggestions: list[ParameterSuggestion], study_name: str
    ) -> str:
        """Render a ready-to-paste Optuna ``objective`` search-space snippet.

        Each ParameterSuggestion is translated to the appropriate
        ``trial.suggest_*`` call.  Parameters with action REMOVE are skipped.
        """
        lines: list[str] = [
            f"# Optuna search space for study: {study_name!r}",
            "# Auto-generated by HPF RangeSuggester — review before use.",
            "",
            "def objective(trial):",
            "    params = {}",
        ]

        for s in suggestions:
            if s.action == SuggestionAction.REMOVE:
                lines.append(
                    f"    # REMOVED: {s.name!r} — {s.reasoning}"
                )
                continue

            name = s.name

            if s.param_type == ParameterType.CATEGORICAL:
                kept_choices: list[Any]
                if s.action == SuggestionAction.NARROW and s.original_choices:
                    # Derive kept choices from the reasoning (best-effort) or
                    # fall back to originals — callers should pass filtered
                    # choices explicitly in a richer model; here we use originals.
                    kept_choices = s.original_choices
                else:
                    kept_choices = s.original_choices or []
                choices_repr = repr(kept_choices)
                lines.append(
                    f"    params[{name!r}] = "
                    f"trial.suggest_categorical({name!r}, {choices_repr})"
                )
                continue

            # Numeric
            low = s.suggested_low if s.suggested_low is not None else s.original_low
            high = s.suggested_high if s.suggested_high is not None else s.original_high
            use_log = s.suggested_scale == ParameterScale.LOG

            if s.param_type == ParameterType.INT:
                log_arg = f", log={use_log}" if use_log else ""
                lines.append(
                    f"    params[{name!r}] = "
                    f"trial.suggest_int({name!r}, {_fmt(low)}, {_fmt(high)}{log_arg})"
                )
            else:
                log_arg = f", log={use_log}" if use_log else ""
                lines.append(
                    f"    params[{name!r}] = "
                    f"trial.suggest_float({name!r}, {_fmt(low)}, {_fmt(high)}{log_arg})"
                )

        lines += [
            "    # … your model training here …",
            "    return score",
            "",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _keep(
        self,
        stats: ParameterStats,
        low: float | None,
        high: float | None,
        scale: ParameterScale,
        confidence: str,
        reasoning: str,
    ) -> ParameterSuggestion:
        return ParameterSuggestion(
            name=stats.name,
            param_type=stats.param_type,
            action=SuggestionAction.KEEP,
            original_low=low,
            original_high=high,
            original_scale=scale,
            suggested_low=low,
            suggested_high=high,
            suggested_scale=scale,
            original_choices=stats.original_choices,
            confidence=confidence,
            reasoning=reasoning,
        )


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def _fmt(value: float | None) -> str:
    """Format a numeric boundary for Optuna code output."""
    if value is None:
        return "None"
    # Use integer notation when the value is a whole number.
    if value == int(value):
        return str(int(value))
    # Scientific notation for very small / very large values.
    if abs(value) < 1e-3 or abs(value) >= 1e6:
        return f"{value:.4e}"
    return f"{value:.6g}"
