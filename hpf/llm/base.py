"""Abstract base class for all HPF LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hpf.models import AnalysisResult, ParameterSuggestion, ParameterType, SuggestionAction

# ---------------------------------------------------------------------------
# Model-specific tips embedded at prompt-build time
# ---------------------------------------------------------------------------

_MODEL_TIPS: dict[str, str] = {
    "xgboost": """
### XGBoost-Specific Guidance
- **learning_rate**: Typical good range is [0.01, 0.3] on a log scale. Values below 0.01 rarely improve
  results and greatly increase training time; values above 0.3 tend to overfit.
- **max_depth**: [3, 10] covers most use-cases. Shallow trees (3–5) generalise better on tabular data;
  deeper trees (6–10) can capture complex interactions but risk overfitting.
- **n_estimators / num_boost_round**: Tune jointly with learning_rate — lower learning rates need more
  rounds. Use early stopping rather than fixing this as a hyperparameter.
- **subsample** and **colsample_bytree**: Keep in [0.5, 1.0]. Values below 0.5 add too much noise.
- **min_child_weight**: [1, 10] on a log scale; increase if you see overfitting on small leaf nodes.
- **gamma / reg_alpha / reg_lambda**: Use log scale for regularisation terms; start with [1e-4, 10].
""",
    "lightgbm": """
### LightGBM-Specific Guidance
- **learning_rate**: [0.01, 0.3] log scale. Pair with `n_estimators` and early stopping.
- **num_leaves**: Key driver of model complexity. [20, 300]; stay below `2^max_depth`.
- **max_depth**: [-1 (unlimited) or 4–12]. With `num_leaves`, prefer controlling via `num_leaves` directly.
- **min_child_samples**: [5, 100]; increase to reduce overfitting on small datasets.
- **feature_fraction** / **bagging_fraction**: [0.5, 1.0]; values below 0.5 are rarely beneficial.
- **reg_alpha** / **reg_lambda**: Log scale [1e-4, 10].
- **subsample_freq**: Set to 1 or small positive int when `bagging_fraction` < 1.
""",
    "catboost": """
### CatBoost-Specific Guidance
- **learning_rate**: [0.01, 0.3] log scale.
- **depth**: [4, 10]; CatBoost is sensitive to depth — keep it ≤ 10.
- **l2_leaf_reg**: Log scale [1e-3, 10].
- **bagging_temperature**: [0, 1]; controls Bayesian bootstrap strength.
- **random_strength**: [1e-3, 10] log scale; controls overfitting via leaf-value randomisation.
- **border_count**: [32, 255]; higher values slow training; 128 is a solid default.
""",
    "neural_net": """
### Neural Network-Specific Guidance
- **learning_rate**: [1e-5, 1e-1] log scale is almost always the right regime.
- **batch_size**: Powers of 2 (32–512) as a categorical or integer on log scale.
- **dropout**: [0.0, 0.5]; values above 0.5 starve the network of information.
- **weight_decay / l2**: Log scale [1e-6, 1e-2].
- **hidden_size / n_units**: Log scale [32, 1024]; tune depth and width jointly.
- **n_layers**: [1, 5] for most tasks; deeper networks need more regularisation.
- **activation**: Categorical — ReLU, GELU, and SiLU are usually the strongest contenders.
- **optimizer**: Adam / AdamW dominate; SGD with momentum can match with proper tuning.
""",
    "random_forest": """
### Random Forest-Specific Guidance
- **n_estimators**: [50, 500]; more trees rarely hurt but yield diminishing returns past ~300.
- **max_depth**: [None (unlimited), 5, 10, 20] as categorical, or integer [3, 30].
- **min_samples_split**: [2, 20]; increase to reduce overfitting.
- **min_samples_leaf**: [1, 10]; increase for smoother decision boundaries.
- **max_features**: ["sqrt", "log2", 0.3–0.8 float] — "sqrt" is a reliable default for classification.
- **bootstrap**: Usually True; set False if the dataset is small.
""",
    "svm": """
### SVM-Specific Guidance
- **C**: Log scale [1e-3, 1e3]; controls regularisation strength.
- **gamma** (RBF kernel): Log scale [1e-5, 1]; "scale" and "auto" heuristics are good starting points.
- **kernel**: Categorical ["rbf", "poly", "linear"] — RBF covers most non-linear problems.
- **degree** (poly kernel): [2, 5]; degree 3 handles most cases.
- **epsilon** (SVR): Log scale [1e-3, 1].
""",
}


class LLMClient(ABC):
    """Abstract base for all HPF LLM back-ends.

    Concrete subclasses only need to implement :meth:`explain`; they should
    call :meth:`_build_prompt` to obtain the ready-made prompt string.
    """

    @abstractmethod
    def explain(
        self,
        analysis: AnalysisResult,
        suggestions: list[ParameterSuggestion],
        model_type: str | None = None,
    ) -> str:
        """Return a rich Markdown explanation for the ML engineer.

        Parameters
        ----------
        analysis:
            Full statistical summary of the Optuna study.
        suggestions:
            Per-parameter suggestions produced by the HPF suggester.
        model_type:
            Optional free-form model label (e.g. "xgboost", "lightgbm").
            When provided, model-specific tips are appended to the prompt.
        """
        ...

    # ------------------------------------------------------------------
    # Shared prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        analysis: AnalysisResult,
        suggestions: list[ParameterSuggestion],
        model_type: str | None,
    ) -> str:
        """Build the full system + user prompt string.

        The returned string is formatted so it can be sent either as a single
        user message (Ollama) or split at the ``---SYSTEM/USER BOUNDARY---``
        marker by clients that support separate system messages.
        """
        system_block = self._system_message()
        user_block = self._user_message(analysis, suggestions, model_type)
        return f"{system_block}\n\n---SYSTEM/USER BOUNDARY---\n\n{user_block}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _system_message() -> str:
        return (
            "You are a senior ML engineer with deep expertise in hyperparameter "
            "optimisation using Optuna. Your job is to interpret statistical "
            "evidence from a completed hyperparameter search and produce clear, "
            "actionable guidance for another ML engineer who will run the next "
            "search. Be concise, specific, and always ground your recommendations "
            "in the data. Avoid generic ML advice — focus on what *this* study "
            "reveals. Format your response in Markdown with headers and bullet "
            "points."
        )

    @staticmethod
    def _format_action(action: SuggestionAction) -> str:
        labels = {
            SuggestionAction.EXPAND: "Expand range",
            SuggestionAction.NARROW: "Narrow range",
            SuggestionAction.KEEP: "Keep as-is",
            SuggestionAction.LOG_SCALE: "Switch to log scale",
            SuggestionAction.REMOVE: "Remove parameter",
        }
        return labels.get(action, action.value)

    def _user_message(
        self,
        analysis: AnalysisResult,
        suggestions: list[ParameterSuggestion],
        model_type: str | None,
    ) -> str:
        lines: list[str] = []

        # ---- Study summary -----------------------------------------------
        lines.append("## Study Summary")
        lines.append(f"- **Study name**: {analysis.study_name}")
        lines.append(f"- **Direction**: {analysis.direction}")
        lines.append(f"- **Total trials**: {analysis.n_trials}")
        lines.append(f"- **Complete trials**: {analysis.n_complete_trials}")
        lines.append(
            f"- **Top-K cohort**: top {analysis.top_k_percent:.0f}% "
            f"= {analysis.n_top_trials} trials"
        )
        best_val_str = (
            f"{analysis.best_value:.6g}"
            if analysis.best_value == analysis.best_value  # not NaN
            else "N/A"
        )
        lines.append(f"- **Best objective value**: {best_val_str}")
        if model_type:
            lines.append(f"- **Model type**: {model_type}")
        lines.append("")

        # ---- Per-parameter details ----------------------------------------
        lines.append("## Parameter Analysis")
        lines.append(
            "Below is the statistical evidence for each hyperparameter. "
            "Use this to explain *why* each change is recommended.\n"
        )

        for sug in suggestions:
            lines.append(f"### `{sug.name}`")
            lines.append(f"- **Type**: {sug.param_type.value}")
            lines.append(f"- **Original range**: {sug.original_range_str}")
            lines.append(f"- **Suggested range**: {sug.suggested_range_str}")
            lines.append(f"- **Action**: {self._format_action(sug.action)}")
            lines.append(f"- **Confidence**: {sug.confidence}")
            lines.append(f"- **Statistical reasoning**: {sug.reasoning}")

            # Attach raw stats from the matching ParameterStats object if found
            param_stats = next(
                (p for p in analysis.parameters if p.name == sug.name), None
            )
            if param_stats and param_stats.best_values:
                best_vals = param_stats.best_values
                if param_stats.param_type != ParameterType.CATEGORICAL:
                    mn = min(best_vals)
                    mx = max(best_vals)
                    mean = sum(best_vals) / len(best_vals)
                    lines.append(
                        f"- **Top-K value range**: [{mn:.4g}, {mx:.4g}] "
                        f"(mean {mean:.4g}, n={len(best_vals)})"
                    )
            lines.append("")

        # ---- Task description for the LLM ---------------------------------
        lines.append("---")
        lines.append("")
        lines.append(
            "Using the study summary and per-parameter data above, produce a "
            "Markdown report with the following sections:\n"
        )
        lines.append(
            "1. **Study Overview** — 2–4 sentences summarising what the study found "
            "overall (e.g. convergence quality, how many trials ran, best result).\n"
        )
        lines.append(
            "2. **Parameter-by-Parameter Recommendations** — For *each* parameter "
            "listed above, explain in 1–3 sentences *why* the recommended change "
            "makes sense in terms an ML engineer understands. Reference the "
            "statistical evidence (top-K range, confidence, reasoning). Do not just "
            "repeat the raw numbers — interpret them.\n"
        )
        lines.append(
            "3. **What to Do Next** — A prioritised bullet-point checklist of "
            "concrete actions for the next Optuna search. Order from highest to "
            "lowest impact. Include any parameter interactions worth watching.\n"
        )

        if model_type:
            normalised = model_type.lower().replace("-", "_").replace(" ", "_")
            tips = _MODEL_TIPS.get(normalised)
            if tips:
                lines.append(
                    f"4. **{model_type.title()}-Specific Tips** — Incorporate the "
                    "following known-good ranges and caveats into your advice where "
                    "relevant:\n"
                )
                lines.append(tips)
            else:
                lines.append(
                    f"4. **{model_type.title()}-Specific Tips** — Provide 3–5 "
                    f"model-specific tips for tuning {model_type} hyperparameters "
                    "based on your expertise. Focus on typical good ranges and "
                    "common pitfalls.\n"
                )

        lines.append(
            "\nKeep the report concise and actionable. Do not include generic ML "
            "advice that does not relate to this study's results."
        )

        return "\n".join(lines)
