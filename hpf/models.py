"""Shared data models for HPF."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ParameterScale(str, Enum):
    LINEAR = "linear"
    LOG = "log"


class ParameterType(str, Enum):
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"


class SuggestionAction(str, Enum):
    EXPAND = "expand"
    NARROW = "narrow"
    KEEP = "keep"
    LOG_SCALE = "log_scale"
    REMOVE = "remove"


@dataclass
class ParameterStats:
    name: str
    param_type: ParameterType
    original_low: Optional[float]
    original_high: Optional[float]
    original_choices: Optional[list[Any]]
    original_scale: ParameterScale
    best_values: list[float]        # values from top-K trials
    all_values: list[float]         # values from all complete trials
    best_objective_values: list[float]  # objective values for top-K trials


@dataclass
class AnalysisResult:
    n_trials: int
    n_complete_trials: int
    n_top_trials: int
    top_k_percent: float
    direction: str                  # "minimize" or "maximize"
    best_value: float
    model_type: Optional[str]       # e.g. "xgboost", "lightgbm", "neural_net"
    study_name: str
    parameters: list[ParameterStats] = field(default_factory=list)


@dataclass
class ParameterSuggestion:
    name: str
    param_type: ParameterType
    action: SuggestionAction
    original_low: Optional[float]
    original_high: Optional[float]
    original_scale: ParameterScale
    suggested_low: Optional[float]
    suggested_high: Optional[float]
    suggested_scale: ParameterScale
    original_choices: Optional[list[Any]]
    confidence: str                 # "high", "medium", "low"
    reasoning: str                  # short human-readable reason

    @property
    def original_range_str(self) -> str:
        if self.param_type == ParameterType.CATEGORICAL:
            return str(self.original_choices)
        scale = f", log={self.original_scale == ParameterScale.LOG}"
        return f"[{self.original_low}, {self.original_high}{scale}]"

    @property
    def suggested_range_str(self) -> str:
        if self.param_type == ParameterType.CATEGORICAL:
            return str(self.original_choices)
        if self.suggested_low is None or self.suggested_high is None:
            return self.original_range_str
        scale = f", log={self.suggested_scale == ParameterScale.LOG}"
        return f"[{self.suggested_low}, {self.suggested_high}{scale}]"


@dataclass
class HPFReport:
    analysis: AnalysisResult
    suggestions: list[ParameterSuggestion]
    llm_explanation: Optional[str]
    optuna_code: str                # ready-to-use Optuna search space snippet
