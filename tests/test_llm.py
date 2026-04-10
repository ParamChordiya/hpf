"""Tests for LLM components (no real API calls)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hpf.llm.base import LLMClient
from hpf.llm.ollama_client import OllamaClient
from hpf.llm.openai_client import OpenAIClient
from hpf.llm.setup_wizard import SetupWizard
from hpf.models import (
    AnalysisResult,
    ParameterScale,
    ParameterStats,
    ParameterSuggestion,
    ParameterType,
    SuggestionAction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis(study_name: str = "my_study") -> AnalysisResult:
    return AnalysisResult(
        n_trials=30,
        n_complete_trials=30,
        n_top_trials=6,
        top_k_percent=20.0,
        direction="minimize",
        best_value=0.05,
        model_type="xgboost",
        study_name=study_name,
        parameters=[
            ParameterStats(
                name="lr",
                param_type=ParameterType.FLOAT,
                original_low=1e-4,
                original_high=1e-1,
                original_choices=None,
                original_scale=ParameterScale.LINEAR,
                best_values=[0.01, 0.02, 0.015],
                all_values=[0.01, 0.02, 0.015, 0.05, 0.08, 0.001],
                best_objective_values=[0.05, 0.06, 0.055],
            ),
            ParameterStats(
                name="max_depth",
                param_type=ParameterType.INT,
                original_low=2.0,
                original_high=10.0,
                original_choices=None,
                original_scale=ParameterScale.LINEAR,
                best_values=[4, 5, 6],
                all_values=[2, 3, 4, 5, 6, 7, 8],
                best_objective_values=[0.05, 0.06, 0.055],
            ),
        ],
    )


def _make_suggestions() -> list[ParameterSuggestion]:
    return [
        ParameterSuggestion(
            name="lr",
            param_type=ParameterType.FLOAT,
            action=SuggestionAction.KEEP,
            original_low=1e-4,
            original_high=1e-1,
            original_scale=ParameterScale.LINEAR,
            suggested_low=1e-4,
            suggested_high=1e-1,
            suggested_scale=ParameterScale.LINEAR,
            original_choices=None,
            confidence="medium",
            reasoning="Well distributed across range.",
        ),
        ParameterSuggestion(
            name="max_depth",
            param_type=ParameterType.INT,
            action=SuggestionAction.NARROW,
            original_low=2.0,
            original_high=10.0,
            original_scale=ParameterScale.LINEAR,
            suggested_low=3.0,
            suggested_high=7.0,
            suggested_scale=ParameterScale.LINEAR,
            original_choices=None,
            confidence="high",
            reasoning="Best trials cluster in [3, 7].",
        ),
    ]


# ---------------------------------------------------------------------------
# Concrete subclass of LLMClient for testing _build_prompt
# ---------------------------------------------------------------------------

class _FakeClient(LLMClient):
    """Minimal concrete subclass that never makes real API calls."""

    def explain(self, analysis, suggestions, model_type=None) -> str:
        return self._build_prompt(analysis, suggestions, model_type)


# ---------------------------------------------------------------------------
# 1. _build_prompt contains study name
# ---------------------------------------------------------------------------

def test_build_prompt_contains_study_name():
    analysis = _make_analysis(study_name="iris_xgb_v2")
    suggestions = _make_suggestions()

    client = _FakeClient()
    prompt = client._build_prompt(analysis, suggestions, model_type=None)

    assert "iris_xgb_v2" in prompt


# ---------------------------------------------------------------------------
# 2. _build_prompt contains parameter names
# ---------------------------------------------------------------------------

def test_build_prompt_contains_param_names():
    analysis = _make_analysis()
    suggestions = _make_suggestions()

    client = _FakeClient()
    prompt = client._build_prompt(analysis, suggestions, model_type=None)

    assert "lr" in prompt
    assert "max_depth" in prompt


# ---------------------------------------------------------------------------
# 3. OllamaClient.list_available_models — unreachable host returns []
# ---------------------------------------------------------------------------

def test_ollama_list_models_unreachable():
    # Port 99999 is guaranteed to be unreachable.
    result = OllamaClient.list_available_models("http://localhost:99999")
    assert result == []


# ---------------------------------------------------------------------------
# 4. OpenAIClient with empty key raises RuntimeError (or returns error string)
# ---------------------------------------------------------------------------

def test_openai_client_missing_key():
    client = OpenAIClient("")
    analysis = _make_analysis()
    suggestions = _make_suggestions()

    # The implementation raises RuntimeError on API failure.
    with pytest.raises(RuntimeError):
        client.explain(analysis, suggestions)


# ---------------------------------------------------------------------------
# 5. SetupWizard.load_config returns None when config file doesn't exist
# ---------------------------------------------------------------------------

def test_setup_wizard_load_config_missing(tmp_path):
    wizard = SetupWizard()
    # Point CONFIG_PATH to a non-existent file inside tmp_path.
    missing_path = tmp_path / "no_config.json"
    wizard.CONFIG_PATH = missing_path

    result = wizard.load_config()
    assert result is None


# ---------------------------------------------------------------------------
# 6. get_client returns None when provider is "none"
# ---------------------------------------------------------------------------

def test_get_client_none_provider():
    wizard = SetupWizard()
    config = {
        "provider": "none",
        "ollama_model": "llama3",
        "ollama_host": "http://localhost:11434",
        "openai_api_key": "",
        "openai_model": "gpt-4o-mini",
        "openai_base_url": None,
    }
    client = wizard.get_client(config)
    assert client is None
