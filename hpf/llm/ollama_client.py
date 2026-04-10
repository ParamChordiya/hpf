"""Ollama-backed LLM client for HPF."""

from __future__ import annotations

from hpf.llm.base import LLMClient
from hpf.models import AnalysisResult, ParameterSuggestion

_BOUNDARY = "---SYSTEM/USER BOUNDARY---"


class OllamaClient(LLMClient):
    """Use a locally-running Ollama instance to generate HPF explanations.

    Parameters
    ----------
    model:
        The Ollama model tag to use, e.g. ``"llama3"``, ``"mistral"``,
        ``"deepseek-coder"``.
    host:
        Base URL of the Ollama API server.
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.host = host

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        analysis: AnalysisResult,
        suggestions: list[ParameterSuggestion],
        model_type: str | None = None,
    ) -> str:
        """Generate a Markdown explanation via Ollama chat.

        Raises
        ------
        RuntimeError
            When Ollama is not reachable or the model returns an error.
        """
        try:
            import ollama  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'ollama' package is not installed. "
                "Run: pip install ollama"
            ) from exc

        prompt = self._build_prompt(analysis, suggestions, model_type)
        system_msg, user_msg = self._split_prompt(prompt)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            client = ollama.Client(host=self.host)
            response = client.chat(model=self.model, messages=messages)
        except Exception as exc:  # noqa: BLE001
            # Catch all connection/API errors and surface a clear message.
            exc_str = str(exc)
            if "connection" in exc_str.lower() or "refused" in exc_str.lower():
                raise RuntimeError(
                    f"Could not connect to Ollama at {self.host}. "
                    "Make sure Ollama is running (`ollama serve`) and that "
                    f"the model '{self.model}' has been pulled "
                    f"(`ollama pull {self.model}`)."
                ) from exc
            raise RuntimeError(
                f"Ollama request failed: {exc}"
            ) from exc

        # The ollama package returns either a dict-like object or a typed
        # response; handle both gracefully.
        try:
            content: str = response["message"]["content"]
        except (TypeError, KeyError):
            try:
                content = response.message.content  # type: ignore[union-attr]
            except AttributeError:
                content = str(response)

        return content.strip()

    @staticmethod
    def list_available_models(host: str = "http://localhost:11434") -> list[str]:
        """Return a list of locally available Ollama model names.

        Returns an empty list (rather than raising) when Ollama is not running,
        so callers can check ``len(models) == 0`` as a reachability probe.
        """
        try:
            import ollama  # type: ignore[import-untyped]
        except ImportError:
            return []

        try:
            client = ollama.Client(host=host)
            result = client.list()
        except Exception:  # noqa: BLE001
            return []

        # result is either a dict with a "models" key or a typed object.
        try:
            models_raw = result["models"]
        except (TypeError, KeyError):
            try:
                models_raw = result.models  # type: ignore[union-attr]
            except AttributeError:
                return []

        names: list[str] = []
        for entry in models_raw:
            try:
                name = entry["name"]
            except (TypeError, KeyError):
                try:
                    name = entry.name  # type: ignore[union-attr]
                except AttributeError:
                    continue
            # Strip the ":latest" tag for cleaner display, but keep custom tags.
            if name.endswith(":latest"):
                name = name[: -len(":latest")]
            names.append(name)

        return sorted(names)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _split_prompt(prompt: str) -> tuple[str, str]:
        """Split a combined prompt string into (system, user) parts."""
        if _BOUNDARY in prompt:
            parts = prompt.split(_BOUNDARY, maxsplit=1)
            return parts[0].strip(), parts[1].strip()
        # Fallback: treat the whole prompt as the user message.
        return (
            "You are a helpful ML engineering assistant.",
            prompt.strip(),
        )
