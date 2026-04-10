"""OpenAI-compatible LLM client for HPF.

Works with OpenAI, Groq, Together AI, Anyscale, and any other provider
that exposes an OpenAI-compatible chat-completions endpoint.
"""

from __future__ import annotations

from hpf.llm.base import LLMClient
from hpf.models import AnalysisResult, ParameterSuggestion

_BOUNDARY = "---SYSTEM/USER BOUNDARY---"


class OpenAIClient(LLMClient):
    """Use an OpenAI-compatible API to generate HPF explanations.

    Parameters
    ----------
    api_key:
        The API key for the provider (e.g. ``"sk-..."`` for OpenAI,
        or a Groq / Together key).
    model:
        Model ID recognised by the provider, e.g. ``"gpt-4o-mini"``,
        ``"llama-3-70b-versatile"`` (Groq), ``"mistral-7b-instruct"`` (Together).
    base_url:
        Override the API base URL.  Leave as ``None`` to use OpenAI's
        official endpoint.  Pass e.g. ``"https://api.groq.com/openai/v1"``
        for Groq or ``"https://api.together.xyz/v1"`` for Together AI.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        analysis: AnalysisResult,
        suggestions: list[ParameterSuggestion],
        model_type: str | None = None,
    ) -> str:
        """Generate a Markdown explanation via an OpenAI-compatible API.

        Raises
        ------
        RuntimeError
            When the ``openai`` package is missing or the API call fails.
        """
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is not installed. "
                "Run: pip install openai"
            ) from exc

        prompt = self._build_prompt(analysis, suggestions, model_type)
        system_msg, user_msg = self._split_prompt(prompt)

        # Build client kwargs — only pass base_url when explicitly set.
        client_kwargs: dict[str, object] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        try:
            client = openai.OpenAI(**client_kwargs)  # type: ignore[arg-type]
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,  # Keep output factual and focused
            )
        except openai.AuthenticationError as exc:
            raise RuntimeError(
                "OpenAI authentication failed. Check that your API key is "
                "correct and has not expired."
            ) from exc
        except openai.NotFoundError as exc:
            raise RuntimeError(
                f"Model '{self.model}' was not found. Check the model name "
                "and make sure it is available for your account / provider."
            ) from exc
        except openai.RateLimitError as exc:
            raise RuntimeError(
                "Rate limit exceeded. Wait a moment and retry, or switch to "
                "a model with a higher rate limit."
            ) from exc
        except openai.APIConnectionError as exc:
            endpoint = self.base_url or "https://api.openai.com"
            raise RuntimeError(
                f"Could not connect to the API endpoint ({endpoint}). "
                "Check your network connection and that the base_url is correct."
            ) from exc
        except openai.APIStatusError as exc:
            raise RuntimeError(
                f"API returned an error (HTTP {exc.status_code}): {exc.message}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected error during API call: {exc}") from exc

        content = response.choices[0].message.content or ""
        return content.strip()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _split_prompt(prompt: str) -> tuple[str, str]:
        """Split a combined prompt string into (system, user) parts."""
        if _BOUNDARY in prompt:
            parts = prompt.split(_BOUNDARY, maxsplit=1)
            return parts[0].strip(), parts[1].strip()
        return (
            "You are a helpful ML engineering assistant.",
            prompt.strip(),
        )
