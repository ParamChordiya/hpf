"""Interactive setup wizard for HPF's LLM back-end configuration."""

from __future__ import annotations

import getpass
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from hpf.llm.base import LLMClient
from hpf.llm.ollama_client import OllamaClient
from hpf.llm.openai_client import OpenAIClient

_console = Console()

# ---------------------------------------------------------------------------
# Default config schema
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "provider": "none",
    "ollama_model": "llama3",
    "ollama_host": "http://localhost:11434",
    "openai_api_key": "",
    "openai_model": "gpt-4o-mini",
    "openai_base_url": None,
}

_OLLAMA_INSTALL_INSTRUCTIONS = """
[bold yellow]Ollama does not appear to be running.[/bold yellow]

To install and start Ollama:

  [bold]macOS / Linux[/bold]
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve            # start the server (runs in the background)
    ollama pull llama3      # download a model

  [bold]Windows[/bold]
    Download the installer from https://ollama.com/download/windows
    Then run: ollama pull llama3

After starting Ollama, re-run [bold cyan]hpf setup[/bold cyan] to complete configuration.
"""


class SetupWizard:
    """Guide the user through configuring an LLM back-end for HPF.

    Configuration is persisted as JSON at :attr:`CONFIG_PATH`.
    """

    CONFIG_PATH: Path = Path.home() / ".hpf" / "config.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the interactive setup flow.

        Returns the saved configuration dictionary.
        """
        self._print_banner()

        provider = self._ask_provider()

        if provider == "ollama":
            config = self._configure_ollama()
        elif provider == "openai":
            config = self._configure_openai()
        else:
            config = {**_DEFAULT_CONFIG, "provider": "none"}
            _console.print(
                "\n[dim]Skipping LLM setup. HPF will use statistical analysis only.[/dim]"
            )

        self._save_config(config)
        self._print_success(config)
        return config

    def load_config(self) -> dict[str, Any] | None:
        """Load the saved configuration.

        Returns ``None`` when no configuration file exists or the file is
        invalid.
        """
        if not self.CONFIG_PATH.exists():
            return None
        try:
            with self.CONFIG_PATH.open() as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
        # Merge with defaults so older config files gain new keys gracefully.
        merged = {**_DEFAULT_CONFIG, **data}
        return merged

    def test_connection(self, config: dict[str, Any]) -> tuple[bool, str]:
        """Test that the configured LLM provider is reachable.

        Returns
        -------
        (success, message)
            ``success`` is True when the provider responds correctly.
        """
        provider = config.get("provider", "none")

        if provider == "none":
            return True, "No LLM provider configured — nothing to test."

        if provider == "ollama":
            return self._test_ollama(
                host=config.get("ollama_host", _DEFAULT_CONFIG["ollama_host"]),
                model=config.get("ollama_model", _DEFAULT_CONFIG["ollama_model"]),
            )

        if provider == "openai":
            return self._test_openai(
                api_key=config.get("openai_api_key", ""),
                model=config.get("openai_model", _DEFAULT_CONFIG["openai_model"]),
                base_url=config.get("openai_base_url"),
            )

        return False, f"Unknown provider '{provider}'."

    def get_client(self, config: dict[str, Any] | None = None) -> LLMClient | None:
        """Return the appropriate :class:`~hpf.llm.base.LLMClient` for *config*.

        If *config* is ``None``, :meth:`load_config` is called automatically.
        Returns ``None`` when no provider is configured or the config is
        missing.
        """
        if config is None:
            config = self.load_config()
        if config is None:
            return None

        provider = config.get("provider", "none")

        if provider == "ollama":
            return OllamaClient(
                model=config.get("ollama_model", _DEFAULT_CONFIG["ollama_model"]),
                host=config.get("ollama_host", _DEFAULT_CONFIG["ollama_host"]),
            )

        if provider == "openai":
            api_key = config.get("openai_api_key", "")
            if not api_key:
                return None
            return OpenAIClient(
                api_key=api_key,
                model=config.get("openai_model", _DEFAULT_CONFIG["openai_model"]),
                base_url=config.get("openai_base_url") or None,
            )

        return None

    # ------------------------------------------------------------------
    # Banner & prompts
    # ------------------------------------------------------------------

    @staticmethod
    def _print_banner() -> None:
        title = Text("HPF — Hyperparameter Finder", style="bold cyan")
        subtitle = Text("LLM Setup Wizard", style="dim")
        content = Text.assemble(title, "\n", subtitle, justify="center")
        _console.print(Panel(content, padding=(1, 4)))
        _console.print()

    @staticmethod
    def _ask_provider() -> str:
        _console.print("[bold]How would you like to use AI explanations?[/bold]\n")
        _console.print("  [bold cyan][1][/bold cyan]  Local model via Ollama  [dim](free, private)[/dim]")
        _console.print("  [bold cyan][2][/bold cyan]  OpenAI API or compatible  [dim](e.g. Groq, Together)[/dim]")
        _console.print("  [bold cyan][3][/bold cyan]  Skip — use statistical analysis only")
        _console.print()

        while True:
            choice = Prompt.ask(
                "[bold]Enter choice[/bold]",
                choices=["1", "2", "3"],
                default="1",
            )
            if choice == "1":
                return "ollama"
            if choice == "2":
                return "openai"
            return "none"

    # ------------------------------------------------------------------
    # Ollama flow
    # ------------------------------------------------------------------

    def _configure_ollama(self) -> dict[str, Any]:
        _console.print()
        _console.print(Rule("[bold]Ollama Configuration[/bold]"))
        _console.print()

        host = Prompt.ask(
            "Ollama host URL",
            default=_DEFAULT_CONFIG["ollama_host"],
        )

        _console.print(f"\n[dim]Checking Ollama at {host}...[/dim]")
        available_models = OllamaClient.list_available_models(host=host)

        if not available_models:
            _console.print()
            _console.print(_OLLAMA_INSTALL_INSTRUCTIONS)
            # Still allow the user to save a config for when they start Ollama later.
            model = Prompt.ask(
                "Model name to use once Ollama is running",
                default=_DEFAULT_CONFIG["ollama_model"],
            )
            return {
                **_DEFAULT_CONFIG,
                "provider": "ollama",
                "ollama_host": host,
                "ollama_model": model,
            }

        _console.print(
            f"\n[green]Ollama is running[/green] — "
            f"found [bold]{len(available_models)}[/bold] model(s):\n"
        )
        for i, name in enumerate(available_models, start=1):
            _console.print(f"  [bold cyan][{i}][/bold cyan]  {name}")
        _console.print(f"  [bold cyan][{len(available_models) + 1}][/bold cyan]  Enter a custom model name")
        _console.print()

        model = self._pick_model(available_models)

        _console.print(f"\n[dim]Testing model '{model}'...[/dim]")
        success, msg = self._test_ollama(host=host, model=model)
        if success:
            _console.print(f"[green]Connection test passed.[/green] {msg}")
        else:
            _console.print(f"[yellow]Warning:[/yellow] {msg}")
            _console.print("[dim]Config will be saved anyway — fix any issues and retry.[/dim]")

        return {
            **_DEFAULT_CONFIG,
            "provider": "ollama",
            "ollama_host": host,
            "ollama_model": model,
        }

    @staticmethod
    def _pick_model(available: list[str]) -> str:
        """Let the user select from the list or type a custom name."""
        custom_idx = len(available) + 1

        while True:
            raw = Prompt.ask(
                f"Select model [1-{custom_idx}] or press Enter for default",
                default="1",
            )
            raw = raw.strip()
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(available):
                    return available[idx - 1]
                if idx == custom_idx:
                    return Prompt.ask("Enter custom model name", default="llama3")
            # Treat non-numeric input as a direct model name.
            if raw:
                return raw
            return available[0] if available else "llama3"

    # ------------------------------------------------------------------
    # OpenAI flow
    # ------------------------------------------------------------------

    def _configure_openai(self) -> dict[str, Any]:
        _console.print()
        _console.print(Rule("[bold]OpenAI-Compatible API Configuration[/bold]"))
        _console.print()
        _console.print(
            "[dim]This works with OpenAI, Groq, Together AI, Anyscale, and any "
            "OpenAI-compatible endpoint.[/dim]\n"
        )

        # API key — masked input
        api_key = self._ask_api_key()

        # Base URL
        _console.print()
        _console.print("[dim]Leave blank to use the official OpenAI API (api.openai.com).[/dim]")
        _console.print("[dim]Examples:[/dim]")
        _console.print("[dim]  Groq:      https://api.groq.com/openai/v1[/dim]")
        _console.print("[dim]  Together:  https://api.together.xyz/v1[/dim]")
        raw_url = Prompt.ask("\nBase URL (press Enter to use OpenAI default)", default="")
        base_url: str | None = raw_url.strip() or None

        # Model name
        _console.print()
        default_model = "gpt-4o-mini" if base_url is None else "llama-3-70b-versatile"
        model = Prompt.ask("Model name", default=default_model)

        _console.print(f"\n[dim]Testing connection to '{model}'...[/dim]")
        success, msg = self._test_openai(api_key=api_key, model=model, base_url=base_url)
        if success:
            _console.print(f"[green]Connection test passed.[/green] {msg}")
        else:
            _console.print(f"[yellow]Warning:[/yellow] {msg}")
            _console.print("[dim]Config will be saved anyway — fix any issues and retry.[/dim]")

        return {
            **_DEFAULT_CONFIG,
            "provider": "openai",
            "openai_api_key": api_key,
            "openai_model": model,
            "openai_base_url": base_url,
        }

    @staticmethod
    def _ask_api_key() -> str:
        """Prompt for an API key with masked echo."""
        _console.print("[bold]API key[/bold] [dim](input will be hidden)[/dim]")
        try:
            # getpass hides input on all platforms.
            key = getpass.getpass(prompt="  API key: ")
        except (EOFError, KeyboardInterrupt):
            key = ""
        return key.strip()

    # ------------------------------------------------------------------
    # Connection tests
    # ------------------------------------------------------------------

    @staticmethod
    def _test_ollama(host: str, model: str) -> tuple[bool, str]:
        """Send a minimal chat request to verify Ollama is working."""
        try:
            import ollama  # type: ignore[import-untyped]
        except ImportError:
            return False, "The 'ollama' Python package is not installed (pip install ollama)."

        try:
            client = ollama.Client(host=host)
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            )
            # Any successful response means the connection works.
            _ = response  # result not used
            return True, f"Model '{model}' responded successfully."
        except Exception as exc:  # noqa: BLE001
            exc_str = str(exc)
            if "connection" in exc_str.lower() or "refused" in exc_str.lower():
                return False, f"Cannot reach Ollama at {host}. Is it running?"
            if "not found" in exc_str.lower() or "does not exist" in exc_str.lower():
                return (
                    False,
                    f"Model '{model}' is not available locally. "
                    f"Run: ollama pull {model}",
                )
            return False, f"Ollama error: {exc}"

    @staticmethod
    def _test_openai(
        api_key: str,
        model: str,
        base_url: str | None,
    ) -> tuple[bool, str]:
        """Send a minimal chat request to verify the OpenAI-compatible API."""
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError:
            return False, "The 'openai' Python package is not installed (pip install openai)."

        client_kwargs: dict[str, object] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        try:
            client = openai.OpenAI(**client_kwargs)  # type: ignore[arg-type]
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
                max_tokens=5,
                temperature=0,
            )
            reply = (response.choices[0].message.content or "").strip()
            return True, f"Model '{model}' responded: '{reply}'."
        except openai.AuthenticationError:
            return False, "Authentication failed — check your API key."
        except openai.NotFoundError:
            return False, f"Model '{model}' not found. Check the model name."
        except openai.RateLimitError:
            return False, "Rate limit hit — but the key and model appear valid."
        except openai.APIConnectionError:
            endpoint = base_url or "https://api.openai.com"
            return False, f"Cannot connect to {endpoint}. Check network / base URL."
        except openai.APIStatusError as exc:
            return False, f"API error {exc.status_code}: {exc.message}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Unexpected error: {exc}"

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _save_config(self, config: dict[str, Any]) -> None:
        """Write *config* to :attr:`CONFIG_PATH` as JSON."""
        self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self.CONFIG_PATH.open("w") as fh:
            json.dump(config, fh, indent=2)
            fh.write("\n")

    def _print_success(self, config: dict[str, Any]) -> None:
        provider = config.get("provider", "none")
        _console.print()
        _console.print(Rule())

        if provider == "ollama":
            detail = (
                f"[bold]Provider:[/bold] Ollama\n"
                f"[bold]Model:[/bold]    {config['ollama_model']}\n"
                f"[bold]Host:[/bold]     {config['ollama_host']}"
            )
        elif provider == "openai":
            key_preview = config.get("openai_api_key", "")
            masked = (
                key_preview[:4] + "..." + key_preview[-4:]
                if len(key_preview) > 8
                else "****"
            )
            base = config.get("openai_base_url") or "https://api.openai.com"
            detail = (
                f"[bold]Provider:[/bold] OpenAI-compatible\n"
                f"[bold]Model:[/bold]    {config['openai_model']}\n"
                f"[bold]Endpoint:[/bold] {base}\n"
                f"[bold]API key:[/bold]  {masked}"
            )
        else:
            detail = "[bold]Provider:[/bold] None (statistical analysis only)"

        _console.print(
            Panel(
                f"[green bold]Setup complete![/green bold]\n\n{detail}\n\n"
                f"[dim]Config saved to {self.CONFIG_PATH}[/dim]",
                title="[green]HPF LLM Ready[/green]",
                padding=(1, 4),
            )
        )
        _console.print()
