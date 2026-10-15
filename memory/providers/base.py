"""The LLM provider interface: one narrow protocol, no vendor inside it.

Phase 4's intelligence features (extraction, conflict adjudication) want a
language model, but the system must remain fully usable without one. So the
contract here is deliberately minimal -- a single ``complete`` call that
takes a prompt and returns text -- and every caller is written to work when
:func:`load_provider` returns ``None``.

Three rules this module exists to enforce:

1. **No vendor is hardcoded and nothing is called by default.** The built-in
   providers are ``none`` (refuses, with an actionable message) and ``echo``
   (deterministic, offline, for tests). An HTTP provider exists but requires
   an explicit ``base_url``, so no network call can happen by accident.
2. **Configuration is data, not code.** A JSON file under ``data/config/``
   or environment variables; both are inspectable with ``mem provider show``.
3. **Absence is a first-class state.** ``load_provider`` returning ``None``
   is normal, not an error. Callers either degrade to a deterministic path or
   raise :class:`ProviderUnavailable`, whose message says exactly what to set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: Environment variables, so a provider can be configured without a file.
ENV_PROVIDER = "MEMORY_LLM_PROVIDER"
ENV_MODEL = "MEMORY_LLM_MODEL"
ENV_BASE_URL = "MEMORY_LLM_BASE_URL"
ENV_API_KEY = "MEMORY_LLM_API_KEY"
ENV_TIMEOUT = "MEMORY_LLM_TIMEOUT"

#: Config file, relative to ``data/``. Not under ``index/`` -- it is input,
#: not derived state, so a rebuild must never touch it.
CONFIG_RELPATH = "config/llm.json"


class ProviderError(RuntimeError):
    """A provider was configured but could not produce a completion."""


class ProviderUnavailable(ProviderError):
    """No provider is configured, and this operation requires one.

    The message is the whole point: it must tell the user what to set, since
    this is the error they will actually hit.
    """

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(
            f"{operation} requires an LLM provider, and none is configured. "
            f"Either set ${ENV_PROVIDER} (with ${ENV_BASE_URL} and ${ENV_MODEL} "
            f"for an OpenAI-compatible local runtime), write data/{CONFIG_RELPATH}, "
            f"or run the deterministic path instead (see --no-llm)."
        )


@dataclass(slots=True)
class Completion:
    """One model response. ``text`` is all any caller is allowed to need."""

    text: str
    model: str | None = None
    provider: str = "unknown"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "chars": len(self.text),
        }


@runtime_checkable
class LLMProvider(Protocol):
    """What Phase 4 needs from a language model, and nothing more.

    ``complete`` is synchronous on purpose: there is no server here, commands
    are short-lived, and an async surface would leak into the CLI for no gain.
    """

    name: str
    model: str | None

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class ProviderConfig:
    """Resolved provider settings, with where each value came from."""

    provider: str = "none"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    options: dict[str, Any] = field(default_factory=dict)
    origin: str = "default"

    def to_json(self) -> dict[str, Any]:
        # The key is never echoed, only whether one is present: this payload
        # goes to stdout, where an agent or a log may capture it.
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_set": bool(self.api_key),
            "timeout": self.timeout,
            "options": self.options,
            "origin": self.origin,
        }


def config_path(data_dir: Path) -> Path:
    return data_dir / CONFIG_RELPATH


def read_config(data_dir: Path | None = None) -> ProviderConfig:
    """Resolve provider settings from the config file, then the environment.

    Environment wins per field, so a shell can point one command at a
    different model without editing the file.
    """
    cfg = ProviderConfig()
    origins: list[str] = []

    if data_dir is not None:
        path = config_path(data_dir)
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise ProviderError(f"cannot read {path}: {exc}") from None
            if not isinstance(raw, dict):
                raise ProviderError(f"{path}: expected a JSON object")
            cfg.provider = str(raw.get("provider") or cfg.provider)
            cfg.model = _opt_str(raw.get("model"))
            cfg.base_url = _opt_str(raw.get("base_url"))
            cfg.api_key = _opt_str(raw.get("api_key"))
            if raw.get("timeout") is not None:
                cfg.timeout = float(raw["timeout"])
            cfg.options = dict(raw.get("options") or {})
            origins.append("file")

    env_provider = os.environ.get(ENV_PROVIDER)
    if env_provider:
        cfg.provider = env_provider
        origins.append("env")
    for var, attr in (
        (ENV_MODEL, "model"),
        (ENV_BASE_URL, "base_url"),
        (ENV_API_KEY, "api_key"),
    ):
        value = os.environ.get(var)
        if value:
            setattr(cfg, attr, value)
            if "env" not in origins:
                origins.append("env")
    timeout_env = os.environ.get(ENV_TIMEOUT)
    if timeout_env:
        try:
            cfg.timeout = float(timeout_env)
        except ValueError:
            raise ProviderError(f"${ENV_TIMEOUT}: not a number: {timeout_env!r}") from None
        if "env" not in origins:
            origins.append("env")

    # An API key alone never implies a provider: silently activating one
    # because a key happens to be in the environment is exactly the
    # "calls a network service by default" failure mode.
    cfg.origin = "+".join(origins) if origins else "default"
    return cfg


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_provider(
    data_dir: Path | None = None, *, config: ProviderConfig | None = None
) -> LLMProvider | None:
    """Build the configured provider, or ``None`` when none is configured.

    ``None`` is the expected result on a fresh install. Callers must handle
    it; they must not treat it as a failure to report.
    """
    cfg = config or read_config(data_dir)
    name = cfg.provider.strip().casefold()

    if name in ("", "none", "off", "disabled"):
        return None
    if name == "echo":
        from .echo import EchoProvider

        return EchoProvider(model=cfg.model, options=cfg.options)
    if name in ("openai-compatible", "openai_compatible", "http", "local"):
        from .http import OpenAICompatibleProvider

        return OpenAICompatibleProvider(cfg)
    raise ProviderError(
        f"unknown LLM provider {cfg.provider!r}; built in: none, echo, openai-compatible"
    )


def require_provider(
    data_dir: Path | None = None,
    *,
    operation: str = "this operation",
    config: ProviderConfig | None = None,
) -> LLMProvider:
    """Like :func:`load_provider`, but raises with an actionable message."""
    provider = load_provider(data_dir, config=config)
    if provider is None:
        raise ProviderUnavailable(operation)
    return provider


def write_config(data_dir: Path, cfg: ProviderConfig) -> Path:
    """Persist provider settings. Plain JSON, inspectable and hand-editable."""
    path = config_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "timeout": cfg.timeout,
        "options": cfg.options,
    }
    if cfg.api_key:
        payload["api_key"] = cfg.api_key
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "CONFIG_RELPATH",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "ENV_PROVIDER",
    "ENV_TIMEOUT",
    "Completion",
    "LLMProvider",
    "ProviderConfig",
    "ProviderError",
    "ProviderUnavailable",
    "config_path",
    "load_provider",
    "read_config",
    "require_provider",
    "write_config",
]
