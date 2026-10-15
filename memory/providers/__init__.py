"""Pluggable, model-agnostic LLM access. Entirely optional.

Nothing in this package runs unless a provider is explicitly configured, and
every Phase 4 feature either degrades to a deterministic path or raises
:class:`ProviderUnavailable` with instructions. See :mod:`memory.providers.base`.
"""

from __future__ import annotations

from .base import (
    CONFIG_RELPATH,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PROVIDER,
    ENV_TIMEOUT,
    Completion,
    LLMProvider,
    ProviderConfig,
    ProviderError,
    ProviderUnavailable,
    config_path,
    load_provider,
    read_config,
    require_provider,
    write_config,
)
from .echo import EchoProvider
from .json_io import JSONResponseError, extract_json_object

__all__ = [
    "CONFIG_RELPATH",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "ENV_PROVIDER",
    "ENV_TIMEOUT",
    "Completion",
    "EchoProvider",
    "JSONResponseError",
    "LLMProvider",
    "ProviderConfig",
    "ProviderError",
    "ProviderUnavailable",
    "config_path",
    "extract_json_object",
    "load_provider",
    "read_config",
    "require_provider",
    "write_config",
]
