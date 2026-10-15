"""An OpenAI-compatible chat-completions client, over the standard library.

The chat-completions wire format is what local inference runtimes converge
on -- llama.cpp's server, Ollama, vLLM, LM Studio, text-generation-webui all
speak it -- so one small client covers local inference without naming a
vendor or adding a dependency.

``base_url`` has no default. That is the whole safety property: this provider
cannot send a request anywhere the user did not name, so "no cloud by
default" is structural rather than a policy comment.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import Completion, ProviderConfig, ProviderError

#: Appended to ``base_url`` unless the configured URL already names a path.
CHAT_PATH = "chat/completions"


class OpenAICompatibleProvider:
    """Talk to any server speaking the chat-completions format."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.base_url:
            raise ProviderError(
                "the openai-compatible provider requires a base_url "
                "(e.g. http://localhost:11434/v1 for Ollama, "
                "http://localhost:8080/v1 for llama.cpp); "
                "set MEMORY_LLM_BASE_URL or base_url in data/config/llm.json"
            )
        if not config.model:
            raise ProviderError(
                "the openai-compatible provider requires a model name; "
                "set MEMORY_LLM_MODEL or model in data/config/llm.json"
            )
        self.config = config
        self.name = "openai-compatible"
        self.model: str | None = config.model

    @property
    def url(self) -> str:
        base = (self.config.base_url or "").rstrip("/")
        # A URL already ending in the endpoint is passed through: users paste
        # whatever their runtime's docs printed, and double-appending the
        # path would 404 with no clue why.
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/{CHAT_PATH}"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **dict(self.config.options.get("request") or {}),
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"{self.url} returned HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                f"cannot reach {self.url}: {exc}; is the local runtime running?"
            ) from None

        return _parse_response(raw, provider=self.name, model=self.model)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "url": self.url,
            "deterministic": False,
            "network": True,
            "timeout": self.config.timeout,
        }


def _parse_response(raw: str, *, provider: str, model: str | None) -> Completion:
    """Pull the text out of a chat-completions response.

    Tolerant about where the text sits, because runtimes differ in small
    ways, but explicit when there is no text at all: a silent empty string
    would read downstream as "the model found nothing", which is a very
    different fact from "the server replied with something unexpected".
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise ProviderError(f"provider returned non-JSON: {raw[:200]!r}") from None
    if not isinstance(parsed, dict):
        raise ProviderError(f"provider returned {type(parsed).__name__}, expected an object")

    if isinstance(parsed.get("error"), dict):
        message = parsed["error"].get("message") or parsed["error"]
        raise ProviderError(f"provider error: {message}")

    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError(f"provider response has no choices: {raw[:200]!r}")
    first = choices[0]
    if not isinstance(first, dict):
        raise ProviderError("provider response choice is not an object")

    message = first.get("message")
    text: Any = None
    if isinstance(message, dict):
        text = message.get("content")
    if text is None:
        text = first.get("text")
    if not isinstance(text, str):
        raise ProviderError(f"provider response has no text content: {raw[:200]!r}")

    raw_usage = parsed.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    return Completion(
        text=text,
        model=str(parsed.get("model") or model or ""),
        provider=provider,
        prompt_tokens=_opt_int(usage.get("prompt_tokens")),
        completion_tokens=_opt_int(usage.get("completion_tokens")),
        meta={"finish_reason": first.get("finish_reason")},
    )


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


__all__ = ["CHAT_PATH", "OpenAICompatibleProvider"]
