"""The provider interface: optional, configurable, and never implicit.

The load-bearing property under test is that nothing reaches the network
unless a user named a URL. Everything else here is about degrading clearly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memory.providers import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PROVIDER,
    ENV_TIMEOUT,
    EchoProvider,
    JSONResponseError,
    LLMProvider,
    ProviderConfig,
    ProviderError,
    ProviderUnavailable,
    config_path,
    extract_json_object,
    load_provider,
    read_config,
    require_provider,
    write_config,
)
from memory.providers.http import OpenAICompatibleProvider, _parse_response


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider config is read from the environment, so isolate it."""
    for var in (ENV_PROVIDER, ENV_MODEL, ENV_BASE_URL, ENV_API_KEY, ENV_TIMEOUT):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------- absence


def test_no_provider_by_default(tmp_path: Path) -> None:
    assert load_provider(tmp_path) is None


def test_require_provider_explains_how_to_configure(tmp_path: Path) -> None:
    with pytest.raises(ProviderUnavailable) as exc:
        require_provider(tmp_path, operation="extraction")
    message = str(exc.value)
    assert "extraction" in message
    assert ENV_PROVIDER in message
    # The message is the user's only instruction; it must name the file too.
    assert "llm.json" in message


def test_api_key_alone_does_not_activate_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stray key in the environment must never turn on network calls."""
    monkeypatch.setenv(ENV_API_KEY, "sk-not-a-real-key")
    monkeypatch.setenv(ENV_BASE_URL, "http://localhost:9/v1")
    monkeypatch.setenv(ENV_MODEL, "whatever")
    assert load_provider(tmp_path) is None


def test_unknown_provider_names_the_built_ins(tmp_path: Path) -> None:
    cfg = ProviderConfig(provider="some-vendor")
    with pytest.raises(ProviderError) as exc:
        load_provider(tmp_path, config=cfg)
    assert "echo" in str(exc.value)


# ---------------------------------------------------------------- config


def test_config_file_is_read(tmp_path: Path) -> None:
    path = config_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"provider": "echo", "model": "m1"}), encoding="utf-8")
    cfg = read_config(tmp_path)
    assert (cfg.provider, cfg.model, cfg.origin) == ("echo", "m1", "file")


def test_environment_overrides_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_config(tmp_path, ProviderConfig(provider="echo", model="from-file"))
    monkeypatch.setenv(ENV_MODEL, "from-env")
    cfg = read_config(tmp_path)
    assert cfg.model == "from-env"
    assert cfg.provider == "echo"
    assert "env" in cfg.origin


def test_write_config_round_trips(tmp_path: Path) -> None:
    written = write_config(
        tmp_path,
        ProviderConfig(
            provider="openai-compatible", model="m", base_url="http://x/v1", timeout=5.0
        ),
    )
    assert written.is_file()
    cfg = read_config(tmp_path)
    assert (cfg.provider, cfg.model, cfg.base_url, cfg.timeout) == (
        "openai-compatible",
        "m",
        "http://x/v1",
        5.0,
    )


def test_config_never_echoes_the_api_key(tmp_path: Path) -> None:
    cfg = ProviderConfig(provider="echo", api_key="sk-secret-value")
    payload = cfg.to_json()
    assert payload["api_key_set"] is True
    assert "sk-secret-value" not in json.dumps(payload)


def test_malformed_config_file_is_reported(tmp_path: Path) -> None:
    path = config_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not json at all", encoding="utf-8")
    with pytest.raises(ProviderError):
        read_config(tmp_path)


def test_bad_timeout_env_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_TIMEOUT, "soon")
    with pytest.raises(ProviderError):
        read_config(tmp_path)


# ---------------------------------------------------------------- echo


def test_echo_provider_satisfies_the_protocol(tmp_path: Path) -> None:
    provider = load_provider(tmp_path, config=ProviderConfig(provider="echo"))
    assert isinstance(provider, LLMProvider)


def test_echo_returns_scripted_replies(tmp_path: Path) -> None:
    provider = EchoProvider(replies={"needle": '{"extract": [{"title": "x"}]}'})
    out = provider.complete("a prompt containing a needle somewhere")
    assert json.loads(out.text)["extract"][0]["title"] == "x"
    assert provider.calls  # the prompt is recorded, which is what tests assert on


def test_echo_shapes_its_empty_answer_by_task(tmp_path: Path) -> None:
    provider = EchoProvider()
    assert json.loads(provider.complete("memory-task: extract\n...").text) == {"extract": []}
    assert json.loads(provider.complete("memory-task: consolidate\n").text) == {"consolidate": []}
    assert json.loads(provider.complete("memory-task: adjudicate\n").text)["verdict"] == "unclear"


def test_echo_never_touches_the_network(tmp_path: Path) -> None:
    assert EchoProvider().describe()["network"] is False


def test_echo_options_can_script_replies(tmp_path: Path) -> None:
    provider = load_provider(
        tmp_path,
        config=ProviderConfig(provider="echo", options={"replies": {"k": "v"}}),
    )
    assert provider is not None
    assert provider.complete("has k in it").text == "v"


# ---------------------------------------------------------------- http


def test_http_provider_refuses_without_a_base_url() -> None:
    with pytest.raises(ProviderError) as exc:
        OpenAICompatibleProvider(ProviderConfig(provider="openai-compatible", model="m"))
    assert "base_url" in str(exc.value)


def test_http_provider_refuses_without_a_model() -> None:
    with pytest.raises(ProviderError) as exc:
        OpenAICompatibleProvider(
            ProviderConfig(provider="openai-compatible", base_url="http://localhost:1/v1")
        )
    assert "model" in str(exc.value)


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("http://localhost:11434/v1", "http://localhost:11434/v1/chat/completions"),
        ("http://localhost:11434/v1/", "http://localhost:11434/v1/chat/completions"),
        (
            "http://localhost:8080/v1/chat/completions",
            "http://localhost:8080/v1/chat/completions",
        ),
    ],
)
def test_http_url_construction(base: str, expected: str) -> None:
    provider = OpenAICompatibleProvider(
        ProviderConfig(provider="openai-compatible", model="m", base_url=base)
    )
    assert provider.url == expected


def test_http_parses_a_standard_response() -> None:
    raw = json.dumps(
        {
            "model": "local-model",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
    )
    out = _parse_response(raw, provider="openai-compatible", model="m")
    assert (out.text, out.prompt_tokens, out.completion_tokens) == ("hello", 10, 2)


def test_http_parses_a_legacy_text_choice() -> None:
    raw = json.dumps({"choices": [{"text": "hi"}]})
    assert _parse_response(raw, provider="p", model=None).text == "hi"


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps({"choices": []}),
        json.dumps({"choices": [{"message": {}}]}),
        json.dumps({"error": {"message": "model not found"}}),
    ],
)
def test_http_rejects_unusable_responses(raw: str) -> None:
    """An unexpected reply must not read downstream as 'the model found nothing'."""
    with pytest.raises(ProviderError):
        _parse_response(raw, provider="p", model=None)


# ---------------------------------------------------------------- json recovery


@pytest.mark.parametrize(
    "text",
    [
        '{"extract": []}',
        '```json\n{"extract": []}\n```',
        'Sure! Here you go:\n```\n{"extract": []}\n```\nHope that helps.',
        'Here is the result: {"extract": []} -- let me know.',
    ],
)
def test_extract_json_object_tolerates_wrapping(text: str) -> None:
    assert extract_json_object(text) == {"extract": []}


def test_extract_json_object_handles_braces_inside_strings() -> None:
    payload = extract_json_object('{"note": "a } brace and a { brace", "n": 1}')
    assert payload["n"] == 1


def test_extract_json_object_promotes_a_bare_array() -> None:
    assert extract_json_object('[{"title": "x"}]')["items"][0]["title"] == "x"


@pytest.mark.parametrize("text", ["", "   ", "no json here", "{unbalanced"])
def test_extract_json_object_raises_rather_than_returning_empty(text: str) -> None:
    with pytest.raises(JSONResponseError):
        extract_json_object(text)


def test_json_response_error_truncates_the_raw_reply() -> None:
    with pytest.raises(JSONResponseError) as exc:
        extract_json_object("x" * 10_000)
    assert len(exc.value.raw) <= 2000


def test_describe_is_json_serializable(tmp_path: Path) -> None:
    """``mem provider show`` puts this in an envelope, so it must serialize."""
    provider: Any = EchoProvider(model="m")
    json.dumps(provider.describe())
