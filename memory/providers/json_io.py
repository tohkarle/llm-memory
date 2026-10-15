"""Recover a JSON object from a model's reply.

Models asked for JSON return JSON wrapped in things: a ``json`` fence, a
sentence of preamble, a trailing explanation. Local models do it more often
than large hosted ones, and this system's whole point is local models.

Parsing is therefore layered, strictest first, and the last layer is a brace
scan that respects string literals -- a regex cannot, and a nested object
inside a quoted string would otherwise truncate the match.
"""

from __future__ import annotations

import json
from typing import Any


class JSONResponseError(ValueError):
    """A model reply did not contain a usable JSON object."""

    def __init__(self, message: str, *, raw: str) -> None:
        # Truncated: a reply can be long, and this message lands in a CLI
        # error envelope.
        self.raw = raw[:2000]
        super().__init__(message)


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object in ``text``.

    Raises :class:`JSONResponseError` rather than returning an empty dict: an
    unparseable reply and a reply meaning "nothing found" must not collapse
    into the same value, or a broken provider looks like a clean corpus.
    """
    stripped = text.strip()
    if not stripped:
        raise JSONResponseError("provider returned an empty response", raw=text)

    for candidate in _candidates(stripped):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            # A bare array is a common slip when the prompt asks for
            # {"extract": [...]}. Accepting it under a conventional key beats
            # rejecting a reply whose content is perfectly usable.
            return {"items": parsed}
    raise JSONResponseError("no JSON object found in the provider response", raw=text)


def _candidates(text: str) -> list[str]:
    out = [text]
    fenced = _strip_fence(text)
    if fenced is not None:
        out.append(fenced)
    scanned = _scan_braces(text)
    if scanned is not None:
        out.append(scanned)
    # A fenced block can itself carry prose around the object.
    if fenced is not None:
        inner = _scan_braces(fenced)
        if inner is not None:
            out.append(inner)
    return out


def _strip_fence(text: str) -> str | None:
    if "```" not in text:
        return None
    _, _, rest = text.partition("```")
    if rest.startswith("json"):
        rest = rest[4:]
    elif "\n" in rest.split("```", 1)[0][:20]:
        rest = rest.split("\n", 1)[1] if "\n" in rest else rest
    body, _, _ = rest.partition("```")
    return body.strip() or None


def _scan_braces(text: str) -> str | None:
    """First balanced ``{...}`` run, ignoring braces inside string literals."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


__all__ = ["JSONResponseError", "extract_json_object"]
