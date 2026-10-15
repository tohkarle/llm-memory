"""A deterministic, offline provider.

Exists so the LLM-shaped code paths are testable without a model. It does no
inference: it answers from scripted replies keyed by a marker the prompt
carries, and otherwise returns a valid-but-empty response of the shape the
caller asked for.

Two reasons this is a real provider rather than a test double:

* the test suite must never need a live model, and a fixture living in
  ``tests/`` could not be reached by ``mem extract --provider echo``, which
  is how a user verifies their wiring before pointing it at a real runtime;
* it pins the response contract. If a caller's parser only tolerates what a
  particular model happens to emit, this provider's bare-minimum output
  breaks it here rather than in front of a user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .base import Completion

#: Prompts tagged with one of these markers get the matching canned shape.
#: The markers are the ``task`` values Phase 4's prompts declare.
TASK_MARKER = "memory-task:"


@dataclass(slots=True)
class EchoProvider:
    """Returns scripted or empty-but-well-formed responses.

    ``replies`` maps a task name (or any substring of the prompt) to the text
    to return, which is what lets a test drive extraction end to end.
    """

    model: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    name: str = "echo"
    replies: dict[str, str] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        scripted = self.options.get("replies")
        if isinstance(scripted, dict):
            for key, value in scripted.items():
                self.replies.setdefault(str(key), str(value))

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        self.calls.append(prompt)
        for key, reply in self.replies.items():
            if key in prompt:
                return self._completion(reply)
        return self._completion(self._default_for(prompt))

    def _completion(self, text: str) -> Completion:
        return Completion(
            text=text,
            model=self.model or "echo",
            provider=self.name,
            prompt_tokens=None,
            completion_tokens=None,
        )

    def _default_for(self, prompt: str) -> str:
        """An empty answer in the right shape for the declared task.

        Returning ``{}`` for a prompt expecting a list would make every
        caller's "no suggestions" path untested, so the shape matters.
        """
        task = _declared_task(prompt)
        if task in ("extract", "consolidate"):
            return json.dumps({task: []})
        if task == "adjudicate":
            return json.dumps({"verdict": "unclear", "reason": "echo provider does no inference"})
        return ""

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model or "echo",
            "deterministic": True,
            "network": False,
            "scripted_replies": sorted(self.replies),
        }


def _declared_task(prompt: str) -> str | None:
    marker = prompt.find(TASK_MARKER)
    if marker < 0:
        return None
    rest = prompt[marker + len(TASK_MARKER) :].strip()
    return rest.split(None, 1)[0].strip().strip(".,;:") or None if rest else None


__all__ = ["TASK_MARKER", "EchoProvider"]
