"""LLM client abstraction for the memory writer.

The writer model is the independent variable of the Memory Writing Model
experiments, so every client reports token usage; `LLMClient.usage` is what the
per-system writer-cost table (design doc 3.3 note) is built from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .tokens import count_tokens


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tag: str = ""


@dataclass
class Usage:
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_tag: dict[str, int] = field(default_factory=dict)

    def add(self, r: LLMResponse) -> None:
        self.n_calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.by_tag[r.tag] = self.by_tag.get(r.tag, 0) + 1

    def to_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "by_tag": dict(self.by_tag),
        }


class LLMClient:
    """Base class. Subclasses implement `_complete`."""

    model: str = "unknown"

    def __init__(self):
        self.usage = Usage()

    def _complete(self, system: str, user: str, tag: str) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    def complete(self, system: str, user: str, tag: str = "") -> LLMResponse:
        r = self._complete(system, user, tag)
        r.tag = tag
        if not r.prompt_tokens:
            r.prompt_tokens = count_tokens(system) + count_tokens(user)
        if not r.completion_tokens:
            r.completion_tokens = count_tokens(r.text)
        self.usage.add(r)
        return r


class CallableLLM(LLMClient):
    """Wrap any `fn(system, user, tag) -> str`. Used by the demo's stub writer."""

    def __init__(self, fn: Callable[[str, str, str], str], model: str = "callable"):
        super().__init__()
        self.fn = fn
        self.model = model

    def _complete(self, system: str, user: str, tag: str) -> LLMResponse:
        return LLMResponse(text=self.fn(system, user, tag))


class ScriptedLLM(LLMClient):
    """Deterministic canned responses for tests.

    `responses` may be a list (consumed in order) or a dict keyed by tag, whose
    values are a single response or a list consumed in order for that tag.
    """

    def __init__(self, responses, model: str = "scripted", default: str = '{"ops": []}'):
        super().__init__()
        self.model = model
        self.default = default
        self._queue = list(responses) if isinstance(responses, list) else None
        self._by_tag = None
        if isinstance(responses, dict):
            self._by_tag = {k: (list(v) if isinstance(v, list) else [v]) for k, v in responses.items()}
        self.calls: list[tuple[str, str]] = []  # (tag, user) for assertions

    def _complete(self, system: str, user: str, tag: str) -> LLMResponse:
        self.calls.append((tag, user))
        text = self.default
        if self._by_tag is not None:
            q = self._by_tag.get(tag)
            if q:
                text = q.pop(0) if len(q) > 1 else q[0]
        elif self._queue:
            text = self._queue.pop(0)
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        return LLMResponse(text=text)


class OpenAIChatClient(LLMClient):  # pragma: no cover - needs network
    """Works with the OpenAI API and any OpenAI-compatible server (vLLM/SGLang),
    which is how Qwen3.5-9B / 27B are served locally."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        super().__init__()
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout)

    def _complete(self, system: str, user: str, tag: str) -> LLMResponse:
        last = None
        for _ in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                u = getattr(resp, "usage", None)
                return LLMResponse(
                    text=resp.choices[0].message.content or "",
                    prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                )
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last}")
