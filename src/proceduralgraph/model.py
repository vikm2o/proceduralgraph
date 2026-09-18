# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Provider-agnostic chat model protocol.

A host implements :class:`ChatModel` once. Every role call (the offline refiner, the online guidance model) is a
single request carrying a system prompt and one user message made of text and image parts, and expects plain text
back. Nothing in this package needs native tool calling (REQUIREMENTS E5); the ``tools`` / ``tool_calls`` fields exist
only so a host's ``skillwiki`` adapter serves both packages by duck typing, and are ignored here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    """A base64-encoded image. ``ref`` names it in text placeholders once it is no longer attached."""

    media_type: str
    data_base64: str
    ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"media_type": self.media_type, "data_base64": self.data_base64, "ref": self.ref}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ImagePart:
        return cls(media_type=value["media_type"], data_base64=value["data_base64"], ref=value.get("ref", ""))


Part = TextPart | ImagePart


@dataclass(frozen=True)
class ToolSpec:
    """Shape compatibility with skillwiki adapters only; this package never offers tools."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolInvocation:
    """Shape compatibility with skillwiki adapters only; this package never reads tool calls."""

    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ModelRequest:
    role: str
    system: str
    parts: Sequence[Part]
    max_tokens: int = 4096
    tools: Sequence[ToolSpec] = ()  # always empty in this package

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def images(self) -> list[ImagePart]:
        return [p for p in self.parts if isinstance(p, ImagePart)]


@dataclass(frozen=True)
class ModelResponse:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    tool_calls: Sequence[ToolInvocation] = ()


class ChatModel(Protocol):
    """``supports_tools`` may exist on an adapter shared with skillwiki; this package never sets ``tools``."""

    async def complete(self, request: ModelRequest) -> ModelResponse: ...


class ScriptedChatModel:
    """Deterministic model for tests and examples.

    ``script`` is either a list of responses consumed in order, or a callable that receives the request and returns
    the response text. Every request is recorded in ``calls``.
    """

    def __init__(self, script: Sequence[str] | Callable[[ModelRequest], str | Awaitable[str]]):
        self._script = list(script) if not callable(script) else script
        self.calls: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if callable(self._script):
            result = self._script(request)
            if hasattr(result, "__await__"):
                result = await result
            text = str(result)
        elif not self._script:
            raise RuntimeError("ScriptedChatModel ran out of scripted responses")
        else:
            text = self._script.pop(0)
        return ModelResponse(text=text, usage={"output_tokens": max(1, len(text) // 4)})


__all__ = [
    "ChatModel",
    "ImagePart",
    "ModelRequest",
    "ModelResponse",
    "Part",
    "ScriptedChatModel",
    "TextPart",
    "ToolInvocation",
    "ToolSpec",
]
