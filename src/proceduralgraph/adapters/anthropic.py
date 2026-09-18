# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""ChatModel over the Anthropic SDK (``pip install proceduralgraph[anthropic]``). Text protocol only: one system prompt,
one user message of text and image parts, text back. ``request.tools`` is ignored (E5)."""

from __future__ import annotations

from ..model import ModelRequest, ModelResponse


class AnthropicChatModel:
    supports_tools = False

    def __init__(self, model: str, *, client=None, **client_kwargs):
        import anthropic

        self.model = model
        self.client = client or anthropic.AsyncAnthropic(**client_kwargs)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        content: list[dict] = [{"type": "text", "text": request.text}]
        for image in request.images:
            content.append({"type": "image", "source": {"type": "base64", "media_type": image.media_type, "data": image.data_base64}})
        kwargs: dict = {"model": self.model, "max_tokens": request.max_tokens, "messages": [{"role": "user", "content": content}]}
        if request.system:
            kwargs["system"] = request.system
        message = await self.client.messages.create(**kwargs)
        text = "".join(getattr(block, "text", "") for block in message.content)
        usage = getattr(message, "usage", None)
        return ModelResponse(text=text, usage=usage.model_dump() if hasattr(usage, "model_dump") else {})


__all__ = ["AnthropicChatModel"]
