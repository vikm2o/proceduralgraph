# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""ChatModel over any OpenAI-compatible chat completions API (``pip install proceduralgraph[openai]``). Text protocol
only: one system message, one user message of text and image parts, text back. ``request.tools`` is ignored (E5)."""

from __future__ import annotations

from ..model import ModelRequest, ModelResponse


class OpenAIChatModel:
    supports_tools = False

    def __init__(self, model: str, *, client=None, **client_kwargs):
        import openai

        self.model = model
        self.client = client or openai.AsyncOpenAI(**client_kwargs)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        content: list[dict] = [{"type": "text", "text": request.text}]
        for image in request.images:
            content.append({"type": "image_url", "image_url": {"url": f"data:{image.media_type};base64,{image.data_base64}"}})
        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": content})
        response = await self.client.chat.completions.create(model=self.model, max_tokens=request.max_tokens, messages=messages)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return ModelResponse(text=choice.message.content or "", usage=usage.model_dump() if hasattr(usage, "model_dump") else {})


__all__ = ["OpenAIChatModel"]
