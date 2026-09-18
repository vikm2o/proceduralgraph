# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""H9: the provider adapters, driven by fake clients so no network or SDK is needed. Text protocol only (E5)."""

from types import SimpleNamespace

import pytest

from proceduralgraph.model import ImagePart, ModelRequest, TextPart, ToolSpec

anthropic = pytest.importorskip("anthropic", reason="anthropic extra not installed")  # noqa: F841 - the adapter imports it lazily
openai = pytest.importorskip("openai", reason="openai extra not installed")  # noqa: F841

from proceduralgraph.adapters.anthropic import AnthropicChatModel  # noqa: E402
from proceduralgraph.adapters.openai import OpenAIChatModel  # noqa: E402


class FakeUsage:
    def __init__(self, **values):
        self.values = values

    def model_dump(self):
        return dict(self.values)


class FakeAnthropic:
    def __init__(self):
        self.kwargs = None
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="hello "), SimpleNamespace(type="text", text="world")],
                               usage=FakeUsage(input_tokens=10, output_tokens=2))


class FakeOpenAI:
    def __init__(self):
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="reply"))], usage=FakeUsage(completion_tokens=3))


def request(system="sys"):
    return ModelRequest(role="refiner", system=system, parts=[TextPart("hi"), ImagePart("image/png", "AAAA")], max_tokens=77,
                        tools=[ToolSpec("ignored", "never sent", {})])


async def test_anthropic_adapter_text_only():
    client = FakeAnthropic()
    model = AnthropicChatModel("some-claude-model", client=client)
    response = await model.complete(request())
    assert response.text == "hello world" and response.usage == {"input_tokens": 10, "output_tokens": 2} and response.tool_calls == ()
    assert client.kwargs["model"] == "some-claude-model" and client.kwargs["max_tokens"] == 77 and client.kwargs["system"] == "sys"
    assert "tools" not in client.kwargs and client.kwargs["messages"][0]["content"][1]["type"] == "image"
    assert model.supports_tools is False
    await AnthropicChatModel("some-claude-model", client=client).complete(request(system=""))
    assert "system" not in client.kwargs


async def test_openai_adapter_text_only():
    client = FakeOpenAI()
    model = OpenAIChatModel("some-model", client=client)
    response = await model.complete(request())
    assert response.text == "reply" and response.usage == {"completion_tokens": 3}
    assert client.kwargs["model"] == "some-model" and "tools" not in client.kwargs
    assert [m["role"] for m in client.kwargs["messages"]] == ["system", "user"]
    assert client.kwargs["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    await model.complete(request(system=""))
    assert [m["role"] for m in client.kwargs["messages"]] == ["user"]
