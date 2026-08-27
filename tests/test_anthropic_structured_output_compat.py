import inspect
import json

import httpx
from anthropic import Anthropic

from pipeline.utils import anthropic_structured_output_request_kwargs


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}


def test_current_sdk_uses_direct_output_config_keyword():
    def create(*, output_config=None):
        return output_config

    kwargs = anthropic_structured_output_request_kwargs(
        SCHEMA,
        create_method=create,
    )

    assert set(kwargs) == {"output_config"}
    assert kwargs["output_config"]["format"]["schema"] == SCHEMA


def test_legacy_sdk_forwards_output_config_through_extra_body():
    installed_client = Anthropic(api_key="test-key")
    installed_parameters = inspect.signature(
        installed_client.messages.create
    ).parameters
    if "output_config" in installed_parameters:
        # Simulate the legacy SDK signature without constructing a current
        # SDK client with the legacy SDK's httpx transport type.
        def legacy_create(*, extra_body=None):
            return extra_body

        kwargs = anthropic_structured_output_request_kwargs(
            SCHEMA,
            create_method=legacy_create,
        )
        assert set(kwargs) == {"extra_body"}
        assert kwargs["extra_body"]["output_config"]["format"]["schema"] == SCHEMA
        return

    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "{}"}],
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = Anthropic(
        api_key="test-key",
        base_url="https://mock.local",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    create_method = client.messages.create
    kwargs = {
        "model": "claude-sonnet-5",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "test"}],
    }
    kwargs.update(
        anthropic_structured_output_request_kwargs(
            SCHEMA,
            create_method=create_method,
        )
    )
    create_method(**kwargs)

    assert captured["body"]["output_config"]["format"]["type"] == "json_schema"
    assert captured["body"]["output_config"]["format"]["schema"] == SCHEMA
