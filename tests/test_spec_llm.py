from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import spec_llm  # noqa: E402
from spec_config import DEFAULT_MODEL  # noqa: E402
from spec_config import DEFAULT_REASONING  # noqa: E402
from spec_config import OPENROUTER_REQUEST_TIMEOUT_SECONDS  # noqa: E402


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_client(response):
    responses = FakeResponses(response)
    client = SimpleNamespace(responses=responses)
    return client, responses


def test_request_text_uses_openrouter_responses_shape():
    client, responses = fake_client(SimpleNamespace(error=None, output_text="  done  "))

    result = spec_llm.request_text(client, "prompt", "unit")

    assert result == "done"
    assert responses.kwargs == {
        "model": DEFAULT_MODEL,
        "reasoning": DEFAULT_REASONING,
        "input": "prompt",
        "timeout": OPENROUTER_REQUEST_TIMEOUT_SECONDS,
    }


def test_request_tool_call_uses_openrouter_responses_tool_shape():
    response = SimpleNamespace(
        error=None,
        output=[
            SimpleNamespace(
                type="function_call",
                name="edit_notes",
                arguments='{"action":"edit","edits":[]}',
            )
        ],
    )
    client, responses = fake_client(response)
    tool_schema = {
        "type": "function",
        "name": "edit_notes",
        "description": "Edit checked-out notes.",
        "strict": True,
        "parameters": {"type": "object", "properties": {}},
    }

    tool_call = spec_llm.request_tool_call(client, "prompt", [tool_schema], "unit")

    assert tool_call.name == "edit_notes"
    assert tool_call.arguments == '{"action":"edit","edits":[]}'
    assert responses.kwargs == {
        "model": DEFAULT_MODEL,
        "reasoning": DEFAULT_REASONING,
        "input": "prompt",
        "tools": [tool_schema],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "timeout": OPENROUTER_REQUEST_TIMEOUT_SECONDS,
    }


def test_parse_tool_call_arguments_rejects_non_object_payload():
    call = spec_llm.ToolCall("edit_notes", "[]")

    try:
        spec_llm.parse_tool_call_arguments(call)
    except RuntimeError as error:
        assert "must be a JSON object" in str(error)
    else:
        raise AssertionError("Expected non-object tool arguments to be rejected.")
