from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import spec_llm  # noqa: E402
from spec_config import DEFAULT_MODEL  # noqa: E402
from spec_config import OPENROUTER_REQUEST_TIMEOUT_SECONDS  # noqa: E402


class FakeCompletions:
    def __init__(self, message):
        self.message = message
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])


def fake_client(message):
    completions = FakeCompletions(message)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


def test_request_text_uses_openrouter_chat_completion_shape():
    client, completions = fake_client(SimpleNamespace(content="  done  "))

    result = spec_llm.request_text(client, "prompt", "unit")

    assert result == "done"
    assert completions.kwargs == {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "prompt"}],
        "timeout": OPENROUTER_REQUEST_TIMEOUT_SECONDS,
    }


def test_request_tool_call_converts_response_tool_schema_to_chat_tool_schema():
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(
                    name="edit_notes",
                    arguments='{"action":"edit","edits":[]}',
                )
            )
        ],
    )
    client, completions = fake_client(message)
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
    assert completions.kwargs == {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "prompt"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "edit_notes",
                    "description": "Edit checked-out notes.",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": True,
                },
            }
        ],
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
