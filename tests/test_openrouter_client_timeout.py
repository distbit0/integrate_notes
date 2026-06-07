import json
import sys
from pathlib import Path
from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import integrate_notes  # noqa: E402
import spec_llm  # noqa: E402
from spec_config import (  # noqa: E402
    OPENROUTER_REQUEST_TIMEOUT_SECONDS,
    OPENROUTER_SDK_MAX_RETRIES,
)


def test_integrate_notes_openrouter_client_uses_bounded_timeout(monkeypatch) -> None:
    captured_kwargs = {}
    client = object()

    def openai_client(**kwargs):
        captured_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(integrate_notes, "OpenAI", openai_client)
    monkeypatch.setattr(integrate_notes, "load_dotenv", lambda *_, **__: None)
    monkeypatch.setenv(integrate_notes.ENV_API_KEY, "test-key")

    assert integrate_notes.create_openrouter_client() is client
    assert (
        captured_kwargs["timeout"]
        == integrate_notes.OPENROUTER_REQUEST_TIMEOUT_SECONDS
    )
    assert captured_kwargs["max_retries"] == integrate_notes.OPENROUTER_SDK_MAX_RETRIES


def test_spec_openrouter_client_uses_bounded_timeout(monkeypatch) -> None:
    captured_kwargs = {}
    client = object()

    def openai_client(**kwargs):
        captured_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(spec_llm, "OpenAI", openai_client)
    monkeypatch.setattr(spec_llm, "load_dotenv", lambda *_, **__: None)
    monkeypatch.setenv(spec_llm.ENV_API_KEY, "test-key")

    assert spec_llm.create_openrouter_client() is client
    assert captured_kwargs["timeout"] == OPENROUTER_REQUEST_TIMEOUT_SECONDS
    assert captured_kwargs["max_retries"] == OPENROUTER_SDK_MAX_RETRIES


def test_integration_request_sets_per_call_timeout() -> None:
    captured_kwargs = {}
    tool_arguments = json.dumps(
        {
            "action": "integrate",
            "patches": [
                {
                    "search": "- Create space for the other person to talk.\n- Ask open-ended follow-up questions.",
                    "replace": (
                        "- Create space for the other person to talk.\n"
                        "- Ask people to tell you more rather than immediately giving advice or your opinion.\n"
                        '- Use verbal acknowledgments while they are speaking, e.g., "yeah that makes sense," "uh huh."\n'
                        "- Ask open-ended follow-up questions."
                    ),
                }
            ],
            "duplications": [],
        }
    )

    class Responses:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(error=None, output_text=f"```json\n{tool_arguments}\n```")

    client = SimpleNamespace(responses=Responses())

    response_text = integrate_notes.request_integration(client, "prompt", "unit")
    instructions, duplications = integrate_notes.parse_integration_payload(
        response_text
    )

    assert instructions == [
        integrate_notes.PatchInstruction(
            search_text="- Create space for the other person to talk.\n- Ask open-ended follow-up questions.",
            replace_text=(
                "- Create space for the other person to talk.\n"
                "- Ask people to tell you more rather than immediately giving advice or your opinion.\n"
                '- Use verbal acknowledgments while they are speaking, e.g., "yeah that makes sense," "uh huh."\n'
                "- Ask open-ended follow-up questions."
            ),
        )
    ]
    assert duplications == []
    assert captured_kwargs["reasoning"] == integrate_notes.DEFAULT_REASONING
    assert captured_kwargs["text"] == {
        "format": integrate_notes.INTEGRATION_RESPONSE_FORMAT
    }
    assert (
        captured_kwargs["timeout"]
        == integrate_notes.OPENROUTER_REQUEST_TIMEOUT_SECONDS
    )
