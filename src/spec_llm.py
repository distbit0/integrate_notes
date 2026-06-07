from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import sleep
from typing import Iterable

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from spec_config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_REASONING,
    ENV_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_REQUEST_TIMEOUT_SECONDS,
    OPENROUTER_SDK_MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    RETRY_INITIAL_DELAY_SECONDS,
    repo_root,
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str | None


def create_openrouter_client() -> OpenAI:
    load_dotenv(repo_root() / ".env", override=True)
    api_key = os.getenv(ENV_API_KEY)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {ENV_API_KEY} is required for GPT access."
        )
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=OPENROUTER_REQUEST_TIMEOUT_SECONDS,
        max_retries=OPENROUTER_SDK_MAX_RETRIES,
    )


def execute_with_retry(
    operation,
    description: str,
    max_attempts: int = DEFAULT_MAX_RETRIES,
    initial_delay_seconds: float = RETRY_INITIAL_DELAY_SECONDS,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
):
    attempt = 1
    delay = initial_delay_seconds
    while True:
        try:
            return operation()
        except Exception as error:
            if attempt >= max_attempts:
                logger.exception(
                    f"OpenRouter {description} failed after {max_attempts} attempt(s): {error}"
                )
                raise
            logger.warning(
                f"OpenRouter {description} attempt {attempt} failed: {error}. Retrying in {delay:.1f}s."
            )
            sleep(delay)
            attempt += 1
            delay *= backoff_factor


def request_text(client: OpenAI, prompt: str, context_label: str) -> str:
    def perform_request() -> str:
        response = client.responses.create(
            model=DEFAULT_MODEL,
            reasoning=DEFAULT_REASONING,
            input=prompt,
            timeout=OPENROUTER_REQUEST_TIMEOUT_SECONDS,
        )
        if getattr(response, "error", None):
            raise RuntimeError(f"OpenRouter error for {context_label}: {response.error}")
        output_text = response.output_text
        if not output_text.strip():
            raise RuntimeError(f"Received empty response for {context_label}.")
        return output_text.strip()

    return execute_with_retry(perform_request, context_label)


def request_tool_call(
    client: OpenAI, prompt: str, tools: Iterable[dict], context_label: str
) -> ToolCall:
    def perform_request() -> ToolCall:
        response = client.responses.create(
            model=DEFAULT_MODEL,
            reasoning=DEFAULT_REASONING,
            input=prompt,
            tools=list(tools),
            tool_choice="required",
            parallel_tool_calls=False,
            timeout=OPENROUTER_REQUEST_TIMEOUT_SECONDS,
        )
        if getattr(response, "error", None):
            raise RuntimeError(f"OpenRouter error for {context_label}: {response.error}")
        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:
            raise RuntimeError(f"No tool call returned for {context_label}.")
        if len(tool_calls) > 1:
            raise RuntimeError(
                f"Expected a single tool call for {context_label}, got {len(tool_calls)}."
            )
        tool_call = tool_calls[0]
        return ToolCall(tool_call.name, tool_call.arguments)

    return execute_with_retry(perform_request, context_label)


def parse_tool_call_arguments(call: ToolCall) -> dict:
    if not call.arguments:
        raise RuntimeError(f"Tool call {call.name} missing arguments.")
    try:
        payload = json.loads(call.arguments)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Tool call {call.name} arguments are not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Tool call {call.name} arguments must be a JSON object.")
    return payload
