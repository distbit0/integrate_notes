from __future__ import annotations

import json
from typing import List

from loguru import logger

from spec_config import MAX_CHUNKING_ATTEMPTS, SpecConfig
from spec_llm import request_text
from spec_markdown import count_words


def build_chunking_prompt(
    numbered_paragraphs: List[str], sample_filenames: List[str], config: SpecConfig
) -> str:
    paragraphs_block = "\n".join(numbered_paragraphs)
    samples_block = "\n".join(f"- {name}" for name in sample_filenames)

    instructions = (
        "Group the numbered paragraphs into semantically coherent chunks. "
        "Paragraphs in a chunk need not be contiguous. "
        "Do not split a paragraph. "
        f"Each chunk must be at most {config.max_chunk_words} words. "
        "Return JSON only in the form: {\"groups\": [[1,2],[3]]}. "
        "Include every paragraph number exactly once. "
        "The order of groups should reflect the order you want them processed; do not sort."
    )

    return (
        "<instructions>\n"
        f"{instructions}\n"
        "</instructions>\n\n"
        "<paragraphs>\n"
        f"{paragraphs_block}\n"
        "</paragraphs>\n\n"
        "<sample_filenames>\n"
        f"{samples_block}\n"
        "</sample_filenames>"
    )


def _parse_group_payload(payload: str, total_paragraphs: int) -> List[List[int]]:
    data = json.loads(payload)
    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError("Chunking response must be a JSON object with a 'groups' key.")
    groups = data["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("Chunking response 'groups' must be a non-empty list.")

    seen: set[int] = set()
    parsed_groups: List[List[int]] = []

    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("Each chunk group must be a non-empty list of integers.")
        parsed_group: List[int] = []
        for value in group:
            if not isinstance(value, int):
                raise ValueError("Chunk group entries must be integers.")
            if value < 1 or value > total_paragraphs:
                raise ValueError(
                    f"Paragraph number {value} is out of range 1..{total_paragraphs}."
                )
            if value in seen:
                raise ValueError(f"Paragraph number {value} appears in multiple groups.")
            seen.add(value)
            parsed_group.append(value)
        parsed_groups.append(parsed_group)

    if len(seen) != total_paragraphs:
        missing = [str(i) for i in range(1, total_paragraphs + 1) if i not in seen]
        raise ValueError(f"Chunking response missing paragraphs: {', '.join(missing)}")

    return parsed_groups


def request_chunk_groups(
    client,
    paragraphs: List[str],
    sample_filenames: List[str],
    config: SpecConfig,
) -> List[List[int]]:
    numbered_paragraphs = [f"{index + 1}) {text}" for index, text in enumerate(paragraphs)]
    feedback: str | None = None

    for attempt in range(1, MAX_CHUNKING_ATTEMPTS + 1):
        prompt = build_chunking_prompt(numbered_paragraphs, sample_filenames, config)
        if feedback:
            prompt += (
                "\n\n<previous_attempt_feedback>\n"
                f"{feedback}\n"
                "</previous_attempt_feedback>"
            )
        response_text = request_text(client, prompt, f"chunking attempt {attempt}")
        try:
            groups = _parse_group_payload(response_text, len(paragraphs))
        except Exception as error:  # noqa: BLE001
            feedback = f"Parsing error: {error}"
            logger.warning(f"Chunking response invalid on attempt {attempt}: {error}")
            continue

        invalid_group = None
        for group in groups:
            words = sum(count_words(paragraphs[index - 1]) for index in group)
            if words > config.max_chunk_words:
                invalid_group = (group, words)
                break
        if invalid_group:
            group, words = invalid_group
            feedback = (
                f"Chunk {group} has {words} words, exceeding max {config.max_chunk_words}."
            )
            logger.warning(
                f"Chunking response exceeded word limit on attempt {attempt}: {feedback}"
            )
            continue

        return groups

    raise RuntimeError("Unable to obtain valid chunk grouping from the model.")
