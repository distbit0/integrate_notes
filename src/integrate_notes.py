import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable, List, Sequence, Tuple
from threading import Event, Lock, Thread
from uuid import uuid4

import shutil
import subprocess

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

SCRATCHPAD_HEADING = "# -- SCRATCHPAD"
GROUPING_FIELD = "grouping"
ORGANISE_FIELD = "organise"
CONTINUOUS_ORGANISE_VALUE = "continuous"
DEFAULT_GROUPING = "Group points according to what you think the most useful/interesting/relevant groupings are. Ensure similar, related and contradictory points are adjacent."
DEFAULT_NOTES_ROOT = Path("/home/pimania/notes")
DEFAULT_CHUNK_PARAGRAPHS = 30
DEFAULT_CHUNK_MAX_WORDS = 400
ENV_API_KEY = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "minimax/minimax-m3"
DEFAULT_REASONING = {"effort": "high"}
DEFAULT_MAX_RETRIES = 3
RETRY_INITIAL_DELAY_SECONDS = 2.0
RETRY_BACKOFF_FACTOR = 2.0
OPENROUTER_REQUEST_TIMEOUT_SECONDS = 120.0
OPENROUTER_SDK_MAX_RETRIES = 0
PENDING_VERIFICATION_PROMPTS_PATH = (
    Path(__file__).resolve().parent / "pending_verification_prompts.json"
)
MAX_CONCURRENT_VERIFICATIONS = 4
INSTRUCTIONS_PROMPT = """# Instructions

- Integrate the provided notes into the document body, following the specified grouping approach.
- Ensure related points are adjacent, according to the grouping approach.
- Break content into relatively atomic bullet points; each bullet should express one idea.
- Use nested bullets when a point is naturally a sub-point of another.
- Make minor grammar edits as needed so ideas read cleanly as bullet points.
- If text to integrate is already well-formatted, punctuated, grammatical and bullet-pointed, avoid altering its wording while integrating/inserting it.
- De-duplicate overlapping points without losing any nuance or detail.
- Keep wording succinct and remove filler words (e.g., "you know", "basically", "essentially", "uh").
- Add new headings, sub-headings, or parent bullet points for new items, and reuse existing ones where appropriate.
- Refactor existing content as needed to smoothly integrate the new notes.


# Rules

- PRESERVE/DO NOT LEAVE OUT ANY NUANCE, DETAILS, POINTS, CONCLUSIONS, IDEAS, ARGUMENTS, OR QUALIFICATIONS from the notes.
- PRESERVE ALL EXPLANATIONS FROM THE NOTES.
- Do not materially alter meaning.
- If new items do not match existing items in the document body, add them appropriately.
- Preserve questions as questions; do not convert them into statements.
- Do not guess acronym expansions if they are not specified.
- Do not modify tone (e.g., confidence/certainty) or add hedging.
- Do not omit any wikilinks, URLs, diagrams, ASCII art, mathematics, tables, figures, or other non-text content.
- Move each link/URL/etc. to the section where it is most relevant based on its surrounding context and its URL text.
    - Do not move links to a separate “resources” or “links” section.
- Do not modify any wikilinks or URLs.
- Any SEARCH or DUPLICATE text you emit must be a single contiguous span copied from the document body; do not concatenate non-adjacent fragments.


# Formatting

- Use nested markdown headings ("#", "##", "###", "####", etc.) for denoting groups and sub-groups, except if heading text is a [[wikilink]].
    - unless document body already employs a different convention, or the grouping approach specifies otherwise.
- Use "- " as the bullet prefix (not "* ", "-  ", or anything else).
    - Use four spaces for each level of bullet-point nesting.


# Before finishing: check your work

- Confirm every item from the provided notes is now represented in the document body without loss of detail.
- Ensure nothing from the original document body was lost.
- If anything is missing, integrate it in appropriately.
"""


LOG_FILE_PATH = Path(__file__).resolve().parent / "logs" / "integrate_notes.log"
LOG_FILE_ROTATION_BYTES = 2 * 1024 * 1024

PATCH_BLOCK_START = "<<<<<<< SEARCH"
PATCH_BLOCK_DIVIDER = "======="
PATCH_BLOCK_END = ">>>>>>> REPLACE"
DUPLICATION_BLOCK_START = "<<<<<<< DUPLICATE"
DUPLICATION_BLOCK_END = ">>>>>>> DUPLICATE"
INTEGRATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["integrate"]},
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "search": {"type": "string"},
                    "replace": {"type": "string"},
                },
                "required": ["search", "replace"],
                "additionalProperties": False,
            },
        },
        "duplications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "notes": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["notes", "body"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["action", "patches", "duplications"],
    "additionalProperties": False,
}
INTEGRATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "integration_response",
    "strict": True,
    "schema": INTEGRATION_RESPONSE_SCHEMA,
}
INTEGRATION_RESPONSE_SCHEMA_TEXT = json.dumps(
    INTEGRATION_RESPONSE_SCHEMA,
    ensure_ascii=False,
    indent=2,
)
MAX_PATCH_ATTEMPTS = 3


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=False)
    try:
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            f"Failed to prepare log directory {LOG_FILE_PATH.parent}: {error}"
        ) from error
    logger.add(
        LOG_FILE_PATH,
        level="DEBUG",
        rotation=LOG_FILE_ROTATION_BYTES,
        enqueue=False,
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate scratchpad notes into a markdown document."
    )
    parser.add_argument(
        "--source", required=False, help="Path to the source markdown document."
    )
    parser.add_argument(
        "--grouping",
        required=False,
        help="Grouping approach to record in frontmatter.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Integrate all notes marked organise: continuous with non-empty scratchpads.",
    )
    parser.add_argument(
        "--notes-root",
        default=str(DEFAULT_NOTES_ROOT),
        help="Notes vault root for --continuous scans.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_PARAGRAPHS,
        help="Max paragraphs per scratchpad chunk integration request.",
    )
    parser.add_argument(
        "--max-chunk-words",
        type=int,
        default=DEFAULT_CHUNK_MAX_WORDS,
        help="Max words per scratchpad chunk integration request.",
    )
    parser.add_argument(
        "--disable-verification",
        action="store_true",
        help="Disable verification prompts and background verification checks.",
    )
    return parser.parse_args()


def resolve_source_path(provided_path: str | None) -> Path:
    if provided_path:
        path = Path(provided_path).expanduser().resolve()
    else:
        user_input = input("Enter path to the source markdown document: ").strip()
        if not user_input:
            raise ValueError("Document path is required to proceed.")
        path = Path(user_input).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source document not found at {path}.")
    if not path.is_file():
        raise ValueError(f"Source path {path} is not a file.")
    return path


def split_document_sections(content: str) -> Tuple[str, str]:
    if SCRATCHPAD_HEADING not in content:
        raise ValueError(f"Document must contain the heading '{SCRATCHPAD_HEADING}'.")
    heading_index = content.index(SCRATCHPAD_HEADING)
    body = content[:heading_index].rstrip()
    scratchpad = content[heading_index + len(SCRATCHPAD_HEADING) :].lstrip("\n")
    return body, scratchpad


def split_frontmatter(content: str) -> tuple[list[str], str] | None:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line == "---":
            body = "\n".join(lines[index + 1 :])
            if content.endswith("\n"):
                body += "\n"
            return lines[1:index], body
    raise ValueError("Frontmatter starts with '---' but has no closing delimiter.")


def _is_top_level_field(line: str) -> bool:
    return bool(line.strip()) and not line.startswith((" ", "\t", "-")) and ":" in line


def read_frontmatter_field(content: str, field_name: str) -> str | None:
    frontmatter_parts = split_frontmatter(content)
    if frontmatter_parts is None:
        return None

    metadata_lines, _ = frontmatter_parts
    field_prefix = f"{field_name}:"
    for index, line in enumerate(metadata_lines):
        if not line.startswith(field_prefix):
            continue
        raw_value = line.split(":", 1)[1].strip()
        if raw_value in {"|", "|-", "|+", ">", ">-", ">+"}:
            block_lines: list[str] = []
            for block_line in metadata_lines[index + 1 :]:
                if _is_top_level_field(block_line):
                    break
                block_lines.append(
                    block_line[2:] if block_line.startswith("  ") else block_line
                )
            return "\n".join(block_lines).strip("\n")
        return raw_value.strip("\"'")
    return None


def _skip_frontmatter_field(lines: list[str], start_index: int) -> int:
    line = lines[start_index]
    value = line.split(":", 1)[1].strip()
    next_index = start_index + 1
    if value in {"|", "|-", "|+", ">", ">-", ">+"}:
        while next_index < len(lines) and not _is_top_level_field(lines[next_index]):
            next_index += 1
    return next_index


def _without_frontmatter_field(lines: list[str], field_name: str) -> list[str]:
    filtered: list[str] = []
    index = 0
    field_prefix = f"{field_name}:"
    while index < len(lines):
        if lines[index].startswith(field_prefix):
            index = _skip_frontmatter_field(lines, index)
            continue
        filtered.append(lines[index])
        index += 1
    return filtered


def _render_block_field(field_name: str, value: str) -> list[str]:
    stripped_value = value.strip()
    if not stripped_value:
        raise ValueError(f"{field_name} cannot be empty.")
    return [f"{field_name}: |"] + [
        f"  {line}" if line else "" for line in stripped_value.splitlines()
    ]


def set_frontmatter_block_field(content: str, field_name: str, value: str) -> str:
    frontmatter_parts = split_frontmatter(content)
    if frontmatter_parts is None:
        metadata_lines: list[str] = []
        body = content
    else:
        metadata_lines, body = frontmatter_parts

    updated_metadata = _without_frontmatter_field(metadata_lines, field_name)
    if updated_metadata and updated_metadata[-1].strip():
        updated_metadata.append("")
    updated_metadata.extend(_render_block_field(field_name, value))

    frontmatter = "\n".join(updated_metadata).rstrip()
    body = body.lstrip("\n")
    return f"---\n{frontmatter}\n---\n{body}"


def frontmatter_field_equals(content: str, field_name: str, value: str) -> bool:
    field_value = read_frontmatter_field(content, field_name)
    return field_value is not None and field_value.strip().lower() == value


def prompt_for_grouping() -> str:
    prompt = (
        "Grouping not found. Provide the text for the frontmatter "
        f"{GROUPING_FIELD} field.\n"
        "Enter multiline text and finish with a single line containing only a '.'.\n"
        "Examples:\n"
        '- Group points according to what problem each idea/proposal/mechanism/concept addresses/are trying to solve, which you will need to figure out yourself based on context. Do not combine multiple goals/problems into one group. Keep goals/problems specific. Ensure groups are mutually exclusive and collectively exhaustive. Avoid overlap between group\'s goals/problems. sub-headings should be per-mechanism/per-solution i.e. according to which "idea"/solution each point relates to.\n'
        "- Group points according to what you think the most useful/interesting/relevant groupings are. Ensure similar, related and contradictory points are adjacent.\n"
        "Your input:\n"
    )
    print(prompt, end="")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == ".":
            break
        lines.append(line)
    grouping = "\n".join(lines)
    if not grouping.strip():
        grouping = DEFAULT_GROUPING
    return grouping


def normalize_paragraphs(text: str) -> List[str]:
    stripped_text = text.strip()
    if not stripped_text:
        return []
    paragraphs = [
        block.strip() for block in re.split(r"\n\s*\n", stripped_text) if block.strip()
    ]
    return paragraphs


def format_duration(seconds: float) -> str:
    remaining_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(remaining_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def count_words(text: str) -> int:
    return len(text.split())


def chunk_paragraphs(
    paragraphs: List[str],
    max_paragraphs_per_chunk: int,
    max_words_per_chunk: int,
) -> List[List[str]]:
    if max_paragraphs_per_chunk <= 0:
        raise ValueError("Chunk size must be positive.")
    if max_words_per_chunk <= 0:
        raise ValueError("Max chunk words must be positive.")

    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = count_words(paragraph)

        if paragraph_word_count > max_words_per_chunk:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_word_count = 0
            logger.warning(
                f"Paragraph of {paragraph_word_count} words exceeds max chunk word limit {max_words_per_chunk}; placing in its own chunk."
            )
            chunks.append([paragraph])
            continue

        if not current_chunk:
            current_chunk = [paragraph]
            current_word_count = paragraph_word_count
            continue

        prospective_paragraph_count = len(current_chunk) + 1
        prospective_word_count = current_word_count + paragraph_word_count

        if (
            prospective_paragraph_count > max_paragraphs_per_chunk
            or prospective_word_count > max_words_per_chunk
        ):
            chunks.append(current_chunk)
            current_chunk = [paragraph]
            current_word_count = paragraph_word_count
        else:
            current_chunk.append(paragraph)
            current_word_count += paragraph_word_count

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def create_openrouter_client() -> OpenAI:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
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


NOTIFY_SEND_PATH = shutil.which("notify-send")
_NOTIFY_SEND_UNAVAILABLE_WARNING_EMITTED = False


def notify_missing_verification(
    chunk_index: int, total_chunks: int, assessment: str
) -> None:
    global _NOTIFY_SEND_UNAVAILABLE_WARNING_EMITTED
    title = "Integration verification missing content"
    body = f"Chunk {chunk_index + 1}/{total_chunks}: {assessment}"
    if NOTIFY_SEND_PATH:
        try:
            subprocess.run(
                [
                    NOTIFY_SEND_PATH,
                    "--app-name=IntegrateNotes",
                    title,
                    body,
                ],
                check=True,
            )
        except Exception as error:
            logger.warning(
                f"notify-send failed for verification chunk {chunk_index + 1}: {error}"
            )
    else:
        if not _NOTIFY_SEND_UNAVAILABLE_WARNING_EMITTED:
            logger.warning(
                "notify-send not available; desktop alerts for verification issues disabled."
            )
            _NOTIFY_SEND_UNAVAILABLE_WARNING_EMITTED = True


def execute_with_retry(
    operation: Callable[[], str],
    description: str,
    max_attempts: int = DEFAULT_MAX_RETRIES,
    initial_delay_seconds: float = RETRY_INITIAL_DELAY_SECONDS,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
) -> str:
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


def build_integration_prompt(
    grouping: str,
    current_body: str,
    chunk_text: str,
    failed_patches: List["PatchFailure"] | None = None,
    failed_duplications: List["DuplicationFailure"] | None = None,
    failed_formatting: str | None = None,
    previous_response: str | None = None,
) -> str:
    clarifications = (
        "You are integrating notes into the main body of the document incrementally. "
        f"Maintain the grouping approach: {grouping}."
    )
    response_instructions = (
        "Return exactly one JSON object matching the schema below. "
        "Use patches for edits and duplications for notes already present in the current document body. "
        "For each patch, search must be the exact text to find and replace must be the complete replacement text. "
        "For insertions, include the anchor text in both search and replace. "
        "For each duplication proof, notes must be the exact notes text already covered and body must be the exact current document body text that contains it. "
        "Emit patches in the order they should be applied. "
        "If any notes are already present in the document body and therefore do not need a patch, you must include a duplication proof entry for them. "
        "SEARCH and DUPLICATE/BODY text must each be a single contiguous span copied from the current document body; do not concatenate separate sections. "
        "Do not add commentary, numbering, markdown fences, or explanations outside the JSON object. "
        "Use empty patches and duplications arrays only when no changes are required and no duplication proofs are needed."
        f"\n<json_schema>\n{INTEGRATION_RESPONSE_SCHEMA_TEXT}\n</json_schema>"
    )

    sections = [
        f"<instructions>\n{INSTRUCTIONS_PROMPT.strip()}\n</instructions>",
        f"<clarifications>\n{clarifications}\n</clarifications>",
        "<context>",
        f"<current_document_body>\n{current_body}\n</current_document_body>",
        f"<scratchpad_chunk>\n{chunk_text}\n</scratchpad_chunk>",
        "</context>",
        f"<response_directive>{response_instructions}</response_directive>",
    ]

    if failed_formatting or failed_patches or failed_duplications:
        feedback_lines: List[str] = []
        if failed_formatting:
            feedback_lines.append(
                "The previous JSON response could not be parsed. Fix the issues below and re-emit only a valid JSON object."
            )
            feedback_lines.append(f"Error: {failed_formatting}")
        if failed_patches:
            feedback_lines.append(
                "The previous patch attempt failed because the search text below did not match the current document."
            )
            for failure in failed_patches:
                feedback_lines.append(
                    f"Patch {failure.index} SEARCH text (please adjust so it matches exactly):"
                )
                feedback_lines.append(failure.search_text)
                feedback_lines.append(f"Reason: {failure.reason}")
        if failed_duplications:
            feedback_lines.append(
                "The previous duplication proof attempt failed because the BODY text below did not match the current document."
            )
            for failure in failed_duplications:
                feedback_lines.append(
                    f"Duplication {failure.index} BODY text (please adjust so it matches exactly):"
                )
                feedback_lines.append(failure.body_text)
                feedback_lines.append(f"Reason: {failure.reason}")
        sections.append(
            "<previous_attempt_feedback>\n"
            + "\n\n".join(feedback_lines)
            + "\n</previous_attempt_feedback>"
        )

    if previous_response:
        sections.append(
            "<previous_json_response>\n"
            + previous_response
            + "\n</previous_json_response>"
        )

    return "\n\n\n\n\n".join(sections)


def request_integration(client: OpenAI, prompt: str, context_label: str) -> str:
    def perform_request() -> str:
        response = client.responses.create(
            model=DEFAULT_MODEL,
            reasoning=DEFAULT_REASONING,
            input=prompt,
            text={"format": INTEGRATION_RESPONSE_FORMAT},
            timeout=OPENROUTER_REQUEST_TIMEOUT_SECONDS,
        )
        if getattr(response, "error", None):
            raise RuntimeError(f"OpenRouter error for {context_label}: {response.error}")
        output_text = response.output_text
        if not output_text.strip():
            raise RuntimeError("Received empty response from GPT integration call.")
        return output_text.strip()

    return execute_with_retry(perform_request, f"integration {context_label}")


@dataclass(frozen=True)
class PatchInstruction:
    search_text: str
    replace_text: str


@dataclass(frozen=True)
class DuplicationProof:
    notes_text: str
    body_text: str


@dataclass(frozen=True)
class PatchFailure:
    index: int
    search_text: str
    reason: str


@dataclass(frozen=True)
class DuplicationFailure:
    index: int
    body_text: str
    reason: str


class IntegrationParseError(RuntimeError):
    def __init__(self, message: str, block_text: str) -> None:
        super().__init__(message)
        self.block_text = block_text


def _json_payload_text(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError:
        return repr(payload)


def _strip_json_code_fence(response_text: str) -> str:
    stripped = response_text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 3:
        return stripped

    opening = lines[0].strip().lower()
    if opening not in {"```", "```json"} or lines[-1].strip() != "```":
        return stripped

    logger.warning(
        "Integration response was wrapped in a markdown JSON fence; parsing the fenced JSON body."
    )
    return "\n".join(lines[1:-1]).strip()


def parse_integration_payload(
    response_text: str,
) -> tuple[List[PatchInstruction], List[DuplicationProof]]:
    if not response_text.strip():
        raise IntegrationParseError("Integration response is empty.", response_text)
    json_text = _strip_json_code_fence(response_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise IntegrationParseError(
            f"Integration response is not valid JSON: {error}",
            response_text,
        ) from error
    if not isinstance(payload, dict):
        raise IntegrationParseError(
            "Integration response must be a JSON object.",
            _json_payload_text(payload),
        )

    if payload.get("action") != "integrate":
        raise IntegrationParseError(
            "Integration JSON response must include action='integrate'.",
            _json_payload_text(payload),
        )

    patches = payload.get("patches")
    if not isinstance(patches, list):
        raise IntegrationParseError(
            "Integration JSON response must include a patches array.",
            _json_payload_text(payload),
        )
    duplication_payloads = payload.get("duplications")
    if not isinstance(duplication_payloads, list):
        raise IntegrationParseError(
            "Integration JSON response must include a duplications array.",
            _json_payload_text(payload),
        )

    instructions: List[PatchInstruction] = []
    duplications: List[DuplicationProof] = []

    for patch in patches:
        if not isinstance(patch, dict):
            raise IntegrationParseError(
                "Each patch must be an object.",
                _json_payload_text(payload),
            )
        search_text = patch.get("search")
        if not isinstance(search_text, str) or not search_text.strip():
            raise IntegrationParseError(
                "Each patch must include non-empty search text.",
                _json_payload_text(payload),
            )
        replace_text = patch.get("replace")
        if not isinstance(replace_text, str):
            raise IntegrationParseError(
                "Each patch must include string replace text.",
                _json_payload_text(payload),
            )
        instructions.append(
            PatchInstruction(search_text=search_text, replace_text=replace_text)
        )

    for duplication_payload in duplication_payloads:
        if not isinstance(duplication_payload, dict):
            raise IntegrationParseError(
                "Each duplication proof must be an object.",
                _json_payload_text(payload),
            )
        notes_text = duplication_payload.get("notes")
        body_text = duplication_payload.get("body")
        if not isinstance(notes_text, str) or not notes_text.strip():
            raise IntegrationParseError(
                "Each duplication proof must include non-empty notes text.",
                _json_payload_text(payload),
            )
        if not isinstance(body_text, str) or not body_text.strip():
            raise IntegrationParseError(
                "Each duplication proof must include non-empty body text.",
                _json_payload_text(payload),
            )
        duplications.append(
            DuplicationProof(notes_text=notes_text, body_text=body_text)
        )

    return instructions, duplications


def _build_whitespace_pattern(text: str, allow_zero: bool) -> re.Pattern[str]:
    if not text:
        raise ValueError("Cannot build whitespace pattern for empty text.")

    pieces: List[str] = []
    whitespace_token = r"\s*" if allow_zero else r"\s+"
    in_whitespace = False

    for char in text:
        if char.isspace():
            if not in_whitespace:
                pieces.append(whitespace_token)
                in_whitespace = True
        else:
            pieces.append(re.escape(char))
            in_whitespace = False

    pattern = "".join(pieces)
    if not pattern:
        pattern = whitespace_token
    return re.compile(pattern, flags=re.MULTILINE)


def _replace_slice(body: str, start: int, end: int, replacement: str) -> str:
    return body[:start] + replacement + body[end:]


def _format_patch_block(instruction: PatchInstruction) -> str:
    return (
        f"{PATCH_BLOCK_START}\n"
        f"{instruction.search_text}\n"
        f"{PATCH_BLOCK_DIVIDER}\n"
        f"{instruction.replace_text}\n"
        f"{PATCH_BLOCK_END}"
    )


def _format_duplication_block(proof: DuplicationProof) -> str:
    return (
        f"{DUPLICATION_BLOCK_START}\n"
        f"{proof.notes_text}\n"
        f"{PATCH_BLOCK_DIVIDER}\n"
        f"{proof.body_text}\n"
        f"{DUPLICATION_BLOCK_END}"
    )


def _locate_search_text(
    body: str, search_text: str
) -> tuple[int | None, int | None, str]:
    attempted_descriptions: List[str] = []

    index = body.find(search_text)
    attempted_descriptions.append("exact match")
    if index != -1:
        next_index = body.find(search_text, index + len(search_text))
        if next_index != -1:
            reason = (
                "SEARCH text matched multiple locations using exact match; "
                "increase SEARCH text length to match a longer, more specific span."
            )
            return None, None, reason
        return index, index + len(search_text), ""

    trimmed_newline_search = search_text.strip("\n")
    if trimmed_newline_search and trimmed_newline_search != search_text:
        attempted_descriptions.append("trimmed newline boundaries")
        index = body.find(trimmed_newline_search)
        if index != -1:
            next_index = body.find(
                trimmed_newline_search, index + len(trimmed_newline_search)
            )
            if next_index != -1:
                reason = (
                    "SEARCH text matched multiple locations using trimmed newline "
                    "boundaries; increase SEARCH text length to match a longer, more specific span."
                )
                return None, None, reason
            return index, index + len(trimmed_newline_search), ""

    trimmed_whitespace_search = search_text.strip()
    if trimmed_whitespace_search and trimmed_whitespace_search not in {
        search_text,
        trimmed_newline_search,
    }:
        attempted_descriptions.append("trimmed outer whitespace")
        index = body.find(trimmed_whitespace_search)
        if index != -1:
            next_index = body.find(
                trimmed_whitespace_search, index + len(trimmed_whitespace_search)
            )
            if next_index != -1:
                reason = (
                    "SEARCH text matched multiple locations using trimmed outer "
                    "whitespace; increase SEARCH text length to match a longer, more specific span."
                )
                return None, None, reason
            return index, index + len(trimmed_whitespace_search), ""

    if search_text.strip():
        pattern_whitespace = _build_whitespace_pattern(search_text, allow_zero=False)
        attempted_descriptions.append("normalized whitespace gaps")
        matches = list(pattern_whitespace.finditer(body))
        if matches:
            if len(matches) > 1:
                reason = (
                    "SEARCH text matched multiple locations using normalized whitespace "
                    "gaps; increase SEARCH text length to match a longer, more specific span."
                )
                return None, None, reason
            match = matches[0]
            return match.start(), match.end(), ""

        pattern_relaxed = _build_whitespace_pattern(search_text, allow_zero=True)
        attempted_descriptions.append("removed whitespace gaps")
        matches = list(pattern_relaxed.finditer(body))
        if matches:
            if len(matches) > 1:
                reason = (
                    "SEARCH text matched multiple locations using removed whitespace "
                    "gaps; increase SEARCH text length to match a longer, more specific span."
                )
                return None, None, reason
            match = matches[0]
            return match.start(), match.end(), ""

    reason = "SEARCH text not found after attempts: " + ", ".join(
        attempted_descriptions
    )
    return None, None, reason


def try_apply_patch(body: str, instruction: PatchInstruction) -> tuple[bool, str, str]:
    search_text = instruction.search_text
    replace_text = instruction.replace_text

    start, end, reason = _locate_search_text(body, search_text)
    if start is None or end is None:
        return False, body, reason

    updated = _replace_slice(body, start, end, replace_text)
    return True, updated, ""


def apply_patches_to_body(
    current_body: str, instructions: List[PatchInstruction], context_label: str
) -> tuple[str, List[PatchFailure]]:
    if not instructions:
        logger.debug(
            f"No patch content for {context_label}; retaining document body unchanged."
        )
        return current_body, []

    candidate_body = current_body
    for index, instruction in enumerate(instructions, start=1):
        success, updated, reason = try_apply_patch(candidate_body, instruction)
        if not success:
            failure = PatchFailure(
                index=index, search_text=instruction.search_text, reason=reason
            )
            logger.warning(f"Patch {index} failed for {context_label}: {reason}")
            logger.info(
                f"Failed patch block for {context_label} (patch {index}):\n"
                f"{_format_patch_block(instruction)}"
            )
            return current_body, [failure]
        candidate_body = updated

    return candidate_body, []


def validate_duplication_proofs(
    current_body: str, proofs: List[DuplicationProof], context_label: str
) -> List[DuplicationFailure]:
    failures: List[DuplicationFailure] = []
    for index, proof in enumerate(proofs, start=1):
        start, end, reason = _locate_search_text(current_body, proof.body_text)
        if start is None or end is None:
            failure = DuplicationFailure(
                index=index, body_text=proof.body_text, reason=reason
            )
            logger.warning(
                f"Duplication proof {index} failed for {context_label}: {reason}"
            )
            logger.info(
                f"Failed duplication block for {context_label} (duplication {index}):\n"
                f"{_format_duplication_block(proof)}"
            )
            failures.append(failure)
    return failures


def integrate_chunk_with_patches(
    client: OpenAI,
    grouping: str,
    base_body: str,
    chunk_text: str,
    context_label: str,
) -> tuple[str, List[PatchInstruction], List[DuplicationProof]]:
    failed_patches: List[PatchFailure] | None = None
    failed_duplications: List[DuplicationFailure] | None = None
    failed_formatting: str | None = None
    previous_response: str | None = None

    for attempt in range(1, MAX_PATCH_ATTEMPTS + 1):
        attempt_label = (
            context_label if attempt == 1 else f"{context_label} attempt {attempt}"
        )
        prompt = build_integration_prompt(
            grouping,
            base_body,
            chunk_text,
            failed_patches=failed_patches,
            failed_duplications=failed_duplications,
            failed_formatting=failed_formatting,
            previous_response=(
                previous_response
                if failed_formatting or failed_patches or failed_duplications
                else None
            ),
        )
        response_text = request_integration(client, prompt, attempt_label)
        previous_response = response_text

        try:
            instructions, duplications = parse_integration_payload(response_text)
        except IntegrationParseError as error:
            failed_formatting = str(error)
            failed_patches = None
            failed_duplications = None
            logger.warning(
                f"Integration response formatting failed for {attempt_label}: {error}"
            )
            logger.info(
                f"Invalid integration response for {attempt_label} on attempt {attempt}; "
                f"reason: {error}\nFailed payload:\n{error.block_text}"
            )
            logger.info(
                f"Retrying {context_label}; response formatting was invalid on attempt {attempt}."
            )
            continue
        except RuntimeError as error:
            failed_formatting = str(error)
            failed_patches = None
            failed_duplications = None
            logger.warning(
                f"Integration response formatting failed for {attempt_label}: {error}"
            )
            logger.info(
                f"Invalid integration response for {attempt_label} on attempt {attempt}; "
                f"reason: {error}"
            )
            logger.info(
                f"Retrying {context_label}; response formatting was invalid on attempt {attempt}."
            )
            continue
        failed_formatting = None
        updated_body, failures = apply_patches_to_body(
            base_body, instructions, attempt_label
        )

        if not failures:
            duplication_failures = validate_duplication_proofs(
                updated_body, duplications, attempt_label
            )
            if not duplication_failures:
                if failed_patches or failed_duplications:
                    logger.info(
                        f"Patches and duplication proofs succeeded for {context_label} on attempt {attempt}."
                    )
                return updated_body, instructions, duplications
            failures = []
            failed_duplications = duplication_failures
        else:
            failed_duplications = None

        failed_patches = failures
        logger.info(
            f"Retrying {context_label}; "
            f"{len(failed_patches)} patch(es) and "
            f"{len(failed_duplications or [])} duplication proof(s) failed to match."
        )

    raise RuntimeError(
        f"Unable to apply integration patches for {context_label} after {MAX_PATCH_ATTEMPTS} attempt(s)."
    )


def build_document(body: str, remaining_paragraphs: List[str]) -> str:
    trimmed_body = body.rstrip()
    document_parts = [trimmed_body, SCRATCHPAD_HEADING]
    if remaining_paragraphs:
        scratchpad_text = "\n\n".join(remaining_paragraphs).rstrip()
        document_parts.append(scratchpad_text)
    document = "\n\n".join(part for part in document_parts if part)
    if not document.endswith("\n"):
        document += "\n"
    return document


def format_verification_assessment(assessment: str) -> str:
    return (
        assessment.replace(" - Notes:", "\nNotes:")
        .replace(" Body:", "\nBody:")
        .replace(" Explanation:", "\nExplanation:")
    )


class VerificationManager:
    def __init__(self, client: OpenAI, target_file: Path) -> None:
        self.client = client
        self.pending_path = PENDING_VERIFICATION_PROMPTS_PATH
        self.lock = Lock()
        self.active_lock = Lock()
        self.active_ids: set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VERIFICATIONS)
        self.new_prompt_event = Event()
        self.stop_requested = False
        self.tracked_file_name = Path(target_file).resolve().name
        self.worker = Thread(
            target=self._run,
            name="VerificationManager",
            daemon=True,
        )
        self.worker.start()

    def enqueue_prompt(
        self,
        prompt: str,
        context_label: str | None,
        chunk_index: int | None,
        total_chunks: int | None,
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Verification prompt must be a non-empty string.")

        entry = {
            "id": str(uuid4()),
            "prompt": prompt,
            "context_label": context_label,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "file_name": self.tracked_file_name,
        }
        with self.lock:
            entries = self._read_entries_locked()
            entries.append(entry)
            self._write_entries_locked(entries)
        self.new_prompt_event.set()

    def shutdown(self) -> None:
        self.stop_requested = True
        self.new_prompt_event.set()
        if self.worker.is_alive():
            self.worker.join()
        self.executor.shutdown(wait=True)

    def _run(self) -> None:
        while True:
            try:
                self._dispatch_pending()
            except Exception as error:
                logger.exception(
                    f"Verification dispatcher encountered an error: {error}"
                )
            if self.stop_requested and not self._has_pending_work():
                break
            self.new_prompt_event.wait(timeout=0.5)
            self.new_prompt_event.clear()

    def _dispatch_pending(self) -> None:
        with self.lock:
            all_entries = self._read_entries_locked()
            entries = self._entries_for_current_file_locked(all_entries)

        for entry in entries:
            entry_id = entry.get("id")
            if not entry_id:
                continue
            with self.active_lock:
                if entry_id in self.active_ids:
                    continue
                self.active_ids.add(entry_id)

            future = self.executor.submit(self._send_prompt, entry)
            future.add_done_callback(
                lambda fut, data=entry: self._handle_result(data, fut)
            )

    def _send_prompt(self, entry: dict[str, Any]) -> str:
        context_label = entry.get("context_label") or "verification"
        prompt = entry["prompt"]
        return request_verification(self.client, prompt, context_label)

    def _handle_result(self, entry: dict[str, Any], future) -> None:
        entry_id = entry.get("id")
        try:
            assessment = future.result()
        except Exception as error:  # noqa: BLE001
            context_label = entry.get("context_label") or "verification"
            logger.exception(f"Verification for {context_label} failed: {error}")
            if entry_id:
                with self.active_lock:
                    self.active_ids.discard(entry_id)
            self.new_prompt_event.set()
            return

        self._log_assessment(entry, assessment)

        if entry_id:
            self._remove_entry(entry_id)
            with self.active_lock:
                self.active_ids.discard(entry_id)

        self.new_prompt_event.set()

    def _log_assessment(self, entry: dict[str, Any], assessment: str) -> None:
        chunk_index = entry.get("chunk_index")
        total_chunks = entry.get("total_chunks")
        context_label = entry.get("context_label") or "verification"
        file_name = entry.get("file_name")

        if not file_name:
            raise RuntimeError(
                "Verification entry missing required file_name; pending prompts file may be corrupted."
            )

        base_header = f'Verification "{file_name}"'

        if (
            isinstance(chunk_index, int)
            and isinstance(total_chunks, int)
            and 0 <= chunk_index < total_chunks
        ):
            if "MISSING" in assessment:
                notify_missing_verification(chunk_index, total_chunks, assessment)
            chunk_header = f"{base_header}:"
            if assessment.startswith(chunk_header):
                logger.info(assessment)
            else:
                logger.info(f"{chunk_header}\n{assessment}")
        else:
            if context_label != "verification":
                header = f"{base_header} ({context_label}):"
            else:
                header = f"{base_header}:"
            if assessment.startswith(header):
                logger.info(assessment)
            else:
                logger.info(f"{header}\n{assessment}")

    def _remove_entry(self, entry_id: str) -> None:
        with self.lock:
            entries = self._read_entries_locked()
            remaining = [item for item in entries if item.get("id") != entry_id]
            self._write_entries_locked(remaining)

    def _read_entries_locked(self) -> List[dict[str, Any]]:
        if not self.pending_path.exists():
            return []
        raw = self.pending_path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Pending verification prompts file {self.pending_path} is corrupted: {error}"
            ) from error
        if not isinstance(data, list):
            raise RuntimeError(
                f"Pending verification prompts file {self.pending_path} must contain a list."
            )
        return data

    def _write_entries_locked(self, entries: List[dict[str, Any]]) -> None:
        payload = json.dumps(entries, ensure_ascii=True, indent=2)
        self.pending_path.write_text(payload, encoding="utf-8")

    def _has_pending_work(self) -> bool:
        with self.lock:
            entries = self._read_entries_locked()
            has_entries = bool(self._entries_for_current_file_locked(entries))
        with self.active_lock:
            has_active = bool(self.active_ids)
        return has_entries or has_active

    def _entries_for_current_file_locked(
        self, entries: List[dict[str, Any]]
    ) -> List[dict[str, Any]]:
        invalid_entries: List[dict[str, Any]] = []
        relevant_entries: List[dict[str, Any]] = []

        for entry in entries:
            file_name = entry.get("file_name")
            entry_id = entry.get("id")
            if not file_name or not entry_id:
                invalid_entries.append(entry)
                continue
            if file_name == self.tracked_file_name:
                relevant_entries.append(entry)

        if invalid_entries:
            invalid_count = len(invalid_entries)
            suffix = "y" if invalid_count == 1 else "ies"
            logger.warning(
                f"Removed {invalid_count} invalid verification prompt entr{suffix} missing file metadata or IDs."
            )
            cleaned_entries = [
                entry for entry in entries if entry not in invalid_entries
            ]
            self._write_entries_locked(cleaned_entries)

        return relevant_entries


def build_verification_prompt(
    chunk_text: str,
    patch_replacements: Sequence[str],
    duplication_proofs: Sequence[DuplicationProof],
    context_label: str | None = None,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> str:
    response_instructions = (
        "Report whether any note content is missing or materially altered."
        " Respond with a concise single paragraph beginning with 'OK -' if everything is covered"
        " or 'MISSING -' followed by details of any omissions."
        " Seperate each omission by two newlines and for each omission, provide the following:\n"
        '    Notes:"..."\n'
        '    Body:"..."\n'
        '    Explanation: "..."\n'
        '    Proposed Fix: "..."\n'
        'Quote the exact text from the notes chunk containing the missing detail and quote the exact passage from the patch replacements or duplication proofs that should cover it (or state Body:"<not present>" if nothing is relevant).'
        " Explain precisely what information is still missing or altered without omitting any nuance."
    )

    if patch_replacements:
        replacement_sections = []
        for index, replacement_text in enumerate(patch_replacements, start=1):
            replacement_sections.append(
                f"[Patch {index} Replacement]\n{replacement_text}"
            )
        replacements_block = "\n\n".join(replacement_sections)
    else:
        replacements_block = "<no patch replacements provided>"

    if duplication_proofs:
        duplication_sections = []
        for index, proof in enumerate(duplication_proofs, start=1):
            duplication_sections.append(
                "[Duplication {index} Proof]\nNotes:\n{notes}\n\nBody:\n{body}".format(
                    index=index, notes=proof.notes_text, body=proof.body_text
                )
            )
        duplications_block = "\n\n".join(duplication_sections)
    else:
        duplications_block = "<no duplication proofs provided>"

    sections = [
        (
            "<task>"
            "You are verifying that every idea/point/concept/argument/detail/url/[[wikilink]]/diagram etc. "
            "from the provided notes chunk has been integrated into the document body."
            " Use the patch replacements to understand what will be inserted or rewritten."
            " Duplication proofs are not edits; they are evidence of existing body text that already covers notes."
            " Use duplication proofs as evidence of where notes are already present in the body without a patch."
            " If a duplication proof does not fully cover the notes text, treat the missing detail as missing."
            "</task>"
        ),
        f"<notes_chunk>\n{chunk_text}\n</notes_chunk>",
        f"<patch_replacements>\n{replacements_block}\n</patch_replacements>",
        f"<duplication_proofs>\n{duplications_block}\n</duplication_proofs>",
        f"<response_guidelines>\n{response_instructions}\n</response_guidelines>",
    ]
    prompt = "\n\n\n\n\n".join(sections)
    return prompt


def request_verification(client: OpenAI, prompt: str, context_label: str) -> str:
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
            raise RuntimeError("Received empty response from GPT verification call.")
        return output_text.strip()

    return execute_with_retry(perform_request, f"verification {context_label}")


def resolve_git_root(source_path: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Failed to locate git repository for {source_path}: {error}"
        ) from error
    return Path(result.stdout.strip()).resolve()


def commit_and_push_original(source_path: Path) -> None:
    repo_root = resolve_git_root(source_path)
    try:
        relative_source = source_path.resolve().relative_to(repo_root)
    except ValueError as error:
        raise RuntimeError(
            f"Source file {source_path} is not within git repository {repo_root}."
        ) from error

    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "add", str(relative_source)], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "commit",
                "--allow-empty",
                "-m",
                f"chore: checkpoint before integrating {source_path.name}",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo_root), "push"], check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Failed to commit and push before integration: {error}"
        ) from error


def refresh_scratchpad_paragraphs(
    source_path: Path, expected_prefix: List[str] | None
) -> List[str]:
    latest_content = source_path.read_text(encoding="utf-8")
    _, scratchpad = split_document_sections(latest_content)
    scratchpad_paragraphs = normalize_paragraphs(scratchpad)
    if expected_prefix is not None:
        prefix_length = len(expected_prefix)
        if scratchpad_paragraphs[:prefix_length] != expected_prefix:
            raise RuntimeError(
                "Scratchpad changed in a non-append-only way while integration was running."
            )
    return scratchpad_paragraphs


def integrate_notes(
    source_path: Path,
    grouping: str | None,
    max_paragraphs_per_chunk: int,
    max_words_per_chunk: int,
    disable_verification: bool,
) -> Path:
    source_content = source_path.read_text(encoding="utf-8")
    source_body, source_scratchpad = split_document_sections(source_content)
    resolved_grouping = grouping or read_frontmatter_field(source_body, GROUPING_FIELD)
    if not resolved_grouping:
        resolved_grouping = prompt_for_grouping()
        logger.info("Recorded new grouping approach from user input.")

    working_body = (
        source_body
        if grouping is None and read_frontmatter_field(source_body, GROUPING_FIELD)
        else set_frontmatter_block_field(source_body, GROUPING_FIELD, resolved_grouping)
    )
    commit_and_push_original(source_path)
    scratchpad_paragraphs = normalize_paragraphs(source_scratchpad)
    client = create_openrouter_client()
    verification_manager = (
        None if disable_verification else VerificationManager(client, source_path)
    )

    try:
        if not scratchpad_paragraphs:
            logger.info(
                "No scratchpad notes to integrate; ensuring scratchpad heading remains present."
            )
            source_path.write_text(
                build_document(working_body, []),
                encoding="utf-8",
            )
            return source_path

        current_body = working_body
        last_written_remaining: List[str] | None = scratchpad_paragraphs.copy()
        chunks_completed = 0
        integration_start = perf_counter()

        while True:
            scratchpad_paragraphs = refresh_scratchpad_paragraphs(
                source_path, last_written_remaining
            )
            remaining_paragraphs = scratchpad_paragraphs
            if not remaining_paragraphs:
                break

            paragraph_chunks = chunk_paragraphs(
                remaining_paragraphs,
                max_paragraphs_per_chunk,
                max_words_per_chunk,
            )
            total_chunks = chunks_completed + len(paragraph_chunks)
            chunk = paragraph_chunks[0]
            chunk_text = "\n\n".join(chunk)
            chunk_word_count = sum(count_words(paragraph) for paragraph in chunk)
            chunk_index = chunks_completed
            chunk_label = f"chunk {chunks_completed + 1}/{total_chunks}"
            logger.info(
                f"Integrating chunk {chunks_completed + 1} of {total_chunks} containing {len(chunk)} paragraphs and {chunk_word_count} words."
            )
            updated_body, patch_instructions, duplication_proofs = (
                integrate_chunk_with_patches(
                    client,
                    resolved_grouping,
                    current_body,
                    chunk_text,
                    chunk_label,
                )
            )
            if verification_manager is not None:
                patch_replacements = [
                    instruction.replace_text for instruction in patch_instructions
                ]
                verification_prompt = build_verification_prompt(
                    chunk_text,
                    patch_replacements,
                    duplication_proofs,
                    chunk_label,
                    chunk_index,
                    total_chunks,
                )
                verification_manager.enqueue_prompt(
                    verification_prompt,
                    chunk_label,
                    chunk_index,
                    total_chunks,
                )

            current_body = updated_body
            refreshed_paragraphs = refresh_scratchpad_paragraphs(
                source_path, last_written_remaining
            )
            remaining_paragraphs = refreshed_paragraphs[len(chunk) :]
            integrated_document = build_document(current_body, remaining_paragraphs)
            source_path.write_text(integrated_document, encoding="utf-8")
            logger.info(
                f'Chunk {chunks_completed + 1} integration written to "{source_path}".'
            )
            last_written_remaining = remaining_paragraphs
            chunks_completed += 1
            remaining_chunks = total_chunks - chunks_completed
            if remaining_chunks > 0:
                elapsed_seconds = perf_counter() - integration_start
                average_duration = elapsed_seconds / chunks_completed
                estimated_seconds_remaining = average_duration * remaining_chunks
                logger.info(
                    f"Estimated time remaining: {format_duration(estimated_seconds_remaining)}"
                    f" for {remaining_chunks} remaining chunk(s)."
                )

        logger.info("All scratchpad notes integrated; scratchpad section cleared.")
        return source_path
    finally:
        if verification_manager is not None:
            verification_manager.shutdown()


def continuous_organise_paths(notes_root: Path) -> list[Path]:
    if not notes_root.exists():
        raise FileNotFoundError(f"Notes root not found: {notes_root}")
    if not notes_root.is_dir():
        raise NotADirectoryError(f"Notes root is not a directory: {notes_root}")

    paths: list[Path] = []
    for path in sorted(notes_root.rglob("*.md")):
        if any(part.startswith(".") for part in path.relative_to(notes_root).parts):
            continue
        content = path.read_text(encoding="utf-8")
        if not frontmatter_field_equals(
            content, ORGANISE_FIELD, CONTINUOUS_ORGANISE_VALUE
        ):
            continue
        if not read_frontmatter_field(content, GROUPING_FIELD):
            content = set_frontmatter_block_field(
                content, GROUPING_FIELD, DEFAULT_GROUPING
            )
            path.write_text(content, encoding="utf-8")
            logger.warning(
                f"{path} is marked {ORGANISE_FIELD}: {CONTINUOUS_ORGANISE_VALUE} "
                f"but had no {GROUPING_FIELD} frontmatter; added default grouping."
            )
        _, scratchpad = split_document_sections(content)
        if normalize_paragraphs(scratchpad):
            paths.append(path)
    return paths


def integrate_continuous_notes(
    notes_root: Path,
    max_paragraphs_per_chunk: int,
    max_words_per_chunk: int,
    disable_verification: bool,
) -> list[Path]:
    source_paths = continuous_organise_paths(notes_root)
    for source_path in source_paths:
        logger.info(f"Integrating continuously organised note {source_path}.")
        integrate_notes(
            source_path,
            grouping=None,
            max_paragraphs_per_chunk=max_paragraphs_per_chunk,
            max_words_per_chunk=max_words_per_chunk,
            disable_verification=disable_verification,
        )
    return source_paths


def main() -> None:
    configure_logging()
    try:
        args = parse_arguments()
        if args.continuous:
            integrated_paths = integrate_continuous_notes(
                Path(args.notes_root).expanduser().resolve(),
                args.chunk_size,
                args.max_chunk_words,
                args.disable_verification,
            )
            logger.info(
                f"Continuous integration completed for {len(integrated_paths)} note(s)."
            )
        else:
            source_path = resolve_source_path(args.source)
            integrated_path = integrate_notes(
                source_path,
                args.grouping,
                args.chunk_size,
                args.max_chunk_words,
                args.disable_verification,
            )
            logger.info(
                f"Integration completed. Updated document available at {integrated_path}."
            )
    except Exception as error:
        logger.exception(f"Integration failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
