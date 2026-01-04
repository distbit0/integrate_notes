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

from loguru import logger
from openai import OpenAI

SCRATCHPAD_HEADING = "# -- SCRATCHPAD"
GROUPING_PREFIX = "Grouping approach: "
DEFAULT_CHUNK_PARAGRAPHS = 30
DEFAULT_CHUNK_MAX_WORDS = 400
ENV_API_KEY = "OPENAI_API_KEY"
DEFAULT_MAX_RETRIES = 3
RETRY_INITIAL_DELAY_SECONDS = 2.0
RETRY_BACKOFF_FACTOR = 2.0
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
        help="Grouping approach to record at the top of the document.",
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


def extract_grouping(body: str) -> str | None:
    lines = body.splitlines()
    line_index = 0

    if lines and lines[0].strip() == "---":
        line_index += 1
        # Skip YAML front matter so grouping detection does not stop there.
        while line_index < len(lines):
            if lines[line_index].strip() == "---":
                line_index += 1
                break
            line_index += 1

    for line in lines[line_index:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(GROUPING_PREFIX.lower()):
            return stripped[len(GROUPING_PREFIX) :].strip()
    return None


def ensure_grouping_record(body: str, grouping: str) -> str:
    normalized_grouping = grouping.strip()
    if not normalized_grouping:
        raise ValueError("Grouping approach cannot be empty.")

    grouping_line = f"{GROUPING_PREFIX}{normalized_grouping}"

    if not body.strip():
        return f"{grouping_line}\n"

    lines = body.splitlines()

    insertion_index = 0
    total_lines = len(lines)
    while insertion_index < total_lines and not lines[insertion_index].strip():
        insertion_index += 1

    if insertion_index < total_lines and lines[insertion_index].strip() == "---":
        insertion_index += 1
        while insertion_index < total_lines and lines[insertion_index].strip() != "---":
            insertion_index += 1
        if insertion_index < total_lines:
            insertion_index += 1
        while insertion_index < total_lines and not lines[insertion_index].strip():
            insertion_index += 1

    existing_indices = [
        index
        for index, line in enumerate(lines)
        if line.strip().lower().startswith(GROUPING_PREFIX.lower())
    ]

    if existing_indices:
        first_index = existing_indices[0]
        for duplicate_index in sorted(existing_indices[1:], reverse=True):
            lines.pop(duplicate_index)
        if first_index < insertion_index:
            insertion_index -= 1
        lines.pop(first_index)

    lines.insert(insertion_index, grouping_line)

    next_index = insertion_index + 1
    if next_index >= len(lines) or lines[next_index].strip():
        lines.insert(next_index, "")

    return "\n".join(lines)


def prompt_for_grouping() -> str:
    grouping = input(
        f"""Grouping not found. Provide the text that should follow {GROUPING_PREFIX} at the top of the document.
Examples:
- Grouping approach: Group points according to what problem each idea/proposal/mechanism/concept addresses/are trying to solve, which you will need to figure out yourself based on context. Do not combine multiple goals/problems into one group. Keep goals/problems specific. Ensure groups are mutually exclusive and collectively exhaustive. Avoid overlap between group's goals/problems. sub-headings should be per-mechanism/per-solution i.e. according to which "idea"/solution each point relates to.
- Group points according to what you think the most useful/interesting/relevant groupings are. Ensure similar, related and contradictory points are adjacent.
Your input: """
    ).strip()
    if not grouping:
        grouping = "Group points according to what you think the most useful/interesting/relevant groupings are. Ensure similar, related and contradictory points are adjacent."
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


def create_openai_client() -> OpenAI:
    api_key = os.getenv(ENV_API_KEY)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {ENV_API_KEY} is required for GPT access."
        )
    return OpenAI(api_key=api_key)


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
                    f"OpenAI {description} failed after {max_attempts} attempt(s): {error}"
                )
                raise
            logger.warning(
                f"OpenAI {description} attempt {attempt} failed: {error}. Retrying in {delay:.1f}s."
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
        "Return only patch instructions and duplication proofs using the exact structures below."
        " For each patch:"
        f"\n{PATCH_BLOCK_START}\n<text to find>\n{PATCH_BLOCK_DIVIDER}\n<replacement text>\n{PATCH_BLOCK_END}"
        "\nFor each duplication proof (use this when notes are already present in the current document body, so no patch is needed):"
        f"\n{DUPLICATION_BLOCK_START}\n<notes text already covered>\n{PATCH_BLOCK_DIVIDER}\n<body text that already contains it>\n{DUPLICATION_BLOCK_END}"
        "\nEmit the blocks back-to-back in the order they should be applied or checked. "
        "If any notes are already present in the document body and therefore do not need a patch, you must include a duplication proof block for them. "
        "Do not add commentary, numbering, markdown fences, or explanations. "
        "If no changes are required and no duplication proofs are needed, return an empty string."
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
                "The previous response could not be parsed. Fix the formatting issues below and re-emit only valid blocks."
            )
            feedback_lines.append(f"Error: {failed_formatting}")
        if failed_patches:
            feedback_lines.append(
                "The previous patch attempt failed because the SEARCH block(s) below did not match the current document."
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
            "<previous_patch_response>\n"
            + previous_response
            + "\n</previous_patch_response>"
        )

    return "\n\n\n\n\n".join(sections)


def request_integration(client: OpenAI, prompt: str, context_label: str) -> str:
    def perform_request() -> str:
        response = client.responses.create(
            model="gpt-5.2",
            reasoning={"effort": "medium"},
            input=prompt,
        )
        output_text = response.output_text
        if not output_text.strip():
            raise RuntimeError("Received empty response from GPT integration call.")
        patch_text = extract_patch_text_from_response(output_text)
        # logger.debug(f"Integration patches for {context_label}:\n{patch_text}")
        return patch_text

    return execute_with_retry(perform_request, f"integration {context_label}")


def extract_patch_text_from_response(response_text: str) -> str:
    stripped = response_text.strip()
    if not stripped:
        return ""

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if not lines:
            return ""
        lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines.pop()
        stripped = "\n".join(lines).strip()

    return stripped


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


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sanitize_patch_segment(segment: str) -> str:
    cleaned = _normalize_line_endings(segment)
    if cleaned.startswith("\n"):
        cleaned = cleaned[1:]
    if cleaned.endswith("\n"):
        cleaned = cleaned[:-1]
    return cleaned


def _find_next_block_start(
    text: str, position: int, markers: Sequence[str]
) -> tuple[int, str] | None:
    candidates: List[tuple[int, str]] = []
    for marker in markers:
        index = text.find(marker, position)
        if index != -1:
            candidates.append((index, marker))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])


def parse_integration_blocks(
    response_text: str,
) -> tuple[List[PatchInstruction], List[DuplicationProof]]:
    if not response_text.strip():
        return [], []

    cleaned = _normalize_line_endings(response_text)
    instructions: List[PatchInstruction] = []
    duplications: List[DuplicationProof] = []
    position = 0

    while True:
        next_block = _find_next_block_start(
            cleaned, position, [PATCH_BLOCK_START, DUPLICATION_BLOCK_START]
        )
        if next_block is None:
            remaining = cleaned[position:].strip()
            if remaining:
                logger.warning(
                    "Ignoring unexpected content outside integration blocks: {}".format(
                        remaining[:120]
                    )
                )
            break

        start_index, block_start = next_block
        divider_index = cleaned.find(
            PATCH_BLOCK_DIVIDER, start_index + len(block_start)
        )
        if divider_index == -1:
            raise RuntimeError(
                "Integration block is missing the divider '{}'.".format(
                    PATCH_BLOCK_DIVIDER
                )
            )

        if block_start == PATCH_BLOCK_START:
            block_end = PATCH_BLOCK_END
        else:
            block_end = DUPLICATION_BLOCK_END

        end_index = cleaned.find(block_end, divider_index + len(PATCH_BLOCK_DIVIDER))
        if end_index == -1:
            raise RuntimeError(
                "Integration block is missing the end marker '{}'.".format(block_end)
            )

        first_segment = cleaned[start_index + len(block_start) : divider_index]
        second_segment = cleaned[divider_index + len(PATCH_BLOCK_DIVIDER) : end_index]

        first_text = _sanitize_patch_segment(first_segment)
        second_text = _sanitize_patch_segment(second_segment)

        if block_start == PATCH_BLOCK_START:
            if not first_text.strip():
                raise RuntimeError(
                    "Patch SEARCH text must contain non-whitespace characters."
                )
            instructions.append(
                PatchInstruction(search_text=first_text, replace_text=second_text)
            )
        else:
            if not first_text.strip() or not second_text.strip():
                raise RuntimeError(
                    "Duplication block must include non-whitespace notes and body text."
                )
            duplications.append(
                DuplicationProof(notes_text=first_text, body_text=second_text)
            )

        position = end_index + len(block_end)

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


def _locate_search_text(body: str, search_text: str) -> tuple[int | None, int | None, str]:
    attempted_descriptions: List[str] = []

    index = body.find(search_text)
    attempted_descriptions.append("exact match")
    if index != -1:
        return index, index + len(search_text), ""

    trimmed_newline_search = search_text.strip("\n")
    if trimmed_newline_search and trimmed_newline_search != search_text:
        attempted_descriptions.append("trimmed newline boundaries")
        index = body.find(trimmed_newline_search)
        if index != -1:
            return index, index + len(trimmed_newline_search), ""

    trimmed_whitespace_search = search_text.strip()
    if trimmed_whitespace_search and trimmed_whitespace_search not in {
        search_text,
        trimmed_newline_search,
    }:
        attempted_descriptions.append("trimmed outer whitespace")
        index = body.find(trimmed_whitespace_search)
        if index != -1:
            return index, index + len(trimmed_whitespace_search), ""

    if search_text.strip():
        pattern_whitespace = _build_whitespace_pattern(search_text, allow_zero=False)
        attempted_descriptions.append("normalized whitespace gaps")
        match = pattern_whitespace.search(body)
        if match:
            return match.start(), match.end(), ""

        pattern_relaxed = _build_whitespace_pattern(search_text, allow_zero=True)
        attempted_descriptions.append("removed whitespace gaps")
        match = pattern_relaxed.search(body)
        if match:
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
            previous_response=previous_response
            if failed_formatting or failed_patches or failed_duplications
            else None,
        )
        patch_text = request_integration(client, prompt, attempt_label)
        previous_response = patch_text

        try:
            instructions, duplications = parse_integration_blocks(patch_text)
        except RuntimeError as error:
            failed_formatting = str(error)
            failed_patches = None
            failed_duplications = None
            logger.warning(
                f"Integration response formatting failed for {attempt_label}: {error}"
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
        "Quote the exact text from the notes chunk containing the missing detail and quote the exact passage from the patch replacements or duplication proofs that should cover it (or state Body:\"<not present>\" if nothing is relevant)."
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
            model="gpt-5.2",
            reasoning={"effort": "medium"},
            input=prompt,
        )
        output_text = response.output_text
        if not output_text.strip():
            raise RuntimeError("Received empty response from GPT verification call.")
        return output_text.strip()

    return execute_with_retry(perform_request, f"verification {context_label}")


def commit_and_push_original(source_path: Path) -> None:
    try:
        subprocess.run(["git", "add", str(source_path)], check=True)
        subprocess.run(
            [
                "git",
                "commit",
                "--allow-empty",
                "-m",
                f"chore: checkpoint before integrating {source_path.name}",
            ],
            check=True,
        )
        subprocess.run(["git", "push"], check=True)
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
) -> Path:
    source_content = source_path.read_text(encoding="utf-8")
    source_body, source_scratchpad = split_document_sections(source_content)
    working_body = source_body

    resolved_grouping = (
        grouping or extract_grouping(working_body) or extract_grouping(source_body)
    )
    if not resolved_grouping:
        resolved_grouping = prompt_for_grouping()
        logger.info("Recorded new grouping approach from user input.")

    working_body = ensure_grouping_record(working_body, resolved_grouping)
    commit_and_push_original(source_path)
    scratchpad_paragraphs = normalize_paragraphs(source_scratchpad)
    client = create_openai_client()
    verification_manager = VerificationManager(client, source_path)

    try:
        if not scratchpad_paragraphs:
            logger.info(
                "No scratchpad notes to integrate; ensuring scratchpad heading remains present."
            )
            source_path.write_text(build_document(working_body, []), encoding="utf-8")
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
        verification_manager.shutdown()


def main() -> None:
    configure_logging()
    try:
        args = parse_arguments()
        source_path = resolve_source_path(args.source)
        integrated_path = integrate_notes(
            source_path,
            args.grouping,
            args.chunk_size,
            args.max_chunk_words,
        )
        logger.info(
            f"Integration completed. Updated document available at {integrated_path}."
        )
    except Exception as error:
        logger.exception(f"Integration failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
