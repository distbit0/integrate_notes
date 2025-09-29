import argparse
import json
import os
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable, List, Tuple

import shutil
import subprocess

from loguru import logger
from openai import OpenAI

SCRATCHPAD_HEADING = "# -- SCRATCHPAD"
GROUPING_PREFIX = "Grouping approach: "
DEFAULT_CHUNK_PARAGRAPHS = 30
DEFAULT_CHUNK_MAX_WORDS = 900
ENV_API_KEY = "OPENAI_API_KEY"
DEFAULT_MAX_RETRIES = 3
RETRY_INITIAL_DELAY_SECONDS = 2.0
RETRY_BACKOFF_FACTOR = 2.0
LAST_VERIFICATION_PROMPT_PATH = (
    Path(__file__).resolve().parent / ".last_verification_prompt.json"
)
INSTRUCTIONS_PROMPT = """# Instructions

- Integrate the provided notes into the document body, following the document's existing structure and the specified grouping approach.
- Break content into relatively atomic bullet points; each bullet should express one idea.
- Use nested bullets when a point is naturally a sub-point of another.
- Make minor grammar edits as needed so ideas read cleanly as bullet points.
- De-duplicate overlapping points without losing any nuance or detail.
- Keep wording succinct and remove filler words (e.g., "you know", "basically", "essentially", "uh").
- Add new headings, sub-headings, or parent bullet points for new items, and reuse existing ones where appropriate.
- Refactor existing content as needed to smoothly integrate the new notes.


# Rules

- Do not omit any details, nuance, points, conclusions, ideas, arguments, or qualifications.
- Do not materially alter meaning.
- DO NOT LEAVE OUT ANY DETAILS, EVEN IF THERE ARE A VERY LARGE NUMBER OF NOTES.
- If new items do not match existing items in the document body, add them appropriately.
- Preserve questions as questions; do not convert them into statements.
- Do not guess acronym expansions if they are not specified.
- Do not modify tone (e.g., confidence/certainty) or add hedging.
- Do not omit any links, wikilinks, URLs, diagrams, ASCII art, mathematics, tables, figures, or other non-text content.
- Keep each link/URL/etc. in the section where it is most relevant based on its surrounding context and its URL text.
    - Do not move links to a separate “resources” or “links” section.
- Do not modify any wikilinks or URLs.


# Formatting

- Use markdown headings with "#", "##", "###", etc.
- Use "- " as the bullet prefix (not "* ", "-  ", or anything else).
    - Use four spaces for each level of bullet-point nesting.


# Before finishing: check your work

- Confirm every item from the provided notes is now represented in the document body without loss of detail.
- Ensure nothing from the original document body was lost.
- If anything is missing, integrate it in appropriately.
"""


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=True)


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


def ensure_integrated_copy(source_path: Path) -> Tuple[Path, bool]:
    integrated_name = f"{source_path.stem}_integrated{source_path.suffix}"
    integrated_path = source_path.with_name(integrated_name)
    if integrated_path.exists():
        return integrated_path, False
    integrated_path.write_text(
        source_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    logger.info(f"Created working copy at {integrated_path}.")
    return integrated_path, True


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
    lines = body.splitlines()
    first_non_empty_index = None
    for index, line in enumerate(lines):
        if line.strip():
            first_non_empty_index = index
            break
    grouping_line = f"{GROUPING_PREFIX} {grouping}"
    if first_non_empty_index is None:
        updated_lines = [grouping_line, ""]
        return "\n".join(updated_lines)
    existing_line = lines[first_non_empty_index]
    stripped = existing_line.strip()
    lowered = stripped.lower()
    if lowered.startswith(GROUPING_PREFIX.lower()):
        lines[first_non_empty_index] = grouping_line
        return "\n".join(lines)
    lines.insert(first_non_empty_index, grouping_line)
    return "\n".join(lines)


def prompt_for_grouping() -> str:
    grouping = input(
        f"Grouping not found. Provide the text that should follow {GROUPING_PREFIX} at the top of the document: "
    ).strip()
    if not grouping:
        raise ValueError("Grouping approach cannot be empty.")
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
) -> str:
    clarifications = (
        "You are integrating notes into the main body of the document incrementally. "
        "For this request, integrate the provided chunk of notes and return only the updated document body text. "
        f"Maintain the grouping approach: {grouping}."
    )
    sections = [
        f"<instructions>\n{INSTRUCTIONS_PROMPT.strip()}\n</instructions>",
        f"<clarifications>\n{clarifications}\n</clarifications>",
        "<context>",
        f"<current_document_body>\n{current_body}\n</current_document_body>",
        f"<scratchpad_chunk>\n{chunk_text}\n</scratchpad_chunk>",
        "</context>",
        "<response_directive>Return only the updated document body.</response_directive>",
    ]
    return "\n\n\n\n\n".join(sections)


def request_integration(client: OpenAI, prompt: str, context_label: str) -> str:
    def perform_request() -> str:
        response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "medium"},
            input=prompt,
        )
        output_text = response.output_text
        if not output_text.strip():
            raise RuntimeError("Received empty response from GPT integration call.")
        return output_text.strip()

    return execute_with_retry(perform_request, f"integration {context_label}")


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


def persist_pending_verification_prompt(
    prompt: str,
    context_label: str | None,
    chunk_index: int | None,
    total_chunks: int | None,
) -> None:
    payload = {
        "prompt": prompt,
        "context_label": context_label,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    try:
        LAST_VERIFICATION_PROMPT_PATH.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )
    except OSError as error:
        raise RuntimeError(
            f"Failed to persist verification prompt to {LAST_VERIFICATION_PROMPT_PATH}: {error}"
        ) from error


def load_pending_verification_prompt() -> dict | None:
    if not LAST_VERIFICATION_PROMPT_PATH.exists():
        return None
    try:
        raw = LAST_VERIFICATION_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Pending verification prompt file {LAST_VERIFICATION_PROMPT_PATH} is corrupted: {error}"
        ) from error
    if "prompt" not in data:
        raise RuntimeError(
            f"Pending verification prompt file {LAST_VERIFICATION_PROMPT_PATH} is missing required fields."
        )
    return data


def clear_pending_verification_prompt(
    expected_prompt: str,
    expected_context_label: str | None,
) -> None:
    if not LAST_VERIFICATION_PROMPT_PATH.exists():
        return
    try:
        raw = LAST_VERIFICATION_PROMPT_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except FileNotFoundError:
        return
    except Exception as error:  # noqa: BLE001
        logger.warning(
            f"Unable to inspect pending verification prompt for cleanup: {error}"
        )
        return
    if (
        data.get("prompt") == expected_prompt
        and data.get("context_label") == expected_context_label
    ):
        try:
            LAST_VERIFICATION_PROMPT_PATH.unlink()
        except OSError as error:
            logger.warning(
                f"Failed to delete pending verification prompt file: {error}"
            )


def build_verification_prompt(
    chunk_text: str,
    updated_body: str,
    context_label: str | None = None,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> str:
    response_instructions = (
        "Report whether any note content is missing or materially altered."
        " Respond with a concise single paragraph beginning with 'OK -' if everything is covered"
        " or 'MISSING -' followed by details of any omissions."
        ' For each omission, include a sequence such as Notes:"..." Body:"..." Explanation: ... .'
        ' Quote the exact text from the notes chunk containing the missing detail and quote the exact passage from the updated document body that should cover it (or state Body:"<not present>" if nothing is relevant).'
        " Explain precisely what information is still missing or altered without omitting any nuance."
    )

    sections = [
        (
            "<task>"
            "You are verifying that every idea/point/concept/argument/detail/url/[[wikilink]]/diagram etc. "
            "from the provided notes chunk has been integrated into the updated document body."
            "</task>"
        ),
        f"<notes_chunk>\n{chunk_text}\n</notes_chunk>",
        f"<updated_document_body>\n{updated_body}\n</updated_document_body>",
        f"<response_guidelines>\n{response_instructions}\n</response_guidelines>",
    ]
    prompt = "\n\n\n\n\n".join(sections)
    persist_pending_verification_prompt(
        prompt,
        context_label,
        chunk_index,
        total_chunks,
    )
    if context_label:
        logger.info(f"Verification prompt for {context_label}:\n{prompt}")
    else:
        logger.info(f"Verification prompt:\n{prompt}")
    return prompt


def request_verification(client: OpenAI, prompt: str, context_label: str) -> str:
    def perform_request() -> str:
        response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "medium"},
            input=prompt,
        )
        output_text = response.output_text
        if not output_text.strip():
            raise RuntimeError("Received empty response from GPT verification call.")
        return output_text.strip()

    return execute_with_retry(perform_request, f"verification {context_label}")


def resume_pending_verification_if_needed(
    client: OpenAI, pending_data: dict | None = None
) -> None:
    if pending_data is None:
        pending_data = load_pending_verification_prompt()
    if not pending_data:
        return

    prompt = pending_data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError(
            f"Pending verification prompt file {LAST_VERIFICATION_PROMPT_PATH} is missing the prompt text."
        )
    context_label = pending_data.get("context_label")
    chunk_index = pending_data.get("chunk_index")
    total_chunks = pending_data.get("total_chunks")

    display_context = context_label or "pending verification"
    logger.info(
        f"Replaying pending verification prompt from {LAST_VERIFICATION_PROMPT_PATH} for {display_context}."
    )
    assessment = request_verification(client, prompt, display_context)

    if (
        isinstance(chunk_index, int)
        and isinstance(total_chunks, int)
        and 0 <= chunk_index < total_chunks
    ):
        if "MISSING" in assessment:
            notify_missing_verification(chunk_index, total_chunks, assessment)
        logger.info(
            f"Verification chunk {chunk_index + 1}/{total_chunks}: {assessment}"
        )
    else:
        logger.info(f"Resumed verification {display_context}: {assessment}")

    clear_pending_verification_prompt(prompt, context_label)


def drain_completed_verifications(
    tasks: List[Tuple[Future[str], int, str, str]],
    total_chunks: int,
    wait_for_all: bool = False,
) -> List[Tuple[Future[str], int, str, str]]:
    pending: List[Tuple[Future[str], int, str, str]] = []
    for future, chunk_index, prompt, context_label in tasks:
        if wait_for_all or future.done():
            try:
                assessment = future.result()
                if "MISSING" in assessment:
                    notify_missing_verification(chunk_index, total_chunks, assessment)
                logger.info(
                    f"Verification chunk {chunk_index + 1}/{total_chunks}: {assessment}"
                )
                clear_pending_verification_prompt(prompt, context_label)
            except Exception as error:
                logger.exception(
                    f"Verification chunk {chunk_index + 1}/{total_chunks} failed: {error}"
                )
        else:
            pending.append((future, chunk_index, prompt, context_label))
    return pending


def integrate_notes(
    source_path: Path,
    grouping: str | None,
    max_paragraphs_per_chunk: int,
    max_words_per_chunk: int,
) -> Path:
    integrated_path, newly_created = ensure_integrated_copy(source_path)
    source_content = source_path.read_text(encoding="utf-8")
    source_body, source_scratchpad = split_document_sections(source_content)
    working_content = integrated_path.read_text(encoding="utf-8")
    working_body, working_scratchpad = split_document_sections(working_content)

    resolved_grouping = (
        grouping or extract_grouping(working_body) or extract_grouping(source_body)
    )
    if not resolved_grouping:
        resolved_grouping = prompt_for_grouping()
        logger.info("Recorded new grouping approach from user input.")

    working_body = ensure_grouping_record(working_body, resolved_grouping)

    if newly_created:
        scratchpad_source = source_scratchpad
    else:
        scratchpad_source = working_scratchpad
        if not scratchpad_source.strip() and source_scratchpad.strip():
            logger.info(
                "Working copy scratchpad is empty; new notes in the source document will need to be synced manually."
            )
    scratchpad_paragraphs = normalize_paragraphs(scratchpad_source)
    pending_verification_data = load_pending_verification_prompt()
    if not newly_created and working_scratchpad.strip():
        logger.info(
            f"Resuming integration with {len(scratchpad_paragraphs)} scratchpad paragraphs remaining."
        )
    if not scratchpad_paragraphs and not pending_verification_data:
        logger.info(
            "No scratchpad notes to integrate; ensuring scratchpad heading remains present."
        )
        integrated_path.write_text(build_document(working_body, []), encoding="utf-8")
        return integrated_path

    paragraph_chunks = chunk_paragraphs(
        scratchpad_paragraphs,
        max_paragraphs_per_chunk,
        max_words_per_chunk,
    )
    total_chunks = len(paragraph_chunks)

    client = create_openai_client()

    if pending_verification_data:
        resume_pending_verification_if_needed(client, pending_verification_data)

    if not scratchpad_paragraphs:
        logger.info(
            "No scratchpad notes to integrate; ensuring scratchpad heading remains present."
        )
        integrated_path.write_text(build_document(working_body, []), encoding="utf-8")
        return integrated_path

    current_body = working_body
    remaining_paragraphs = scratchpad_paragraphs.copy()
    verification_tasks: List[Tuple[Future[str], int, str, str]] = []
    integration_start = perf_counter()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for chunk_index, chunk in enumerate(paragraph_chunks):
            verification_tasks = drain_completed_verifications(
                verification_tasks, total_chunks
            )

            chunk_text = "\n\n".join(chunk)
            chunk_word_count = sum(count_words(paragraph) for paragraph in chunk)
            prompt = build_integration_prompt(
                resolved_grouping,
                current_body,
                chunk_text,
            )
            chunk_label = f"chunk {chunk_index + 1}/{total_chunks}"
            logger.info(
                f"Integrating chunk {chunk_index + 1} of {total_chunks} containing {len(chunk)} paragraphs and {chunk_word_count} words."
            )
            updated_body = request_integration(client, prompt, chunk_label)
            verification_prompt = build_verification_prompt(
                chunk_text,
                updated_body,
                chunk_label,
                chunk_index,
                total_chunks,
            )
            verification_future = executor.submit(
                request_verification, client, verification_prompt, chunk_label
            )
            verification_tasks.append(
                (verification_future, chunk_index, verification_prompt, chunk_label)
            )

            current_body = updated_body
            remaining_paragraphs = remaining_paragraphs[len(chunk) :]
            integrated_document = build_document(current_body, remaining_paragraphs)
            integrated_path.write_text(integrated_document, encoding="utf-8")
            logger.info(
                f'Chunk {chunk_index + 1} integration written to "{integrated_path}".'
            )
            chunks_completed = chunk_index + 1
            remaining_chunks = total_chunks - chunks_completed
            if remaining_chunks > 0:
                elapsed_seconds = perf_counter() - integration_start
                average_duration = elapsed_seconds / chunks_completed
                estimated_seconds_remaining = average_duration * remaining_chunks
                logger.info(
                    f"Estimated time remaining: {format_duration(estimated_seconds_remaining)}"
                    f" for {remaining_chunks} remaining chunk(s)."
                )
            verification_tasks = drain_completed_verifications(
                verification_tasks, total_chunks
            )

        verification_tasks = drain_completed_verifications(
            verification_tasks, total_chunks, wait_for_all=True
        )

    logger.info("All scratchpad notes integrated; scratchpad section cleared.")
    return integrated_path


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
