import argparse
import os
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable, Iterable, List, Tuple

from loguru import logger
from openai import OpenAI

SCRATCHPAD_HEADING = "# -- SCRATCHPAD"
GROUPING_PREFIX = "Grouping approach: "
DEFAULT_CHUNK_PARAGRAPHS = 15
ENV_API_KEY = "OPENAI_API_KEY"
DEFAULT_MAX_RETRIES = 3
RETRY_INITIAL_DELAY_SECONDS = 2.0
RETRY_BACKOFF_FACTOR = 2.0
INSTRUCTIONS_PROMPT = """# Instructions

- Integrate the provided notes into the document body, following the document's existing structure and the recorded grouping approach.
- Break content into relatively atomic bullet points; each bullet should express one idea.
- Use nested bullets when a point is naturally a sub-point of another.
- Make minor grammar edits as needed so ideas read cleanly as bullet points.
- Do not convert notes into bullets "programmatically"; use your judgment for each note.
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
        help="Number of paragraphs per scratchpad chunk integration request.",
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


def iter_chunks(items: List[str], chunk_size: int) -> Iterable[Tuple[int, List[str]]]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    total = len(items)
    for start in range(0, total, chunk_size):
        index = start // chunk_size
        yield index, items[start : start + chunk_size]


def create_openai_client() -> OpenAI:
    api_key = os.getenv(ENV_API_KEY)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {ENV_API_KEY} is required for GPT access."
        )
    return OpenAI(api_key=api_key)


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
    chunk_number: int,
    total_chunks: int,
    original_filename: str,
) -> str:
    clarifications = (
        "You are integrating notes into the main body of the document incrementally. "
        "For this request, integrate the provided chunk of notes and return only the updated document body text. "
        f"Maintain the grouping approach: {grouping}. "
    )
    payload = (
        f"{INSTRUCTIONS_PROMPT}\n\n"
        f"{clarifications}\n\n"
        "Current document body:\n"
        f"{current_body}\n\n"
        "Scratchpad chunk to integrate:\n"
        f"{chunk_text}\n\n"
        "Return only the updated document body."
    )
    return payload


def request_integration(client: OpenAI, prompt: str, context_label: str) -> str:
    def perform_request() -> str:
        response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "high"},
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


def build_verification_prompt(
    original_filename: str,
    chunk_number: int,
    total_chunks: int,
    chunk_text: str,
    updated_body: str,
) -> str:
    return (
        "You are verifying that every idea/point/concept/argument/detail from the provided notes was integrated into the updated document body.\n"
        "Notes chunk:\n"
        f"{chunk_text}\n\n"
        "Updated document body after integration:\n"
        f"{updated_body}\n\n"
        "Report whether any note content is missing or materially altered."
        " Respond with a concise single paragraph beginning with 'OK -' if everything is covered"
        " or 'MISSING -' followed by details of any omissions."
        ' For each omission, include a sequence such as Notes:"..." Body:"..." Explanation: ... .'
        ' Quote the exact text from the notes chunk containing the missing detail and quote the exact passage from the updated document body that should cover it (or state Body:"<not present>" if nothing is relevant).'
        " Explain precisely what information is still missing or altered without omitting any nuance."
    )


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


def drain_completed_verifications(
    tasks: List[Tuple[Future[str], int]],
    total_chunks: int,
    wait_for_all: bool = False,
) -> List[Tuple[Future[str], int]]:
    pending: List[Tuple[Future[str], int]] = []
    for future, chunk_index in tasks:
        if wait_for_all or future.done():
            try:
                assessment = future.result()
                logger.info(
                    f"Verification chunk {chunk_index + 1}/{total_chunks}: {assessment}"
                )
            except Exception as error:
                logger.exception(
                    f"Verification chunk {chunk_index + 1}/{total_chunks} failed: {error}"
                )
        else:
            pending.append((future, chunk_index))
    return pending


def integrate_notes(source_path: Path, grouping: str | None, chunk_size: int) -> Path:
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
    if not newly_created and working_scratchpad.strip():
        logger.info(
            f"Resuming integration with {len(scratchpad_paragraphs)} scratchpad paragraphs remaining."
        )
    if not scratchpad_paragraphs:
        logger.info(
            "No scratchpad notes to integrate; ensuring scratchpad heading remains present."
        )
        integrated_path.write_text(build_document(working_body, []), encoding="utf-8")
        return integrated_path

    client = create_openai_client()
    total_chunks = (len(scratchpad_paragraphs) + chunk_size - 1) // chunk_size
    original_filename = source_path.name

    current_body = working_body
    remaining_paragraphs = scratchpad_paragraphs.copy()
    verification_tasks: List[Tuple[Future[str], int]] = []
    integration_start = perf_counter()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for chunk_index, chunk in iter_chunks(scratchpad_paragraphs, chunk_size):
            verification_tasks = drain_completed_verifications(
                verification_tasks, total_chunks
            )

            chunk_text = "\n\n".join(chunk)
            prompt = build_integration_prompt(
                resolved_grouping,
                current_body,
                chunk_text,
                chunk_index,
                total_chunks,
                original_filename,
            )
            chunk_label = f"chunk {chunk_index + 1}/{total_chunks}"
            logger.info(
                f"Integrating chunk {chunk_index + 1} of {total_chunks} containing {len(chunk)} paragraphs."
            )
            updated_body = request_integration(client, prompt, chunk_label)
            verification_prompt = build_verification_prompt(
                original_filename,
                chunk_index,
                total_chunks,
                chunk_text,
                updated_body,
            )
            verification_future = executor.submit(
                request_verification, client, verification_prompt, chunk_label
            )
            verification_tasks.append((verification_future, chunk_index))

            current_body = updated_body
            remaining_paragraphs = remaining_paragraphs[len(chunk) :]
            integrated_document = build_document(current_body, remaining_paragraphs)
            integrated_path.write_text(integrated_document, encoding="utf-8")
            logger.info(
                f"Chunk {chunk_index + 1} integration written to {integrated_path}."
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
        integrated_path = integrate_notes(source_path, args.grouping, args.chunk_size)
        logger.info(
            f"Integration completed. Updated document available at {integrated_path}."
        )
    except Exception as error:
        logger.exception(f"Integration failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
