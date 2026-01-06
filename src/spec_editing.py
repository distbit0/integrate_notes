from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from loguru import logger

from spec_config import MAX_PATCH_ATTEMPTS
from spec_llm import parse_tool_call_arguments, request_tool_call


EDIT_TOOL_SCHEMA = {
    "type": "function",
    "name": "edit_notes",
    "description": "Provide find/replace edits for checked-out files.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["edit"]},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "find": {"type": "string"},
                        "replace": {"type": "string"},
                        "is_duplicate": {"type": "boolean"},
                    },
                    "required": ["file", "find", "is_duplicate"],
                },
            },
        },
        "required": ["action", "edits"],
    },
}

INSTRUCTIONS_PROMPT = """# Instructions

- Integrate the provided notes into the checked-out files.
- Ensure related points are adjacent.
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
- If new items do not match existing items in the checked-out files, add them appropriately.
- Preserve questions as questions; do not convert them into statements.
- Do not guess acronym expansions if they are not specified.
- Do not modify tone (e.g., confidence/certainty) or add hedging.
- Do not omit any wikilinks, URLs, diagrams, ASCII art, mathematics, tables, figures, or other non-text content.
- Move each link/URL/etc. to the section where it is most relevant based on its surrounding context and its URL text.
    - Do not move links to a separate "resources" or "links" section.
- Do not modify any wikilinks or URLs.


# Formatting

- Use nested markdown headings ("#", "##", "###", "####", etc.) for denoting groups and sub-groups, except if heading text is a [[wikilink]].
    - unless existing content already employs a different convention.
- Use "- " as the bullet prefix (not "* ", "-  ", or anything else).
    - Use four spaces for each level of bullet-point nesting.


# Before finishing: check your work

- Confirm every item from the provided notes is now represented in the checked-out files without loss of detail.
- Ensure nothing from the original checked-out files was lost.
- If anything is missing, integrate it in appropriately.
"""


@dataclass(frozen=True)
class EditInstruction:
    file_path: Path
    find_text: str
    replace_text: str | None
    is_duplicate: bool


@dataclass(frozen=True)
class EditFailure:
    index: int
    file_path: Path
    find_text: str
    reason: str


@dataclass(frozen=True)
class EditApplication:
    updated_contents: Dict[Path, str]
    patch_replacements: List[str]
    duplicate_texts: List[str]


class EditParseError(RuntimeError):
    pass


def build_edit_prompt(
    chunk_text: str,
    checked_out_contents: Dict[Path, str],
    failed_edits: List[EditFailure] | None = None,
    failed_formatting: str | None = None,
    previous_response: str | None = None,
) -> str:
    file_sections = []
    for path, content in checked_out_contents.items():
        file_sections.append(f"## [{path.name}]\n\n{content}")

    instructions = (
        "You are integrating the notes chunk into the checked-out files. "
        "Return only a tool call to edit_notes with edits targeting the listed files. "
        "Use is_duplicate=true only when the notes are already fully covered by existing text. "
        "For edits, 'find' must be a single contiguous span copied from the file content. "
        "For insertions, include the anchor text in both find and replace. "
        "Do not include any commentary or additional text."
    )

    prompt = (
        "<instructions>\n"
        f"{instructions}\n\n{INSTRUCTIONS_PROMPT.strip()}\n"
        "</instructions>\n\n"
        "<notes_chunk>\n"
        f"{chunk_text}\n"
        "</notes_chunk>\n\n"
        "<checked_out_files>\n"
        f"{'\n\n'.join(file_sections)}\n"
        "</checked_out_files>"
    )

    if failed_formatting or failed_edits:
        feedback_lines: List[str] = []
        if failed_formatting:
            feedback_lines.append(
                "The previous response could not be parsed. Fix the issues below and re-emit a valid tool call."
            )
            feedback_lines.append(f"Error: {failed_formatting}")
        if failed_edits:
            feedback_lines.append(
                "The previous edits failed to match the current file contents. Adjust only the failing edits."
            )
            for failure in failed_edits:
                feedback_lines.append(
                    f"Edit {failure.index} ({failure.file_path.name}) find text must match exactly once."
                )
                feedback_lines.append(failure.find_text)
                feedback_lines.append(f"Reason: {failure.reason}")
        prompt += (
            "\n\n<previous_attempt_feedback>\n"
            + "\n\n".join(feedback_lines)
            + "\n</previous_attempt_feedback>"
        )

    if previous_response:
        prompt += (
            "\n\n<previous_edit_response>\n"
            + previous_response
            + "\n</previous_edit_response>"
        )

    return prompt


def parse_edit_instructions(
    payload: dict,
    checked_out_paths: Iterable[Path],
) -> List[EditInstruction]:
    action = payload.get("action")
    if action != "edit":
        raise EditParseError("Edit tool payload must include action='edit'.")

    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        raise EditParseError("Edit tool payload must include a non-empty edits list.")

    checked_out_map = {path.name.lower(): path for path in checked_out_paths}
    instructions: List[EditInstruction] = []

    for edit in edits:
        if not isinstance(edit, dict):
            raise EditParseError("Each edit must be an object.")
        file_name = edit.get("file")
        if not isinstance(file_name, str) or not file_name.strip():
            raise EditParseError("Each edit must include a non-empty file name.")
        path = checked_out_map.get(file_name.strip().lower())
        if path is None:
            raise EditParseError(
                f"Edit file '{file_name}' is not in the checked-out file list."
            )
        find_text = edit.get("find")
        if not isinstance(find_text, str) or not find_text.strip():
            raise EditParseError("Each edit must include non-empty find text.")
        is_duplicate = edit.get("is_duplicate")
        if not isinstance(is_duplicate, bool):
            raise EditParseError("Each edit must include a boolean is_duplicate flag.")
        replace_text = edit.get("replace")
        if is_duplicate:
            replace_text = None
        else:
            if not isinstance(replace_text, str):
                raise EditParseError(
                    "Non-duplicate edits must include a string replace value."
                )
        instructions.append(
            EditInstruction(
                file_path=path,
                find_text=find_text,
                replace_text=replace_text,
                is_duplicate=is_duplicate,
            )
        )

    return instructions


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


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


def _locate_search_text(body: str, search_text: str) -> tuple[int | None, int | None, str]:
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

    reason = "SEARCH text not found after attempts: " + ", ".join(attempted_descriptions)
    return None, None, reason


def _replace_slice(body: str, start: int, end: int, replacement: str) -> str:
    return body[:start] + replacement + body[end:]


def apply_edits(
    file_contents: Dict[Path, str],
    edits: List[EditInstruction],
) -> tuple[EditApplication | None, List[EditFailure]]:
    updated_contents = {path: _normalize_line_endings(content) for path, content in file_contents.items()}
    failures: List[EditFailure] = []
    patch_replacements: List[str] = []
    duplicate_texts: List[str] = []

    for index, edit in enumerate(edits, start=1):
        content = updated_contents[edit.file_path]
        start, end, reason = _locate_search_text(content, edit.find_text)
        if start is None or end is None:
            failures.append(
                EditFailure(
                    index=index,
                    file_path=edit.file_path,
                    find_text=edit.find_text,
                    reason=reason,
                )
            )
            continue
        if edit.is_duplicate:
            duplicate_texts.append(edit.find_text)
            continue
        replacement = edit.replace_text or ""
        updated_contents[edit.file_path] = _replace_slice(content, start, end, replacement)
        patch_replacements.append(replacement)

    if failures:
        return None, failures

    return EditApplication(updated_contents, patch_replacements, duplicate_texts), []


def request_and_apply_edits(
    client,
    chunk_text: str,
    checked_out_contents: Dict[Path, str],
    checked_out_paths: Iterable[Path],
    context_label: str,
) -> EditApplication:
    failed_edits: List[EditFailure] | None = None
    failed_formatting: str | None = None
    previous_response: str | None = None

    for attempt in range(1, MAX_PATCH_ATTEMPTS + 1):
        attempt_label = (
            context_label if attempt == 1 else f"{context_label} attempt {attempt}"
        )
        prompt = build_edit_prompt(
            chunk_text,
            checked_out_contents,
            failed_edits=failed_edits,
            failed_formatting=failed_formatting,
            previous_response=previous_response,
        )

        tool_call = request_tool_call(
            client, prompt, [EDIT_TOOL_SCHEMA], f"edit {attempt_label}"
        )
        previous_response = tool_call.arguments

        try:
            payload = parse_tool_call_arguments(tool_call)
            edit_instructions = parse_edit_instructions(payload, checked_out_paths)
        except Exception as error:  # noqa: BLE001
            failed_formatting = str(error)
            failed_edits = None
            logger.warning(
                f"Edit response invalid for {attempt_label}: {error}"
            )
            continue

        failed_formatting = None
        application, failures = apply_edits(checked_out_contents, edit_instructions)
        if not failures:
            return application
        failed_edits = failures
        logger.info(
            f"Retrying {context_label}; {len(failed_edits)} edit(s) failed to match."
        )

    raise RuntimeError(
        f"Unable to apply edits for {context_label} after {MAX_PATCH_ATTEMPTS} attempt(s)."
    )
