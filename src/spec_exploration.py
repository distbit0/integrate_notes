from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from spec_config import MAX_TOOL_ATTEMPTS, SpecConfig
from spec_llm import parse_tool_call_arguments, request_tool_call
from spec_notes import NoteRepository, ViewedNote


VIEW_TOOL_SCHEMA = {
    "type": "function",
    "name": "view_files",
    "description": "Request additional files to view.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["view"]},
            "files": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action", "files"],
    },
}

CHECKOUT_TOOL_SCHEMA = {
    "type": "function",
    "name": "checkout_files",
    "description": "Select viewed files to check out for editing.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["checkout"]},
            "files": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action", "files"],
    },
}


@dataclass(frozen=True)
class ExplorationState:
    viewed: Dict[Path, ViewedNote]
    available: Dict[Path, str]


class ExplorationError(RuntimeError):
    pass


def format_viewed_note(note: ViewedNote) -> str:
    headings = "\n".join(f"- {heading}" for heading in note.headings) or "- <no headings>"
    links = (
        "\n".join(f"- [[{path.name}]] — {summary}" for path, summary in note.link_summaries)
        or "- <no links>"
    )
    return (
        f"## [{note.path.name}]\n\n"
        f"**Summary:** {note.summary}\n\n"
        f"**Headings:**\n{headings}\n\n"
        f"**Links to:**\n{links}"
    )


def format_available_note(path: Path, summary: str) -> str:
    return f"- [[{path.name}]] — {summary}"


def build_exploration_prompt(
    chunk_text: str,
    viewed_notes: Iterable[ViewedNote],
    available_notes: Iterable[Tuple[Path, str]],
    remaining_rounds: int,
    config: SpecConfig,
    feedback: str | None = None,
) -> str:
    viewed_blocks = [format_viewed_note(note) for note in viewed_notes]
    available_blocks = [format_available_note(path, summary) for path, summary in available_notes]

    instructions = (
        "You are exploring notes to decide which files to view next or to checkout. "
        "Respond with a tool call to view_files selecting up to "
        f"{config.max_files_viewed_per_round} AVAILABLE files, or call checkout_files "
        "to select up to {max_checkout} VIEWED files for editing. "
        "Only choose files from the provided lists."
    ).format(max_checkout=config.max_files_checked_out)

    prompt = (
        "<instructions>\n"
        f"{instructions}\n"
        "</instructions>\n\n"
        "<notes_chunk>\n"
        f"{chunk_text}\n"
        "</notes_chunk>\n\n"
        "<viewed_files>\n"
        f"{'\n\n'.join(viewed_blocks) if viewed_blocks else '<none>'}\n"
        "</viewed_files>\n\n"
        "<available_files>\n"
        f"{'\n'.join(available_blocks) if available_blocks else '<none>'}\n"
        "</available_files>\n\n"
        f"<remaining_rounds>{remaining_rounds}</remaining_rounds>"
    )
    if feedback:
        prompt += f"\n\n<previous_attempt_feedback>\n{feedback}\n</previous_attempt_feedback>"
    return prompt


def _parse_file_list(payload: dict, action: str) -> List[str]:
    if payload.get("action") != action:
        raise ExplorationError(f"Tool payload must include action='{action}'.")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ExplorationError("Tool payload must include a files list.")
    file_names: List[str] = []
    for value in files:
        if not isinstance(value, str) or not value.strip():
            raise ExplorationError("Each file entry must be a non-empty string.")
        file_names.append(value.strip())
    return file_names


def _resolve_requested_paths(
    names: Iterable[str],
    mapping: Dict[str, Path],
    label: str,
) -> List[Path]:
    resolved: List[Path] = []
    for name in names:
        key = name.lower()
        path = mapping.get(key)
        if path is None:
            raise ExplorationError(f"Requested {label} file '{name}' is not available.")
        resolved.append(path)
    return resolved


def explore_until_checkout(
    client,
    chunk_text: str,
    root_path: Path,
    root_summary: str,
    root_headings: List[str],
    root_links: List[Path],
    root_link_summaries: List[tuple[Path, str]],
    summary_map: Dict[Path, str],
    repo: NoteRepository,
    config: SpecConfig,
) -> List[Path]:
    viewed: Dict[Path, ViewedNote] = {}
    available: Dict[Path, str] = {}

    viewed[root_path] = ViewedNote(
        path=root_path,
        summary=root_summary,
        headings=root_headings,
        links=root_links,
        link_summaries=root_link_summaries,
    )
    for path in root_links:
        if path not in viewed and path in summary_map:
            available[path] = summary_map[path]

    rounds_left = config.max_exploration_rounds
    total_viewed_limit = config.max_files_viewed_total

    while rounds_left > 0:
        needs_checkout = len(viewed) >= total_viewed_limit or not available
        feedback = None
        attempts_left = MAX_TOOL_ATTEMPTS

        while attempts_left > 0:
            prompt = build_exploration_prompt(
                chunk_text,
                viewed.values(),
                available.items(),
                rounds_left,
                config,
                feedback=feedback,
            )
            tools = [CHECKOUT_TOOL_SCHEMA] if needs_checkout else [VIEW_TOOL_SCHEMA, CHECKOUT_TOOL_SCHEMA]
            tool_call = request_tool_call(
                client,
                prompt,
                tools,
                f"exploration round {config.max_exploration_rounds - rounds_left + 1}",
            )
            payload = parse_tool_call_arguments(tool_call)
            try:
                if tool_call.name == "checkout_files":
                    requested = _parse_file_list(payload, "checkout")
                    if len(requested) > config.max_files_checked_out:
                        raise ExplorationError(
                            "Checkout request exceeds max files allowed."
                        )
                    view_map = {path.name.lower(): path for path in viewed.keys()}
                    checkout_paths = _resolve_requested_paths(
                        requested, view_map, "viewed"
                    )
                    if not checkout_paths:
                        raise ExplorationError("Checkout request must include at least one file.")
                    return checkout_paths

                if needs_checkout:
                    raise ExplorationError(
                        "No additional files are available to view; you must checkout."
                    )
                requested = _parse_file_list(payload, "view")
                if len(requested) > config.max_files_viewed_per_round:
                    raise ExplorationError("View request exceeds max files allowed.")
                available_map = {path.name.lower(): path for path in available.keys()}
                requested_paths = _resolve_requested_paths(
                    requested, available_map, "available"
                )
            except ExplorationError as error:
                feedback = str(error)
                attempts_left -= 1
                if attempts_left == 0:
                    raise
                continue

            for path in requested_paths:
                summary = available.pop(path)
                headings = repo.get_headings(path)
                links = repo.get_links(path)
                link_summaries = [
                    (link_path, summary_map[link_path])
                    for link_path in links
                    if link_path in summary_map
                ]
                viewed[path] = ViewedNote(
                    path=path,
                    summary=summary,
                    headings=headings,
                    links=links,
                    link_summaries=link_summaries,
                )
                for link_path in links:
                    if link_path not in viewed and link_path in summary_map:
                        available[link_path] = summary_map[link_path]

                if len(viewed) >= total_viewed_limit:
                    break

            rounds_left -= 1
            break

    feedback = None
    attempts_left = MAX_TOOL_ATTEMPTS
    while attempts_left > 0:
        prompt = build_exploration_prompt(
            chunk_text,
            viewed.values(),
            available.items(),
            rounds_left,
            config,
            feedback=feedback,
        )
        tool_call = request_tool_call(
            client,
            prompt,
            [CHECKOUT_TOOL_SCHEMA],
            "exploration checkout",
        )
        payload = parse_tool_call_arguments(tool_call)
        try:
            requested = _parse_file_list(payload, "checkout")
            if len(requested) > config.max_files_checked_out:
                raise ExplorationError("Checkout request exceeds max files allowed.")
            view_map = {path.name.lower(): path for path in viewed.keys()}
            checkout_paths = _resolve_requested_paths(requested, view_map, "viewed")
            if not checkout_paths:
                raise ExplorationError("Checkout request must include at least one file.")
            return checkout_paths
        except ExplorationError as error:
            feedback = str(error)
            attempts_left -= 1
            if attempts_left == 0:
                raise

    raise ExplorationError("Unable to select checkout files.")
