from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from time import perf_counter

from loguru import logger

from spec_chunking import request_chunk_groups
from spec_config import (
    SCRATCHPAD_HEADING,
    SpecConfig,
    default_log_path,
    load_config,
    repo_root,
)
from spec_editing import request_and_apply_edits
from spec_exploration import explore_until_checkout
from spec_llm import create_openai_client
from spec_logging import configure_logging
from spec_markdown import (
    build_document,
    format_duration,
    normalize_paragraphs,
    split_document_sections,
)
from spec_notes import NoteRepository
from spec_summary import SummaryService
from spec_verification import VerificationManager, build_verification_prompt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate scratchpad notes into a markdown repository (SPEC flow)."
    )
    parser.add_argument(
        "--source", required=False, help="Path to the root markdown document."
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
        user_input = input("Enter path to the root markdown document: ").strip()
        if not user_input:
            raise ValueError("Document path is required to proceed.")
        path = Path(user_input).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source document not found at {path}.")
    if not path.is_file():
        raise ValueError(f"Source path {path} is not a file.")
    return path


def _select_sample_filenames(
    repo: NoteRepository, reachable: list[Path], config: SpecConfig
) -> list[str]:
    candidates = []
    for path in reachable:
        if repo.is_index_note(path, config.index_filename_suffix):
            continue
        if repo.get_word_count(path) < config.granularity_sample_min_words:
            continue
        candidates.append(path.name)

    if not candidates:
        return []

    if len(candidates) <= config.granularity_sample_size:
        return candidates

    return random.sample(candidates, config.granularity_sample_size)


def _ensure_scratchpad_matches(
    source_path: Path, expected_paragraphs: list[str]
) -> tuple[str, list[str]]:
    content = source_path.read_text(encoding="utf-8")
    body, scratchpad = split_document_sections(content, SCRATCHPAD_HEADING)
    paragraphs = normalize_paragraphs(scratchpad)
    if paragraphs != expected_paragraphs:
        raise RuntimeError(
            "Scratchpad changed while integration was running; aborting to avoid data loss."
        )
    return body, paragraphs


def _write_updated_files(
    source_path: Path,
    root_body: str,
    remaining_paragraphs: list[str],
    updated_files: dict[Path, str],
    repo: NoteRepository,
    summaries: SummaryService,
) -> None:
    for path, content in updated_files.items():
        if path == source_path:
            root_body = content
        else:
            path.write_text(content, encoding="utf-8")
            repo.invalidate_content(path)
            summaries.invalidate(path)

    document = build_document(root_body, SCRATCHPAD_HEADING, remaining_paragraphs)
    source_path.write_text(document, encoding="utf-8")
    repo.set_root_body(root_body)


def integrate_notes_spec(source_path: Path, disable_verification: bool) -> Path:
    config = load_config(repo_root() / "config.json")
    source_content = source_path.read_text(encoding="utf-8")
    source_body, source_scratchpad = split_document_sections(
        source_content, SCRATCHPAD_HEADING
    )
    scratchpad_paragraphs = normalize_paragraphs(source_scratchpad)

    repo = NoteRepository(source_path, source_body, source_path.parent)
    client = create_openai_client()
    summaries = SummaryService(repo, client, config)
    verification_manager = (
        None if disable_verification else VerificationManager(client, source_path)
    )

    try:
        if not scratchpad_paragraphs:
            logger.info(
                "No scratchpad notes to integrate; ensuring scratchpad heading remains present."
            )
            source_path.write_text(
                build_document(source_body, SCRATCHPAD_HEADING, []),
                encoding="utf-8",
            )
            return source_path

        reachable = repo.iter_reachable_paths()
        sample_filenames = _select_sample_filenames(repo, reachable, config)
        chunk_groups = request_chunk_groups(
            client, scratchpad_paragraphs, sample_filenames, config
        )

        remaining_indices = set(range(1, len(scratchpad_paragraphs) + 1))
        total_chunks = len(chunk_groups)
        chunks_completed = 0
        integration_start = perf_counter()
        current_body = source_body

        for group in chunk_groups:
            if any(index not in remaining_indices for index in group):
                raise RuntimeError(
                    "Chunk references paragraphs that were already integrated; aborting."
                )
            chunk_paragraphs = [scratchpad_paragraphs[index - 1] for index in group]
            chunk_text = "\n\n".join(chunk_paragraphs)

            expected_remaining = [
                scratchpad_paragraphs[index - 1]
                for index in sorted(remaining_indices)
            ]
            file_body, _ = _ensure_scratchpad_matches(
                source_path, expected_remaining
            )
            if file_body != current_body:
                raise RuntimeError(
                    "Root document body changed while integration was running; aborting."
                )

            repo.set_root_body(current_body)
            reachable = repo.iter_reachable_paths()
            summary_map = summaries.get_summaries(reachable)

            root_summary = summary_map[source_path]
            root_headings = repo.get_headings(source_path)
            root_links = repo.get_links(source_path)
            root_link_summaries = [
                (path, summary_map[path])
                for path in root_links
                if path in summary_map
            ]

            chunk_label = f"chunk {chunks_completed + 1}/{total_chunks}"
            checkout_paths = explore_until_checkout(
                client,
                chunk_text,
                source_path,
                root_summary,
                root_headings,
                root_links,
                root_link_summaries,
                summary_map,
                repo,
                config,
            )

            checked_out_contents = {
                path: repo.get_note_content(path) for path in checkout_paths
            }
            edit_application = request_and_apply_edits(
                client,
                chunk_text,
                checked_out_contents,
                checkout_paths,
                chunk_label,
            )

            for path, content in edit_application.updated_contents.items():
                if path == source_path:
                    current_body = content
            for path in edit_application.updated_contents:
                if path != source_path:
                    repo.invalidate_content(path)

            for index in group:
                remaining_indices.remove(index)
            remaining_paragraphs = [
                scratchpad_paragraphs[index - 1]
                for index in sorted(remaining_indices)
            ]

            _write_updated_files(
                source_path,
                current_body,
                remaining_paragraphs,
                edit_application.updated_contents,
                repo,
                summaries,
            )

            if verification_manager is not None:
                verification_prompt = build_verification_prompt(
                    chunk_text,
                    edit_application.patch_replacements,
                    edit_application.duplicate_texts,
                )
                verification_manager.enqueue_prompt(
                    verification_prompt,
                    chunk_label,
                    chunks_completed,
                    total_chunks,
                )

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
        summaries.shutdown()
        if verification_manager is not None:
            verification_manager.shutdown()


def main() -> None:
    configure_logging(default_log_path())
    try:
        args = parse_arguments()
        source_path = resolve_source_path(args.source)
        integrated_path = integrate_notes_spec(
            source_path,
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
